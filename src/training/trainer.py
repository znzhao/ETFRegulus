"""Assembling and running one curriculum stage.

PPO itself comes from SB3 (D2) -- what lives here is everything around it: building the
vectorized environment for a stage, warm-starting from the previous stage's checkpoint,
wiring the diagnostics, and running the gate afterwards.

The warm-start is the part with a sharp edge. Loading a policy across stages is only sound
because the observation space is identical at every rung: the risk-envelope toggle and the
lock range change the environment's *dynamics*, not its observation. `warm_start` checks
that rather than trusting it, because the failure mode is a silently reinitialized network
that looks like a stage which simply learned nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from src.agents.normalization import load_normalizer, save_bundle, wrap_normalizer
from src.agents.policy import SharedAssetActorCritic, count_parameters, policy_kwargs_from
from src.env.etf_env import EnvConfig
from src.env.factory import EnvBundle, make_vec_env
from src.training.callbacks import (
    BaselineReferenceCallback,
    ConstraintMonitor,
    DiagnosticCallback,
    EntropyGuard,
)
from src.training.curriculum import CurriculumStage, evaluate_gate

DEFAULT_PPO = dict(
    n_steps=2048, batch_size=4096, n_epochs=10, gamma=0.999, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
    learning_rate=3.0e-4, target_kl=0.02,
)


@dataclass
class StageOutcome:
    stage: int
    name: str
    run_dir: Path
    timesteps: int
    diagnostics: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    wall_seconds: float = 0.0
    n_parameters: int = 0

    @property
    def passed(self) -> bool:
        return bool(self.gate.get("passed"))


def stage_env_config(bundle: EnvBundle, stage: CurriculumStage,
                     overrides: dict | None = None) -> EnvConfig:
    """Apply a rung's overrides on top of the config's environment settings.

    `strict` is forced off: the Stage 6 smoke test exists precisely so that training does
    not have to pay for per-step invariant assertions, which roughly double the step cost.
    Violations are still caught -- `ConstraintMonitor` reads the same counters out of
    `info` and stops the run.
    """
    base = dict(bundle.env_cfg.__dict__)
    base.update(stage.env)
    base.update(overrides or {})
    base["strict"] = False
    return EnvConfig(**base)


def ppo_kwargs(resolved: dict) -> dict:
    cfg = dict(DEFAULT_PPO)
    cfg.update({k: v for k, v in (resolved.get("ppo", {}) or {}).items()
                if k in DEFAULT_PPO})
    return cfg


def build_model(bundle: EnvBundle, vec, resolved: dict, *, seed: int,
                tensorboard: Path | None) -> PPO:
    training = resolved.get("training", {}) or {}
    return PPO(
        SharedAssetActorCritic, vec,
        policy_kwargs=policy_kwargs_from(resolved, bundle.spec),
        device=training.get("device", "cpu"), seed=seed, verbose=0,
        tensorboard_log=str(tensorboard) if tensorboard else None,
        **ppo_kwargs(resolved),
    )


def warm_start(model: PPO, previous: Path, log=print) -> bool:
    """Load the previous rung's weights into this rung's freshly built model.

    Parameters are transferred rather than the whole `PPO` object, so this rung keeps its
    own environment, schedules and hyperparameters -- warm-starting means inheriting what
    was learned, not inheriting the previous stage's setup.
    """
    policy_path = Path(previous) / "policy.zip"
    if not policy_path.exists():
        raise FileNotFoundError(f"no policy to warm-start from at {policy_path}")
    old = PPO.load(policy_path, device=model.device)

    new_state = model.policy.state_dict()
    old_state = old.policy.state_dict()
    mismatched = [k for k in new_state
                  if k not in old_state or old_state[k].shape != new_state[k].shape]
    if mismatched:
        raise ValueError(
            f"cannot warm-start: {len(mismatched)} parameter(s) differ in shape, "
            f"e.g. {mismatched[:3]}. The observation space must be identical across "
            "curriculum stages -- the risk toggle changes dynamics, not observations."
        )
    model.policy.load_state_dict(old_state)
    log(f"warm-started {len(old_state)} tensors from {policy_path}")
    return True


def train_stage(
    bundle: EnvBundle, stage: CurriculumStage, resolved: dict, *,
    run_dir: Path, seed: int, total_timesteps: int, n_envs: int,
    subproc: bool = True, resume_from: Path | None = None,
    baselines: dict | None = None, env_overrides: dict | None = None,
    checkpoint_every: int = 0, log=print,
) -> StageOutcome:
    """Train one rung and evaluate its gate."""
    import time

    env_cfg = stage_env_config(bundle, stage, env_overrides)
    vec = make_vec_env(bundle, n_envs, base_seed=seed, subproc=subproc, env_cfg=env_cfg)
    vec = wrap_normalizer(vec, gamma=ppo_kwargs(resolved)["gamma"])

    stage_dir = run_dir / f"stage{stage.index}_{stage.name}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(bundle, vec, resolved, seed=seed,
                        tensorboard=run_dir / "tensorboard")
    n_params = count_parameters(model.policy)
    log(f"stage {stage.index} ({stage.name}): {stage.what_is_added}")
    log(f"  N={env_cfg.hold_days_values}  D_max={env_cfg.max_drawdown_values}  "
        f"risk={env_cfg.risk_enabled}  flat_start={env_cfg.flat_start}  "
        f"episodes={env_cfg.episode_lengths}")
    log(f"  {n_params:,} parameters, {n_envs} envs, {total_timesteps:,} timesteps")

    if resume_from is not None:
        warm_start(model, resume_from, log=log)

    diagnostics = DiagnosticCallback()
    monitor = ConstraintMonitor()
    callbacks = [monitor, diagnostics, EntropyGuard(verbose=1)]
    if baselines:
        callbacks.append(BaselineReferenceCallback(baselines))
    if checkpoint_every:
        callbacks.append(CheckpointCallback(
            save_freq=max(checkpoint_every // n_envs, 1),
            save_path=str(stage_dir / "checkpoints"), name_prefix="ppo"))

    started = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks),
                    tb_log_name=f"stage{stage.index}", progress_bar=False)
    finally:
        elapsed = time.time() - started

    save_bundle(model, vec, stage_dir)
    snapshot = diagnostics.snapshot()
    # The monitor RAISES on the first violation, so reaching here means zero. Reading its
    # counter anyway keeps the gate honest if that behaviour is ever relaxed.
    gate = evaluate_gate(stage.index, snapshot, cash_log_return=0.0,
                         lock_violations=monitor.total)
    vec.close()

    outcome = StageOutcome(
        stage=stage.index, name=stage.name, run_dir=stage_dir,
        timesteps=total_timesteps, diagnostics=snapshot, gate=gate.to_dict(),
        wall_seconds=round(elapsed, 1), n_parameters=n_params)
    (stage_dir / "metrics.json").write_text(
        json.dumps({"stage": stage.index, "name": stage.name,
                    "what_is_added": stage.what_is_added,
                    "timesteps": total_timesteps, "wall_seconds": outcome.wall_seconds,
                    "steps_per_second": round(total_timesteps / max(elapsed, 1e-9), 1),
                    "n_parameters": n_params,
                    "env": {k: str(v) for k, v in env_cfg.__dict__.items()},
                    "ppo": ppo_kwargs(resolved),
                    "diagnostics": snapshot, "gate": gate.to_dict()},
                   indent=2, default=str), encoding="utf-8")

    log(f"  {total_timesteps / max(elapsed, 1e-9):,.0f} steps/s, "
        f"{elapsed:.0f}s wall")
    log(f"  {gate.explain()}")
    for note in gate.notes:
        log(f"  note: {note}")
    return outcome
