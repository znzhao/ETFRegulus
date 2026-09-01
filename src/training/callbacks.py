"""What to watch while PPO trains, and what to stop for.

reference/rl-training.md section 6 lists the signals and what healthy looks like. All of
them are logged to TensorBoard and carried into `metrics.json`, because a training curve
that only ever existed in a TensorBoard event file is not a result anyone can cite.

The one non-negotiable is `ConstraintMonitor`. A lock or feasibility violation during
training does not mean the policy is bad -- it means the constraint layer is broken, and
every subsequent timestep is spent learning against a simulator that is lying. So it raises
rather than logs, immediately, with the step it happened on.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ConstraintViolation(RuntimeError):
    """The constraint layer failed during training. Not a policy problem."""


class ConstraintMonitor(BaseCallback):
    """Hard stop on any lock or feasibility violation. Never a warning."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.total = 0

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            n = int(info.get("lock_violations", 0) or 0)
            if n:
                self.total += n
                raise ConstraintViolation(
                    f"{n} lock violation(s) in env {i} at timestep "
                    f"{self.num_timesteps}, session {info.get('session')}. "
                    "This is a constraint-layer defect, not a bad policy: stop and fix "
                    "it rather than training against a simulator that is not enforcing "
                    "its own rules."
                )
        return True


class DiagnosticCallback(BaseCallback):
    """Roll the per-step `info` diagnostics into TensorBoard scalars.

    Everything here comes out of `env.step`'s `info`, which is the same dict the Stage 4
    trajectory rows are built from -- so a number on a training curve and the same number
    in an evaluation artifact are computed by the same code.
    """

    #: Flags averaged into rates.
    RATES = ("safety_intervened", "capital_preservation", "infeasible_fallback")

    def __init__(self, window: int = 4096, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self.buffers: dict[str, deque] = {
            k: deque(maxlen=window) for k in
            (*self.RATES, "proj_distance", "cash_weight", "drawdown", "turnover",
             "drawdown_budget", "de_risk_alpha")
        }
        self.episode_returns: deque = deque(maxlen=256)
        self._running = None

    def _on_training_start(self) -> None:
        self._running = np.zeros(self.training_env.num_envs)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        for i, info in enumerate(infos):
            for key in self.RATES:
                if key in info:
                    self.buffers[key].append(float(bool(info[key])))
            for key in ("proj_distance", "turnover", "drawdown", "drawdown_budget",
                        "de_risk_alpha"):
                if key in info:
                    self.buffers[key].append(float(info[key]))
            if "cash" in info and info.get("nav"):
                self.buffers["cash_weight"].append(float(info["cash"]) / float(info["nav"]))

        # Episode return in RAW reward units. `VecNormalize` scales the reward PPO sees,
        # so the training curve would otherwise be in arbitrary units and incomparable to
        # a baseline.
        if rewards is not None and self._running is not None:
            raw = self.training_env.get_original_reward() \
                if hasattr(self.training_env, "get_original_reward") else rewards
            self._running += np.asarray(raw, dtype=float)
            if dones is not None:
                for i, done in enumerate(np.atleast_1d(dones)):
                    if done:
                        self.episode_returns.append(float(self._running[i]))
                        self._running[i] = 0.0
        return True

    def _on_rollout_end(self) -> None:
        for key, buf in self.buffers.items():
            if buf:
                self.logger.record(f"diag/{key}", float(np.mean(buf)))
        if self.episode_returns:
            # Log return per episode -> the growth an episode actually delivered.
            self.logger.record("diag/episode_log_return",
                               float(np.mean(self.episode_returns)))
            self.logger.record("diag/episode_growth_pct",
                               100.0 * float(np.mean(np.expm1(self.episode_returns))))

    def snapshot(self) -> dict:
        out = {k: (float(np.mean(v)) if v else 0.0) for k, v in self.buffers.items()}
        out["episode_log_return"] = (float(np.mean(self.episode_returns))
                                     if self.episode_returns else 0.0)
        out["episode_growth_pct"] = (100.0 * float(np.mean(np.expm1(self.episode_returns)))
                                     if self.episode_returns else 0.0)
        out["n_episodes"] = len(self.episode_returns)
        return out


class BaselineReferenceCallback(BaseCallback):
    """Draw the Stage 5 baselines as flat reference lines on the reward chart.

    "Is this good" should be answerable at a glance rather than after an analysis, and the
    bar that matters is not `cash` -- it is `spy_tlt_60_40`, a two-line static allocation
    anyone could implement.
    """

    def __init__(self, baselines: dict[str, float], verbose: int = 0):
        super().__init__(verbose)
        self.baselines = baselines

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        for name, value in self.baselines.items():
            self.logger.record(f"baseline/{name}", float(value))


class EntropyGuard(BaseCallback):
    """Warn once when the policy collapses.

    A softmax over 25 assets collapses to a single-asset corner readily, and a collapsed
    policy looks perfectly stable while learning nothing -- the failure mode is quiet, so
    something has to say it out loud.
    """

    def __init__(self, floor: float = -3.0, verbose: int = 0):
        super().__init__(verbose)
        self.floor = floor
        self.fired = False

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.fired:
            return
        with_std = getattr(self.model.policy, "log_std", None)
        if with_std is None:
            return
        mean_log_std = float(with_std.mean().detach().cpu())
        self.logger.record("diag/mean_log_std", mean_log_std)
        if mean_log_std < self.floor:
            self.fired = True
            if self.verbose:
                print(f"WARN policy log_std collapsed to {mean_log_std:.2f} "
                      f"(< {self.floor}); consider raising ent_coef")
