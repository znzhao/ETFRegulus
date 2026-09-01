"""Wiring the trainer: normalization, warm-starting, and one real (tiny) training run.

The normalization tests are the ones to read first. `norm_obs=False` is a deliberate
correction to reference/rl-training.md section 2, and the reason is exactly the failure
that section 3 names as the most likely cause of a flat `proj_distance` — so it is pinned
here rather than left as a comment someone later "fixes".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.agents.normalization import VECNORM_NAME, freeze, save_bundle, wrap_normalizer
from src.env.etf_env import EnvConfig
from src.training.curriculum import STAGES_BY_INDEX
from src.training.trainer import stage_env_config

pytest.importorskip("stable_baselines3")


# --------------------------------------------------------------- normalization


def _dummy_vec(n: int = 2):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    class Tiny(gym.Env):
        observation_space = gym.spaces.Box(-10.0, 10.0, (4,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.arange(4, dtype=np.float32) * 100.0, {}

        def step(self, action):
            return (np.arange(4, dtype=np.float32) * 100.0, 0.001, False, False, {})

    return DummyVecEnv([lambda: Tiny() for _ in range(n)])


def test_observations_are_not_renormalized():
    """The fold scaler already normalized them, and `VecNormalize` would flatten the
    availability masks the policy reads to mask its own logits."""
    vec = wrap_normalizer(_dummy_vec(), gamma=0.99)
    assert vec.norm_obs is False
    assert vec.norm_reward is True
    obs = vec.reset()
    # The raw magnitudes survive: a normalizer would have centred these on zero.
    assert obs.max() == pytest.approx(300.0)
    vec.close()


def test_freezing_stops_the_statistics_and_the_reward_scaling():
    """A `VecNormalize` still updating during evaluation is a full-sample scaling leak
    wearing a disguise, and evaluation must read raw returns."""
    vec = freeze(wrap_normalizer(_dummy_vec(), gamma=0.99))
    assert vec.training is False
    assert vec.norm_reward is False
    vec.close()


def test_the_normalizer_is_saved_next_to_the_policy(tmp_path: Path):
    """A policy restored without its normalizer is meaningless, so they move together."""
    from stable_baselines3 import PPO

    vec = wrap_normalizer(_dummy_vec(), gamma=0.99)
    model = PPO("MlpPolicy", vec, n_steps=8, batch_size=8, n_epochs=1, device="cpu")
    paths = save_bundle(model, vec, tmp_path / "stage1")
    assert (tmp_path / "stage1" / "policy.zip").exists()
    assert (tmp_path / "stage1" / VECNORM_NAME).exists()
    assert "vecnormalize" in paths
    vec.close()


# ------------------------------------------------------------ stage env config


class _FakeBundle:
    env_cfg = EnvConfig(hold_days_values=(15, 21, 30, 42, 60),
                        hold_days_weights=(0.15, 0.20, 0.30, 0.20, 0.15),
                        max_drawdown_values=(0.05, 0.10, 0.15, 0.20, 0.25),
                        episode_lengths=(63, 126, 252, 504), strict=True)


def test_a_rungs_overrides_land_on_the_env_config():
    cfg = stage_env_config(_FakeBundle(), STAGES_BY_INDEX[2])
    assert cfg.hold_days_values == (21, 30, 42)
    assert cfg.max_drawdown_values == (0.20,)
    assert cfg.risk_enabled is False
    assert cfg.flat_start is True


def test_strict_mode_is_forced_off_during_training():
    """Stage 6 exists so training does not pay for per-step invariant assertions; the
    `ConstraintMonitor` catches violations from the same counters instead."""
    for rung in STAGES_BY_INDEX.values():
        assert stage_env_config(_FakeBundle(), rung).strict is False


def test_rung_five_uses_the_reservoir_and_sampled_episode_lengths():
    cfg = stage_env_config(_FakeBundle(), STAGES_BY_INDEX[5])
    assert cfg.flat_start is False
    assert cfg.risk_enabled is True
    assert len(cfg.episode_lengths) == 4


def test_an_explicit_override_beats_the_rung():
    cfg = stage_env_config(_FakeBundle(), STAGES_BY_INDEX[1],
                           {"episode_lengths": (30,)})
    assert cfg.episode_lengths == (30,)


# -------------------------------------------------------- a real training run


@pytest.mark.slow
def test_a_rung_trains_end_to_end_with_zero_violations():
    """The integration test: real environment, real policy, real PPO update.

    Marked `slow` and deselected by default because it costs ~15s against a 60s suite
    budget. It is the test that would catch the whole stack failing to fit together, so
    run it with `-m slow` before trusting a training run.
    """
    from src.env.factory import build_bundle
    from src.training.trainer import train_stage

    if not Path("data/features/feature_manifest.json").exists():
        pytest.skip("run Stages 1-3 first")

    bundle = build_bundle("config/experiments/ppo_stage1.yaml",
                          start="2008-01-02", end="2009-12-31")
    resolved = dict(bundle.resolved)
    resolved["ppo"] = {"n_steps": 64, "batch_size": 64, "n_epochs": 1}

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        outcome = train_stage(
            bundle, STAGES_BY_INDEX[1], resolved, run_dir=Path(tmp), seed=3,
            total_timesteps=256, n_envs=2, subproc=False, log=lambda *_: None)
        # Inside the block: the artifacts live in `tmp` and vanish when it closes.
        assert outcome.gate["lock_violations"] == 0
        assert outcome.gate["feasibility_violations"] == 0
        assert outcome.n_parameters > 0
        assert (outcome.run_dir / "policy.zip").exists()
        assert (outcome.run_dir / "vecnormalize.pkl").exists()
        assert (outcome.run_dir / "metrics.json").exists()


@pytest.mark.slow
def test_warm_starting_transfers_weights_and_rejects_a_shape_mismatch():
    from stable_baselines3 import PPO

    from src.env.factory import build_bundle, make_vec_env
    from src.training.trainer import build_model, stage_env_config, warm_start

    if not Path("data/features/feature_manifest.json").exists():
        pytest.skip("run Stages 1-3 first")

    import tempfile

    bundle = build_bundle("config/experiments/ppo_stage1.yaml",
                          start="2008-01-02", end="2008-12-31")
    resolved = dict(bundle.resolved)
    resolved["ppo"] = {"n_steps": 32, "batch_size": 32, "n_epochs": 1}
    cfg = stage_env_config(bundle, STAGES_BY_INDEX[1])
    vec = make_vec_env(bundle, 1, base_seed=0, subproc=False, env_cfg=cfg)

    with tempfile.TemporaryDirectory() as tmp:
        first = build_model(bundle, vec, resolved, seed=1, tensorboard=None)
        first.save(Path(tmp) / "policy.zip")
        second = build_model(bundle, vec, resolved, seed=2, tensorboard=None)
        # Different seeds -> different initial weights, so a transfer is observable.
        key = "mlp_extractor.asset_head.0.weight"
        before = second.policy.state_dict()[key].clone()
        warm_start(second, Path(tmp), log=lambda *_: None)
        after = second.policy.state_dict()[key]
        assert not np.allclose(before.numpy(), after.numpy()), "nothing was transferred"

        expected = PPO.load(Path(tmp) / "policy.zip", device="cpu")
        assert np.allclose(after.numpy(), expected.policy.state_dict()[key].numpy())
    vec.close()
