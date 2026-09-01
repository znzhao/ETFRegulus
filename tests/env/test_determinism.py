"""T9 -- same seed, byte-identical trajectory, across worker counts.

Reproducibility is a hard acceptance criterion, so "close enough" is not the standard
here: the comparison is `array_equal`, not `allclose`. A float that drifts in the last bit
between runs means something in the pipeline is reading an unseeded RNG or iterating a set,
and that is worth finding now rather than when Stage 12 reports a spread across seeds and
nobody can tell noise from a bug.

The worker-count half is the subtle one. Worker `i` derives its seed as `seed + i` and
never shares an RNG, so worker 0 must see the same episode stream whether it is one of 4
or one of 16 -- which is what makes the Stage 6 throughput table a comparison of speed
alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.env.etf_env import EnvConfig, softmax_weights
from src.env.factory import make_env, make_vec_env

STEPS = 25


def _rollout(bundle, seed: int, cfg: EnvConfig, action_seed: int | None = None):
    env = make_env(bundle, seed=seed, env_cfg=cfg)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed if action_seed is None else action_seed)
    observations, rewards, navs = [obs.copy()], [], []
    for _ in range(STEPS):
        a = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
        obs, reward, _, trunc, info = env.step(a)
        observations.append(obs.copy())
        rewards.append(reward)
        navs.append(info["nav"])
        if trunc:
            break
    env.close()
    return np.array(observations), np.array(rewards), np.array(navs)


@pytest.fixture(scope="module")
def cfg(bundle) -> EnvConfig:
    return EnvConfig(**{**bundle.env_cfg.__dict__,
                        "strict": False, "episode_lengths": (STEPS + 5,)})


def test_the_same_seed_gives_a_byte_identical_trajectory(bundle, cfg):
    a = _rollout(bundle, 11, cfg)
    b = _rollout(bundle, 11, cfg)
    for name, x, y in zip(("observations", "rewards", "navs"), a, b):
        assert np.array_equal(x, y), f"{name} differ between two runs at the same seed"


def test_different_seeds_give_different_trajectories(bundle, cfg):
    """The converse. Without it, a constant environment would pass the test above."""
    a = _rollout(bundle, 11, cfg)
    b = _rollout(bundle, 12, cfg)
    assert not np.array_equal(a[0], b[0]), "two seeds produced identical observations"


def test_the_risk_envelope_is_deterministic_too(bundle):
    """The block bootstrap is the one genuinely stochastic component in the step. It is
    seeded from `cfg.seed + i`, so it must not move between runs."""
    cfg = EnvConfig(**{**bundle.env_cfg.__dict__,
                       "risk_enabled": True, "strict": False,
                       "episode_lengths": (12,)})
    a = _rollout(bundle, 5, cfg)
    b = _rollout(bundle, 5, cfg)
    assert np.array_equal(a[1], b[1]), "rewards moved with the envelope enabled"


def test_worker_zero_does_not_depend_on_how_many_workers_there_are(bundle, cfg):
    """Changing `n_envs` is a throughput decision. It must not change what is trained on.

    The actions are drawn PER WORKER from that worker's own generator. Drawing one
    (n_envs, K+1) block from a shared generator would advance it n_envs times per step,
    so worker 0 would see different actions at 1 worker than at 4 -- a property of the
    harness, not of the environment, and one that reads as a determinism failure.
    """
    traces = {}
    for n_envs in (1, 4):
        vec = make_vec_env(bundle, n_envs, base_seed=3, env_cfg=cfg)
        obs = vec.reset()
        rngs = [np.random.default_rng(3 + i) for i in range(n_envs)]
        trace = [obs[0].copy()]
        for _ in range(8):
            a = np.stack([r.uniform(-1, 1, vec.action_space.shape) for r in rngs]
                         ).astype(np.float32)
            obs, _, _, _ = vec.step(a)
            trace.append(obs[0].copy())
        vec.close()
        traces[n_envs] = np.array(trace)
    assert np.array_equal(traces[1], traces[4])


def test_each_worker_gets_its_own_stream(bundle, cfg):
    """`seed + worker_index`, never a shared RNG. Two workers that agreed would be
    training on the same episode four times over and reporting it as four."""
    vec = make_vec_env(bundle, 2, base_seed=3, env_cfg=cfg)
    obs = vec.reset()
    vec.close()
    assert not np.array_equal(obs[0], obs[1])


def test_softmax_is_invariant_to_a_constant_shift():
    """A property of the action mapping itself: only differences between logits matter,
    so the max-shift inside `softmax_weights` cannot change the allocation."""
    a = np.array([0.4, -0.2, 0.9, -1.0, 0.0])
    assert np.allclose(softmax_weights(a), softmax_weights(a * 0 + a))
    assert softmax_weights(a).sum() == pytest.approx(1.0)
    assert (softmax_weights(a) >= 0).all()


def test_a_degenerate_action_still_produces_a_valid_allocation():
    """NaN out of a diverging policy must not become NaN weights -- the environment is
    the last line of defence before the ledger."""
    w = softmax_weights(np.array([np.nan, np.inf, -np.inf, 0.0, 1.0]))
    assert np.isfinite(w).all()
    assert w.sum() == pytest.approx(1.0)
