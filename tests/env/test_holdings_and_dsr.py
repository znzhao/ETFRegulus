"""The holdings limit as an episode parameter, and the differential-Sharpe reward."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.env.etf_env import EnvConfig
from src.env.factory import make_env

from .conftest import needs_data


def _env(bundle, **over):
    spec = dataclasses.replace(bundle.spec, holdings_param=True, names=())
    spec = dataclasses.replace(spec, names=spec.build_names())
    b = dataclasses.replace(bundle, spec=spec)
    cfg = EnvConfig(**{**bundle.env_cfg.__dict__, "risk_enabled": False, "episode_lengths": (40,),
                       **over})
    return make_env(b, seed=3, env_cfg=cfg), spec


@needs_data
@pytest.mark.parametrize("k", [3, 5])
def test_the_limit_is_observed_and_never_exceeded(bundle, k):
    env, spec = _env(bundle, max_holdings_values=(k,), flat_start=True)
    obs, info = env.reset(seed=1)
    assert obs.shape == (spec.size,)
    assert obs[spec.index_of("p:param_max_holdings")] == pytest.approx(k / spec.n_assets)
    assert info["episode"]["max_holdings"] == k
    rng = np.random.default_rng(0)
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(rng.normal(size=env.action_space.shape))
        held = sum(env.state.ledger.get(t) > 1e-9 for t in env.market.universe)
        assert held <= k
        done = term or trunc


@needs_data
def test_no_limit_reads_as_one(bundle):
    env, spec = _env(bundle, max_holdings_values=(0,))
    obs, _ = env.reset(seed=2)
    assert obs[spec.index_of("p:param_max_holdings")] == pytest.approx(1.0)


@needs_data
def test_dsr_matches_the_formula_and_log_return_is_unchanged(bundle):
    env, _ = _env(bundle, reward="dsr", flat_start=True)
    _, info = env.reset(seed=4)
    nav0 = info["nav"]
    a, b, eta, floor = 0.0, env.cfg.dsr_var_init, env.cfg.dsr_eta, env.cfg.dsr_var_floor
    rng = np.random.default_rng(1)
    for _ in range(15):
        _, r, _, _, info = env.step(rng.normal(size=env.action_space.shape))
        x = float(np.log(info["nav"] / nav0))
        nav0 = info["nav"]
        da, db = x - a, x * x - b
        assert r == pytest.approx((b * da - 0.5 * a * db) / max(b - a * a, floor) ** 1.5, rel=1e-6, abs=1e-9)
        a, b = a + eta * da, b + eta * db

    plain, _ = _env(bundle, flat_start=True)
    _, info = plain.reset(seed=4)
    _, r, _, _, info2 = plain.step(np.zeros(plain.action_space.shape))
    assert r == pytest.approx(np.log(info2["nav"] / info["nav"]))


@needs_data
def test_observation_noise_touches_market_features_only(bundle):
    from src.env.etf_env import market_feature_mask
    quiet, spec = _env(bundle, flat_start=True)
    noisy, _ = _env(bundle, flat_start=True, obs_noise=0.1)
    a, _ = quiet.reset(seed=5)
    b, _ = noisy.reset(seed=5)
    mask = market_feature_mask(spec)
    assert mask.sum() > 0 and (~mask).sum() > 0
    np.testing.assert_array_equal(a[~mask], b[~mask])
    assert (b[mask] - a[mask]).std() == pytest.approx(0.1, rel=0.2)
