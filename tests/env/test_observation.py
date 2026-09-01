"""T14 -- the observation matches the feature manifest, and every mandatory field is there.

The failure this guards against is a silent index shift: a feature added, removed or
reordered upstream, after which the policy keeps training happily on a vector whose
meaning has moved by one position. Nothing crashes, the numbers stay plausible, and the
result is quietly wrong. So the layout is asserted by NAME at specific indices, not by
shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config.loader import resolve_config
from src.env.observation import (
    MANDATORY_FIELDS,
    MANDATORY_GLOBAL,
    MANDATORY_PARAMS,
    MANDATORY_PER_ASSET,
    ObservationError,
    ObservationSpec,
)

MANIFEST = Path("data/features/feature_manifest.json")


def test_every_declared_column_exists_in_the_manifest(obs_spec):
    """The whole point of the manifest: a typo is a startup error, not a zero column."""
    report = obs_spec.validate()
    assert report["obs_dim"] == obs_spec.size
    assert report["selected"]["per_asset_etf"] > 0


def test_a_misspelled_feature_is_rejected_with_a_hint(obs_spec):
    resolved = {
        "observation": {"per_asset": {"etf": ["ret_1", "ret_9999"]},
                        "global": {"macro": ["VIX_level"]}},
    }
    with pytest.raises(ObservationError) as exc:
        ObservationSpec.from_config(resolved, obs_spec.tickers)
    msg = str(exc.value)
    assert "ret_9999" in msg
    assert "did you mean" in msg, "a rejection should name the near misses"


def test_reordering_tickers_is_rejected(obs_spec):
    """Per-asset blocks are positional. Silently reordering them invalidates a trained
    policy while leaving every shape identical."""
    swapped = (obs_spec.tickers[1], obs_spec.tickers[0], *obs_spec.tickers[2:])
    resolved = resolve_config(Path("config/training.yaml"))
    with pytest.raises(ObservationError, match="positional"):
        ObservationSpec.from_config(resolved, swapped)


def test_every_mandatory_markov_field_is_present(obs_spec):
    """reference/env-mdp.md section 1: if any of these is absent the problem is not
    harder, it is non-Markov, and PPO's assumptions break silently."""
    names = obs_spec.build_names()
    for field in MANDATORY_FIELDS:
        assert any(n.split(":")[-1] == field for n in names), f"{field} missing"


def test_feasibility_is_reconstructible_from_the_observation(obs_spec):
    """The specific sufficiency requirement: the agent must be able to tell, from the
    observation alone, which actions the projection will alter. That needs the lock, the
    availability mask and the drawdown headroom -- all three, per asset where relevant."""
    for field in ("pf_locked", "pf_lock_remaining", "pf_available"):
        assert field in MANDATORY_PER_ASSET
    assert "pf_drawdown_budget" in MANDATORY_GLOBAL
    assert set(MANDATORY_PARAMS) == {"param_hold_days", "param_max_drawdown"}


def test_the_blocks_tile_the_vector_exactly(obs_spec):
    """No gap, no overlap, and the declared order from env-mdp.md section 1."""
    s = obs_spec
    assert s.macro_slice.start == 0
    assert s.macro_slice.stop == s.per_asset_slice.start
    assert s.per_asset_slice.stop == s.portfolio_slice.start
    assert s.portfolio_slice.stop == s.param_slice.start
    assert s.param_slice.stop == s.size


def test_names_are_unique_and_addressable(obs_spec):
    names = obs_spec.build_names()
    assert len(set(names)) == len(names)
    assert len(names) == obs_spec.size
    # A named lookup must agree with the arithmetic, or `index_of` is decoration.
    first = obs_spec.tickers[0]
    assert obs_spec.index_of(f"a:{first}:pf_weight") in range(
        obs_spec.per_asset_slice.start, obs_spec.per_asset_slice.stop)
    assert obs_spec.index_of("p:param_hold_days") == obs_spec.param_slice.start


def test_asset_slice_lands_on_that_asset(obs_spec):
    names = obs_spec.build_names()
    for ticker in (obs_spec.tickers[0], obs_spec.tickers[-1]):
        sl = obs_spec.asset_slice(ticker)
        assert all(n.startswith(f"a:{ticker}:") for n in names[sl])
        assert len(names[sl]) == obs_spec.n_per_asset


# ------------------------------------------------------- against a live observation


def test_a_live_observation_carries_the_parameters_it_was_reset_with(env):
    """The conditioning parameters must be readable back out of the vector -- normalized
    by fixed constants, so training and deployment semantics are identical."""
    spec = env.obs_spec
    obs, _ = env.reset(seed=1, options={"hold_days": 42, "max_drawdown": 0.20})
    assert obs[spec.index_of("p:param_hold_days")] == pytest.approx(
        42 / spec.hold_days_divisor, abs=1e-6)
    assert obs[spec.index_of("p:param_max_drawdown")] == pytest.approx(0.20, abs=1e-6)


def test_unavailable_assets_are_zeroed_and_masked(env):
    """The mask carries the information; the zeros are filler (env-mdp.md section 1)."""
    spec = env.obs_spec
    obs, _ = env.reset(seed=2)
    n_market = len(spec.per_asset_market)
    for j, ticker in enumerate(spec.tickers):
        block = obs[spec.asset_slice(ticker)]
        available = block[spec.index_of(f"a:{ticker}:pf_available")
                          - spec.asset_slice(ticker).start]
        if available == 0.0:
            assert np.all(block[:n_market] == 0.0), (
                f"{ticker} is unavailable but its market block is not zeroed")


def test_the_drawdown_budget_is_dmax_minus_drawdown(env):
    spec = env.obs_spec
    obs, info = env.reset(seed=3, options={"max_drawdown": 0.15})
    budget = obs[spec.index_of("p:pf_drawdown_budget")]
    drawdown = obs[spec.index_of("p:pf_drawdown")]
    assert budget == pytest.approx(0.15 - drawdown, abs=1e-5)


def test_observations_are_finite_float32_inside_the_declared_box(env):
    obs, _ = env.reset(seed=4)
    rng = np.random.default_rng(0)
    for _ in range(15):
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all()
        assert env.observation_space.contains(obs)
        obs, _, _, trunc, _ = env.step(rng.uniform(-1, 1, env.action_space.shape))
        if trunc:
            break


def test_the_per_asset_lock_fields_track_the_lock_manager(env):
    """The observation's lock view and the lock manager must not be able to disagree."""
    spec = env.obs_spec
    obs, _ = env.reset(seed=5, options={"hold_days": 30})
    rng = np.random.default_rng(1)
    for _ in range(10):
        obs, _, _, trunc, _ = env.step(rng.uniform(-1, 1, env.action_space.shape))
        session = env.market.sessions[env._row].date()
        for ticker in spec.tickers:
            locked = obs[spec.index_of(f"a:{ticker}:pf_locked")]
            assert bool(locked) == env.state.lock_manager.is_locked(ticker, session), (
                f"{ticker}: observation says locked={locked}"
            )
        if trunc:
            break
