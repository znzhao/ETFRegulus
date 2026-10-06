"""T8 -- the reset sampler produces only reachable states, and `D_t <= D_max` normally.

Reachability is the whole point. Naively randomizing weights and lock clocks produces
states no legal trajectory could reach -- a position locked for 200 days under `N = 30`,
or a holding that predates its ETF's inception -- and training on those wastes capacity
and distorts the value function.

So the tests come in two halves: the generators must produce legal states, and the
*checker* must actually reject illegal ones. A reachability check that has never been
seen to fail is untested infrastructure, so several tests here hand it a deliberately
broken state and assert it complains.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from src.env.reset_sampler import (
    ConstructiveSampler,
    InitialState,
    ReservoirSampler,
    is_reachable,
)
from src.portfolio.ledger import Ledger
from src.portfolio.lock_manager import LockManager

GRID = [(n, d) for n in (15, 30, 60) for d in (0.05, 0.15, 0.25)]


@pytest.fixture(scope="module")
def market(bundle):
    return bundle.market


# ------------------------------------------------------------ the checker itself


def _legal(market, session=None) -> InitialState:
    session = session or market.sessions[100]
    row = int(market.sessions.get_indexer([session])[0])
    j = int(np.flatnonzero(market.available[row])[0])
    ticker = market.universe[j]
    ledger = Ledger(cash=500_000.0)
    ledger.set_shares(ticker, 1000.0)
    return InitialState(session=session, ledger=ledger,
                        lock_manager=LockManager(universe=list(market.universe)),
                        peak=1_100_000.0, nav=1_000_000.0, drawdown=0.0909,
                        source="test")


def test_the_checker_accepts_a_legal_state(market):
    assert is_reachable(_legal(market), market, hold_days=30, max_drawdown=0.15) == []


def test_a_lock_longer_than_n_is_rejected(market):
    """The unlock date is set to exactly `exec + N` and only counts down, so a remaining
    clock above N could not have been produced at this N."""
    state = _legal(market)
    ticker = state.ledger.holdings()[0]
    state.lock_manager.unlock_dates[ticker] = state.session.date() + dt.timedelta(days=200)
    reasons = is_reachable(state, market, hold_days=30, max_drawdown=0.15)
    assert any("lock remaining" in r for r in reasons), reasons


def test_an_unlock_date_without_a_position_is_rejected(market):
    """L2: the lock manager clears these on every execution, so one cannot persist."""
    state = _legal(market)
    absent = [t for t in market.universe if state.ledger.get(t) == 0][0]
    state.lock_manager.unlock_dates[absent] = state.session.date() + dt.timedelta(days=5)
    reasons = is_reachable(state, market, hold_days=30, max_drawdown=0.15)
    assert any("no position" in r for r in reasons), reasons


def test_a_pre_inception_holding_is_rejected(market):
    """I5. XLRE did not exist in 2008; holding it then is not a state, it is a bug."""
    row = 100
    session = market.sessions[row]
    absent = [t for j, t in enumerate(market.universe) if not market.available[row, j]]
    if not absent:
        pytest.skip("every asset is available in this window")
    state = _legal(market, session)
    state.ledger.set_shares(absent[0], 10.0)
    reasons = is_reachable(state, market, hold_days=30, max_drawdown=0.15)
    assert any("inception" in r for r in reasons), reasons


def test_a_breached_drawdown_is_rejected_normally_but_allowed_under_stress(market):
    state = _legal(market)
    state.drawdown = 0.40
    assert any("exceeds D_max" in r for r in
               is_reachable(state, market, hold_days=30, max_drawdown=0.15))
    # Deliberately-breached states are a separate stress environment, never a default.
    assert is_reachable(state, market, hold_days=30, max_drawdown=0.15,
                        stress=True) == []


def test_a_peak_below_the_nav_it_caps_is_rejected(market):
    state = _legal(market)
    state.peak = state.nav - 1.0
    assert any("peak below" in r for r in
               is_reachable(state, market, hold_days=30, max_drawdown=0.15))


# ------------------------------------------------------- Approach A, the reservoir


def test_the_reservoir_is_reachable_by_construction(market):
    """Approach A's claim: every state came out of a legal trajectory through this very
    simulator, so there is nothing to verify. This asserts the claim anyway."""
    try:
        sampler = ReservoirSampler.latest(market)
    except FileNotFoundError:
        pytest.skip("run Stage 5 first")
    rng = np.random.default_rng(0)
    for n, d in GRID:
        state = sampler.sample(rng, hold_days=n, max_drawdown=d)
        if state is None:
            continue
        assert is_reachable(state, market, hold_days=n, max_drawdown=d) == []
        assert state.drawdown <= d + 1e-9


def test_the_reservoir_filters_by_the_episodes_own_parameters(market):
    """A state recorded at N = 90 carries lock clocks that N = 15 could never create, so
    admissibility is re-checked against the episode's parameters, not the ones it was
    recorded under."""
    try:
        sampler = ReservoirSampler.latest(market)
    except FileNotFoundError:
        pytest.skip("run Stage 5 first")
    tight = len(sampler.candidates(hold_days=15, max_drawdown=0.05))
    loose = len(sampler.candidates(hold_days=60, max_drawdown=0.25))
    assert tight <= loose, "a tighter parameter cell admitted more states than a looser one"
    assert loose > 0


# --------------------------------------------------- Approach B, the constructive


def test_the_constructive_generator_is_legal_across_the_grid(market):
    sampler = ConstructiveSampler(market)
    rng = np.random.default_rng(1)
    for n, d in GRID:
        for _ in range(6):
            state = sampler.sample(rng, hold_days=n, max_drawdown=d)
            assert state is not None, f"no state for N={n}, D_max={d}"
            assert is_reachable(state, market, hold_days=n, max_drawdown=d) == []


def test_the_constructive_generator_covers_the_corners_the_reservoir_misses(market):
    """Its whole reason to exist: high locked fractions and near-ceiling drawdowns, which
    a replayed baseline trajectory almost never supplies."""
    sampler = ConstructiveSampler(market)
    rng = np.random.default_rng(2)
    deep = 0
    for _ in range(30):
        state = sampler.sample(rng, hold_days=30, max_drawdown=0.15,
                               min_locked_fraction=0.9, min_drawdown=0.14)
        assert state is not None
        assert is_reachable(state, market, hold_days=30, max_drawdown=0.15) == []
        deep += state.drawdown >= 0.14
    assert deep >= 25, f"only {deep}/30 draws landed near the ceiling"


def test_no_lock_means_no_lock_state(market):
    """N = 0 is a legal stress value. Nothing may be locked under it."""
    sampler = ConstructiveSampler(market)
    rng = np.random.default_rng(3)
    for _ in range(10):
        state = sampler.sample(rng, hold_days=0, max_drawdown=0.15)
        assert state is not None
        assert state.lock_manager.unlock_dates == {}


# --------------------------------------------------------------- through the env


def test_every_reset_satisfies_dmax_across_the_whole_grid(env):
    """The guarantee that actually matters at training time."""
    for n, d in GRID:
        for k in range(3):
            _, info = env.reset(seed=k, options={"hold_days": n, "max_drawdown": d})
            assert info["drawdown"] <= d + 1e-9, (
                f"N={n} D_max={d}: reset at drawdown {info['drawdown']:.4f}")


def test_the_peak_is_inherited_not_reset_to_the_nav(env):
    """Setting `peak = nav` at reset would hand every episode a fresh zero drawdown and
    teach the agent that drawdown resets for free (env-mdp.md section 6, step 5)."""
    started_under_water = 0
    for k in range(25):
        _, info = env.reset(seed=k, options={"max_drawdown": 0.25})
        assert info["peak"] >= info["nav"] - 1e-6
        started_under_water += info["drawdown"] > 1e-6
    assert started_under_water > 0, (
        "no episode began inside a drawdown; the peak is being reset to the NAV")
