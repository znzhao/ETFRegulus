"""R-1, the no-trade band (MODEL_REDESIGN_PLAN.md section 15).

An ETF whose target weight is within `no_trade_band` of its current weight is not traded:
its share count stays and it is not relocked. Off (0) the simulator is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baselines.strategies import BASELINES
from src.config.loader import load_typed
from src.config.schema import ConstraintsConfig, UniverseConfig
from src.constraints.risk_envelope import RiskEnvelope
from src.evaluation.violations import replay_feasibility
from src.sim.engine import DecisionClock, state_from
from src.sim.runner import make_projector
from src.sim.simulator import MarketData, SimulationConfig, simulate

CURATED = Path("data/curated/prices.parquet")
pytestmark = pytest.mark.skipif(not CURATED.exists(), reason="run Stage 2 first")


@pytest.fixture(scope="module")
def market() -> MarketData:
    ucfg, _ = load_typed("config/universe.yaml", UniverseConfig)
    prices = pd.read_parquet(CURATED)
    prices = prices[prices["is_tradable"]]
    return MarketData.from_curated(prices, ucfg.tradable_tickers, start="2018-01-02", end="2021-12-31")


@pytest.fixture(scope="module")
def constraints() -> ConstraintsConfig:
    cfg, _ = load_typed("config/constraints.yaml", ConstraintsConfig)
    return cfg


def envelope(constraints) -> RiskEnvelope:
    r = constraints.risk
    return RiskEnvelope(quantile=r.quantile, horizon_days=r.horizon_days, block_length=r.block_length,
                        aggregation=r.aggregation, estimators=tuple(r.estimators),
                        crisis_windows=dict(r.crisis_windows), n_bootstrap_paths=64)


def run(market, constraints, name, band, *, risk=False, d_max=0.15, fn=None, k=0.0, window=0):
    cfg = SimulationConfig(hold_days=30, max_drawdown=d_max, risk_enabled=risk, no_trade_band=band,
                           peak_window=window)
    proj = make_projector(over_budget_rule="no_risk_increase", derisk_step=k)
    fn = fn or BASELINES[name].build(market, {})
    return simulate(market, fn, cfg, projector=proj, envelope=envelope(constraints))


def trade_days(traj) -> int:
    return int((traj["turnover"] > 0).sum())


def test_a_monthly_strategy_trades_about_monthly_with_the_band(market, constraints):
    off = run(market, constraints, "momentum", 0.0).trajectory
    on = run(market, constraints, "momentum", 0.01).trajectory
    months = len(set(zip(on.index.year, on.index.month)))
    assert trade_days(off) > 3 * months, "the daily drift-trading this fixes is present without it"
    assert trade_days(on) <= 2 * months
    assert on["band_suppressed"].sum() > 0


def test_band_zero_changes_nothing(market, constraints):
    a = run(market, constraints, "equal_weight", 0.0).trajectory
    b_cfg = SimulationConfig(hold_days=30, max_drawdown=0.15, risk_enabled=False)
    b = simulate(market, BASELINES["equal_weight"].build(market, {}), b_cfg,
                 projector=make_projector(over_budget_rule="no_risk_increase"),
                 envelope=envelope(constraints)).trajectory
    np.testing.assert_allclose(a["nav"].to_numpy(), b["nav"].to_numpy(), rtol=0, atol=0)


def test_small_changes_are_skipped_and_large_ones_executed(market, constraints):
    """A source that nudges SPY by 0.5 pp and moves TLT by 5 pp every 42 sessions (TLT is
    unlocked by then, so the projection never has to squeeze SPY to respect a lock)."""
    spy, tlt = market.universe.index("SPY") + 1, market.universe.index("TLT") + 1

    def fn(session, ctx):
        w = np.asarray(ctx.current_weights, float).copy()
        if ctx.step == 0:
            w[:] = 0.0
            w[spy], w[tlt], w[0] = 0.5, 0.3, 0.2
            return w
        if ctx.step % 42 == 0:
            w[spy] += 0.005
            w[tlt] += 0.05 if w[0] >= 0.06 else -0.05
            w[0] = 1.0 - w[1:].sum()
        return w

    t = run(market, constraints, None, 0.01, fn=fn).trajectory
    spy_sh = t["shares_SPY"].to_numpy()
    tlt_sh = t["shares_TLT"].to_numpy()
    # SPY: bought once at the start, never again (dividend accretion aside)
    assert (np.diff(spy_sh) >= -1e-9).all() and spy_sh[-1] / spy_sh[0] < 1.08
    assert np.unique(t["unlock_date_SPY"].dropna()).size == 1
    # TLT: traded at the 42-session marks
    assert (np.abs(np.diff(tlt_sh)) > 1.0).sum() >= 10


def test_constraints_hold_with_the_band_and_risk_on(market, constraints):
    for name in ("momentum", "equal_weight", "spy_tlt_60_40"):
        res = run(market, constraints, name, 0.01, risk=True, d_max=0.05, k=0.2, window=252)
        assert res.diagnostics["lock_violations"] == 0
        assert res.diagnostics["feasibility_violations"] == 0
        findings = replay_feasibility(res.trajectory, market.universe, d_max=0.05, market=market)
        assert not findings, findings[:3]


def test_o2_cuts_drawdown_and_is_never_softened_by_the_band(market, constraints):
    plain = run(market, constraints, "spy_buy_hold", 0.01, risk=True, d_max=0.05, window=252).trajectory
    o2 = run(market, constraints, "spy_buy_hold", 0.01, risk=True, d_max=0.05, k=0.2, window=252).trajectory
    assert o2["derisk_forced"].any()
    # Forced decisions never buy back into a position they are cutting, so they never
    # relock it: through March 2020 SPY's unlock date (set by a genuine purchase in
    # February) does not move, and once it unlocks O-2 sells it down quickly.
    march = o2.loc["2020-03-01":"2020-03-31"]
    assert march["derisk_forced"].sum() >= 15
    assert march["unlock_date_SPY"].nunique() == 1
    assert o2.loc["2020-03-31", "w_SPY"] < 0.6 * plain.loc["2020-03-31", "w_SPY"]


def test_preservation_daily_clock():
    class Ctx:
        def __init__(self, step, dd):
            self.step, self.drawdown, self.max_drawdown = step, dd, 0.05
            self.current_weights = np.array([1.0, 0.0])
            self.lock_remaining_days = np.array([0.0])

    c = DecisionClock(5, preservation_daily=True)
    assert c.due(Ctx(0, 0.0))
    assert not c.due(Ctx(1, 0.0))
    assert c.due(Ctx(2, 0.07))      # flip
    assert c.due(Ctx(3, 0.08))      # still past the ceiling: every session
    plain = DecisionClock(5)
    plain.due(Ctx(0, 0.0)); plain.due(Ctx(2, 0.07))
    assert not plain.due(Ctx(3, 0.08))


def test_rolling_peak_state_restores_the_nav_window():
    from src.portfolio.ledger import Ledger
    from src.portfolio.lock_manager import LockManager
    cfg = SimulationConfig(peak_window=252)
    lm = LockManager(universe=["SPY"], scope="per_etf")
    st = state_from(Ledger(cash=100.0), lm, cfg, nav=100.0, peak=120.0, navs=[110.0, 120.0, 100.0])
    assert st.peak == 120.0 and list(st.navs) == [110.0, 120.0, 100.0]
    st = state_from(Ledger(cash=100.0), lm, cfg, nav=100.0, peak=120.0)
    assert list(st.navs) == [120.0, 100.0]
    st = state_from(Ledger(cash=100.0), lm, SimulationConfig(), nav=100.0, peak=120.0)
    assert st.navs is None and st.peak == 120.0


def test_a_window_ending_mid_month_does_not_make_its_last_day_a_month_end(market):
    """The live web app's data always ends today; a monthly strategy must not treat every
    today as a month-end and rebalance daily."""
    from src.baselines.strategies import _month_ends
    s = market.sessions
    full = _month_ends(s)
    assert s[-1] in full                                   # 2021-12-31 really is one
    cut = s[s <= pd.Timestamp("2021-11-15")]
    ends = _month_ends(cut)
    assert cut[-1] not in ends
    assert pd.Timestamp("2021-10-29") in ends
