"""Stage 4 integration: the invariants asserted over real trajectories.

The unit suites prove each component in isolation. This one proves they compose -- that a
full run over real prices holds I1, I2, I3, I5 at every step, is deterministic, and
produces the documented trajectory schema.

The four baseline expectations from reference/baselines.md section 7 live here too,
because each is a **test of the simulator disguised as a benchmark**: each has an obvious
correct answer that a subtle accounting bug would break.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baselines.strategies import BASELINES
from src.config.loader import load_typed
from src.config.schema import ConstraintsConfig, UniverseConfig
from src.constraints.risk_envelope import RiskEnvelope
from src.portfolio.ledger import Ledger
from src.sim.simulator import MarketData, SimulationConfig, simulate

CURATED = Path("data/curated/prices.parquet")
pytestmark = pytest.mark.skipif(not CURATED.exists(), reason="run Stage 2 first")

# A window short enough to keep the suite fast, long enough to contain a real crisis
# (COVID) and a real inception (XLRE, 2015-10-08).
START, END = "2015-01-02", "2021-12-31"


@pytest.fixture(scope="module")
def market() -> MarketData:
    ucfg, _ = load_typed("config/universe.yaml", UniverseConfig)
    prices = pd.read_parquet(CURATED)
    prices = prices[prices["is_tradable"]]
    return MarketData.from_curated(prices, ucfg.tradable_tickers, start=START, end=END)


@pytest.fixture(scope="module")
def constraints() -> ConstraintsConfig:
    cfg, _ = load_typed("config/constraints.yaml", ConstraintsConfig)
    return cfg


def envelope(constraints) -> RiskEnvelope:
    r = constraints.risk
    return RiskEnvelope(quantile=r.quantile, horizon_days=r.horizon_days,
                        block_length=r.block_length, aggregation=r.aggregation,
                        estimators=tuple(r.estimators),
                        crisis_windows=dict(r.crisis_windows), n_bootstrap_paths=64)


def run(market, constraints, name, *, hold_days=30, d_max=0.15, risk=False, seed=42,
        params=None):
    """Risk envelope OFF by default.

    The portfolio invariants must hold with or without it, and running them without it
    both isolates the failure (a violation is then unambiguously a ledger/lock bug) and
    keeps the suite inside the 60-second budget. Tests that are actually about the
    envelope pass `risk=True` explicitly.
    """
    fn = BASELINES[name].build(market, params or {})
    cfg = SimulationConfig(hold_days=hold_days, max_drawdown=d_max, risk_enabled=risk,
                           seed=seed, cost_bps=constraints.execution.cost_bps)
    return simulate(market, fn, cfg, envelope=envelope(constraints))


# ---------------------------------------------------------------- the invariants


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_every_baseline_runs_with_zero_violations(market, constraints, name):
    """The Stage 5 gate, asserted per baseline.

    A baseline that violates the lock or leverage is a bug in the CONSTRAINT LAYER, not a
    bad strategy -- and finding it here, against a strategy whose correct behaviour is
    obvious, is far cheaper than finding it during training.
    """
    res = run(market, constraints, name, risk=True)
    assert res.diagnostics["lock_violations"] == 0
    assert res.diagnostics["feasibility_violations"] == 0


@pytest.mark.parametrize("name", ["equal_weight", "momentum", "spy_tlt_60_40"])
def test_portfolio_and_weight_invariants_hold_every_step(market, constraints, name):
    """I1 and I2 over a real trajectory."""
    res = run(market, constraints, name)
    traj = res.trajectory
    universe = market.universe

    assert (traj["nav"] > 0).all(), "I1: NAV went non-positive"
    assert (traj["cash"] >= -1e-6).all(), "I1: negative cash"

    share_cols = [f"shares_{t}" for t in universe]
    assert (traj[share_cols].to_numpy() >= -1e-9).all(), "I1: negative shares"

    w_cols = [f"w_{t}" for t in universe]
    total = traj[w_cols].to_numpy().sum(axis=1) + (traj["cash"] / traj["nav"]).to_numpy()
    assert np.allclose(total, 1.0, atol=1e-8), "I2: weights and cash do not sum to 1"
    assert (traj[w_cols].to_numpy() >= -1e-9).all(), "long-only violated"


def test_drawdown_and_peak_are_consistent(market, constraints):
    res = run(market, constraints, "equal_weight")
    traj = res.trajectory
    assert traj["peak_nav"].is_monotonic_increasing, "the running peak fell"
    assert np.allclose(traj["drawdown"], 1.0 - traj["nav"] / traj["peak_nav"], atol=1e-12)
    assert (traj["drawdown"] >= -1e-12).all() and (traj["drawdown"] < 1.0).all()


def test_reward_is_log_nav_growth(market, constraints):
    """D11: `r_t = log(V_{t+1} / V_t)`, and nothing else."""
    res = run(market, constraints, "equal_weight")
    traj = res.trajectory
    expected = np.log(traj["nav"] / traj["nav"].shift(1)).iloc[1:]
    assert np.allclose(traj["reward"].iloc[1:], expected, atol=1e-12)


def test_no_position_exists_before_inception(market, constraints):
    """I5 over a real trajectory. XLRE launched 2015-10-08, inside this window."""
    res = run(market, constraints, "equal_weight")
    traj = res.trajectory
    j = market.universe.index("XLRE")
    first = market.sessions[np.flatnonzero(market.available[:, j])[0]]
    assert first == pd.Timestamp("2015-10-08")

    before = traj.loc[traj.index < first]
    assert (before["shares_XLRE"] == 0).all(), "a position existed before inception"
    assert (before["w_XLRE"] == 0).all()
    after = traj.loc[traj.index > first + pd.Timedelta(days=30)]
    assert (after["shares_XLRE"] > 0).any(), "XLRE was never bought after inception"


def test_a_locked_position_never_shrinks(market, constraints):
    """I3 read straight off the trajectory.

    The authoritative record is `unlock_date_<t>`, not the `locked_<t>` flag: the flag is
    the state at the row's own session, while the trade that changed the share count
    executed at the NEXT session's open. Between the two the lock can expire -- and it
    routinely does, because `N` is in calendar days while sessions are not. A sale on
    2019-11-04 against an unlock date of Saturday 2019-11-02 is legal, and comparing
    against the flag would flag it wrongly.
    """
    res = run(market, constraints, "momentum", hold_days=60)
    traj = res.trajectory
    sessions = traj.index
    for ticker in market.universe:
        shares = traj[f"shares_{ticker}"].to_numpy()
        unlock = traj[f"unlock_date_{ticker}"]
        for i in np.flatnonzero(np.diff(shares) < -1e-9):
            held_until = unlock.iloc[i]
            if pd.isna(held_until):
                continue
            exec_session = sessions[i + 1]
            assert exec_session >= held_until, (
                f"{ticker}: sold at {exec_session.date()} while locked until "
                f"{held_until.date()} -- an illegal sale reached execution"
            )


def test_the_run_is_deterministic(market, constraints):
    """T9 in its Stage 4 form: same seed -> identical trajectory."""
    a = run(market, constraints, "momentum", seed=7, risk=True).trajectory
    b = run(market, constraints, "momentum", seed=7, risk=True).trajectory
    pd.testing.assert_frame_equal(a, b)


def test_a_different_seed_changes_only_the_stochastic_estimator(market, constraints):
    """The bootstrap is seeded; everything else is deterministic. So a seed change may
    move the risk numbers, but must not move a run with the envelope switched off."""
    a = run(market, constraints, "momentum", seed=1, risk=False).trajectory
    b = run(market, constraints, "momentum", seed=2, risk=False).trajectory
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------ trajectory schema


def test_trajectory_matches_the_documented_schema(market, constraints):
    """architecture.md section 4. Every downstream metric function reads this."""
    traj = run(market, constraints, "equal_weight").trajectory
    required = {"nav", "peak_nav", "drawdown", "cash", "proj_distance",
                "safety_intervened", "capital_preservation", "n_param", "dmax_param",
                "reward"}
    assert required <= set(traj.columns), f"missing {sorted(required - set(traj.columns))}"
    for ticker in market.universe:
        for prefix in ("w_", "shares_", "locked_", "unlock_date_"):
            assert f"{prefix}{ticker}" in traj.columns
    assert traj.index.name == "session"
    assert traj.index.is_monotonic_increasing


# -------------------------------------------- the four baseline simulator checks


def test_cash_has_exactly_zero_drawdown_and_turnover(market, constraints):
    """B4. Two exact expectations at every `N` and `D_max`; any deviation is a ledger bug.

    It also proves the lock never *forces* risk-taking.
    """
    for hold_days in (0, 30, 180):
        res = run(market, constraints, "cash", hold_days=hold_days)
        traj = res.trajectory
        assert traj["drawdown"].max() == pytest.approx(0.0, abs=1e-12), "cash drew down"
        assert traj["turnover"].sum() == pytest.approx(0.0, abs=1e-12), "cash traded"
        assert traj["nav"].nunique() == 1, "cash NAV moved"
        assert res.diagnostics["lock_violations"] == 0


def test_spy_buy_hold_is_identical_across_every_n(market, constraints):
    """B1 is the lock-isolation control. One buy, no sells ever, so `N` cannot matter.

    If these differ, the lock manager is corrupting state it should not touch.
    """
    ref = run(market, constraints, "spy_buy_hold", hold_days=0, risk=False).trajectory
    for hold_days in (7, 30, 60, 180):
        other = run(market, constraints, "spy_buy_hold", hold_days=hold_days,
                    risk=False).trajectory
        pd.testing.assert_series_equal(ref["nav"], other["nav"],
                                       check_names=False, rtol=0, atol=0)


def test_spy_buy_hold_is_a_control_for_n_but_not_for_dmax(market, constraints):
    """B1 isolates the LOCK, not the risk envelope.

    reference/baselines.md section 2 claimed B1 was unaffected by `D_max` because "the
    envelope can never force a sale". That is false as implemented, and the implementation
    is right: the action-level budget constrains `L_stress(w)` for the proposed portfolio
    itself (risk-envelope.md section 3), not merely increases in it. 100% SPY is a
    high-stress portfolio, so under a tight ceiling it gets de-risked -- which is the
    envelope doing its job.

    What never forces a sale is CAPITAL PRESERVATION (section 2), which caps positions at
    current levels. The two are different layers and only the second has that property.
    """
    with_risk_loose = run(market, constraints, "spy_buy_hold", d_max=0.25,
                          risk=True).trajectory
    with_risk_tight = run(market, constraints, "spy_buy_hold", d_max=0.05,
                          risk=True).trajectory
    assert with_risk_tight["drawdown"].max() < with_risk_loose["drawdown"].max(), (
        "a tighter D_max did not reduce B1's realized drawdown; the envelope is inert"
    )

    # With the envelope off, D_max cannot matter at all -- there is no other channel.
    a = run(market, constraints, "spy_buy_hold", d_max=0.25, risk=False).trajectory
    b = run(market, constraints, "spy_buy_hold", d_max=0.05, risk=False).trajectory
    pd.testing.assert_series_equal(a["nav"], b["nav"], check_names=False, rtol=0, atol=0)


def test_momentum_shows_the_lock_actually_binding(market, constraints):
    """B2 proves the lock BINDS.

    Monthly rebalancing means selling last month's losers, but every buy relocks. The
    effect is overwhelming at the `0 -> N>0` boundary and noisy beyond it, because the
    lock interacts with a *discrete* rebalance calendar -- so what is asserted is that the
    lock bites hard and that turnover trends down, not that it is strictly monotone. See
    the correction in reference/baselines.md section 3.
    """
    from scipy.stats import spearmanr

    control = [0, 30, 90, 180]
    turnover = {}
    for hold_days in control:
        traj = run(market, constraints, "momentum", hold_days=hold_days,
                   risk=False).trajectory
        turnover[hold_days] = float(traj["turnover"].sum())

    assert turnover[0] > 3 * turnover[30], (
        f"turnover barely changed when the lock was introduced: {turnover}. The lock is "
        f"not binding, which would hide inside a trained policy."
    )
    rho = float(spearmanr(control, [turnover[n] for n in control]).statistic)
    assert rho <= -0.5, f"turnover does not trend down in N (rho={rho:.3f}): {turnover}"


def test_spy_tlt_60_40_shows_a_severe_2022_drawdown(market, constraints):
    """B3 proves the total-return bond pricing is right.

    2022 is the year the negative stock/bond correlation broke: SPY and TLT fell together
    and TLT alone drew down 31%. **If TLT's 2022 looks mild, the adjustment is wrong.**
    """
    ucfg, _ = load_typed("config/universe.yaml", UniverseConfig)
    prices = pd.read_parquet(CURATED)
    prices = prices[prices["is_tradable"]]
    md = MarketData.from_curated(prices, ucfg.tradable_tickers,
                                 start="2021-01-04", end="2022-12-30")

    fn = BASELINES["spy_tlt_60_40"].build(md, {})
    cfg = SimulationConfig(hold_days=30, max_drawdown=0.25, risk_enabled=False)
    traj = simulate(md, fn, cfg).trajectory
    assert traj.loc["2022"]["drawdown"].max() > 0.15, (
        "60/40's 2022 drawdown looks mild; suspect the total-return adjustment"
    )


# ------------------------------------------------------------- initial state


def test_the_peak_is_inherited_rather_than_reset(market, constraints):
    """Setting `peak = nav` at reset would hand every episode a fresh zero drawdown and
    teach the agent that drawdown resets for free."""
    fn = BASELINES["cash"].build(market, {})
    cfg = SimulationConfig(hold_days=30, max_drawdown=0.15, risk_enabled=False,
                           initial_cash=1_000_000.0)
    res = simulate(market, fn, cfg, initial_peak=2_000_000.0)
    traj = res.trajectory
    assert traj["peak_nav"].iloc[0] == pytest.approx(2_000_000.0)
    assert traj["drawdown"].iloc[0] == pytest.approx(0.5), (
        "the episode began at a fresh zero drawdown despite an inherited peak"
    )


def test_a_supplied_initial_ledger_is_honoured(market, constraints):
    ledger = Ledger(cash=500_000.0, shares={"SPY": 100.0})
    fn = BASELINES["cash"].build(market, {})
    cfg = SimulationConfig(hold_days=30, max_drawdown=0.25, risk_enabled=False)
    res = simulate(market, fn, cfg, initial_ledger=ledger)
    # `cash` asks for 100% cash, so SPY is sold at the first open and never rebought.
    assert res.trajectory["shares_SPY"].iloc[0] == pytest.approx(0.0)
    assert res.trajectory["nav"].iloc[0] > 500_000.0
