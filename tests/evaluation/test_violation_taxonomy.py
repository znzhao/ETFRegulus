"""T13 -- the preventable-violation detector fires on a deliberately injected bug.

reference/testing.md is explicit about why this test exists: *an invariant checker that has
never been observed to fail is untested infrastructure*. So the tests here do not merely
check that a clean trajectory reports clean. They break the projection on purpose, in the
specific ways it could really break, and assert the detector catches each one.

The distinction being defended is the one from evaluation.md section 4. A **market-forced**
breach is legitimate -- the ceiling constrains the action, never the realized path. A
**preventable** breach is a project-blocking defect. Conflating them is how a broken safety
layer gets excused as bad luck, so the two are never summed into a single count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.violations import (
    BreachRecord,
    breach_episodes,
    classify,
    replay_feasibility,
    summarize,
)

UNIVERSE = ["SPY", "TLT", "GLD"]


def make_traj(*, nav, shares=None, locked=None, proj=None, capital_preservation=None,
              share_floor_binding=None) -> pd.DataFrame:
    """A legal trajectory in the Stage 4 shape, which tests then corrupt."""
    n = len(nav)
    idx = pd.bdate_range("2020-01-01", periods=n, name="session")
    nav_s = pd.Series(nav, index=idx, dtype=float)
    peak = nav_s.cummax()
    frame = pd.DataFrame({
        "nav": nav_s, "peak_nav": peak, "drawdown": 1.0 - nav_s / peak,
        "cash": nav_s * 0.10, "turnover": 0.0, "safety_intervened": False,
        "capital_preservation": capital_preservation
        if capital_preservation is not None else [False] * n,
        "share_floor_binding": share_floor_binding
        if share_floor_binding is not None else [0] * n,
    }, index=idx)
    default_w = 0.90 / len(UNIVERSE)
    for j, ticker in enumerate(UNIVERSE):
        frame[f"w_{ticker}"] = default_w
        frame[f"shares_{ticker}"] = (shares[j] if shares else [100.0] * n)
        frame[f"locked_{ticker}"] = (locked[j] if locked else [False] * n)
    legal = np.array([0.10, *([default_w] * len(UNIVERSE))])
    frame["proj_weights"] = [np.array(proj[i]) if proj else legal.copy()
                             for i in range(n)]
    return frame


# ------------------------------------------------------- a clean trajectory


def test_a_legal_trajectory_produces_no_findings():
    assert replay_feasibility(make_traj(nav=[100, 101, 102, 103]), UNIVERSE,
                              d_max=0.15) == []


def test_a_trajectory_without_the_projected_action_cannot_be_replayed():
    """Better to refuse than to silently check something weaker."""
    traj = make_traj(nav=[100, 101]).drop(columns=["proj_weights"])
    with pytest.raises(KeyError, match="proj_weights"):
        replay_feasibility(traj, UNIVERSE, d_max=0.15)


# ------------------------------------------------ the injected projection bugs


def test_the_detector_fires_on_a_projection_that_returns_a_negative_weight():
    """Injected bug: a projector that lets a short position through."""
    broken = [0.10, 0.40, 0.40, 0.10]
    broken[2] = -0.05                              # short TLT
    traj = make_traj(nav=[100, 99, 98], proj=[broken, broken, broken])
    findings = replay_feasibility(traj, UNIVERSE, d_max=0.15)
    assert any(f.kind == "negative_weight" for f in findings), findings


def test_the_detector_fires_on_a_projection_that_leaves_the_simplex():
    """Injected bug: weights that no longer sum to one -- leverage, in effect."""
    over = [0.10, 0.60, 0.60, 0.10]                # sums to 1.40
    traj = make_traj(nav=[100, 99], proj=[over, over])
    findings = replay_feasibility(traj, UNIVERSE, d_max=0.15)
    assert any(f.kind == "not_on_simplex" for f in findings)
    assert findings[0].magnitude == pytest.approx(0.40, abs=1e-9)


def test_the_detector_fires_when_a_locked_position_shrinks():
    """Injected bug: the lock floor was not applied, so a locked position was sold."""
    traj = make_traj(
        nav=[100, 95, 90],
        shares=[[100.0, 60.0, 60.0], [50.0] * 3, [50.0] * 3],   # SPY sold at step 2
        locked=[[True] * 3, [False] * 3, [False] * 3])
    findings = replay_feasibility(traj, UNIVERSE, d_max=0.15)
    assert any(f.kind == "lock_floor_breached" for f in findings), findings
    assert any("SPY" in f.detail for f in findings)


def test_the_detector_fires_when_shares_grow_during_capital_preservation():
    """Injected bug: the cap was applied to WEIGHTS rather than share counts, which in a
    falling market reads as an instruction to buy the dip -- the real Phase 1 bug."""
    traj = make_traj(
        nav=[100, 80, 78],
        shares=[[100.0, 100.0, 400.0], [50.0] * 3, [50.0] * 3],
        capital_preservation=[True, True, True])
    findings = replay_feasibility(traj, UNIVERSE, d_max=0.15)
    assert any(f.kind == "preservation_cap_breached" for f in findings), findings


def test_a_clean_trajectory_stays_clean_under_capital_preservation():
    """The converse: preservation itself must not be reported as a violation."""
    traj = make_traj(nav=[100, 80, 78],
                     shares=[[100.0, 100.0, 90.0]] * 3,
                     capital_preservation=[True, True, True])
    assert replay_feasibility(traj, UNIVERSE, d_max=0.15) == []


# ------------------------------------------------------------ classification


def test_breach_episodes_are_contiguous_runs_past_the_ceiling():
    traj = make_traj(nav=[100, 100, 80, 79, 100, 100, 82])
    episodes = breach_episodes(traj, d_max=0.15)
    assert len(episodes) == 2


def test_a_breach_with_no_illegal_action_is_market_forced():
    """The honest consequence of an action-level constraint: legal behaviour, and prices
    that exceeded the limit anyway."""
    traj = make_traj(nav=[100, 100, 80, 79, 85],
                     locked=[[True] * 5, [False] * 5, [False] * 5],
                     share_floor_binding=[0, 0, 3, 2, 0])
    records = classify(traj, UNIVERSE, d_max=0.15)
    assert len(records) == 1
    record = records[0]
    assert record.classification == "market_forced"
    assert record.breach_depth > 0.15
    # Two sessions, not three: the final NAV of 85 is a drawdown of exactly 0.15, which
    # is at the ceiling and not over it. The boundary belongs to the legal side.
    assert record.breach_duration == 2
    # The field that carries the argument: what the agent tried to do and could not.
    assert record.actions_blocked_by_lock == 5
    assert record.locked_exposure_at_breach > 0
    assert 0.0 <= record.available_cash_at_breach <= 1.0


def test_a_breach_containing_an_illegal_action_is_preventable():
    traj = make_traj(
        nav=[100, 100, 80, 79],
        shares=[[100.0, 100.0, 50.0, 50.0], [50.0] * 4, [50.0] * 4],
        locked=[[False, True, True, True], [False] * 4, [False] * 4])
    records = classify(traj, UNIVERSE, d_max=0.15)
    assert records and records[0].classification == "preventable"
    assert records[0].preventable_findings


def test_every_required_breach_field_is_recorded():
    """evaluation.md section 4 lists these explicitly as a required record."""
    required = {"breach_start", "breach_depth", "breach_duration",
                "locked_exposure_at_breach", "available_cash_at_breach",
                "actions_blocked_by_lock"}
    traj = make_traj(nav=[100, 100, 80, 85])
    record = classify(traj, UNIVERSE, d_max=0.15)[0].to_dict()
    assert required <= set(record)


# ---------------------------------------------------------------- the summary


def test_preventable_and_market_forced_are_never_summed():
    traj = make_traj(nav=[100, 100, 80, 79, 85])
    out = summarize(traj, UNIVERSE, d_max=0.15)
    assert "n_preventable" in out and "n_market_forced" in out
    assert "n_violations" not in out, (
        "a single combined count is exactly what the taxonomy exists to prevent")


def test_the_summary_reports_admissibility_for_model_selection():
    """`all_breaches_market_forced` is what Level 1 of the selection rule reads."""
    clean = summarize(make_traj(nav=[100, 100, 80, 85]), UNIVERSE, d_max=0.15)
    assert clean["all_breaches_market_forced"] is True
    assert clean["n_breaches"] >= 1

    broken = summarize(
        make_traj(nav=[100, 100, 80, 79],
                  shares=[[100.0, 100.0, 50.0, 50.0], [50.0] * 4, [50.0] * 4],
                  locked=[[False, True, True, True], [False] * 4, [False] * 4]),
        UNIVERSE, d_max=0.15)
    assert broken["all_breaches_market_forced"] is False


def test_an_illegal_action_outside_any_breach_is_still_reported():
    """The same defect, caught before it happened to cost anything."""
    traj = make_traj(nav=[100, 101, 102],
                     shares=[[100.0, 50.0, 50.0], [50.0] * 3, [50.0] * 3],
                     locked=[[True] * 3, [False] * 3, [False] * 3])
    out = summarize(traj, UNIVERSE, d_max=0.15)
    assert out["n_breaches"] == 0
    assert out["n_replay_findings"] > 0


# ----------------------------------------------- against a real trajectory


@pytest.fixture(scope="module")
def real_market():
    from pathlib import Path

    from src.config.loader import resolve_config
    from src.sim.runner import load_market

    if not Path("data/curated/prices.parquet").exists():
        pytest.skip("run Stage 2 first")
    market, _ = load_market(resolve_config(Path("config/evaluation.yaml")))
    return market


def test_the_stage_5_baselines_contain_no_preventable_violations(real_market):
    """The detector, pointed at real trajectories produced by the real constraint layer.

    `market` is not optional here. Without it the detector cannot separate a reinvested
    distribution from a purchase, and every dividend paid inside a capital-preservation
    window is reported as a cap breach -- which is exactly what happened the first time
    this ran.
    """
    import glob

    runs = sorted(glob.glob("artifacts/reports/baselines/trajectories/*/*.parquet"))
    if not runs:
        pytest.skip("run Stage 12 first")
    from src.config.loader import load_typed
    from src.config.schema import UniverseConfig

    ucfg, _ = load_typed("config/universe.yaml", UniverseConfig)
    for path in runs[::11]:
        traj = pd.read_parquet(path)
        if "proj_weights" not in traj.columns:
            pytest.skip("trajectories predate the proj_weights column; rerun Stage 12")
        out = summarize(traj, list(ucfg.tradable_tickers), d_max=0.15,
                        market=real_market)
        assert out["n_replay_findings"] == 0, f"{path}: {out['replay_findings'][:2]}"
        assert out["n_preventable"] == 0


def test_a_reinvested_distribution_is_not_a_capital_preservation_breach(real_market):
    """The carve-out, on the case that actually caught it.

    XLE paid $0.2637 on 2020-03-23 into a collapsed $11.79 price -- a 2.24% one-day share
    accretion while the portfolio was in capital preservation. Share accretion from a
    distribution is a corporate action, not a trade: `Ledger.accrue_shares` is a separate
    entry point precisely so the lock manager never observes it, and the preservation cap
    must be equally blind to it. No fixed threshold would have separated 2.24% from real
    dip-buying, which is why the detector does the exact inversion instead.
    """
    import glob

    from src.config.loader import load_typed
    from src.config.schema import UniverseConfig

    matches = glob.glob(
        "artifacts/reports/baselines/trajectories/*/2020.parquet")
    if not matches:
        pytest.skip("run Stage 12 first")
    ucfg, _ = load_typed("config/universe.yaml", UniverseConfig)
    universe = list(ucfg.tradable_tickers)

    fired_without_market = 0
    for path in matches:
        traj = pd.read_parquet(path)
        if "proj_weights" not in traj.columns:
            pytest.skip("trajectories predate the proj_weights column")
        assert replay_feasibility(traj, universe, d_max=0.15,
                                  market=real_market) == [], path
        fired_without_market += len(
            replay_feasibility(traj, universe, d_max=0.15))
    assert fired_without_market > 0, (
        "2020 should contain distributions paid during capital preservation; without "
        "the market this detector is expected to false-positive on them, and if it no "
        "longer does, this test has stopped exercising the carve-out")
