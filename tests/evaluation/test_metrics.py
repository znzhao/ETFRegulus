"""The standard metric set, and the Stage 5 artifacts.

Every backtest goes through this code, so a wrong metric is wrong everywhere at once. The
hand-computable cases below are the defence against that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_metrics, drawdown_episodes, summarize


def make_traj(nav: list[float]) -> pd.DataFrame:
    nav_s = pd.Series(nav, index=pd.bdate_range("2020-01-01", periods=len(nav)),
                      name="nav")
    peak = nav_s.cummax()
    return pd.DataFrame({
        "nav": nav_s, "peak_nav": peak, "drawdown": 1.0 - nav_s / peak,
        "cash": 0.0, "turnover": 0.0,
    })


def test_metrics_on_a_hand_computable_path():
    traj = make_traj([100.0, 110.0, 88.0, 121.0])
    m = compute_metrics(traj, d_max=0.10)

    assert m["cumulative_return"] == pytest.approx(0.21)
    assert m["final_nav"] == pytest.approx(121.0)
    # Peak 110, trough 88 -> 20% drawdown.
    assert m["max_drawdown"] == pytest.approx(0.20)
    assert m["worst_1d"] == pytest.approx(-0.20)
    assert m["n_dmax_breaches"] == 1
    assert m["worst_breach_depth"] == pytest.approx(0.20)


def test_a_monotone_path_has_no_drawdown():
    m = compute_metrics(make_traj([100.0, 101.0, 102.0, 103.0]))
    assert m["max_drawdown"] == pytest.approx(0.0)
    assert m["time_under_water"] == pytest.approx(0.0)
    assert m["drawdown_duration_max"] == 0


def test_a_flat_path_has_zero_volatility_and_no_divide_by_zero():
    m = compute_metrics(make_traj([100.0] * 10))
    assert m["volatility"] == pytest.approx(0.0)
    assert m["sharpe"] == 0.0 and m["calmar"] == 0.0 and m["sortino"] == 0.0
    assert np.isfinite(list(m.values())).all() if all(
        isinstance(v, (int, float)) for v in m.values()) else True


def test_drawdown_episodes_are_contiguous_runs():
    dd = pd.Series([0.0, 0.1, 0.2, 0.0, 0.0, 0.05, 0.0],
                   index=pd.bdate_range("2020-01-01", periods=7))
    episodes = drawdown_episodes(dd)
    assert len(episodes) == 2
    assert episodes[0]["depth"] == pytest.approx(0.2)
    assert episodes[1]["depth"] == pytest.approx(0.05)


def test_summarize_carries_the_constraint_counters():
    """`lock_violations == 0` and `feasibility_violations == 0` are hard acceptance
    criteria, so they must survive into the summary rather than being diagnostic colour."""
    traj = make_traj([100.0, 105.0])
    out = summarize(traj, {"lock_violations": 0, "feasibility_violations": 0,
                           "hold_days": 30, "max_drawdown": 0.15}, d_max=0.15)
    assert out["lock_violations"] == 0
    assert out["feasibility_violations"] == 0
    assert out["n_param"] == 30 and out["dmax_param"] == 0.15


# --------------------------------------------------------- the Stage 5 artifacts


def _latest_summary() -> dict | None:
    runs = sorted(Path("artifacts/runs").glob("s05_run_baselines_*/baseline_summary.json"))
    return json.loads(runs[-1].read_text(encoding="utf-8")) if runs else None


@pytest.mark.skipif(_latest_summary() is None, reason="run Stage 5 first")
def test_stage5_produced_all_six_baselines_with_zero_violations():
    s = _latest_summary()
    expected = {"spy_buy_hold", "momentum", "spy_tlt_60_40", "cash", "equal_weight",
                "classical_optimizer"}
    assert expected <= set(s["baselines"])
    for name, m in s["baselines"].items():
        assert m["lock_violations"] == 0, f"{name} violated the lock"
        assert m["feasibility_violations"] == 0, f"{name} violated feasibility"


@pytest.mark.skipif(_latest_summary() is None, reason="run Stage 5 first")
def test_every_stage5_acceptance_check_passed():
    s = _latest_summary()
    failed = [n for n, c in s["checks"].items() if not c["passed"]]
    assert failed == [], f"Stage 5 acceptance checks failed: {failed}"
    assert len(s["checks"]) >= 5


@pytest.mark.skipif(_latest_summary() is None, reason="run Stage 5 first")
def test_the_calibration_closed_q1_with_an_admissible_candidate():
    """Q1: the envelope must be neither ornamental nor paralyzing."""
    s = _latest_summary()
    cal = s["calibration"]
    assert cal is not None, "the calibration did not run"
    assert cal["chosen"] is not None, (
        "no candidate was admissible; the envelope is over-calibrated and Stage 7 would "
        "train an agent that just sits in cash"
    )
    chosen = cal["chosen"]
    assert chosen["intervention_at_primary"] <= cal["ceiling"]
    assert chosen["drawdown_responds_to_dmax"], "a tighter D_max did not reduce drawdown"
    # Selected by design constraint, not by return.
    assert chosen["horizon_days"] >= 5, "the horizon must span more than the overnight gap"
    assert chosen["aggregation"] == "max", "a hard constraint takes the conservative aggregation"


@pytest.mark.skipif(_latest_summary() is None, reason="run Stage 5 first")
def test_the_calibration_used_annual_windows():
    """On one 21-year path the intervention rate just measures time under water, because
    the peak never resets. See reference/risk-envelope.md section 7."""
    cal = _latest_summary()["calibration"]
    assert cal["window"] == "annual"
    assert cal["n_cells"] > 100


@pytest.mark.skipif(_latest_summary() is None, reason="run Stage 5 first")
def test_the_reset_reservoir_is_populated_and_reachable():
    """Approach A: reachable BY CONSTRUCTION, because every state came out of a legal
    trajectory through this very simulator (reference/env-mdp.md section 5)."""
    from src.portfolio.ledger import Ledger
    from src.portfolio.lock_manager import LockManager

    runs = sorted(Path("artifacts/runs").glob("s05_run_baselines_*/reset_reservoir.json"))
    states = json.loads(runs[-1].read_text(encoding="utf-8"))
    assert len(states) > 100, f"only {len(states)} reservoir states"

    for state in states[:: max(1, len(states) // 50)]:
        ledger = Ledger.from_dict(state["ledger"])
        lm = LockManager.from_dict(state["lock_manager"])
        ledger.check()
        assert state["peak"] >= state["nav"] - 1e-6, "a peak below the NAV it caps"
        assert 0.0 <= state["drawdown"] < 1.0
        # L2: an unlock date exists iff a position is held.
        for ticker in lm.unlock_dates:
            assert ledger.get(ticker) > 0, (
                f"{ticker} carries an unlock date with no position -- unreachable state"
            )
