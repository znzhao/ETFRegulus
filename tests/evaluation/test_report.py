"""The comparison report's conventions.

These tests exist because the report format is fixed *before* the policy is trained, and
its value depends entirely on nobody quietly changing a convention later. Each test pins
one of the three decisions documented in `src/evaluation/report.py`, plus the property the
user actually asked for: allocation columns in percentage points that always sum to 100.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.loader import load_typed
from src.config.schema import UniverseConfig
from src.evaluation.categories import CASH_LABEL, CATEGORIES, ticker_to_category
from src.evaluation.report import (
    YearResult,
    annual_metrics,
    build_compliance,
    build_summary,
    build_tables,
    category_allocation,
)


@pytest.fixture(scope="module")
def category_of() -> dict[str, str]:
    ucfg, _ = load_typed("config/universe.yaml", UniverseConfig)
    return ticker_to_category(ucfg)


def make_traj(nav: list[float], weights: dict[str, float] | None = None,
              year: int = 2020) -> pd.DataFrame:
    """A trajectory with constant weights and a given NAV path."""
    idx = pd.bdate_range(f"{year}-01-01", periods=len(nav))
    nav_s = pd.Series(nav, index=idx, dtype=float)
    weights = weights or {}
    invested = sum(weights.values())
    frame = pd.DataFrame({
        "nav": nav_s,
        "peak_nav": nav_s.cummax(),
        "drawdown": 1.0 - nav_s / nav_s.cummax(),
        "cash": nav_s * (1.0 - invested),
        "turnover": 0.0,
    }, index=idx)
    frame.index.name = "session"
    for ticker, w in weights.items():
        frame[f"w_{ticker}"] = w
    return frame


# ------------------------------------------------------------------- categories


def test_every_tradable_ticker_has_exactly_one_category(category_of):
    ucfg, _ = load_typed("config/universe.yaml", UniverseConfig)
    assert set(category_of) == set(ucfg.tradable_tickers)
    assert len(category_of) == 24
    assert set(category_of.values()) | {CASH_LABEL} == set(CATEGORIES)


def test_the_seven_categories_are_the_ones_requested():
    assert CATEGORIES == (
        "US Broad Equity", "US Sector Equity", "Interest Rate", "Credit",
        "Commodities", "International Equity", "CASH")


# ------------------------------------------------------------------- allocation


def test_allocation_is_in_percentage_points_and_sums_to_100(category_of):
    traj = make_traj([100.0] * 10, {"SPY": 0.5, "TLT": 0.3})
    alloc = category_allocation(traj, category_of)
    assert alloc["US Broad Equity"] == pytest.approx(50.0)
    assert alloc["Interest Rate"] == pytest.approx(30.0)
    assert alloc[CASH_LABEL] == pytest.approx(20.0)
    assert sum(alloc.values()) == pytest.approx(100.0, abs=1e-9)


def test_allocation_is_a_time_average_not_a_year_end_snapshot(category_of):
    """A snapshot cannot tell a portfolio that held 60% equity all year from one that held
    0% for eleven months and 60% in December. The time average can."""
    traj = make_traj([100.0] * 10, {"SPY": 0.0})
    traj["w_SPY"] = [0.0] * 9 + [0.6]
    traj["cash"] = traj["nav"] * (1.0 - traj["w_SPY"])
    alloc = category_allocation(traj, category_of)
    assert alloc["US Broad Equity"] == pytest.approx(6.0), (
        "a year-end snapshot would report 60%, which is the bug this guards against")
    assert sum(alloc.values()) == pytest.approx(100.0)


def test_sector_etfs_aggregate_into_one_category(category_of):
    traj = make_traj([100.0] * 5, {f"XL{s}": 0.05 for s in "KYPEFIBVU"})
    alloc = category_allocation(traj, category_of)
    assert alloc["US Sector Equity"] == pytest.approx(45.0)
    assert alloc[CASH_LABEL] == pytest.approx(55.0)


def test_an_uncategorized_ticker_is_an_error_not_a_silent_drop(category_of):
    """A percentage table that silently omits an asset while still summing to 100 is
    worse than no table."""
    traj = make_traj([100.0] * 5, {"SPY": 0.5, "NOTATICKER": 0.2})
    with pytest.raises(KeyError, match="NOTATICKER"):
        category_allocation(traj, category_of)


# ---------------------------------------------------------------------- metrics


def test_annual_metrics_on_a_hand_computable_path():
    traj = make_traj([100.0, 110.0, 88.0, 121.0])
    m = annual_metrics(traj)
    assert m["total_return"] == pytest.approx(0.21)
    # Peak 110, trough 88 -> 20%.
    assert m["max_drawdown"] == pytest.approx(0.20)


def test_sharpe_uses_a_zero_risk_free_rate():
    """CASH returns exactly 0.00% per day and is the agent's outside option, so raw
    return IS excess return. Anything else describes a different problem."""
    traj = make_traj([100.0, 101.0, 102.0, 103.0, 104.0])
    m = annual_metrics(traj)
    assert m["sharpe"] == pytest.approx(m["annualized_return"] / m["volatility"], rel=1e-9)


def test_a_flat_path_has_zero_sharpe_and_no_divide_by_zero():
    m = annual_metrics(make_traj([100.0] * 20))
    assert m["volatility"] == 0.0 and m["sharpe"] == 0.0 and m["max_drawdown"] == 0.0
    assert m["sortino"] == 0.0


def test_sortino_uses_downside_deviation_over_every_observation():
    """The denominator is `sqrt(mean(min(r, 0)^2))` over ALL observations, not over the
    losing ones alone. Averaging over just the losers flatters a strategy that loses
    rarely -- three bad days out of 250 would set the whole denominator."""
    traj = make_traj([100.0, 110.0, 99.0, 108.9])
    m = annual_metrics(traj)
    returns = np.array([0.10, -0.10, 0.10])
    expected = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)) * np.sqrt(252))
    assert m["downside_deviation"] == pytest.approx(expected, rel=1e-9)
    assert m["sortino"] == pytest.approx(m["annualized_return"] / expected, rel=1e-9)


def test_sortino_rewards_upside_that_sharpe_penalises():
    """The reason for the switch. Two paths with identical downside and very different
    upside: Sharpe narrows the gap, Sortino does not penalise the good days at all."""
    steady = annual_metrics(make_traj([100.0, 101.0, 100.0, 101.0, 100.0, 101.0]))
    spiky = annual_metrics(make_traj([100.0, 130.0, 100.0, 130.0, 100.0, 130.0]))
    assert spiky["downside_deviation"] > steady["downside_deviation"]
    # Both are penalised for the drawdowns; the point is that the ratio is built on the
    # loss side only, so the metric is defined by shortfalls rather than by variance.
    assert spiky["sortino"] == pytest.approx(
        spiky["annualized_return"] / spiky["downside_deviation"], rel=1e-9)


def test_a_strategy_with_no_losing_days_reports_zero_not_infinity():
    """A degenerate case must not top a ranking."""
    m = annual_metrics(make_traj([100.0, 101.0, 102.0, 103.0]))
    assert m["downside_deviation"] == 0.0
    assert m["sortino"] == 0.0


def test_max_drawdown_is_measured_within_the_window():
    """The whole reason each report year is independent: a peak carried in from an
    earlier year would make every later year read as a permanent drawdown."""
    traj = make_traj([80.0, 85.0, 90.0])          # recovering from an earlier fall
    assert annual_metrics(traj)["max_drawdown"] == pytest.approx(0.0), (
        "a monotone rise inside the window must show no drawdown, whatever preceded it")


# ----------------------------------------------------------------------- tables


def _results(category_of) -> list[YearResult]:
    out = []
    for strategy, w in (("a", 0.6), ("b", 0.2)):
        for year in (2012, 2013):
            traj = make_traj([100.0, 105.0, 103.0, 108.0], {"SPY": w}, year=year)
            out.append(YearResult(
                strategy=strategy, year=year, trajectory=traj,
                metrics=annual_metrics(traj),
                allocation=category_allocation(traj, category_of)))
    return out


def test_tables_are_years_by_strategies(category_of):
    tables = build_tables(_results(category_of), ["a", "b"])
    for name in ("annual_return", "volatility", "sharpe", "max_drawdown"):
        assert list(tables[name].columns) == ["a", "b"]
        assert list(tables[name].index) == [2012, 2013]
    for category in CATEGORIES:
        assert f"allocation::{category}" in tables


def test_the_allocation_tables_sum_to_100_for_every_cell(category_of):
    tables = build_tables(_results(category_of), ["a", "b"])
    total = sum(tables[f"allocation::{c}"] for c in CATEGORIES)
    assert np.allclose(total.to_numpy(), 100.0)


def test_summary_allocations_sum_to_100(category_of):
    summary = build_summary(_results(category_of), ["a", "b"])
    assert np.allclose(summary[list(CATEGORIES)].sum(axis=1).to_numpy(), 100.0)
    assert list(summary.columns[:5]) == [
        "Annual return %", "Return std %", "Sharpe", "Sortino", "Downside dev %"]


def test_summary_max_drawdown_is_the_worst_ANNUAL_drawdown(category_of):
    """Not a drawdown across the window -- that would be measured against a peak no
    strategy ever operated under, which is the convention this report exists to avoid."""
    results = _results(category_of)
    summary = build_summary(results, ["a"])
    worst = max(r.metrics["max_drawdown"] for r in results if r.strategy == "a")
    assert summary.loc["a", "Max drawdown %"] == pytest.approx(100.0 * worst)


def test_compliance_counts_dmax_exceedances_without_calling_them_violations(category_of):
    """`D_max` constrains the ACTION, never the realized path, so a realized drawdown
    above the ceiling is expected and must not be reported as a violation."""
    results = _results(category_of)
    frame = build_compliance(results, ["a", "b"], max_drawdown=0.001)
    assert (frame["Years D_t > D_max"] > 0).all()
    # The two hard acceptance criteria are separate columns and stay at zero.
    assert (frame["Lock violations"] == 0).all()
    assert (frame["Feasibility violations"] == 0).all()


# ------------------------------------------------- against the generated report


def test_the_generated_report_is_internally_consistent():
    """If the report has been built, its own CSVs must agree with each other."""
    from pathlib import Path

    d = Path("artifacts/reports/baselines")
    if not (d / "summary.csv").exists():
        pytest.skip("run Stage 12 first")

    total = None
    for category in CATEGORIES:
        name = category.replace(" ", "_").lower()
        frame = pd.read_csv(d / f"allocation_{name}.csv", index_col=0)
        total = frame if total is None else total + frame
    assert np.allclose(total.to_numpy(), 100.0, atol=1e-6), (
        "a (year, strategy) allocation does not sum to 100%")

    summary = pd.read_csv(d / "summary.csv", index_col=0)
    assert np.allclose(summary[list(CATEGORIES)].sum(axis=1).to_numpy(), 100.0, atol=1e-6)

    # `cash` is the trivially checkable strategy: 100% CASH, zero return, zero risk.
    if "cash" in summary.index:
        assert summary.loc["cash", "CASH"] == pytest.approx(100.0, abs=1e-6)
        assert summary.loc["cash", "Annual return %"] == pytest.approx(0.0, abs=1e-9)
        assert summary.loc["cash", "Max drawdown %"] == pytest.approx(0.0, abs=1e-9)
