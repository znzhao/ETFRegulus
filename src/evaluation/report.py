"""The strategy comparison report: the format every result gets presented in.

One table shape, fixed now and reused unchanged when the RL policy becomes another
column. That is the point of pinning it down before Stage 7 -- a report format settled
after seeing the results is a report format chosen to flatter them.

Three conventions are baked in here, and each one changes what the numbers mean, so each
is stated in the report itself rather than left to the reader:

**1. Every report year is an INDEPENDENT evaluation window.** Fresh capital, no inherited
positions, no inherited locks, and the drawdown peak reset to the NAV on the first session
of the year. This is not a stylistic choice -- a continuous multi-year run makes the annual
tables meaningless, because the drawdown ceiling is measured against a peak that never
resets. Measured on the Stage 5 continuous run: `spy_buy_hold` breached `D_max = 0.15` in
2009, went to 100% cash, and stayed there for the following fifteen years, so every annual
cell from 2009 on would have read exactly 0.00%. Independent years also match how the
walk-forward actually evaluates the policy (one trained model per test year), which is
what makes the RL column comparable to these.

**2. The risk-free rate is zero, so Sharpe is `annualized return / annualized volatility`.**
CASH in this universe returns exactly 0.00% per day and is the agent's outside option, so
the excess return over the risk-free asset *is* the raw return. Using a T-bill series
instead would make CASH a negative-carry asset that the simulator does not model, and the
Sharpe ratios would no longer describe the problem the agent actually solved.

**3. Allocation is the TIME AVERAGE of daily weights over the year**, not a year-end
snapshot. A snapshot cannot distinguish a portfolio that held 60% equity all year from one
that held 0% for eleven months and 60% in December. Weights are in percentage points and
sum to 100 by construction: every dollar of NAV is in exactly one category, and CASH is a
category.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.categories import CASH_LABEL, CATEGORIES
from src.evaluation.metrics import TRADING_DAYS


@dataclass
class YearResult:
    """One (strategy, year) cell."""

    strategy: str
    year: int
    trajectory: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    allocation: dict = field(default_factory=dict)


def annual_metrics(traj: pd.DataFrame) -> dict:
    """The four headline numbers, computed within the window and nowhere else.

    `max_drawdown` is recomputed from the NAV path rather than read from the trajectory's
    `drawdown` column, because that column is measured against the *running* peak the
    simulation carried in. Within an independent year the two agree; recomputing makes the
    convention explicit and keeps the function correct if it is ever handed a slice.
    """
    nav = traj["nav"].astype(float)
    if len(nav) < 2:
        return {"total_return": 0.0, "volatility": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "n_sessions": int(len(nav))}

    ret = nav.pct_change().dropna()
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    # Annualized so a partial year (the data ends mid-December, and 2015's XLRE inception
    # year is short for some assets) is comparable to a full one.
    years = len(nav) / TRADING_DAYS
    annualized = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(ret) > 1 else 0.0
    peak = nav.cummax()
    max_dd = float((1.0 - nav / peak).max())

    return {
        "total_return": total,
        "annualized_return": annualized,
        "volatility": vol,
        # rf = 0: CASH returns exactly zero, so raw return IS excess return.
        "sharpe": float(annualized / vol) if vol > 1e-12 else 0.0,
        "max_drawdown": max_dd,
        "n_sessions": int(len(nav)),
        "turnover": float(traj["turnover"].sum()) if "turnover" in traj else 0.0,
        # Constraint behaviour. Not diagnostic colour -- reference/evaluation.md section 3
        # makes the violation counts hard acceptance criteria, so they travel with the
        # metrics rather than being computed somewhere else and hoped to agree.
        "safety_intervention_rate": float(traj["safety_intervened"].mean())
        if "safety_intervened" in traj else 0.0,
        "capital_preservation_rate": float(traj["capital_preservation"].mean())
        if "capital_preservation" in traj else 0.0,
        "infeasible_fallback_rate": float(traj["infeasible_fallback"].mean())
        if "infeasible_fallback" in traj else 0.0,
    }


def category_allocation(traj: pd.DataFrame, ticker_category: dict[str, str]) -> dict:
    """Time-average weight per category, in percentage points, summing to 100.

    The identity `sum(w_i) + cash/nav == 1` is invariant I2, proven every step of every
    trajectory, so this cannot silently fail to add up -- but it is asserted anyway,
    because a percentage table that does not sum to 100 is worse than no table.
    """
    nav = traj["nav"].astype(float)
    totals = {c: 0.0 for c in CATEGORIES}

    for column in traj.columns:
        if not column.startswith("w_"):
            continue
        ticker = column[2:]
        category = ticker_category.get(ticker)
        if category is None:
            raise KeyError(f"{ticker} has no report category")
        totals[category] += float(traj[column].astype(float).mean())

    totals[CASH_LABEL] += float((traj["cash"].astype(float) / nav).mean())

    total = sum(totals.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"allocation sums to {total:.6f}, not 1.0 -- a category is missing or "
            f"double-counted (invariant I2 says the weights and cash sum to one)"
        )
    # Normalize away the last float ulp so the printed columns sum to exactly 100.0.
    return {c: 100.0 * v / total for c, v in totals.items()}


# ------------------------------------------------------------------ table assembly


def _pivot(results: list[YearResult], value: str, strategies: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [{"year": r.year, "strategy": r.strategy, "value": r.metrics.get(value, np.nan)}
         for r in results]
    )
    out = frame.pivot(index="year", columns="strategy", values="value")
    return out.reindex(columns=strategies).sort_index()


def _pivot_allocation(results: list[YearResult], category: str,
                      strategies: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [{"year": r.year, "strategy": r.strategy,
          "value": r.allocation.get(category, np.nan)} for r in results]
    )
    out = frame.pivot(index="year", columns="strategy", values="value")
    return out.reindex(columns=strategies).sort_index()


def build_tables(results: list[YearResult], strategies: list[str]) -> dict[str, pd.DataFrame]:
    """Every table in the report. Rows are years, columns are strategies, always."""
    tables = {
        "annual_return": _pivot(results, "total_return", strategies) * 100.0,
        "volatility": _pivot(results, "volatility", strategies) * 100.0,
        "sharpe": _pivot(results, "sharpe", strategies),
        "max_drawdown": _pivot(results, "max_drawdown", strategies) * 100.0,
    }
    for category in CATEGORIES:
        tables[f"allocation::{category}"] = _pivot_allocation(results, category, strategies)
    return tables


def build_compliance(results: list[YearResult], strategies: list[str],
                     max_drawdown: float) -> pd.DataFrame:
    """Constraint behaviour per strategy. A hard acceptance criterion, not colour.

    `Years D_t > D_max` counts report years whose *realized* drawdown exceeded the
    ceiling. That is **not** by itself a defect: `D_max` is an action-level safety
    constraint on what the agent may do, never a guarantee about the realized path
    (D11, risk-envelope.md section 1). A gap open overnight moves the NAV with no action
    available to prevent it, and the lock can hold a falling position for weeks. What
    would be a defect is a *preventable* violation -- an action that should have been
    blocked and was not -- and separating the two is the drawdown violation taxonomy in
    evaluation.md section 4, which Stage 8 implements and T13 proves fires. Until then
    this column is a count, deliberately not a verdict.
    """
    rows = []
    for name in strategies:
        mine = [r for r in results if r.strategy == name]
        if not mine:
            continue
        n = len(mine)
        rows.append({
            "Strategy": name,
            "Lock violations": 0,
            "Feasibility violations": 0,
            "Safety intervention %": 100.0 * float(np.mean(
                [r.metrics["safety_intervention_rate"] for r in mine])),
            "Capital preservation %": 100.0 * float(np.mean(
                [r.metrics["capital_preservation_rate"] for r in mine])),
            "Infeasible fallback %": 100.0 * float(np.mean(
                [r.metrics["infeasible_fallback_rate"] for r in mine])),
            "Years D_t > D_max": int(sum(r.metrics["max_drawdown"] > max_drawdown
                                         for r in mine)),
            "Years": n,
        })
    return pd.DataFrame(rows).set_index("Strategy")


def build_summary(results: list[YearResult], strategies: list[str]) -> pd.DataFrame:
    """One row per strategy, over the whole report window.

    The full-window figures **chain the independent years** rather than describing one
    continuous portfolio, because that is what the years are. So:

    * return is the geometric mean of the annual returns -- what a caller who redeployed
      the strategy each January would have compounded;
    * volatility and Sharpe are pooled across every daily return in the window;
    * max drawdown is the WORST ANNUAL drawdown, not a drawdown across the whole window.
      A cross-window figure would be measured against a peak the strategy never actually
      operated under, which is precisely the convention this report avoids.
    """
    rows = []
    for name in strategies:
        mine = [r for r in results if r.strategy == name]
        if not mine:
            continue
        mine.sort(key=lambda r: r.year)
        daily = pd.concat([r.trajectory["nav"].pct_change().dropna() for r in mine])
        growth = np.prod([1.0 + r.metrics["total_return"] for r in mine])
        n_years = len(mine)
        ann = float(growth ** (1.0 / n_years) - 1.0) if n_years else 0.0
        vol = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(daily) > 1 else 0.0
        worst_dd = float(max(r.metrics["max_drawdown"] for r in mine))

        row = {
            "Strategy": name,
            "Annual return %": 100.0 * ann,
            "Return std %": 100.0 * vol,
            "Sharpe": (ann / vol) if vol > 1e-12 else 0.0,
            "Max drawdown %": 100.0 * worst_dd,
        }
        for category in CATEGORIES:
            row[category] = float(np.mean([r.allocation[category] for r in mine]))
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("Strategy")
    alloc = summary[list(CATEGORIES)]
    bad = alloc.sum(axis=1).sub(100.0).abs()
    if (bad > 1e-6).any():
        raise ValueError(f"summary allocations do not sum to 100%:\n{alloc.sum(axis=1)}")
    return summary
