"""Stage 9 -- the stress suite: crisis windows, parameter sweeps, and the monotonicity gate.

reference/robustness.md section 1. The final model must not report only average
out-of-sample return, so this evaluates the same policy across historical crises and
across the whole `(N, D_max)` grid it claims to serve.

**The gate is section 1.3, and it is the single most important robustness check in the
project:**

> Realized max drawdown must be **monotone non-decreasing in `D_max`**.

A tighter ceiling must not produce a deeper realized drawdown. If it does, the safety layer
is not doing what it claims and every risk number in the report is void. A small tolerance
is allowed for the stochasticity of a single seed -- and a violation beyond tolerance means
run more seeds before concluding it is a bug, then treat it as one.

Two things this module is careful to label rather than bury:

* **Out-of-distribution cells.** `N` is swept over `{0, 7, 90, 180}` as well as the D16
  operating range. The policy was never trained there, so those cells characterize
  degradation; they do not measure performance, and every output says so.
* **Crisis windows overlap training data.** The folds all train from 2004, so a 2008 or
  2020 window is in-sample for any policy this project produces. That is a real limitation
  of a 20-year dataset, and the report states it instead of presenting crisis results as
  out-of-sample evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: D16's operating range plus the out-of-distribution stress points (robustness.md 1.2).
N_SWEEP: tuple[int, ...] = (0, 7, 15, 18, 21, 25, 30, 36, 42, 50, 60, 90, 180)

#: Deliberately outside the training range. Every cell at one of these is reported as OOD.
N_OUT_OF_DISTRIBUTION: frozenset[int] = frozenset({0, 7, 90, 180})

D_SWEEP: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15)

#: The default combined grid (D5): the operating range, so it measures the system where it
#: is meant to run.
GRID_N_DEFAULT: tuple[int, ...] = (15, 30, 60)
GRID_D_DEFAULT: tuple[float, ...] = (0.01, 0.05, 0.15)

#: How much a single seed may violate monotonicity before it is called a bug.
MONOTONICITY_TOLERANCE = 0.02


@dataclass
class Cell:
    """One evaluated `(N, D_max)` combination over one window."""

    hold_days: int
    max_drawdown: float
    window: str
    metrics: dict = field(default_factory=dict)
    out_of_distribution: bool = False

    def to_dict(self) -> dict:
        return {"hold_days": self.hold_days, "max_drawdown": self.max_drawdown,
                "window": self.window,
                "out_of_distribution": self.out_of_distribution, **self.metrics}


def is_ood(hold_days: int) -> bool:
    return int(hold_days) in N_OUT_OF_DISTRIBUTION


def cell_metrics(traj: pd.DataFrame, violations: dict, universe: list[str]) -> dict:
    """The per-cell report line from robustness.md sections 1.2 and 1.3.

    `locked_nav_fraction` and `actions_blocked_by_lock` are what make an `N` sweep
    interpretable: without them a flat sweep is indistinguishable from an inert lock.
    """
    nav = traj["nav"].astype(float)
    locked_cols = [f"locked_{t}" for t in universe]
    weight_cols = [f"w_{t}" for t in universe]
    locked = traj[locked_cols].to_numpy(dtype=bool)
    weights = traj[weight_cols].to_numpy(dtype=float)

    return {
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "max_drawdown": float((1.0 - nav / nav.cummax()).max()),
        "volatility": float(nav.pct_change().std(ddof=1) * np.sqrt(252)),
        "turnover": float(traj["turnover"].sum()),
        "mean_cash_weight": float((traj["cash"].astype(float) / nav).mean()),
        # Fraction of NAV that could not be sold, averaged over the window.
        "locked_nav_fraction": float(np.where(locked, weights, 0.0).sum(axis=1).mean()),
        "actions_blocked_by_lock": int(traj.get("share_floor_binding",
                                                pd.Series([0])).sum()),
        "safety_intervention_rate": float(traj["safety_intervened"].mean()),
        "capital_preservation_rate": float(traj["capital_preservation"].mean()),
        "infeasible_fallback_rate": float(traj["infeasible_fallback"].mean()),
        "n_sessions": int(len(traj)),
        "market_forced_breaches": int(violations.get("n_market_forced", 0)),
        "preventable_breaches": int(violations.get("n_preventable", 0)),
        "n_replay_findings": int(violations.get("n_replay_findings", 0)),
    }


# ------------------------------------------------------------------- the gate


@dataclass
class MonotonicityResult:
    """Whether realized drawdown behaved as a ceiling must."""

    by_d_max: dict
    violations: list[dict]
    tolerance: float
    n_hold_days: int = 0

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict:
        return {"passed": self.passed, "tolerance": self.tolerance,
                "by_d_max": self.by_d_max, "violations": self.violations,
                "n_hold_days_checked": self.n_hold_days}

    def explain(self) -> str:
        if self.passed:
            return ("realized drawdown is monotone non-decreasing in D_max across "
                    f"{self.n_hold_days} value(s) of N (tolerance {self.tolerance})")
        worst = max(self.violations, key=lambda v: v["excess"])
        return (
            f"MONOTONICITY VIOLATED: at N={worst['hold_days']}, tightening D_max from "
            f"{worst['looser_d_max']} to {worst['tighter_d_max']} made realized drawdown "
            f"WORSE ({worst['looser_drawdown']:.4f} -> {worst['tighter_drawdown']:.4f}, "
            f"excess {worst['excess']:.4f} > tolerance {self.tolerance}). "
            "The safety layer is not doing what it claims and every risk number in the "
            "report is void. Run more seeds before concluding it is a bug -- then treat "
            "it as one."
        )


def check_monotonicity(cells: list[Cell],
                       tolerance: float = MONOTONICITY_TOLERANCE) -> MonotonicityResult:
    """Realized drawdown must not get worse as `D_max` tightens.

    Checked WITHIN each `N`, never across it: comparing a cell at `N=15` to one at `N=60`
    would confound the ceiling with the lock, and the lock genuinely can make drawdown
    worse by preventing a sale.
    """
    by_n: dict[int, dict[float, float]] = {}
    for cell in cells:
        by_n.setdefault(cell.hold_days, {})[cell.max_drawdown] = \
            float(cell.metrics.get("max_drawdown", 0.0))

    violations: list[dict] = []
    summary: dict = {}
    for hold_days, series in sorted(by_n.items()):
        ordered = sorted(series.items())
        summary[str(hold_days)] = {str(d): dd for d, dd in ordered}
        for (tight_d, tight_dd), (loose_d, loose_dd) in zip(ordered, ordered[1:]):
            excess = tight_dd - loose_dd
            if excess > tolerance:
                violations.append({
                    "hold_days": hold_days,
                    "tighter_d_max": tight_d, "looser_d_max": loose_d,
                    "tighter_drawdown": tight_dd, "looser_drawdown": loose_dd,
                    "excess": float(excess),
                })
    return MonotonicityResult(by_d_max=summary, violations=violations,
                              tolerance=tolerance, n_hold_days=len(by_n))


# ------------------------------------------------------------------- shaping


def lock_is_binding(cells: list[Cell]) -> dict:
    """Is performance actually responding to `N`, or is the constraint inert?

    robustness.md section 1.2: if performance is *flat* in `N`, the lock is probably not
    binding -- and a silently-inert constraint looks exactly like this. Turnover and the
    locked NAV fraction are the discriminators, because they respond to the lock
    mechanically rather than through the market.
    """
    by_n: dict[int, list[Cell]] = {}
    for cell in cells:
        by_n.setdefault(cell.hold_days, []).append(cell)
    if len(by_n) < 2:
        return {"checked": False, "reason": "fewer than two values of N"}

    ns = sorted(by_n)
    turnover = [float(np.mean([c.metrics["turnover"] for c in by_n[n]])) for n in ns]
    locked = [float(np.mean([c.metrics["locked_nav_fraction"] for c in by_n[n]]))
              for n in ns]

    lo, hi = turnover[0], turnover[-1]
    ratio = (lo / hi) if hi > 1e-12 else float("inf")
    return {
        "checked": True,
        "hold_days": ns,
        "turnover": turnover,
        "locked_nav_fraction": locked,
        "turnover_ratio_low_to_high_N": float(ratio),
        # A real lock cuts turnover sharply and raises the locked fraction.
        "binding": bool(ratio > 1.5 and locked[-1] > locked[0] + 0.05),
        "note": ("turnover should fall and locked NAV should rise as N grows; if both are "
                 "flat the lock lower bounds may not be reaching the projection"),
    }


def to_frame(cells: list[Cell]) -> pd.DataFrame:
    return pd.DataFrame([c.to_dict() for c in cells])
