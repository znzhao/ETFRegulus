"""Scoring one trained policy over one walk-forward window, across the (N, D_max) grid.

Shared by Stage 8 (`scripts/s08_walk_forward.py`) and the incremental-training campaign
(`scripts/s14_incremental.py`), so a checkpoint in the campaign and a one-shot walk-forward
run are scored by the same code and their numbers are the same measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.evaluation.rollout import rollout, window_rows
from src.evaluation.violations import summarize as summarize_violations

#: The default evaluation grid: 3x3 out of the full 5x5 (D5 compute sizing).
DEFAULT_GRID_N = (15, 30, 60)
# Four ceilings, so the reports at 1/5/10/15% all read a cached cell.
DEFAULT_GRID_D = (0.01, 0.05, 0.10, 0.15)


def eval_grid(kind: str, constraints) -> list[tuple[int, float]]:
    if kind == "primary":
        return [(constraints.lock.hold_days.primary,
                 constraints.drawdown.max_drawdown.primary)]
    if kind == "full":
        return [(n, d) for n in constraints.lock.hold_days.values
                for d in constraints.drawdown.max_drawdown.values]
    return [(n, d) for n in DEFAULT_GRID_N for d in DEFAULT_GRID_D]


def cell_key(hold_days: int, d_max: float) -> str:
    return f"N{hold_days}_D{d_max}"


def evaluate_window(model, bundle, window, cells, *, seed: int, label: str,
                    log) -> tuple[dict, dict]:
    """Run a policy over one window across the parameter grid.

    Returns `(per-cell metrics, aggregate)`. The aggregate takes the WORST drawdown and
    the SUM of violations across cells, never the mean: a model that is safe on eight
    cells and broken on the ninth is a broken model.
    """
    start_row, end_row = window_rows(bundle.market, *window)
    per_cell: dict[str, dict] = {}
    trajectories: dict[str, pd.DataFrame] = {}

    for hold_days, d_max in cells:
        result = rollout(model, bundle, hold_days=hold_days, max_drawdown=d_max,
                         start_row=start_row, end_row=end_row, seed=seed)
        traj = result.trajectory
        metrics = compute_metrics(traj, d_max=d_max)
        violations = summarize_violations(traj, bundle.market.universe, d_max=d_max,
                                          market=bundle.market)
        key = cell_key(hold_days, d_max)
        per_cell[key] = {
            **metrics,
            "hold_days": hold_days, "max_drawdown_param": d_max,
            "lock_violations": result.diagnostics["lock_violations"],
            "feasibility_violations": result.diagnostics["feasibility_violations"],
            "preventable_violations": violations["n_preventable"],
            "market_forced_violations": violations["n_market_forced"],
            "n_replay_findings": violations["n_replay_findings"],
            "all_breaches_market_forced": violations["all_breaches_market_forced"],
        }
        trajectories[key] = traj

    log(f"    {label}: {len(cells)} cell(s), "
        f"worst DD {max(c['max_drawdown'] for c in per_cell.values()):.4f}, "
        f"preventable {sum(c['preventable_violations'] for c in per_cell.values())}")

    aggregate = {
        "cumulative_log_return": float(np.mean(
            [np.log1p(c["cumulative_return"]) for c in per_cell.values()])),
        "cumulative_return": float(np.mean(
            [c["cumulative_return"] for c in per_cell.values()])),
        "sharpe": float(np.mean([c["sharpe"] for c in per_cell.values()])),
        # Worst, not mean: safe on eight cells and broken on the ninth is broken.
        "max_drawdown": float(max(c["max_drawdown"] for c in per_cell.values())),
        "lock_violations": int(sum(c["lock_violations"] for c in per_cell.values())),
        "feasibility_violations": int(sum(
            c["feasibility_violations"] for c in per_cell.values())),
        "preventable_violations": int(sum(
            c["preventable_violations"] for c in per_cell.values())),
        "all_breaches_market_forced": all(
            c["all_breaches_market_forced"] for c in per_cell.values()),
        "n_cells": len(cells),
        "cells": per_cell,
    }
    return aggregate, trajectories
