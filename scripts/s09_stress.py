"""Stage 9 -- the stress suite.

    python -m scripts.s09_stress --config config/evaluation.yaml --policy <dir>
    python -m scripts.s09_stress --config config/evaluation.yaml --grid full

Crisis windows, `N` sensitivity, `D_max` sensitivity, and the combined grid
(reference/robustness.md section 1). One trajectory per cell, plus a grid summary.

**The gate:** realized max drawdown must be monotone non-decreasing in `D_max`. A tighter
ceiling producing a deeper realized drawdown means the safety layer is not doing what it
claims, and it blocks everything downstream.

Two labels this stage refuses to omit:

* cells at `N in {0, 7, 90, 180}` are **out-of-distribution** -- the policy was never
  trained there, so they characterize degradation rather than measure performance;
* the crisis windows **overlap the training data** for every policy this project can
  produce, because all folds train from 2004. That is a limitation of a 20-year dataset,
  and it is stated rather than left for a reader to infer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.cli.stage import RUNS, StageContext, StageError, stage
from src.evaluation.rollout import load_policy, rollout, window_rows
from src.evaluation.stress import (
    D_SWEEP,
    GRID_D_DEFAULT,
    GRID_N_DEFAULT,
    MONOTONICITY_TOLERANCE,
    N_SWEEP,
    Cell,
    cell_metrics,
    check_monotonicity,
    is_ood,
    lock_is_binding,
    to_frame,
)
from src.evaluation.violations import summarize as summarize_violations


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy", default=None,
                   help="policy directory; defaults to the newest Stage 8 selection")
    p.add_argument("--grid", choices=("default", "full"), default="default")
    p.add_argument("--window", default=None,
                   help="evaluation window for the sweeps, as START:END")
    p.add_argument("--skip-crisis", action="store_true")
    p.add_argument("--tolerance", type=float, default=MONOTONICITY_TOLERANCE)


def newest_stage8_policy() -> Path:
    """The selected policy from the latest Stage 8 fold -- the one that passed selection."""
    runs = sorted(RUNS.glob("s08_walk_forward_*/walk_forward_summary.json"))
    if not runs:
        raise StageError(
            "no Stage 8 run found, and Stage 9 is gated on it (D12): stress-testing a "
            "policy that has not passed walk-forward measures nothing. Run "
            "`python -m scripts.s08_walk_forward --config config/evaluation.yaml` first.")
    summary = json.loads(runs[-1].read_text(encoding="utf-8"))
    if not summary["hard_acceptance"]["passed"]:
        raise StageError(
            f"the latest Stage 8 run failed hard acceptance: "
            f"{summary['hard_acceptance']}. Phase 3 does not start until it passes.")
    last = summary["folds"][-1]
    chosen = next(c for c in
                  json.loads((runs[-1].parent / "folds" / str(last["test_year"]) /
                              "selection.json").read_text(encoding="utf-8"))["candidates"]
                  if c["name"] == last["selected"])
    return Path(chosen["params"]["policy_dir"])


def evaluate(model, bundle, *, hold_days: int, d_max: float, window, label: str,
             seed: int) -> Cell:
    start_row, end_row = window_rows(bundle.market, *window)
    result = rollout(model, bundle, hold_days=hold_days, max_drawdown=d_max,
                     start_row=start_row, end_row=end_row, seed=seed)
    violations = summarize_violations(result.trajectory, bundle.market.universe,
                                      d_max=d_max, market=bundle.market)
    metrics = cell_metrics(result.trajectory, violations, bundle.market.universe)
    metrics["lock_violations"] = result.diagnostics["lock_violations"]
    metrics["feasibility_violations"] = result.diagnostics["feasibility_violations"]
    return Cell(hold_days=hold_days, max_drawdown=d_max, window=label,
                metrics=metrics, out_of_distribution=is_ood(hold_days))


@stage(
    name="s09_stress",
    config_default="config/evaluation.yaml",
    inputs=["data/curated/prices.parquet"],
    outputs=["artifacts/runs/{run_id}/stress_summary.json"],
    upstream="s08_walk_forward",
    add_args=_add_args,
)
def main(resolved: dict, ctx: StageContext) -> None:
    """Crisis windows, parameter sweeps, and the D_max monotonicity gate."""
    from src.env.factory import build_bundle

    args = ctx.args
    policy_dir = Path(args.policy) if args.policy else newest_stage8_policy()
    ctx.log(f"policy: {policy_dir}")

    # The last fold's scaler, which is the one this policy was trained under.
    fold_id = (resolved.get("environment", {}) or {}).get("fold_id")
    bundle = build_bundle(ctx.config_path, resolved=resolved, fold_id=fold_id)
    model = load_policy(policy_dir)
    sessions = bundle.market.sessions
    ctx.log(f"market {sessions[0].date()}..{sessions[-1].date()}  "
            f"fold={bundle.store.fold_id}  scaler fit {bundle.store.fit_range}")

    if args.window:
        start, end = args.window.split(":")
    else:
        # The sweeps run over the walk-forward test span, so they are comparable to
        # Stage 8's numbers rather than to a different slice of history.
        wf = resolved.get("walk_forward", {}) or {}
        start = f"{wf.get('first_test_year', 2012)}-01-01"
        end = str(sessions[-1].date())
    sweep_window = (start, end)
    ctx.log(f"sweep window: {start} .. {end}")

    primary_n = bundle.constraints.lock.hold_days.primary
    primary_d = bundle.constraints.drawdown.max_drawdown.primary
    report: dict = {"policy": str(policy_dir), "sweep_window": [start, end],
                    "fold_id": bundle.store.fold_id,
                    "scaler_fit_range": list(bundle.store.fit_range or ())}

    # ---------------------------------------------------------- crisis windows
    crisis_cells: list[Cell] = []
    if not args.skip_crisis:
        windows = bundle.constraints.risk.crisis_windows
        ctx.log(f"crisis windows: {len(windows)}")
        for name, (c_start, c_end) in windows.items():
            if pd.Timestamp(c_start) < sessions[0] or pd.Timestamp(c_end) > sessions[-1]:
                ctx.warn(f"  {name}: outside the available calendar, skipped")
                continue
            try:
                cell = evaluate(model, bundle, hold_days=primary_n, d_max=primary_d,
                                window=(c_start, c_end), label=name, seed=ctx.seed)
            except ValueError as exc:
                ctx.warn(f"  {name}: {exc}")
                continue
            crisis_cells.append(cell)
            ctx.log(f"  {name:<14} {c_start}..{c_end}  "
                    f"return {cell.metrics['total_return']:>+7.2%}  "
                    f"maxDD {cell.metrics['max_drawdown']:>6.2%}  "
                    f"locked {cell.metrics['locked_nav_fraction']:>5.1%}  "
                    f"blocked {cell.metrics['actions_blocked_by_lock']:>4}")
        report["crisis"] = [c.to_dict() for c in crisis_cells]
        report["crisis_caveat"] = (
            "Every crisis window overlaps the policy's training data: all walk-forward "
            "folds train from 2004, so no policy this project produces has an "
            "out-of-sample 2008 or 2020. These cells show behaviour under stress, not "
            "out-of-sample performance.")

    # ------------------------------------------------------------- N sensitivity
    ctx.log(f"N sensitivity at D_max={primary_d} ({len(N_SWEEP)} cells)")
    n_cells = []
    for hold_days in N_SWEEP:
        cell = evaluate(model, bundle, hold_days=hold_days, d_max=primary_d,
                        window=sweep_window, label="sweep_N", seed=ctx.seed)
        n_cells.append(cell)
        ctx.log(f"  N={hold_days:<4} return {cell.metrics['total_return']:>+7.2%}  "
                f"maxDD {cell.metrics['max_drawdown']:>6.2%}  "
                f"turnover {cell.metrics['turnover']:>7.1f}  "
                f"locked {cell.metrics['locked_nav_fraction']:>5.1%}  "
                f"blocked {cell.metrics['actions_blocked_by_lock']:>5}"
                f"{'   [OUT-OF-DISTRIBUTION]' if cell.out_of_distribution else ''}")
    report["n_sensitivity"] = [c.to_dict() for c in n_cells]
    report["lock_binding"] = lock_is_binding(n_cells)
    ctx.log(f"  lock binding: {report['lock_binding'].get('binding')} "
            f"(turnover ratio low/high N = "
            f"{report['lock_binding'].get('turnover_ratio_low_to_high_N', 0):.2f})")

    # --------------------------------------------------------- D_max sensitivity
    ctx.log(f"D_max sensitivity at N={primary_n} ({len(D_SWEEP)} cells)")
    d_cells = []
    for d_max in D_SWEEP:
        cell = evaluate(model, bundle, hold_days=primary_n, d_max=d_max,
                        window=sweep_window, label="sweep_D", seed=ctx.seed)
        d_cells.append(cell)
        ctx.log(f"  D_max={d_max:<5} return {cell.metrics['total_return']:>+7.2%}  "
                f"maxDD {cell.metrics['max_drawdown']:>6.2%}  "
                f"cash {cell.metrics['mean_cash_weight']:>5.1%}  "
                f"intervention {cell.metrics['safety_intervention_rate']:>5.1%}")
    report["d_max_sensitivity"] = [c.to_dict() for c in d_cells]

    # ------------------------------------------------------------ combined grid
    grid_n = N_SWEEP if args.grid == "full" else GRID_N_DEFAULT
    grid_d = D_SWEEP if args.grid == "full" else GRID_D_DEFAULT
    ctx.log(f"combined grid: {len(grid_n)} x {len(grid_d)} = "
            f"{len(grid_n) * len(grid_d)} cells")
    grid_cells = []
    for hold_days in grid_n:
        row = []
        for d_max in grid_d:
            cell = evaluate(model, bundle, hold_days=hold_days, d_max=d_max,
                            window=sweep_window, label="grid", seed=ctx.seed)
            grid_cells.append(cell)
            row.append(f"{cell.metrics['max_drawdown']:>6.2%}")
        ctx.log(f"  N={hold_days:<4} maxDD across D_max: {' '.join(row)}"
                f"{'   [OOD]' if is_ood(hold_days) else ''}")
    report["grid"] = [c.to_dict() for c in grid_cells]

    # ------------------------------------------------------------------- the gate
    # Checked on the grid, which varies both parameters, so monotonicity is verified at
    # every N rather than only at the primary one.
    mono = check_monotonicity(grid_cells + d_cells, tolerance=args.tolerance)
    report["monotonicity"] = mono.to_dict()

    out = ctx.out_path("stress_summary.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for name, cells in (("crisis", crisis_cells), ("n_sensitivity", n_cells),
                        ("d_max_sensitivity", d_cells), ("grid", grid_cells)):
        if cells:
            to_frame(cells).to_csv(ctx.out_path(f"{name}.csv"), index=False)

    lock_v = sum(c.metrics["lock_violations"] for c in grid_cells + n_cells + d_cells)
    feas_v = sum(c.metrics["feasibility_violations"]
                 for c in grid_cells + n_cells + d_cells)
    prev_v = sum(c.metrics["preventable_breaches"]
                 for c in grid_cells + n_cells + d_cells)
    report["hard_acceptance"] = {"lock_violations": lock_v,
                                 "feasibility_violations": feas_v,
                                 "preventable_violations": prev_v}
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    ctx.record(n_cells=len(grid_cells) + len(n_cells) + len(d_cells) + len(crisis_cells),
               monotonicity_passed=mono.passed, lock_violations=lock_v,
               feasibility_violations=feas_v, preventable_violations=prev_v)

    ctx.log("")
    ctx.log(mono.explain())
    if lock_v or feas_v or prev_v:
        raise StageError(
            f"hard acceptance failed during stress: {lock_v} lock, {feas_v} feasibility, "
            f"{prev_v} preventable violation(s).")
    if not mono.passed:
        raise StageError(mono.explain() + f"\nFull grid: {out}")
    ctx.log(f"Stage 9 PASSED. Summary: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
