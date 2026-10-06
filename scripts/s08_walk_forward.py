"""Stage 8 -- annual walk-forward. **The Phase 3 gate.**

    python -m scripts.s08_walk_forward --config config/evaluation.yaml
    python -m scripts.s08_walk_forward --config config/evaluation.yaml --folds 2012,2013 --seeds 3

Implements reference/evaluation.md: expanding-window annual folds (train `[2004..Y-2]`,
validate `Y-1`, test `Y`), lexicographic model selection (risk criteria first, return
second), and the preventable-versus-market-forced drawdown violation taxonomy.

Per fold, in this order and no other:

    1. train each candidate on [2004..Y-2], with THAT fold's scaler
    2. score every candidate on the validation year Y-1
    3. select lexicographically -- risk criteria first; return only breaks ties among
       the admissible
    4. freeze, and evaluate once on the test year Y over the (N, D_max) grid
    5. replay every test trajectory to classify any breach

Three prohibitions, each enforced by code here rather than by discipline:

* **no mid-year retraining on test-year data** -- the training window is the fold's, and
  the environment window is clamped to it;
* **no scaler fitted outside the training window** -- checked against the fold before a
  single timestep runs, and the run refuses to start if it fails;
* **no selection on test performance** -- `select()` never sees a test metric, and the
  selection record cites validation metrics only.

Writes `artifacts/runs/<run_id>/folds/<year>/{trajectory.parquet, selection.json}` and
`walk_forward_summary.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.cli.stage import StageContext, StageError, stage
from src.evaluation.fold_eval import eval_grid, evaluate_window
from src.evaluation.rollout import load_policy
from src.evaluation.violations import summarize as summarize_violations
from src.evaluation.walk_forward import (
    assert_folds_valid,
    load_folds,
    scaler_is_legal,
    select_folds,
)
from src.training.curriculum import STAGES_BY_INDEX
from src.training.model_selection import Candidate, default_candidates, select
from src.training.trainer import train_stage


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--folds", default=None, help="comma-separated test years")
    p.add_argument("--first-year", type=int, default=None)
    p.add_argument("--last-year", type=int, default=None)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--candidates", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=None,
                   help="training timesteps per candidate")
    p.add_argument("--grid", choices=("default", "full", "primary"), default="default",
                   help="evaluation (N, D_max) cells: 3x3, 5x5, or the primary cell alone")
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--no-subproc", action="store_true")


def run_fold(fold, bundle, resolved, ctx, *, cells, candidates_n, timesteps,
             n_envs, subproc, seeds) -> dict:
    """One fold, in the prescribed order. Training, validation, selection, then test."""
    log = ctx.log
    log("")
    log(f"=== fold {fold.fold_id}: train [{fold.train_start}..{fold.train_end}]  "
        f"val {fold.val_start[:4]}  test {fold.test_year} ===")

    # Prohibition 2, checked BEFORE any compute is spent.
    problems = scaler_is_legal(*bundle.store.fit_range, fold) if bundle.store.fit_range \
        else ["no scaler is fitted; training on unscaled observations"]
    if problems:
        raise StageError(
            f"{fold.fold_id}: scaler is not legal for this fold:\n  "
            + "\n  ".join(problems)
            + "\nThis is a walk-forward leak, and it is invisible in results -- every "
              "fold would simply look slightly better.")

    fold_dir = ctx.run_dir / "folds" / str(fold.test_year)
    fold_dir.mkdir(parents=True, exist_ok=True)
    d_max_primary = bundle.constraints.drawdown.max_drawdown.primary

    candidates = default_candidates(candidates_n, base_seed=ctx.seed)
    for candidate in candidates:
        log(f"  candidate {candidate.name}: {candidate.params}")
        stage_resolved = dict(resolved)
        stage_resolved["ppo"] = {**(resolved.get("ppo", {}) or {}),
                                 **{k: v for k, v in candidate.params.items()
                                    if k != "seed"}}
        # Each candidate gets its own directory: `train_stage` names its output by
        # curriculum rung, not by candidate, so a shared parent would have them
        # overwriting one another and the selection would compare a model to itself.
        outcome = train_stage(
            bundle, STAGES_BY_INDEX[5], stage_resolved,
            run_dir=fold_dir / "candidates" / candidate.name,
            seed=int(candidate.params["seed"]),
            total_timesteps=timesteps, n_envs=n_envs, subproc=subproc,
            # Prohibition 1: the environment window IS the fold's training window, so
            # no episode can start in the validation or test year.
            env_overrides={"window": (fold.train_start, fold.train_end)},
            log=lambda m: None)
        candidate.params["policy_dir"] = str(outcome.run_dir)

        model = load_policy(outcome.run_dir)
        # Prohibition 3: scored on the VALIDATION year, never the test year.
        aggregate, _ = evaluate_window(
            model, bundle, (fold.val_start, fold.val_end), cells,
            seed=ctx.seed, label=f"val {candidate.name}", log=log)
        candidate.validation = aggregate

    result = select(candidates, d_max=d_max_primary, fold_id=fold.fold_id)
    log(f"  selected: {result.chosen.name}  ({result.criterion})")
    if result.constraint_validation_failure:
        ctx.warn(f"{fold.fold_id}: CONSTRAINT VALIDATION FAILURE -- this mark travels to "
                 "the final report and is not dropped")

    (fold_dir / "selection.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")

    # Frozen. One evaluation on the test year.
    model = load_policy(result.chosen.params["policy_dir"])
    test_aggregate, trajectories = evaluate_window(
        model, bundle, (fold.test_start, fold.test_end), cells,
        seed=ctx.seed, label=f"TEST {fold.test_year}", log=log)

    # Seed spread. The evaluation rollout is deterministic, so the only stochasticity
    # left is training itself -- which is exactly the thing a single number hides. The
    # SELECTED hyperparameters are retrained at fresh seeds and re-evaluated; the
    # selection is not redone, because that would be selecting K times and reporting the
    # best, which is a different and much weaker claim.
    seed_spread = None
    if seeds > 1:
        returns = [test_aggregate["cumulative_return"]]
        drawdowns = [test_aggregate["max_drawdown"]]
        for k in range(1, seeds):
            alt = dict(resolved)
            alt["ppo"] = {**(resolved.get("ppo", {}) or {}),
                          **{key: v for key, v in result.chosen.params.items()
                             if key in ("ent_coef", "learning_rate")}}
            outcome = train_stage(
                bundle, STAGES_BY_INDEX[5], alt,
                run_dir=fold_dir / "seeds" / f"seed{k}",
                seed=int(ctx.seed) + 1000 * k,
                total_timesteps=timesteps, n_envs=n_envs, subproc=subproc,
                env_overrides={"window": (fold.train_start, fold.train_end)},
                log=lambda m: None)
            extra, _ = evaluate_window(
                load_policy(outcome.run_dir), bundle,
                (fold.test_start, fold.test_end), cells,
                seed=ctx.seed, label=f"TEST {fold.test_year} seed{k}", log=log)
            returns.append(extra["cumulative_return"])
            drawdowns.append(extra["max_drawdown"])
        seed_spread = {
            "n_seeds": seeds,
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
            "return_min": float(np.min(returns)), "return_max": float(np.max(returns)),
            "max_drawdown_mean": float(np.mean(drawdowns)),
            "max_drawdown_max": float(np.max(drawdowns)),
            "returns": [float(r) for r in returns],
        }
        log(f"    seed spread over {seeds}: return "
            f"{seed_spread['return_mean']:+.2%} +/- {seed_spread['return_std']:.2%} "
            f"(min {seed_spread['return_min']:+.2%}, max {seed_spread['return_max']:+.2%})")

    primary_key = f"N{bundle.constraints.lock.hold_days.primary}_D{d_max_primary}"
    # `or` would test a DataFrame for truthiness, which raises. Be explicit.
    headline = (trajectories[primary_key] if primary_key in trajectories
                else next(iter(trajectories.values())))
    headline.to_parquet(fold_dir / "trajectory.parquet")
    for key, traj in trajectories.items():
        (fold_dir / "cells").mkdir(exist_ok=True)
        traj.to_parquet(fold_dir / "cells" / f"{key}.parquet")

    taxonomy = summarize_violations(headline, bundle.market.universe,
                                    d_max=d_max_primary, market=bundle.market)
    (fold_dir / "violations.json").write_text(
        json.dumps(taxonomy, indent=2, default=str), encoding="utf-8")

    return {
        "fold_id": fold.fold_id, "test_year": fold.test_year,
        "fold": fold.to_dict(),
        "scaler_fit_range": list(bundle.store.fit_range),
        "selected": result.chosen.name,
        "criterion": result.criterion,
        "constraint_validation_failure": result.constraint_validation_failure,
        "validation": result.chosen.validation,
        "test": test_aggregate,
        "seed_spread": seed_spread,
        "violations": {k: v for k, v in taxonomy.items() if k != "replay_findings"},
    }


@stage(
    name="s08_walk_forward",
    config_default="config/evaluation.yaml",
    inputs=["data/curated/prices.parquet", "data/features/folds.json"],
    outputs=["artifacts/runs/{run_id}/walk_forward_summary.json"],
    upstream="s07_train_ppo",
    add_args=_add_args,
)
def main(resolved: dict, ctx: StageContext) -> None:
    """Expanding-window annual folds with lexicographic selection. The Phase 3 gate."""
    from src.env.factory import build_bundle

    args = ctx.args
    wf = resolved.get("walk_forward", {}) or {}
    training = resolved.get("training", {}) or {}

    folds = load_folds()
    assert_folds_valid(folds)          # T11, before anything else runs
    ctx.log(f"{len(folds)} folds pass walk-forward integrity (T11)")

    only = [int(y) for y in args.folds.split(",")] if args.folds else None
    chosen = select_folds(
        folds, only=only,
        first_year=args.first_year or wf.get("first_test_year"),
        last_year=args.last_year or wf.get("last_test_year"))
    if not chosen:
        raise StageError("no folds selected")

    candidates_n = int(args.candidates or wf.get("candidates", 4))
    seeds = int(args.seeds or wf.get("seeds", 1))
    timesteps = int(args.timesteps or training.get("total_timesteps", 2_000_000))
    n_envs = int(args.n_envs or training.get("n_envs", 8))
    subproc = bool(training.get("subproc", True)) and not args.no_subproc

    probe = build_bundle(ctx.config_path, resolved=resolved)
    cells = eval_grid(args.grid, probe.constraints)
    ctx.log(f"{len(chosen)} fold(s) x {candidates_n} candidate(s) x {timesteps:,} "
            f"timesteps; evaluating {len(cells)} (N, D_max) cell(s)")
    if seeds > 1:
        ctx.log(f"{seeds} seeds per fold; the spread is recorded per fold. A single-seed "
                "result on a stochastic policy over one test year is a data point, not a "
                "conclusion, and the report presents it as such.")

    records = []
    for fold in chosen:
        # Each fold gets its OWN scaler. This is the whole point of the protocol.
        bundle = build_bundle(ctx.config_path, resolved=resolved, fold_id=fold.fold_id)
        records.append(run_fold(fold, bundle, resolved, ctx, cells=cells,
                                candidates_n=candidates_n, timesteps=timesteps,
                                n_envs=n_envs, subproc=subproc, seeds=seeds))

    lock = sum(r["test"]["lock_violations"] for r in records)
    feas = sum(r["test"]["feasibility_violations"] for r in records)
    preventable = sum(r["test"]["preventable_violations"] for r in records)
    failures = [r["fold_id"] for r in records if r["constraint_validation_failure"]]

    summary = {
        "run_id": ctx.run_id, "seed": ctx.seed, "n_folds": len(records),
        "candidates_per_fold": candidates_n, "timesteps_per_candidate": timesteps,
        "eval_cells": [f"N{n}_D{d}" for n, d in cells],
        "hard_acceptance": {
            "lock_violations": lock,
            "feasibility_violations": feas,
            "preventable_dmax_violations": preventable,
            "passed": lock == 0 and feas == 0 and preventable == 0,
        },
        "constraint_validation_failures": failures,
        "folds": records,
    }
    ctx.out_path("walk_forward_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    ctx.record(n_folds=len(records), lock_violations=lock,
               feasibility_violations=feas, preventable_violations=preventable,
               constraint_validation_failures=len(failures))

    ctx.log("")
    spread_col = any(r["seed_spread"] for r in records)
    header = (f"{'fold':<8} {'selected':<20} {'test ret':>9} {'max DD':>8} "
              f"{'prev':>5} {'forced':>7} {'CVF':>4}")
    ctx.log(header + ("  seed sd" if spread_col else ""))
    for r in records:
        t = r["test"]
        line = (f"{r['test_year']:<8} {r['selected']:<20} "
                f"{t['cumulative_return']:>+9.2%} {t['max_drawdown']:>8.2%} "
                f"{t['preventable_violations']:>5} "
                f"{r['violations']['n_market_forced']:>7} "
                f"{'YES' if r['constraint_validation_failure'] else '-':>4}")
        if spread_col and r["seed_spread"]:
            line += f"  {r['seed_spread']['return_std']:>6.2%}"
        ctx.log(line)

    ctx.log("")
    if not summary["hard_acceptance"]["passed"]:
        raise StageError(
            f"HARD ACCEPTANCE FAILED: {lock} lock, {feas} feasibility, {preventable} "
            "preventable D_max violation(s). These must be exactly zero -- there is no "
            "partial credit. Phase 3 does not start until this passes."
        )
    if failures:
        ctx.warn(f"{len(failures)} fold(s) marked CONSTRAINT VALIDATION FAILURE: "
                 f"{failures}. This mark travels into the final report.")
    ctx.log(f"hard acceptance PASSED across {len(records)} fold(s): "
            "zero lock, zero feasibility, zero preventable violations")


if __name__ == "__main__":
    raise SystemExit(main())
