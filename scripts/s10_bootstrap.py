"""Stage 10 -- block bootstrap confidence bands.

    python -m scripts.s10_bootstrap --config config/evaluation.yaml --replicates 1000
    python -m scripts.s10_bootstrap --config config/evaluation.yaml --method moving_block

Stationary / moving-block bootstrap over return blocks -- **never IID resampling**, which
would destroy the volatility clustering the whole risk layer depends on
(reference/robustness.md section 2). Bands are produced for every standard metric, and the
block-length sensitivity is reported whether or not it is flattering.

Runs over the Stage 8 walk-forward trajectories, and over each baseline, so the bands are
comparable: the same resampling machinery applied to the same sessions.

The interpretive caution travels with the output. A bootstrap over historical returns
quantifies sampling uncertainty **within the observed regime distribution**. It is not a
statement about regimes that have not occurred.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cli.stage import RUNS, StageContext, StageError, stage
from src.evaluation.bootstrap import (
    BLOCK_LENGTH_SENSITIVITY,
    DEFAULT_MEAN_BLOCK,
    DEFAULT_REPLICATES,
    block_length_sensitivity,
    bootstrap,
    portfolio_returns,
)

CAVEAT = (
    "A bootstrap over historical returns quantifies sampling uncertainty WITHIN the "
    "observed regime distribution. It is not a statement about regimes that have not "
    "occurred, and it does not re-run the policy on counterfactual prices."
)


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    p.add_argument("--method", choices=("stationary", "moving_block"),
                   default="stationary")
    p.add_argument("--mean-block", type=float, default=DEFAULT_MEAN_BLOCK)
    p.add_argument("--walk-forward", default=None, help="Stage 8 run id or directory")
    p.add_argument("--skip-baselines", action="store_true")


def newest_walk_forward(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else RUNS / explicit
    runs = sorted(RUNS.glob("s08_walk_forward_*/walk_forward_summary.json"))
    if not runs:
        raise StageError(
            "no Stage 8 run found. Stage 10 bootstraps the walk-forward result, so it "
            "needs one: run `python -m scripts.s08_walk_forward`.")
    return runs[-1].parent


def load_fold_trajectories(run_dir: Path) -> list[pd.DataFrame]:
    paths = sorted(run_dir.glob("folds/*/trajectory.parquet"))
    if not paths:
        raise StageError(f"no fold trajectories under {run_dir}")
    return [pd.read_parquet(p) for p in paths]


def load_baseline_returns() -> dict[str, np.ndarray]:
    """Each baseline's walk-forward-comparable return series, from the Stage 12 report."""
    out: dict[str, np.ndarray] = {}
    for directory in sorted(glob.glob("artifacts/reports/baselines/trajectories/*")):
        name = Path(directory).name
        frames = [pd.read_parquet(p)
                  for p in sorted(glob.glob(f"{directory}/*.parquet"))]
        if frames:
            out[name] = portfolio_returns(frames)
    return out


@stage(
    name="s10_bootstrap",
    config_default="config/evaluation.yaml",
    inputs=["data/curated/prices.parquet"],
    outputs=["artifacts/runs/{run_id}/bootstrap_summary.json"],
    upstream="s08_walk_forward",
    add_args=_add_args,
)
def main(resolved: dict, ctx: StageContext) -> None:
    """Confidence bands for every standard metric, plus block-length sensitivity."""
    args = ctx.args
    wf_dir = newest_walk_forward(args.walk_forward)
    ctx.log(f"walk-forward run: {wf_dir}")

    trajectories = load_fold_trajectories(wf_dir)
    returns = portfolio_returns(trajectories)
    ctx.log(f"{len(trajectories)} fold trajectories -> {returns.size} daily returns")
    if returns.size < 100:
        raise StageError(f"only {returns.size} returns; too few to bootstrap")

    ctx.log(f"{args.method} bootstrap, {args.replicates} replicates, "
            f"mean block {args.mean_block}")
    result = bootstrap(returns, replicates=args.replicates, method=args.method,
                       mean_block=args.mean_block, seed=ctx.seed)

    ctx.log("")
    ctx.log(f"{'metric':<20} {'observed':>10} {'q05':>10} {'q50':>10} {'q95':>10}")
    for key in ("annualized_return", "volatility", "sharpe", "max_drawdown",
                "worst_1d", "worst_5d"):
        band = result.bands[key]
        ctx.log(f"{key:<20} {result.observed[key]:>10.4f} {band['q05']:>10.4f} "
                f"{band['q50']:>10.4f} {band['q95']:>10.4f}")

    ctx.log("")
    ctx.log(f"block-length sensitivity over {list(BLOCK_LENGTH_SENSITIVITY)}")
    sensitivity = block_length_sensitivity(
        returns, replicates=max(200, args.replicates // 4), method=args.method,
        seed=ctx.seed)
    for key, spread in sensitivity["band_width_spread"].items():
        ctx.log(f"  {key:<20} 90% band width {spread['min_width']:.4f}.."
                f"{spread['max_width']:.4f}  (ratio {spread['ratio']:.2f})")
    worst_ratio = max(s["ratio"] for s in sensitivity["band_width_spread"].values())

    summary: dict = {
        "walk_forward_run": str(wf_dir),
        "n_folds": len(trajectories),
        "policy": result.to_dict(),
        "block_length_sensitivity": sensitivity,
        "block_length_conclusion_stable": bool(worst_ratio < 2.0),
        "caveat": CAVEAT,
    }

    if not args.skip_baselines:
        ctx.log("")
        ctx.log("baselines, same machinery")
        baselines = {}
        for name, series in load_baseline_returns().items():
            if series.size < 100:
                continue
            b = bootstrap(series, replicates=args.replicates, method=args.method,
                          mean_block=args.mean_block, seed=ctx.seed)
            baselines[name] = b.to_dict()
            ctx.log(f"  {name:<22} ann.return q05 {b.bands['annualized_return']['q05']:>+7.2%} "
                    f"q50 {b.bands['annualized_return']['q50']:>+7.2%} "
                    f"q95 {b.bands['annualized_return']['q95']:>+7.2%}  "
                    f"maxDD q95 {b.bands['max_drawdown']['q95']:>6.2%}")
        summary["baselines"] = baselines

    out = ctx.out_path("bootstrap_summary.json")
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    rows = [{"metric": k, "observed": result.observed[k], **v}
            for k, v in result.bands.items()]
    pd.DataFrame(rows).to_csv(ctx.out_path("bands.csv"), index=False)

    ctx.record(replicates=args.replicates, method=args.method,
               n_returns=int(returns.size),
               block_length_conclusion_stable=summary["block_length_conclusion_stable"])
    ctx.log("")
    if not summary["block_length_conclusion_stable"]:
        ctx.warn(f"band width moves by up to {worst_ratio:.2f}x across block lengths; "
                 "the report must say so rather than quote the tightest choice")
    ctx.log(CAVEAT)
    ctx.log(f"Stage 10 complete. Summary: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
