"""Stage 4 -- the deterministic simulator.  **This is the gate.**

    python -m scripts.s04_simulate --config config/sim/default.yaml
    python -m scripts.s04_simulate --config config/sim/default.yaml \
        --strategy momentum --hold-days 30 --max-drawdown 0.15

Given any legal initial state, any sequence of target weights, and the price history,
produce the complete portfolio trajectory. **No RL anywhere in this stage**, and no RL
code exists anywhere in the project until its invariant suite is green.

Writes `artifacts/runs/<run_id>/trajectory.parquet` in the schema every downstream metric
function reads (reference/architecture.md section 4), plus `diagnostics.json`.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.cli.stage import StageContext, StageError, stage
from src.evaluation.metrics import summarize
from src.sim.runner import (
    build_constraints,
    build_envelope,
    build_sim_config,
    load_market,
    make_projector_from,
    make_weight_fn,
)
from src.sim.simulator import simulate


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", default=None,
                   help="built-in weight source (overrides simulation.strategy)")
    p.add_argument("--weights", default=None,
                   help="parquet weight sequence to replay instead of a strategy")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--hold-days", type=int, default=None, help="N, in calendar days")
    p.add_argument("--max-drawdown", type=float, default=None, help="D_max")
    p.add_argument("--no-risk", action="store_true",
                   help="disable the risk envelope (mechanics only)")


@stage(
    name="s04_simulate",
    config_default="config/sim/default.yaml",
    inputs=["data/curated/prices.parquet"],
    outputs=["artifacts/runs/{run_id}/trajectory.parquet"],
    upstream="s03_build_features",
    add_args=_add_args,
)
def main(resolved: dict, ctx: StageContext) -> None:
    """Run one deterministic backtest and write its trajectory."""
    args = ctx.args
    if args.weights:
        resolved.setdefault("simulation", {})["weights_file"] = args.weights
    if args.no_risk:
        resolved.setdefault("simulation", {})["risk_enabled"] = False

    market, ucfg = load_market(resolved, start=args.start, end=args.end)
    constraints = build_constraints(resolved)
    cfg = build_sim_config(resolved, constraints, hold_days=args.hold_days,
                           max_drawdown=args.max_drawdown, seed=ctx.seed)

    strategy = args.strategy if not args.weights else None
    weight_fn, source = make_weight_fn(resolved, market, strategy=strategy)

    ctx.log(f"universe {market.n_assets} tradables + CASH, "
            f"{len(market.sessions)} sessions "
            f"{market.sessions[0].date()} .. {market.sessions[-1].date()}")
    ctx.log(f"source={source}  N={cfg.hold_days}  D_max={cfg.max_drawdown}  "
            f"risk={cfg.risk_enabled}  cost_bps={cfg.cost_bps}  scope={cfg.lock_scope}")

    result = simulate(
        market, weight_fn, cfg,
        projector=make_projector_from(constraints),
        envelope=build_envelope(constraints),
    )

    traj = result.trajectory
    metrics = summarize(traj, result.diagnostics, d_max=cfg.max_drawdown)

    out = ctx.out_path("trajectory.parquet")
    traj.to_parquet(out)
    ctx.out_path("diagnostics.json").write_text(
        json.dumps({"diagnostics": result.diagnostics, "metrics": metrics,
                    "source": source}, indent=2, default=str), encoding="utf-8")

    ctx.log("")
    ctx.log(f"final NAV      {metrics['final_nav']:>14,.0f}   "
            f"({metrics['cumulative_return']:+.1%} cumulative, "
            f"{metrics['annualized_return']:+.2%} annualized)")
    ctx.log(f"max drawdown   {metrics['max_drawdown']:>14.4f}   "
            f"vs D_max {cfg.max_drawdown}")
    ctx.log(f"volatility     {metrics['volatility']:>14.4f}   "
            f"Sharpe {metrics['sharpe']:.3f}  Calmar {metrics['calmar']:.3f}")
    ctx.log(f"mean cash w    {metrics['mean_cash_weight']:>14.4f}   "
            f"turnover/step {metrics['turnover_mean']:.5f}")
    ctx.log(f"safety rate    {metrics.get('safety_intervention_rate', 0.0):>14.4f}   "
            f"capital preservation {metrics.get('capital_preservation_rate', 0.0):.4f}")
    ctx.log("")
    ctx.log(f"lock violations        {metrics['lock_violations']}")
    ctx.log(f"feasibility violations {metrics['feasibility_violations']}")

    ctx.record(**{k: metrics[k] for k in
                  ("final_nav", "max_drawdown", "annualized_return", "sharpe",
                   "lock_violations", "feasibility_violations")},
               source=source, n_steps=result.diagnostics["n_steps"])

    # The hard engineering criteria. A run breaching either is not a weaker result, it is
    # an invalid one, and it must not be written as though it counted.
    if metrics["lock_violations"] or metrics["feasibility_violations"]:
        raise StageError(
            f"HARD ACCEPTANCE FAILURE: {metrics['lock_violations']} lock violation(s), "
            f"{metrics['feasibility_violations']} feasibility violation(s). The "
            f"constraint layer is broken; fix it rather than tuning around it."
        )
    ctx.log(f"trajectory -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
