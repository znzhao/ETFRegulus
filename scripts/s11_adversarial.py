"""Stage 11 -- adversarial historical scenarios.

    python -m scripts.s11_adversarial --config config/evaluation.yaml
    python -m scripts.s11_adversarial --config config/evaluation.yaml --block 21 --n-blocks 4

Four unfavourable combinations built **within the support of the historical empirical
distribution** (reference/robustness.md section 3): equity shock with credit widening,
duration loss, correlation spike, and diversification breakdown.

Every path is assembled from real historical blocks by chaining their returns, so every
return is one that actually happened; only the ordering and combination are adversarial.
The features travel with their source sessions, so what the policy observes inside a block
is real and self-consistent.

**The disclaimer is part of the deliverable**, and it is written into every artifact this
stage produces: a scenario built by picking the worst historical blocks is, by
construction, not a probability statement about the future.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cli.stage import StageContext, StageError, stage
from src.evaluation.adversarial import (
    DEFAULT_BLOCK,
    DISCLAIMER,
    build_scenarios,
    splice,
)
from src.evaluation.rollout import load_policy, rollout
from src.evaluation.violations import summarize as summarize_violations


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy", default=None)
    p.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--baselines", default="spy_tlt_60_40,equal_weight,cash",
                   help="baselines to run through the same scenarios for reference")


def spliced_store(store, rows: np.ndarray, sessions: pd.DatetimeIndex):
    """A feature store whose rows follow the SOURCE sessions, on the synthetic calendar.

    This is the piece that makes a spliced evaluation honest. The synthetic market reuses
    a stretch of the real calendar for its dates, so a naive lookup would hand the policy
    the features of the calendar date rather than of the block the returns came from --
    real-looking numbers describing the wrong day. Reindexing the feature arrays by the
    same `rows` keeps returns and observations pointing at the same session.
    """
    return dataclasses.replace(
        store, sessions=pd.DatetimeIndex(sessions),
        per_asset=store.per_asset[rows], globals=store.globals[rows])


def run_path(model, bundle, market, store, *, hold_days: int, d_max: float,
             seed: int, weight_fn=None):
    """One scenario path. `weight_fn` overrides the policy, for the baseline references."""
    scenario_bundle = dataclasses.replace(bundle, market=market, store=store)
    if weight_fn is None:
        return rollout(model, scenario_bundle, hold_days=hold_days, max_drawdown=d_max,
                       start_row=0, end_row=len(market.sessions) - 1, seed=seed)

    from src.sim.runner import build_envelope, build_sim_config, make_projector_from
    from src.sim.simulator import simulate

    sim_cfg = build_sim_config(bundle.resolved, bundle.constraints,
                               hold_days=hold_days, max_drawdown=d_max, seed=seed)
    return simulate(market, weight_fn(market), sim_cfg,
                    projector=make_projector_from(bundle.constraints),
                    envelope=build_envelope(bundle.constraints),
                    start_row=0, end_row=len(market.sessions) - 1)


def path_metrics(result, market, d_max: float) -> dict:
    traj = result.trajectory
    nav = traj["nav"].astype(float)
    violations = summarize_violations(traj, market.universe, d_max=d_max, market=market)
    locked = traj[[f"locked_{t}" for t in market.universe]].to_numpy(dtype=bool)
    weights = traj[[f"w_{t}" for t in market.universe]].to_numpy(dtype=float)
    return {
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "max_drawdown": float((1.0 - nav / nav.cummax()).max()),
        "worst_1d": float(nav.pct_change().min()),
        "mean_cash_weight": float((traj["cash"].astype(float) / nav).mean()),
        "locked_nav_fraction": float(np.where(locked, weights, 0.0).sum(axis=1).mean()),
        "actions_blocked_by_lock": int(traj.get("share_floor_binding",
                                                pd.Series([0])).sum()),
        "safety_intervention_rate": float(traj["safety_intervened"].mean()),
        "capital_preservation_rate": float(traj["capital_preservation"].mean()),
        "n_sessions": int(len(traj)),
        "lock_violations": int(result.diagnostics["lock_violations"]),
        "feasibility_violations": int(result.diagnostics["feasibility_violations"]),
        "preventable_violations": int(violations["n_preventable"]),
        "market_forced_breaches": int(violations["n_market_forced"]),
    }


@stage(
    name="s11_adversarial",
    config_default="config/evaluation.yaml",
    inputs=["data/curated/prices.parquet"],
    outputs=["artifacts/runs/{run_id}/adversarial_summary.json"],
    upstream="s09_stress",
    add_args=_add_args,
)
def main(resolved: dict, ctx: StageContext) -> None:
    """Four adversarial historical scenarios, plus baselines through the same paths."""
    from src.env.factory import build_bundle
    from src.sim.runner import weight_source

    args = ctx.args
    policy_dir = Path(args.policy) if args.policy else _newest_policy()
    ctx.log(f"policy: {policy_dir}")

    bundle = build_bundle(ctx.config_path, resolved=resolved,
                          fold_id=(resolved.get("environment", {}) or {}).get("fold_id"))
    model = load_policy(policy_dir)
    market = bundle.market
    primary_n = bundle.constraints.lock.hold_days.primary
    primary_d = bundle.constraints.drawdown.max_drawdown.primary

    # Blocks are drawn from the visible range of the policy's own fold, so a scenario is
    # not assembled out of the future relative to the model that faces it.
    fit_end = pd.Timestamp(bundle.store.fit_range[1]) if bundle.store.fit_range else None
    rows = np.flatnonzero(market.sessions <= fit_end) if fit_end is not None else None
    ctx.log(f"blocks drawn from {market.sessions[0].date()}.."
            f"{(fit_end or market.sessions[-1]).date()} "
            f"({'the fold training window' if rows is not None else 'all history'})")

    scenarios = build_scenarios(market, block=args.block, n_blocks=args.n_blocks,
                                rows=rows)
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    records = []
    for scenario in scenarios:
        if not scenario.blocks:
            ctx.warn(f"{scenario.name}: no blocks selected, skipped")
            continue
        source_rows = scenario.rows()
        synthetic = splice(market, source_rows)
        store = spliced_store(bundle.store, source_rows, synthetic.sessions)

        result = run_path(model, bundle, synthetic, store,
                          hold_days=primary_n, d_max=primary_d, seed=ctx.seed)
        metrics = path_metrics(result, synthetic, primary_d)

        reference = {}
        for name in baselines:
            try:
                ref = run_path(model, bundle, synthetic, store,
                               hold_days=primary_n, d_max=primary_d, seed=ctx.seed,
                               weight_fn=lambda m, n=name: weight_source(
                                   n, m, (resolved.get("baselines", {}) or {})
                                   .get("params", {}).get(n)))
                reference[name] = path_metrics(ref, synthetic, primary_d)
            except Exception as exc:                       # a baseline is a reference,
                reference[name] = {"error": f"{type(exc).__name__}: {exc}"}   # not the test

        ctx.log("")
        ctx.log(f"{scenario.name}  ({len(scenario.blocks)} blocks, "
                f"{len(source_rows)} sessions)")
        for b in scenario.blocks:
            ctx.log(f"    {b.to_dict(market)['start']}..{b.to_dict(market)['end']}  "
                    f"{b.reason}")
        ctx.log(f"    POLICY   return {metrics['total_return']:>+7.2%}  "
                f"maxDD {metrics['max_drawdown']:>6.2%}  "
                f"locked {metrics['locked_nav_fraction']:>5.1%}  "
                f"blocked {metrics['actions_blocked_by_lock']:>4}  "
                f"preventable {metrics['preventable_violations']}")
        for name, ref in reference.items():
            if "error" in ref:
                ctx.log(f"    {name:<8} {ref['error']}")
            else:
                ctx.log(f"    {name:<8} return {ref['total_return']:>+7.2%}  "
                        f"maxDD {ref['max_drawdown']:>6.2%}")

        result.trajectory.to_parquet(ctx.out_path(f"scenarios/{scenario.name}.parquet"))
        records.append({
            "name": scenario.name, "description": scenario.description,
            "blocks": [b.to_dict(market) for b in scenario.blocks],
            "n_sessions": int(len(source_rows)),
            "policy": metrics, "baselines": reference,
        })

    if not records:
        raise StageError("no scenario produced a path")

    lock_v = sum(r["policy"]["lock_violations"] for r in records)
    feas_v = sum(r["policy"]["feasibility_violations"] for r in records)
    prev_v = sum(r["policy"]["preventable_violations"] for r in records)

    summary = {
        "DISCLAIMER": DISCLAIMER,
        "policy": str(policy_dir),
        "block_sessions": args.block, "blocks_per_scenario": args.n_blocks,
        "blocks_drawn_from": ("the fold training window" if rows is not None
                              else "all available history"),
        "construction": ("Blocks are concatenated by CHAINING RETURNS, never price "
                         "levels: a price series stitched from disjoint windows invents "
                         "an enormous return at every seam. Features travel with their "
                         "source sessions."),
        "hard_acceptance": {"lock_violations": lock_v,
                            "feasibility_violations": feas_v,
                            "preventable_violations": prev_v},
        "scenarios": records,
    }
    out = ctx.out_path("adversarial_summary.json")
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    rows_out = []
    for r in records:
        rows_out.append({"scenario": r["name"], "who": "policy", **r["policy"]})
        for name, ref in r["baselines"].items():
            if "error" not in ref:
                rows_out.append({"scenario": r["name"], "who": name, **ref})
    pd.DataFrame(rows_out).to_csv(ctx.out_path("scenarios.csv"), index=False)

    ctx.record(n_scenarios=len(records), lock_violations=lock_v,
               feasibility_violations=feas_v, preventable_violations=prev_v)
    ctx.log("")
    if lock_v or feas_v or prev_v:
        raise StageError(
            f"hard acceptance failed under adversarial paths: {lock_v} lock, {feas_v} "
            f"feasibility, {prev_v} preventable violation(s). The constraint layer must "
            "hold on every path, including ones chosen to be hostile.")
    ctx.log(f"constraint layer held on all {len(records)} adversarial path(s): "
            "zero lock, zero feasibility, zero preventable violations")
    ctx.log("")
    ctx.log(DISCLAIMER)
    ctx.log(f"Stage 11 complete. Summary: {out}")


def _newest_policy() -> Path:
    from scripts.s09_stress import newest_stage8_policy as _f

    return _f()


if __name__ == "__main__":
    raise SystemExit(main())
