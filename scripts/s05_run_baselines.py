"""Stage 5 -- the baselines.

    python -m scripts.s05_run_baselines --config config/evaluation.yaml
    python -m scripts.s05_run_baselines --config config/evaluation.yaml --only cash,momentum
    python -m scripts.s05_run_baselines --config config/evaluation.yaml --skip-calibration

Six baselines, all through the *same* simulator and the *same* constraint layer as the
agent, **before any RL**, so the whole RL development period has a reference point.

Also performs the two jobs Stage 5 owes the later stages:

* the **risk-envelope calibration** (open question Q1) -- the table that decides
  `quantile`, `horizon_days` and `aggregation` before Stage 7 can mean anything;
* the **initial-state reservoir** for the Stage 6/7 reset sampler, reachable by
  construction because every state in it came out of a legal trajectory.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.baselines.strategies import BASELINES, PRIMARY
from src.cli.stage import StageContext, StageError, stage
from src.constraints.risk_envelope import RiskEnvelope
from src.evaluation.metrics import summarize
from src.sim.runner import (
    build_constraints,
    build_envelope,
    build_sim_config,
    load_market,
    make_projector_from,
)
from src.sim.simulator import simulate


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--only", default=None, help="comma-separated subset of baselines")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--skip-controls", action="store_true",
                   help="skip the N-sensitivity control runs (they are part of the gate)")


@stage(
    name="s05_run_baselines",
    config_default="config/evaluation.yaml",
    inputs=["data/curated/prices.parquet"],
    outputs=["artifacts/runs/{run_id}/baseline_summary.json"],
    upstream="s04_simulate",
    add_args=_add_args,
)
def main(resolved: dict, ctx: StageContext) -> None:
    """Run the six baselines, calibrate the envelope, and build the reset reservoir."""
    args = ctx.args
    market, ucfg = load_market(resolved, start=args.start, end=args.end)
    constraints = build_constraints(resolved)
    bcfg = resolved["baselines"]
    projector = make_projector_from(constraints)

    names = [n.strip() for n in args.only.split(",")] if args.only else bcfg["names"]
    unknown = set(names) - set(BASELINES)
    if unknown:
        raise StageError(f"unknown baseline(s): {sorted(unknown)}")

    ctx.log(f"{market.n_assets} tradables + CASH, {len(market.sessions)} sessions "
            f"{market.sessions[0].date()} .. {market.sessions[-1].date()}")

    base_cfg = build_sim_config(resolved, constraints, seed=ctx.seed)
    ctx.log(f"N={base_cfg.hold_days}  D_max={base_cfg.max_drawdown}  "
            f"cost_bps={base_cfg.cost_bps}  scope={base_cfg.lock_scope}")
    ctx.log("")

    # ------------------------------------------------------------ the six runs
    summaries: dict[str, dict] = {}
    reservoir: list[dict] = []
    violations: dict[str, dict] = {}

    for name in names:
        cfg = build_sim_config(resolved, constraints, seed=ctx.seed)
        cfg.reservoir_every = int(bcfg.get("reservoir_every", 0))
        fn = BASELINES[name].build(market, bcfg.get("params", {}).get(name, {}))
        res = simulate(market, fn, cfg, projector=projector,
                       envelope=build_envelope(constraints))

        metrics = summarize(res.trajectory, res.diagnostics, d_max=cfg.max_drawdown)
        summaries[name] = metrics
        reservoir.extend(res.reservoir)
        violations[name] = {
            "lock": metrics["lock_violations"],
            "feasibility": metrics["feasibility_violations"],
        }

        out_dir = ctx.out_path(f"baselines/{name}/trajectory.parquet")
        res.trajectory.to_parquet(out_dir)
        ctx.log(f"{name:<22} NAV {metrics['final_nav']:>12,.0f}  "
                f"ann {metrics['annualized_return']:>+7.2%}  "
                f"maxDD {metrics['max_drawdown']:>6.3f}  "
                f"vol {metrics['volatility']:>5.3f}  "
                f"Sharpe {metrics['sharpe']:>+6.2f}  "
                f"cash {metrics['mean_cash_weight']:>5.2f}  "
                f"viol {metrics['lock_violations']}/{metrics['feasibility_violations']}")

    # ------------------------------------------- the four simulator-checking gates
    ctx.log("")
    ctx.log("Stage 5 acceptance checks (reference/baselines.md section 7):")
    checks: dict[str, dict] = {}

    if "cash" in summaries:
        m = summaries["cash"]
        ok = abs(m["max_drawdown"]) < 1e-12 and abs(m["turnover_total"]) < 1e-12
        checks["cash_zero_drawdown_and_turnover"] = {
            "passed": bool(ok), "max_drawdown": m["max_drawdown"],
            "turnover_total": m["turnover_total"],
        }

    if not args.skip_controls:
        controls = list(bcfg.get("n_control_set", [0, 30, 90, 180]))
        checks.update(_n_controls(ctx, market, resolved, constraints, projector, controls))

    if "spy_tlt_60_40" in summaries:
        traj = pd.read_parquet(ctx.run_dir / "baselines/spy_tlt_60_40/trajectory.parquet")
        in_2022 = traj.loc["2022"] if "2022" in traj.index.astype(str).str[:4].values else None
        depth = float(in_2022["drawdown"].max()) if in_2022 is not None and len(in_2022) else 0.0
        checks["spy_tlt_60_40_severe_2022"] = {
            "passed": bool(depth > 0.15), "drawdown_2022": depth,
            "note": "if 2022 looks mild, the total-return bond adjustment is wrong",
        }

    for name, chk in checks.items():
        ctx.log(f"  {'PASS' if chk['passed'] else 'FAIL'}  {name}: "
                f"{ {k: v for k, v in chk.items() if k != 'passed'} }")

    # ------------------------------------------------------------- calibration
    calibration = None
    if resolved.get("calibration", {}).get("enabled", True) and not args.skip_calibration:
        calibration = _calibrate(ctx, market, resolved, constraints, projector)

    # ---------------------------------------------------------------- reservoir
    if reservoir:
        path = ctx.out_path("reset_reservoir.json")
        path.write_text(json.dumps(reservoir, indent=1), encoding="utf-8")
        ctx.log("")
        ctx.log(f"initial-state reservoir: {len(reservoir)} reachable states -> {path.name}")

    payload = {
        "baselines": summaries, "checks": checks, "calibration": calibration,
        "n_reservoir_states": len(reservoir),
        "hold_days": base_cfg.hold_days, "max_drawdown": base_cfg.max_drawdown,
        "cost_bps": base_cfg.cost_bps,
        "window": [str(market.sessions[0].date()), str(market.sessions[-1].date())],
    }
    summary_path = ctx.out_path("baseline_summary.json")
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    total_lock = sum(v["lock"] for v in violations.values())
    total_feas = sum(v["feasibility"] for v in violations.values())
    ctx.record(n_baselines=len(summaries), lock_violations=total_lock,
               feasibility_violations=total_feas,
               n_reservoir_states=len(reservoir),
               checks_passed=sum(c["passed"] for c in checks.values()),
               checks_total=len(checks))

    failed = [n for n, c in checks.items() if not c["passed"]]
    if total_lock or total_feas:
        raise StageError(
            f"HARD ACCEPTANCE FAILURE: {total_lock} lock and {total_feas} feasibility "
            f"violation(s) across the baselines. A baseline that violates the lock is a "
            f"bug in the CONSTRAINT LAYER, not a bad strategy."
        )
    if failed:
        raise StageError(
            f"Stage 5 acceptance check(s) failed: {failed}. Each of these is a test of "
            f"the simulator with an obvious expected answer; a failure means an "
            f"accounting bug, not a disappointing strategy."
        )
    ctx.log(f"summary -> {summary_path}")


# --------------------------------------------------------------- the N controls


def _n_controls(ctx, market, resolved, constraints, projector, control_set) -> dict:
    """B1 must be `N`-invariant; B2's turnover must fall monotonically as `N` grows."""
    from src.sim.simulator import SimulationConfig

    def run(name, hold_days):
        cfg = build_sim_config(resolved, constraints, hold_days=hold_days, seed=ctx.seed)
        cfg.risk_enabled = False        # isolate the LOCK from the envelope
        fn = BASELINES[name].build(market, resolved["baselines"]["params"].get(name, {}))
        return simulate(market, fn, cfg, projector=projector).trajectory

    navs = {n: run("spy_buy_hold", n)["nav"] for n in control_set}
    ref = navs[control_set[0]]
    worst = max(float((navs[n] - ref).abs().max()) for n in control_set)
    checks = {
        "spy_buy_hold_identical_across_n": {
            "passed": bool(worst < 1e-6), "max_nav_difference": worst,
            "control_set": control_set,
            "note": "B1 makes one buy and no sells, so N cannot matter; if it does, the "
                    "lock manager is corrupting state it should not touch",
        }
    }

    # B2 proves the lock BINDS. reference/baselines.md expected performance to "degrade
    # monotonically" in `N`; measured, that is too strong (see the correction there). The
    # effect is overwhelmingly the 0 -> N>0 transition, and beyond it the discrete
    # rebalance calendar and path dependence produce noise. What IS robust, and what is
    # gated here, is that the lock bites hard and that the trend is decreasing.
    from scipy.stats import spearmanr

    trajs = {n: run("momentum", n) for n in sorted(control_set)}
    turnover = {n: float(t["turnover"].sum()) for n, t in trajs.items()}
    ordered_n = sorted(control_set)
    rho = float(spearmanr(ordered_n, [turnover[n] for n in ordered_n]).statistic)

    positive = [n for n in ordered_n if n > 0]
    ratio = (turnover[0] / turnover[min(positive)]
             if 0 in turnover and positive and turnover[min(positive)] > 0 else float("inf"))

    checks["momentum_lock_binds_hard"] = {
        "passed": bool(ratio > 3.0), "turnover_ratio_unlocked_to_locked": ratio,
        "turnover_by_n": turnover,
        "note": "turnover with no lock must dwarf turnover with one; if it does not, the "
                "lock is not actually binding and that would hide inside a trained policy",
    }
    checks["momentum_turnover_trends_down_in_n"] = {
        "passed": bool(rho <= -0.5), "spearman_rho": rho, "turnover_by_n": turnover,
        "note": "a decreasing TREND, not strict monotonicity -- see the correction in "
                "reference/baselines.md section 3",
    }
    return checks


# --------------------------------------------------------------- calibration (Q1)


def _calibrate(ctx, market, resolved, constraints, projector) -> dict:
    """The table that resolves Q1, and the reasoning that reads off it.

    Too conservative -> the agent sits in cash, intervention near 100%, and there is no
    learning signal about anything else. Too loose -> realized drawdowns exceed `D_max`
    routinely and the constraint is decorative. The choice is the setting where a tighter
    `D_max` measurably reduces realized drawdown while the intervention rate stays under
    the configured ceiling.

    **Calibrated over ANNUAL windows, not one long path** -- see the correction in
    reference/risk-envelope.md section 7. On a single 21-year path the running peak never
    resets, so a breach in 2008 is never recovered and the envelope intervenes on almost
    every later session. Measured: intervention 0.774 with a 0.9984 correlation to "is the
    drawdown already past `D_max`" -- i.e. the number was reporting time-under-water, not
    calibration, and it barely moved across the whole candidate grid. Annual windows match
    how the policy is actually trained (episodes) and evaluated (folds).
    """
    from src.sim.runner import load_market

    cal = resolved["calibration"]
    ceiling = float(cal.get("intervention_rate_ceiling", 0.5))
    grid = list(cal["d_max_grid"])
    primary = constraints.drawdown.max_drawdown.primary
    years = sorted({int(y) for y in market.sessions.year.unique()})[1:]

    ctx.log("")
    ctx.log("Risk-envelope calibration (Q1) -- reference/risk-envelope.md section 7")
    ctx.log(f"over {len(years)} annual windows {years[0]}..{years[-1]}, "
            f"{len(cal['candidates'])} candidates x {len(cal['baselines'])} baselines "
            f"x {len(grid)} D_max")

    # One MarketData per year, built once and reused across every candidate.
    yearly = {}
    for year in years:
        yearly[year] = load_market(resolved, start=f"{year}-01-01", end=f"{year}-12-31")[0]

    rows = []
    for cand in cal["candidates"]:
        for name in cal["baselines"]:
            fn_params = resolved["baselines"]["params"].get(name, {})
            for d_max in grid:
                for year, md in yearly.items():
                    if len(md.sessions) < 30:
                        continue
                    env = RiskEnvelope(
                        quantile=float(cand["quantile"]),
                        horizon_days=int(cand["horizon_days"]),
                        block_length=constraints.risk.block_length,
                        aggregation=cand["aggregation"],
                        estimators=tuple(constraints.risk.estimators),
                        crisis_windows=dict(constraints.risk.crisis_windows),
                        n_bootstrap_paths=128,
                    )
                    cfg = build_sim_config(resolved, constraints, max_drawdown=d_max,
                                           seed=ctx.seed)
                    res = simulate(md, BASELINES[name].build(md, fn_params), cfg,
                                   projector=projector, envelope=env)
                    m = summarize(res.trajectory, res.diagnostics, d_max=d_max)
                    rows.append({
                        "candidate": cand["name"], "baseline": name, "d_max": d_max,
                        "year": year,
                        "quantile": cand["quantile"],
                        "horizon_days": cand["horizon_days"],
                        "aggregation": cand["aggregation"],
                        "realized_max_drawdown": m["max_drawdown"],
                        "intervention_rate": m.get("safety_intervention_rate", 0.0),
                        "capital_preservation_rate": m.get("capital_preservation_rate", 0.0),
                        "mean_cash_weight": m["mean_cash_weight"],
                        "annualized_return": m["annualized_return"],
                    })

    table = pd.DataFrame(rows)
    table.to_parquet(ctx.out_path("calibration.parquet"))

    # A candidate is admissible if, at the primary D_max, mean intervention across folds
    # stays under the ceiling, AND a tighter D_max measurably reduces realized drawdown.
    verdicts = []
    for cand_name, block in table.groupby("candidate"):
        at_primary = block[block["d_max"] == primary]
        interv = float(at_primary["intervention_rate"].mean())
        responsive = []
        for _base, sub in block.groupby("baseline"):
            by_dmax = sub.groupby("d_max")["realized_max_drawdown"].mean().sort_index()
            responsive.append(bool(by_dmax.iloc[0] < by_dmax.iloc[-1] - 1e-6))
        verdicts.append({
            "candidate": cand_name,
            "intervention_at_primary": interv,
            "worst_fold_intervention": float(at_primary["intervention_rate"].max()),
            "under_ceiling": bool(interv <= ceiling),
            "drawdown_responds_to_dmax": bool(all(responsive)),
            "mean_cash_at_primary": float(at_primary["mean_cash_weight"].mean()),
            "mean_annret_at_primary": float(at_primary["annualized_return"].mean()),
            "admissible": bool(interv <= ceiling and all(responsive)),
            "horizon_days": int(block["horizon_days"].iloc[0]),
            "aggregation": str(block["aggregation"].iloc[0]),
            "quantile": float(block["quantile"].iloc[0]),
        })

    # Selection is by DESIGN CONSTRAINT first, return only as a tie-break. Picking the
    # highest-return admissible candidate would be fitting the risk layer to performance,
    # which is the failure the whole envelope exists to avoid -- and here it would have
    # chosen a 2-day horizon over a 5-day one for +1.1pp of annualized return, well inside
    # the fold-to-fold noise.
    #
    # The design constraints, both from reference/risk-envelope.md:
    #   * `horizon_days` must span the overnight gap PLUS several sessions, because the
    #     lock may prevent reacting for up to N calendar days (section 3);
    #   * `max` is the conservative aggregation and the right default for a HARD
    #     constraint, since no single estimator's blind spot can then wave an action
    #     through (section 4).
    min_horizon = int(cal.get("min_horizon_days", 5))
    admissible = [v for v in verdicts if v["admissible"]]
    preferred = [v for v in admissible
                 if v["horizon_days"] >= min_horizon and v["aggregation"] == "max"]
    pool = preferred or admissible
    chosen = max(pool, key=lambda v: v["mean_annret_at_primary"]) if pool else None
    if chosen is not None:
        chosen = dict(chosen, selected_by=(
            "design constraints (horizon >= %d, aggregation=max), return as tie-break"
            % min_horizon) if preferred else "return only -- no candidate met the design constraints")

    ctx.log("")
    ctx.log(f"{'candidate':<14}{'interv@primary':>15}{'worstfold':>11}{'<=ceil':>8}"
            f"{'ddresp':>8}{'cash':>7}{'annret':>9}{'admissible':>12}")
    for v in sorted(verdicts, key=lambda v: -v["mean_annret_at_primary"]):
        ctx.log(f"{v['candidate']:<14}{v['intervention_at_primary']:>15.3f}"
                f"{v['worst_fold_intervention']:>11.3f}"
                f"{str(v['under_ceiling']):>8}{str(v['drawdown_responds_to_dmax']):>8}"
                f"{v['mean_cash_at_primary']:>7.2f}{v['mean_annret_at_primary']:>+9.2%}"
                f"{str(v['admissible']):>12}")

    ctx.log("")
    per_dmax = table.groupby("d_max")[["realized_max_drawdown", "intervention_rate"]].mean()
    ctx.log("mean across candidates and folds, by D_max:")
    for d_max, row in per_dmax.iterrows():
        ctx.log(f"  D_max {d_max:.2f}  realized maxDD {row['realized_max_drawdown']:.3f}  "
                f"intervention {row['intervention_rate']:.3f}")

    ctx.log("")
    if chosen:
        best_return = max(admissible, key=lambda v: v["mean_annret_at_primary"])
        ctx.log(f"CHOSEN: {chosen['candidate']}  q={chosen['quantile']} "
                f"horizon={chosen['horizon_days']} agg={chosen['aggregation']}")
        ctx.log(f"  rule: {chosen['selected_by']}")
        ctx.log(f"  mean fold intervention {chosen['intervention_at_primary']:.3f} "
                f"(ceiling {ceiling:.0%}); realized drawdown responds to D_max")
        if best_return["candidate"] != chosen["candidate"]:
            ctx.log(f"  NOTE: {best_return['candidate']} returned more "
                    f"({best_return['mean_annret_at_primary']:+.2%} vs "
                    f"{chosen['mean_annret_at_primary']:+.2%}) but has "
                    f"horizon={best_return['horizon_days']}/agg={best_return['aggregation']}; "
                    f"not chosen, because tuning the risk layer on return is the failure "
                    f"the envelope exists to avoid")
    else:
        ctx.log(f"NO CANDIDATE ADMISSIBLE at ceiling {ceiling:.0%}. Widen the candidate "
                f"set before Stage 7, or the agent will sit in cash with no learning signal.")

    return {"ceiling": ceiling, "primary_d_max": primary, "verdicts": verdicts,
            "chosen": chosen, "n_cells": len(rows), "window": "annual",
            "years": [years[0], years[-1]]}


if __name__ == "__main__":
    raise SystemExit(main())
