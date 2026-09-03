"""Stage 12 -- the strategy comparison report.

    python -m scripts.s12_report --config config/evaluation.yaml
    python -m scripts.s12_report --config config/evaluation.yaml --first-year 2012 --last-year 2024

Produces the comparison every result in this project is reported in: the RL policy against
every baseline, out of sample, year by year. Right now there is no policy, so it runs the
six baselines alone -- which is the point. **The format is fixed before Stage 7 trains
anything**, because a report format chosen after seeing the results is a report format
chosen to flatter them. When the policy exists it becomes one more column and nothing else
about this changes.

Writes `artifacts/reports/<name>/`:

    report.md               the whole thing, rendered
    report.html             the same, as a standalone page
    summary.csv             one row per strategy, whole window
    annual_return.csv       rows = years, columns = strategies
    volatility.csv
    sharpe.csv
    max_drawdown.csv
    allocation_<category>.csv    seven of them
    trajectories/<strategy>/<year>.parquet

Scope note: [reference/stages.md](../reference/stages.md) gives Stage 12 a second job --
consolidating Stages 8-11 into an acceptance table. That half needs stages that do not
exist yet. This is the strategy-comparison half, complete and runnable now.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cli.stage import StageContext, StageError, stage
from src.config.loader import load_typed
from src.config.schema import UniverseConfig
from src.evaluation.acceptance import build as build_acceptance
from src.evaluation.categories import CATEGORIES, ticker_to_category
from src.evaluation.report import (
    YearResult,
    annual_metrics,
    build_compliance,
    build_summary,
    build_tables,
    category_allocation,
)
from src.sim.runner import (
    build_constraints,
    build_envelope,
    build_sim_config,
    load_market,
    make_projector_from,
    weight_source,
)
from src.evaluation.render import render_page
from src.sim.simulator import simulate

REPORTS = Path("artifacts/reports")
RUNS = Path("artifacts/runs")


def _latest(pattern: str) -> dict | None:
    found = sorted(RUNS.glob(pattern))
    if not found:
        return None
    return json.loads(found[-1].read_text(encoding="utf-8"))


def policy_year_results(spec: str, category_of: dict, ctx,
                        cell: str | None = None) -> list[YearResult]:
    """The Stage 8 per-fold trajectories, as report rows.

    They need no adaptation: walk-forward already evaluates one trained model per test
    year over that year alone, which is exactly this report's independent-window
    convention. The RL column is therefore the same measurement as every baseline column,
    not a differently-computed number placed beside them.

    `cell` selects a specific `(N, D_max)` grid cell instead of the fold's headline
    trajectory. This is not a re-fit and needs no retraining: `N` and `D_max` enter the
    observation, so one policy serves the whole grid by construction (env-mdp.md 4).
    Reading a different cell asks the same trained model what it does under a different
    mandate -- which is the entire point of conditioning on the parameters.
    """
    directory = (sorted(RUNS.glob("s08_walk_forward_*"))[-1] if spec in ("latest", "")
                 else (Path(spec) if Path(spec).exists() else RUNS / spec))
    pattern = f"folds/*/cells/{cell}.parquet" if cell else "folds/*/trajectory.parquet"
    paths = sorted(directory.glob(pattern))
    if not paths:
        available = sorted({q.stem for q in directory.glob("folds/*/cells/*.parquet")})
        raise StageError(
            f"no trajectories matching {pattern!r} under {directory}."
            + (f" Cells available: {available}" if available else ""))

    def year_of(path: Path) -> int:
        return int(path.parent.name if cell is None else path.parent.parent.name)

    ctx.log(f"rl_policy: {len(paths)} fold trajectories from {directory.name}"
            + (f", cell {cell}" if cell else ""))

    out = []
    for path in paths:
        traj = pd.read_parquet(path)
        out.append(YearResult(
            strategy="rl_policy", year=year_of(path), trajectory=traj,
            metrics=annual_metrics(traj),
            allocation=category_allocation(traj, category_of)))
    return out


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--name", default="baselines",
                   help="report directory under artifacts/reports/")
    p.add_argument("--title", default=None,
                   help="page title; defaults to a name derived from --name")
    p.add_argument("--first-year", type=int, default=None)
    p.add_argument("--last-year", type=int, default=None)
    p.add_argument("--only", default=None, help="comma-separated subset of strategies")
    p.add_argument("--hold-days", type=int, default=None)
    p.add_argument("--max-drawdown", type=float, default=None)
    p.add_argument("--policy-runs", default=None,
                   help="Stage 8 run id or directory whose per-fold trajectories become "
                        "the `rl_policy` column; 'latest' picks the newest")
    p.add_argument("--no-acceptance", action="store_true",
                   help="skip the acceptance table (it needs Stages 8-11)")
    p.add_argument("--no-unconstrained", action="store_true",
                   help="omit the plain market benchmarks and report only the "
                        "constraint-layer versions")
    p.add_argument("--policy-cell", default=None,
                   help="which Stage 8 (N, D_max) cell the rl_policy column comes from, "
                        "e.g. N30_D0.05. Defaults to the fold's headline trajectory. The "
                        "policy is parameter-conditioned, so evaluating the SAME trained "
                        "model at another cell is the design working, not a re-fit.")


# --------------------------------------------------------------------- the run


def run_year(market, weight_fn, sim_cfg, projector, envelope, year: int):
    """One strategy, one year, as an INDEPENDENT window.

    `market` spans the full history so a 252-day lookback works on 2 January; only the
    decisions are restricted to the year. Fresh capital, no inherited positions, no
    inherited locks, peak reset to the opening NAV.
    """
    rows = np.flatnonzero(market.sessions.year == year)
    if rows.size < 2:
        return None
    start, end = int(rows[0]), int(rows[-1])
    return simulate(market, weight_fn, sim_cfg, projector=projector, envelope=envelope,
                    start_row=start, end_row=end)


def collect(resolved: dict, ctx: StageContext, strategies: list[str],
            years: list[int], hold_days, max_drawdown, *,
            constrained: bool = True, suffix: str = "") -> list[YearResult]:
    """Run each strategy year by year.

    `constrained=False` produces the **market benchmark**: the rule as an ordinary
    investor would run it, with no holding lock, no drawdown ceiling and no risk
    envelope. That version is what a name like `spy_buy_hold` actually promises, and
    without it the report has no market reference at all -- only rules that the
    constraint layer has already reshaped. Measured on the constrained runs, SPY
    buy-and-hold requests 100% SPY on every single step and is cut to a 14.5% average
    holding in 2020, which is the safety layer working exactly as specified and is also
    not a buy-and-hold benchmark by any reading of the words.
    """
    market, ucfg = load_market(resolved)
    constraints = build_constraints(resolved)
    projector = make_projector_from(constraints)
    envelope = build_envelope(constraints) if constrained else None
    sim_cfg = build_sim_config(
        resolved, constraints,
        hold_days=hold_days if constrained else 0,
        # A ceiling of 1.0 can never be breached, so capital preservation never engages.
        max_drawdown=max_drawdown if constrained else 1.0, seed=ctx.seed)
    if not constrained:
        sim_cfg.risk_enabled = False
    category_of = ticker_to_category(ucfg)
    params = (resolved.get("baselines", {}) or {}).get("params", {})

    ctx.log(f"cell: N={sim_cfg.hold_days}  D_max={sim_cfg.max_drawdown}  "
            f"cost_bps={sim_cfg.cost_bps}  risk={sim_cfg.risk_enabled}"
            f"{'' if constrained else '   [MARKET BENCHMARK -- no lock, no ceiling]'}")

    results: list[YearResult] = []
    for name in strategies:
        for year in years:
            # REBUILT PER YEAR, and that is load-bearing. These strategies carry state --
            # `spy_tlt_60_40` holds a `held` flag, `equal_weight` remembers the universe
            # size -- and every report year is an independent window with fresh capital.
            # Sharing one weight function across years leaks the previous year's state in:
            # only the first year saw its initial deployment, and every later year opened
            # 100% in cash waiting for a month-end rebalance, roughly 20 sessions of
            # unintended cash drag that silently understated every rebalancing baseline.
            weight_fn = weight_source(name, market, params.get(name))
            out = run_year(market, weight_fn, sim_cfg, projector, envelope, year)
            if out is None:
                ctx.warn(f"{name} {year}: too few sessions, skipped")
                continue
            if out.diagnostics["lock_violations"]:
                raise StageError(
                    f"{name} {year}: {out.diagnostics['lock_violations']} lock "
                    "violation(s). A fold failing this is not a weaker result, it is an "
                    "invalid one (evaluation.md section 3)."
                )
            results.append(YearResult(
                strategy=name + suffix, year=year, trajectory=out.trajectory,
                metrics=annual_metrics(out.trajectory),
                allocation=category_allocation(out.trajectory, category_of),
            ))
        done = [r for r in results if r.strategy == name + suffix]
        ctx.log(f"  {name + suffix:<28} {len(done)} years, "
                f"mean return {np.mean([r.metrics['total_return'] for r in done]):+.2%}")
    return results


# ------------------------------------------------------------------ rendering


def _fmt(frame: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    return frame.round(decimals)


def _md_table(frame: pd.DataFrame, decimals: int = 2, index_name: str = "Year") -> str:
    """Render a markdown table without pulling in `tabulate` for it.

    `DataFrame.to_markdown` needs an optional dependency, and a report generator is not a
    good reason to add one to `requirements.txt`. Columns are right-aligned so the digits
    line up, which is most of what makes a numeric table readable.
    """
    def cell(v) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "--"
        return f"{v:.{decimals}f}" if isinstance(v, (int, float, np.floating)) else str(v)

    header = [index_name, *(str(c) for c in frame.columns)]
    rows = [[str(idx), *(cell(v) for v in frame.loc[idx])] for idx in frame.index]
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.rjust(w) for c, w in zip(cells, widths)) + " |"

    sep = "|" + "|".join("-" * (w + 1) + ":" for w in widths) + "|"
    return "\n".join([line(header), sep, *(line(r) for r in rows)])


def render_markdown(summary: pd.DataFrame, tables: dict, meta: dict,
                    compliance: pd.DataFrame) -> str:
    L: list[str] = []
    add = L.append

    add(f"# {meta['title']}")
    add("")
    add(f"Window **{meta['first_year']}–{meta['last_year']}** · "
        f"parameters **N = {meta['hold_days']} calendar days**, "
        f"**D_max = {meta['max_drawdown']:.0%}** · "
        f"{meta['n_strategies']} strategies · generated {meta['generated_at']}")
    add("")
    add("> **This is the reference format.** When the RL policy exists it becomes one more "
        "column in every table below and nothing else changes. The format is fixed before "
        "training deliberately: a report format chosen after seeing the results is a "
        "report format chosen to flatter them.")
    add("")

    add("## How to read this")
    add("")
    add("Three conventions, each of which changes what the numbers mean.")
    add("")
    add("**Every year is an independent evaluation window.** Fresh capital each January, "
        "no inherited positions, no inherited locks, and the drawdown peak reset to the "
        "opening NAV. This is forced, not stylistic: on a continuous multi-year run the "
        "drawdown ceiling is measured against a peak that never resets, and `spy_buy_hold` "
        "breached `D_max = 15%` in 2009, went to 100% cash and stayed there for fifteen "
        "years — every annual cell from 2009 on would read exactly 0.00%. Independent "
        "years also match how walk-forward evaluates the policy (one trained model per "
        "test year), which is what makes the RL column comparable to these.")
    add("")
    add("**The risk-free rate is zero, so Sharpe = annualized return / annualized "
        "volatility.** CASH in this universe returns exactly 0.00% per day and is the "
        "agent's outside option, so excess return over the risk-free asset *is* the raw "
        "return. Substituting a T-bill series would make CASH a negative-carry asset the "
        "simulator does not model.")
    add("")
    add("**Allocation is the time average of daily weights**, in percentage points, "
        "summing to 100 by construction. Not a year-end snapshot: a snapshot cannot tell "
        "a portfolio that held 60% equity all year from one that held 0% for eleven "
        "months and 60% in December.")
    add("")
    add("**These baselines run through the full constraint layer** — the same per-ETF "
        f"{meta['hold_days']}-day holding lock, the same {meta['max_drawdown']:.0%} "
        "drawdown ceiling and the same risk envelope the agent faces. `spy_buy_hold` here "
        "is *not* unconstrained SPY; it is SPY as this system would have been allowed to "
        "hold it. That is what makes it a fair reference point, and it is why the returns "
        "are lower than the index.")
    add("")

    add("## Summary — whole window")
    add("")
    add("Return chains the independent years (what a caller redeploying each January would "
        "have compounded). Volatility pools every daily return. **Max drawdown is the "
        "worst annual drawdown**, not a drawdown across the window — a cross-window figure "
        "would be measured against a peak no strategy operated under.")
    add("")
    perf = summary[["Annual return %", "Return std %", "Sharpe", "Sortino", "Max drawdown %"]]
    add(_md_table(perf, 2, index_name="Strategy"))
    add("")
    add("### Average allocation — whole window (percentage points, rows sum to 100)")
    add("")
    alloc = summary[list(CATEGORIES)].copy()
    alloc["Total"] = alloc.sum(axis=1)
    add(_md_table(alloc, 1, index_name="Strategy"))
    add("")

    add("## Constraint compliance")
    add("")
    add("`Lock violations` and `feasibility violations` are **hard acceptance criteria**: "
        "a result failing either is not a weaker result, it is an invalid one, and this "
        "report refuses to build if any cell shows one.")
    add("")
    add(_md_table(compliance, 2, index_name="Strategy"))
    add("")
    add(f"**`Years D_t > D_max` is a count, not a verdict.** `D_max = "
        f"{meta['max_drawdown']:.0%}` is an action-level constraint on what the agent may "
        "*do*, never a guarantee about the realized path — a gap that opens overnight "
        "moves the NAV with no action available to prevent it, and the lock can hold a "
        "falling position for weeks. So a realized drawdown above the ceiling is expected "
        "and is visible in the table below. What *would* be a defect is a **preventable** "
        "violation: an action that should have been blocked and was not. Separating the "
        "two is the drawdown violation taxonomy "
        "([evaluation.md](../../reference/evaluation.md) §4), which Stage 8 implements and "
        "T13 proves actually fires. Until that exists this column is deliberately "
        "unadjudicated.")
    add("")

    add("## Year by year")
    add("")
    for key, title, decimals, note in (
        ("annual_return", "Annual return (%)", 2, None),
        ("volatility", "Return standard deviation (%, annualized)", 2, None),
        ("sharpe", "Sharpe ratio (rf = 0)", 2,
         "The headline risk-adjusted measure."),
        ("sortino", "Sortino ratio (rf = 0, downside deviation)", 2,
         "For reference: does not penalise upside deviation the way Sharpe does."),
        ("max_drawdown", "Maximum drawdown (%, within year)", 2,
         "Peak resets on the first session of each year."),
    ):
        add(f"### {title}")
        if note:
            add("")
            add(f"*{note}*")
        add("")
        add(_md_table(tables[key], decimals))
        add("")

    add("## Allocation by category")
    add("")
    add("Percentage points, time-averaged within each year. For any (year, strategy) the "
        "seven tables below sum to 100.")
    add("")
    for category in CATEGORIES:
        add(f"### {category}")
        add("")
        add(_md_table(tables[f"allocation::{category}"], 1))
        add("")

    add("## Reproducing this")
    add("")
    add("```bash")
    add(f"python -m scripts.s12_report --config {meta['config']} \\")
    add(f"    --first-year {meta['first_year']} --last-year {meta['last_year']}")
    add("```")
    add("")
    add(f"Run id `{meta['run_id']}` · git `{meta['git_sha']}` · seed {meta['seed']}")
    return "\n".join(L)


# --------------------------------------------------------------------- the stage


@stage(
    name="s12_report",
    config_default="config/evaluation.yaml",
    inputs=["data/curated/prices.parquet"],
    outputs=["artifacts/reports/{run_id}/report.md"],
    upstream="s05_run_baselines",
    add_args=_add_args,
)
def main(resolved: dict, ctx: StageContext) -> None:
    """Build the year-by-year strategy comparison."""
    import datetime as dt

    from src.cli.stage import git_state

    args = ctx.args
    wf = resolved.get("walk_forward", {}) or {}
    first = int(args.first_year or wf.get("first_test_year", 2012))
    last = int(args.last_year or wf.get("last_test_year", 2025))

    market, _ = load_market(resolved)
    available_years = sorted({int(y) for y in market.sessions.year})
    years = [y for y in range(first, last + 1) if y in available_years]
    if not years:
        raise StageError(f"no sessions between {first} and {last}")
    if years[-1] < last:
        ctx.warn(f"data ends in {years[-1]}; reporting {years[0]}-{years[-1]}")

    names = (resolved.get("baselines", {}) or {}).get("names", [])
    strategies = [s.strip() for s in args.only.split(",")] if args.only else list(names)

    constraints = build_constraints(resolved)
    hold_days = args.hold_days or constraints.lock.hold_days.primary
    max_drawdown = (args.max_drawdown if args.max_drawdown is not None
                    else constraints.drawdown.max_drawdown.primary)

    ctx.log(f"{len(strategies)} strategies x {len(years)} years "
            f"({years[0]}-{years[-1]}) = {len(strategies) * len(years)} runs")
    # The market benchmarks first: they are what a reader compares against, and what
    # the plain names promise. The constrained versions carry an explicit suffix.
    results: list[YearResult] = []
    reported = []
    if not args.no_unconstrained:
        results += collect(resolved, ctx, strategies, years, hold_days, max_drawdown,
                           constrained=False)
        reported += list(strategies)
    results += collect(resolved, ctx, strategies, years, hold_days, max_drawdown,
                       constrained=True,
                       suffix="_constrained" if not args.no_unconstrained else "")
    reported += [s + ("_constrained" if not args.no_unconstrained else "")
                 for s in strategies]
    strategies = reported
    if not results:
        raise StageError("no results")

    if args.policy_runs:
        from src.config.loader import load_typed
        from src.config.schema import UniverseConfig

        ucfg, _ = load_typed(resolved["universe_config"], UniverseConfig)
        policy_rows = policy_year_results(args.policy_runs, ticker_to_category(ucfg),
                                          ctx, cell=args.policy_cell)
        # Only the years both sides actually cover, so no column is compared against a
        # different span than its neighbours.
        shared = {r.year for r in policy_rows} & set(years)
        results = [r for r in results if r.year in shared]
        results += [r for r in policy_rows if r.year in shared]
        years = sorted(shared)
        strategies = ["rl_policy", *strategies]
        ctx.log(f"comparing over {len(years)} shared year(s): {years[0]}-{years[-1]}")

    tables = build_tables(results, strategies)
    summary = build_summary(results, strategies)
    compliance = build_compliance(results, strategies, max_drawdown)

    acceptance = None
    if not args.no_acceptance:
        wf = _latest("s08_walk_forward_*/walk_forward_summary.json")
        st = _latest("s09_stress_*/stress_summary.json")
        bs = _latest("s10_bootstrap_*/bootstrap_summary.json")
        adv = _latest("s11_adversarial_*/adversarial_summary.json")
        comparison = None
        if "rl_policy" in summary.index:
            comparison = {"beats": {}}
            rl = summary.loc["rl_policy"]
            for name in summary.index:
                if name == "rl_policy":
                    continue
                better = float(rl["Sharpe"]) > float(summary.loc[name, "Sharpe"])
                note = ("POINT ESTIMATE ONLY, and on 13 years it carries no "
                        "significance: read the Stage 10 bootstrap bands before "
                        "treating a pass here as a result.")
                if name == "spy_tlt_60_40":
                    note = ("the bar that matters -- a two-line static allocation "
                            "anyone could implement. " + note)
                comparison["beats"][name] = {
                    "passed": better,
                    "observed": (f"Sharpe {rl['Sharpe']:.2f} vs "
                                 f"{summary.loc[name, 'Sharpe']:.2f}"),
                    "requirement": "higher Sharpe than the baseline",
                    "note": note,
                }
        acceptance = build_acceptance(wf, st, bs, adv, comparison)
        ctx.log("")
        ctx.log(f"acceptance: {acceptance.to_dict()['n_passed']}/"
                f"{acceptance.to_dict()['n_criteria']} criteria pass; "
                f"blocking failures: {acceptance.blocking_failures or 'none'}")
        (out_dir_early := REPORTS / args.name).mkdir(parents=True, exist_ok=True)
        (out_dir_early / "acceptance.json").write_text(
            json.dumps(acceptance.to_dict(), indent=2), encoding="utf-8")

    sha, dirty = git_state()
    default_title = ("Constrained Baseline Scorecard" if args.name == "baselines"
                     else f"{args.name.replace('_', ' ').title()} Scorecard")
    meta = {
        "title": args.title or default_title, "name": args.name, "first_year": years[0], "last_year": years[-1],
        "hold_days": hold_days, "max_drawdown": max_drawdown,
        "n_strategies": len(strategies), "run_id": ctx.run_id, "git_sha": sha or "?",
        "seed": ctx.seed, "config": str(ctx.config_path).replace("\\", "/"),
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    out_dir = REPORTS / args.name
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(
        render_markdown(summary, tables, meta, compliance), encoding="utf-8")
    (out_dir / "report.html").write_text(
        render_page(summary, tables, meta, compliance, acceptance), encoding="utf-8")
    summary.to_csv(out_dir / "summary.csv")
    compliance.to_csv(out_dir / "compliance.csv")
    for key, frame in tables.items():
        name = key.replace("allocation::", "allocation_").replace(" ", "_").lower()
        frame.to_csv(out_dir / f"{name}.csv")
    for r in results:
        d = out_dir / "trajectories" / r.strategy
        d.mkdir(parents=True, exist_ok=True)
        r.trajectory.to_parquet(d / f"{r.year}.parquet")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # The run dir gets the canonical copy so the stage harness's output contract holds.
    ctx.out_path("report.md").write_text((out_dir / "report.md").read_text(encoding="utf-8"),
                                         encoding="utf-8")

    ctx.record(n_strategies=len(strategies), n_years=len(years),
               n_cells=len(results), report_dir=str(out_dir))
    ctx.log("")
    ctx.log(summary[["Annual return %", "Return std %", "Sharpe",
                     "Max drawdown %"]].round(2).to_string())
    ctx.log("")
    ctx.log(f"report: {out_dir / 'report.md'}  and  {out_dir / 'report.html'}")


if __name__ == "__main__":
    raise SystemExit(main())
