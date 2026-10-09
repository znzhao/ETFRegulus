"""Is more training budget making the policy better? One row per checkpoint.

INCREMENTAL_TRAINING_PLAN.md sections 5-6. The verdict between two checkpoints comes from a
**paired** stationary block bootstrap: the same resampled session blocks are applied to
both checkpoints' return series. Both faced the same market on the same days, so most of
the market noise is common to the two and cancels in the difference -- an unpaired
comparison of two separate bands would call almost everything "no clear change".

The verdict is informational. It never stops the campaign; the user decides that, and the
decision is to be made on the VALIDATION column (plan P5). The test column is shown at
every checkpoint, but choosing the budget that maximizes it would be tuning on the test
set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.bootstrap import (
    DEFAULT_MEAN_BLOCK,
    bootstrap,
    metrics_from_returns,
    portfolio_returns,
    resample_indices,
)

#: Two-sided 90% interval: "improved" means its lower end is above zero.
VERDICT_QUANTILES = (0.05, 0.95)


def paired_difference(before: np.ndarray, after: np.ndarray, *, metric: str = "sharpe",
                      replicates: int = 1000, mean_block: float = DEFAULT_MEAN_BLOCK,
                      seed: int = 42) -> dict:
    """Bootstrap the change `after - before` in one metric, on shared session blocks."""
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.shape != after.shape:
        raise ValueError(f"paired series must cover the same sessions: {before.shape} vs "
                         f"{after.shape}")
    rng = np.random.default_rng(seed)
    diffs = np.empty(replicates)
    for i in range(replicates):
        idx = resample_indices(before.size, method="stationary", mean_block=mean_block,
                               rng=rng)
        diffs[i] = (metrics_from_returns(after[idx])[metric]
                    - metrics_from_returns(before[idx])[metric])
    lo, hi = (float(np.quantile(diffs, q)) for q in VERDICT_QUANTILES)
    return {"metric": metric,
            "observed": float(metrics_from_returns(after)[metric]
                              - metrics_from_returns(before)[metric]),
            "q05": lo, "q50": float(np.median(diffs)), "q95": hi,
            "verdict": verdict(lo, hi), "replicates": replicates}


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "improved"
    if hi < 0:
        return "regressed"
    return "no clear change"


# ----------------------------------------------------------- checkpoint record


def fold_dirs(checkpoint_dir: Path) -> list[Path]:
    return sorted(p for p in (checkpoint_dir / "folds").iterdir() if p.is_dir())


def load_returns(checkpoint_dir: Path, filename: str) -> np.ndarray:
    frames = [pd.read_parquet(d / filename) for d in fold_dirs(checkpoint_dir)]
    return portfolio_returns(frames)


def worst_annual_drawdown(checkpoint_dir: Path, filename: str) -> float:
    """The report's convention: each year is its own window, peak reset each January."""
    worst = 0.0
    for d in fold_dirs(checkpoint_dir):
        nav = pd.read_parquet(d / filename)["nav"].astype(float)
        worst = max(worst, float((1.0 - nav / nav.cummax()).max()))
    return worst


def selections(checkpoint_dir: Path) -> dict[str, str]:
    out = {}
    for d in fold_dirs(checkpoint_dir):
        out[d.name] = json.loads((d / "selection.json").read_text(encoding="utf-8"))["chosen"]
    return out


def mean_candidate_val_sharpe(checkpoint_dir: Path, primary_cell: str) -> float:
    """Validation Sharpe averaged over EVERY candidate, at the primary cell.

    The selected candidate's validation score is biased upward -- it was chosen for being
    the best on that year. The mean over all candidates is not, so it is the cleaner
    measure of whether training itself is improving.
    """
    values = []
    for d in fold_dirs(checkpoint_dir):
        sel = json.loads((d / "selection.json").read_text(encoding="utf-8"))
        for cand in sel["candidates"]:
            cell = cand["validation"].get("cells", {}).get(primary_cell)
            if cell is not None:
                values.append(float(cell["sharpe"]))
    return float(np.mean(values)) if values else float("nan")


#: The ceilings every checkpoint is also reported at (redesign plan, Phase 1 item 3): at 5%
#: the risk layer decides most sessions, so 5% alone says little about the policy itself.
REPORT_CEILINGS = (0.05, 0.1, 0.15)


def _pooled_sharpe(paths: list[Path]) -> float | None:
    if not paths:
        return None
    frames = [pd.read_parquet(p) for p in paths]
    return float(metrics_from_returns(portfolio_returns(frames))["sharpe"])


def _hold_days(primary_cell: str) -> int:
    return int(primary_cell.split("_")[0][1:])


def sharpe_by_ceiling(checkpoint_dir: Path, primary_cell: str,
                      folder: str = "cells") -> dict[str, float | None]:
    """Sharpe of the reported policy at each ceiling, pooled over every fold.

    `cells` holds the test runs; `val_cells` the validation runs, which only an ensemble
    campaign keeps at every ceiling.
    """
    hd = _hold_days(primary_cell)
    return {f"{d:g}": _pooled_sharpe([f / folder / f"N{hd}_D{d}.parquet"
                                      for f in fold_dirs(checkpoint_dir)
                                      if (f / folder / f"N{hd}_D{d}.parquet").exists()])
            for d in REPORT_CEILINGS}


def seed_sharpes(checkpoint_dir: Path, primary_cell: str) -> dict[str, dict] | None:
    """Per-seed Sharpe at each ceiling, validation and test, pooled over every fold.

    Only an ensemble campaign keeps per-seed trajectories; for any other it returns None.
    """
    hd = _hold_days(primary_cell)
    names = sorted({d.name for f in fold_dirs(checkpoint_dir)
                    for d in (f / "seeds").glob("*") if d.is_dir()})
    if not names:
        return None
    out = {}
    for name in names:
        out[name] = {
            part: {f"{d:g}": _pooled_sharpe([f / "seeds" / name / f"{part}_N{hd}_D{d}.parquet"
                                             for f in fold_dirs(checkpoint_dir)
                                             if (f / "seeds" / name / f"{part}_N{hd}_D{d}.parquet").exists()])
                   for d in REPORT_CEILINGS}
            for part in ("val", "test")}
    return out


def baseline_sharpes(report_dir: Path) -> dict[str, float]:
    summary = pd.read_csv(report_dir / "summary.csv", index_col=0)
    return {str(k): float(v) for k, v in summary["Sharpe"].items()}


def build_record(*, label: str, pct: int, checkpoint_dir: Path, report_dir: Path,
                 previous_dir: Path | None, previous_label: str | None,
                 training: dict, hard_acceptance: dict, primary_cell: str,
                 replicates: int = 1000, seed: int = 42) -> dict:
    test = load_returns(checkpoint_dir, "trajectory.parquet")
    val = load_returns(checkpoint_dir, "val_trajectory.parquet")
    sharpes = baseline_sharpes(report_dir)
    policy_sharpe = sharpes.pop("rl_policy")
    unconstrained = {k: v for k, v in sharpes.items() if not k.endswith("_constrained")}
    constrained = {k: v for k, v in sharpes.items() if k.endswith("_constrained")}
    best_u = max(unconstrained.items(), key=lambda kv: kv[1]) if unconstrained else None
    best_c = max(constrained.items(), key=lambda kv: kv[1]) if constrained else None

    band = bootstrap(test, replicates=replicates, seed=seed).bands["sharpe"]
    record = {
        "label": label, "pct": pct,
        "test_sharpe": policy_sharpe,
        "test_sharpe_band": {"q05": band["q05"], "q50": band["q50"], "q95": band["q95"]},
        "test_max_drawdown": worst_annual_drawdown(checkpoint_dir, "trajectory.parquet"),
        "val_sharpe_selected": float(metrics_from_returns(val)["sharpe"]),
        "val_sharpe_all_candidates": mean_candidate_val_sharpe(checkpoint_dir, primary_cell),
        "baselines": sharpes,
        "best_unconstrained": list(best_u) if best_u else None,
        "best_constrained": list(best_c) if best_c else None,
        "gap_to_best_unconstrained": policy_sharpe - best_u[1] if best_u else None,
        "gap_to_best_constrained": policy_sharpe - best_c[1] if best_c else None,
        "selections": selections(checkpoint_dir),
        "hard_acceptance": hard_acceptance,
        "training": training,
        "test_by_ceiling": sharpe_by_ceiling(checkpoint_dir, primary_cell),
        "val_by_ceiling": sharpe_by_ceiling(checkpoint_dir, primary_cell, "val_cells"),
        "seeds": seed_sharpes(checkpoint_dir, primary_cell),
        "vs_previous": None,
    }
    if previous_dir is not None:
        prev_sel = selections(previous_dir)
        record["vs_previous"] = {
            "previous": previous_label,
            "validation": paired_difference(
                load_returns(previous_dir, "val_trajectory.parquet"), val,
                replicates=replicates, seed=seed),
            "test": paired_difference(
                load_returns(previous_dir, "trajectory.parquet"), test,
                replicates=replicates, seed=seed),
            "selection_changes": sum(prev_sel.get(y) != c
                                     for y, c in record["selections"].items()),
        }
    return record


# ------------------------------------------------------------------ rendering


def _f(v, fmt="{:.2f}") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return fmt.format(v)


def _delta(d: dict | None) -> str:
    if not d:
        return "--"
    return f"{d['observed']:+.2f} [{d['q05']:+.2f}, {d['q95']:+.2f}] {d['verdict']}"


def render(records: list[dict], *, campaign: str, primary_cell: str = "N30_D0.05") -> str:
    L: list[str] = []
    add = L.append
    years = sorted(int(y) for y in records[-1]["selections"]) if records else []
    span = (f"test years {years[0]}–{years[-1]}, validation years {years[0] - 1}–"
            f"{years[-1] - 1}" if years else "no folds yet")
    add(f"# Learning curve — campaign `{campaign}`")
    add("")
    add(f"One row per checkpoint, at cell `{primary_cell}`, {span}. Regenerated at every "
        "checkpoint.")
    add("")
    add("**Read the Validation columns to decide anything.** Test is shown so nothing is "
        "hidden, but picking the budget where test looks best is tuning on the test set "
        "(plan P5). Δ is a paired block bootstrap against the previous checkpoint: "
        "observed change, 90% interval, verdict. *No clear change* is common and, on its "
        "own, means little.")
    add("")
    add("| Checkpoint | Budget | Val Sharpe (selected) | Val Sharpe (all candidates) "
        "| Δ val vs previous | Test Sharpe | Test band q05–q95 | Δ test vs previous "
        "| Test max DD | Gap to best market baseline | Gap to best constrained baseline "
        "| Selections changed | Hard acceptance |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        vp = r.get("vs_previous") or {}
        band = r["test_sharpe_band"]
        bu, bc = r.get("best_unconstrained"), r.get("best_constrained")
        ha = r["hard_acceptance"]
        add(f"| {r['label']} | {r['pct']}% | {_f(r['val_sharpe_selected'])} "
            f"| {_f(r['val_sharpe_all_candidates'])} | {_delta(vp.get('validation'))} "
            f"| **{_f(r['test_sharpe'])}** | {_f(band['q05'])}–{_f(band['q95'])} "
            f"| {_delta(vp.get('test'))} | {_f(r['test_max_drawdown'], '{:.1%}')} "
            f"| {_f(r['gap_to_best_unconstrained'], '{:+.2f}')}"
            f"{f' ({bu[0]})' if bu else ''} "
            f"| {_f(r['gap_to_best_constrained'], '{:+.2f}')}"
            f"{f' ({bc[0]})' if bc else ''} "
            f"| {vp.get('selection_changes', '--')} "
            f"| {'PASS' if ha.get('passed') else 'FAIL'} |")
    add("")

    add("## Baselines (Sharpe, same years — unchanged across checkpoints)")
    add("")
    if records:
        base = records[-1]["baselines"]
        add("| Strategy | Sharpe |")
        add("|---|---|")
        for name, value in sorted(base.items(), key=lambda kv: -kv[1]):
            add(f"| {name} | {value:.2f} |")
    add("")

    add("## Training diagnostics")
    add("")
    add("Averaged over all candidates at each checkpoint. `epochs/rollout` is how many of "
        "PPO's configured epochs (`ppo.n_epochs`) ran before the KL early-stop (`target_kl`) "
        "cut the update short; 1.0 means every update stopped in its first epoch. C0's "
        "diagnostics come from the original Stage 8 training run; later ones from each "
        "candidate's last rollout.")
    add("")
    add("| Checkpoint | epochs/rollout (lr 3e-4) | epochs/rollout (lr 1e-4) | proj_distance "
        "| infeasible_fallback | cash weight |")
    add("|---|---|---|---|---|---|")
    for r in records:
        t = r.get("training", {})
        add(f"| {r['label']} | {_f(t.get('epochs_per_rollout_fast'))} "
            f"| {_f(t.get('epochs_per_rollout_slow'))} | {_f(t.get('proj_distance'), '{:.3f}')} "
            f"| {_f(t.get('infeasible_fallback'), '{:.1%}')} "
            f"| {_f(t.get('cash_weight'), '{:.1%}')} |")
    add("")

    add("## By drawdown ceiling (test Sharpe of the reported policy, N fixed)")
    add("")
    add("At 5% the risk layer decides most sessions, so the looser ceilings show more of "
        "the policy itself.")
    add("")
    add("| Checkpoint | " + " | ".join(f"{d:.0%}" for d in REPORT_CEILINGS) + " | Average |")
    add("|---|" + "---|" * (len(REPORT_CEILINGS) + 1))
    for r in records:
        by = r.get("test_by_ceiling") or {}
        vals = [by.get(f"{d:g}") for d in REPORT_CEILINGS]
        known = [v for v in vals if v is not None]
        add(f"| {r['label']} | " + " | ".join(_f(v) for v in vals)
            + f" | {_f(float(np.mean(known)) if known else None)} |")
    add("")

    if any(r.get("seeds") for r in records):
        add("## Seeds (Sharpe averaged over the three ceilings)")
        add("")
        add("Each seed is the same setting trained from a different random start. The spread "
            "between them is the noise floor: a change between checkpoints smaller than it "
            "is not evidence of anything. The ensemble is the reported policy.")
        add("")
        for part, title in (("val", "Validation"), ("test", "Test")):
            names = sorted({n for r in records for n in (r.get("seeds") or {})})
            add(f"**{title}**")
            add("")
            add("| Checkpoint | " + " | ".join(names) + " | Seed mean | Seed range | Ensemble |")
            add("|---|" + "---|" * (len(names) + 3))
            for r in records:
                seeds = r.get("seeds") or {}

                def avg(d):
                    v = [x for x in d.values() if x is not None]
                    return float(np.mean(v)) if v else None

                per = [avg(seeds[n][part]) if n in seeds else None for n in names]
                known = [v for v in per if v is not None]
                ens = avg(r.get("test_by_ceiling" if part == "test" else "val_by_ceiling")
                          or {})
                add(f"| {r['label']} | " + " | ".join(_f(v) for v in per)
                    + f" | {_f(float(np.mean(known)) if known else None)}"
                    + f" | {(_f(min(known)) + '–' + _f(max(known))) if known else '--'}"
                    + f" | {_f(ens)} |")
            add("")

    add("## Selected candidate per test year")
    add("")
    if records:
        years = sorted(records[-1]["selections"])
        add("| Checkpoint | " + " | ".join(years) + " |")
        add("|---|" + "---|" * len(years))
        for r in records:
            add(f"| {r['label']} | "
                + " | ".join(r["selections"].get(y, "--") for y in years) + " |")
    add("")
    return "\n".join(L)
