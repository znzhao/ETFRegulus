"""Re-evaluate an already-trained checkpoint under another rule set (no training).

    python -m scripts.s17_reevaluate --campaign seeds_df --checkpoint C3 \
        --config config/experiments/redesign_v2.yaml --name seeds_df_C3_v2

Runs Stage 8's per-fold ensemble protocol (every seed on validation and test at D_max
5/10/15%, the seed ensemble on validation and on the full test grid) for the campaign's
frozen models at `--checkpoint`, with the simulator rules of `--config`, into
`artifacts/reevaluation/<name>/`. The layout is Stage 8's, so Stages 10 and 12 read it
with `--walk-forward` / `--policy-runs`. With `--full`, Stages 9, 11, 12, 10 and the
acceptance report then run as for any checkpoint's full report.

Why: MODEL_REDESIGN_PLAN.md section 15 changed the rules (no-trade band, 12-month peak,
O-2). Models trained before cannot be retrained for free, but they can be scored fairly
under the corrected rules.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.s14_incremental import COMPLETE, EVAL_SEED, Log, _bundle, _evaluate_fold_ensemble, _fold_keys, _folds
from src.config.loader import resolve_config
from src.evaluation.fold_eval import cell_key, eval_grid
from src.training.incremental import campaign_dir, model_dir

ROOT = Path("artifacts/reevaluation")


def run_full(name: str, title: str, config: str, out: Path, policy_dirs: list[str], first: int, last: int,
             hd: int, dm: float, log) -> None:
    from scripts.s09_stress import main as s09
    from scripts.s10_bootstrap import main as s10
    from scripts.s11_adversarial import main as s11
    from scripts.s12_report import main as s12

    pol = ",".join(policy_dirs)
    cell = cell_key(hd, dm)
    common = ["--config", config, "--force"]
    steps = [
        ("s09", s09, common + ["--policy", pol]),
        ("s11", s11, common + ["--policy", pol]),
        ("s12", s12, common + ["--policy-runs", str(out), "--policy-cell", cell, "--hold-days", str(hd),
                              "--max-drawdown", str(dm), "--first-year", str(first), "--last-year", str(last),
                              "--no-acceptance", "--name", name, "--title", title]),
        ("s10", s10, common + ["--walk-forward", str(out), "--replicates", "1000", "--cell", cell,
                              "--baseline-report", name]),
        ("s12", s12, common + ["--policy-runs", str(out), "--policy-cell", cell, "--hold-days", str(hd),
                              "--max-drawdown", str(dm), "--first-year", str(first), "--last-year", str(last),
                              "--name", name, "--title", title]),
    ]
    for label, fn, args in steps:
        log(f"{label} {' '.join(args)}")
        rc = fn(args)
        if rc not in (0, None):
            log(f"{label} exited {rc} (continuing: a failed gate is a result, not a crash)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--full", action="store_true", help="also run Stages 9-12 (the full report)")
    a = ap.parse_args(argv)

    cdir = campaign_dir(a.campaign)
    camp = json.loads((cdir / "campaign.json").read_text(encoding="utf-8"))
    if camp.get("mode") != "ensemble":
        raise SystemExit("only seed-ensemble campaigns are supported")
    config_path = Path(a.config)
    resolved = resolve_config(config_path)
    out = ROOT / a.name
    log = Log(out / "reevaluate.log")
    log(f"re-evaluating {a.campaign} {a.checkpoint} under {a.config}")
    sess = SimpleNamespace(campaign=camp, log=log, cdir=cdir)

    folds = _folds(camp)
    records = []
    for fold in folds:
        fdir = out / "folds" / str(fold.test_year)
        if not (fdir / COMPLETE).exists():
            t0 = time.time()
            bundle = _bundle(config_path, resolved, fold)
            cells = eval_grid("default", bundle.constraints)
            hd = bundle.constraints.lock.hold_days.primary
            dm = bundle.constraints.drawdown.max_drawdown.primary
            _evaluate_fold_ensemble(sess, fold, bundle, cells, a.checkpoint, fdir, hd, dm)
            (fdir / COMPLETE).write_text(a.checkpoint, encoding="utf-8")
            log(f"  fold {fold.test_year}: {time.time() - t0:.0f}s")
        records.append(json.loads((fdir / "record.json").read_text(encoding="utf-8")))

    lock = sum(r["test"]["lock_violations"] + sum(s["test"]["lock_violations"] for s in r.get("seeds", {}).values())
               for r in records)
    feas = sum(r["test"]["feasibility_violations"]
               + sum(s["test"]["feasibility_violations"] for s in r.get("seeds", {}).values()) for r in records)
    prev = sum(r["test"]["preventable_violations"] for r in records)
    hard = {"lock_violations": lock, "feasibility_violations": feas, "preventable_dmax_violations": prev,
            "passed": lock == 0 and feas == 0 and prev == 0,
            "constraint_validation_failures": [r["fold_id"] for r in records if r["constraint_validation_failure"]]}
    (out / "walk_forward_summary.json").write_text(json.dumps({
        "campaign": camp["campaign"], "checkpoint": a.checkpoint, "config": a.config, "reevaluation": True,
        "n_folds": len(records), "hard_acceptance": hard, "folds": records}, indent=2, default=str), encoding="utf-8")
    log(f"hard acceptance: {hard}")

    if a.full:
        from src.sim.runner import build_constraints
        c = build_constraints(resolved)
        last_fold_models = [str(model_dir(cdir, k) / a.checkpoint)
                            for k in _fold_keys(camp, folds[-1].test_year)]
        run_full(a.name, a.title or a.name, a.config, out, last_fold_models, camp["first_year"], camp["last_year"],
                 c.lock.hold_days.primary, c.drawdown.max_drawdown.primary, log)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
