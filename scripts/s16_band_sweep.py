"""R-Q1: choose the no-trade band width on validation years (MODEL_REDESIGN_PLAN.md section 15).

    python -m scripts.s16_band_sweep --campaign seeds_df --checkpoint C3 \
        --config config/experiments/redesign_v2.yaml --bands 0,0.005,0.01,0.02,0.03

For each fold, the campaign's seed ensemble at the given checkpoint is run over the fold's
VALIDATION year at D_max 5/10/15%, once per band width, with everything else from the
config. Reported per band: mean Sharpe, the share of sessions with any trade, and the mean
share of held positions that are locked. Test years are not touched.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from scripts.s14_incremental import EVAL_SEED, _bundle, _fold_keys, _folds
from src.config.loader import resolve_config
from src.evaluation.fold_eval import window_rows
from src.evaluation.rollout import load_policy, rollout
from src.sim.runner import build_constraints
from src.training.incremental import campaign_dir, model_dir

CEILINGS = (0.05, 0.10, 0.15)


def stats(traj, universe) -> dict:
    nav = traj["nav"].to_numpy()
    r = nav[1:] / nav[:-1] - 1
    sd = r.std()
    held = np.stack([traj[f"shares_{t}"].to_numpy() > 1e-9 for t in universe], axis=1)
    locked = np.stack([traj[f"locked_{t}"].to_numpy() for t in universe], axis=1) & held
    traded = traj["turnover"].to_numpy() > 0
    return {"sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
            "trade_share": float(traded.mean()),
            "locked_share": float((locked.sum(1) / np.maximum(held.sum(1), 1)).mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--bands", default="0,0.005,0.01,0.02,0.03")
    ap.add_argument("--hold-days", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    bands = [float(x) for x in a.bands.split(",")]
    cdir = campaign_dir(a.campaign)
    camp = json.loads((cdir / "campaign.json").read_text(encoding="utf-8"))
    config_path = Path(a.config)
    base = resolve_config(config_path)
    rows = []
    t0 = time.time()
    for fold in _folds(camp):
        bundle = _bundle(config_path, base, fold)
        members = [load_policy(model_dir(cdir, k) / a.checkpoint) for k in _fold_keys(camp, fold.test_year)]
        start, end = window_rows(bundle.market, fold.val_start, fold.val_end)
        for b in bands:
            resolved = json.loads(json.dumps(base))
            resolved.setdefault("execution", {})["no_trade_band"] = b
            bb = dataclasses.replace(bundle, resolved=resolved, constraints=build_constraints(resolved))
            for dm in CEILINGS:
                traj = rollout(members, bb, hold_days=a.hold_days, max_drawdown=dm,
                               start_row=start, end_row=end, seed=EVAL_SEED).trajectory
                rows.append({"val_year": fold.test_year - 1, "band": b, "d_max": dm,
                             **stats(traj, bundle.market.universe)})
        print(f"fold {fold.test_year} done ({time.time() - t0:.0f}s)", flush=True)

    out = Path(a.out or f"artifacts/band_sweep/{a.campaign}_{a.checkpoint}_{config_path.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\n{'band (pp)':>9} {'val Sharpe':>11} {'5%':>6} {'10%':>6} {'15%':>6} {'trade days':>11} {'locked':>7}")
    for b in bands:
        sel = [r for r in rows if r["band"] == b]
        by = {dm: np.mean([r["sharpe"] for r in sel if r["d_max"] == dm]) for dm in CEILINGS}
        print(f"{b * 100:>9.1f} {np.mean([r['sharpe'] for r in sel]):>11.3f} "
              + " ".join(f"{by[dm]:>6.2f}" for dm in CEILINGS)
              + f" {np.mean([r['trade_share'] for r in sel]):>11.2f} {np.mean([r['locked_share'] for r in sel]):>7.2f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
