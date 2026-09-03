"""Stage 3 -- build features.

    python -m scripts.s03_build_features --config config/features.yaml
    python -m scripts.s03_build_features --config config/features.yaml --fold 2012

Writes the three unscaled feature blocks, one fitted scaler per walk-forward fold, and
the feature manifest.

The blocks are stored **unscaled**: scaling is fold-dependent and lives in the scaler
artifact, because a scaler fitted over the full sample is lookahead that contaminates
every fold at once, invisibly.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.cli.stage import StageContext, StageError, stage
from src.config.loader import load_typed
from src.config.schema import FeaturesConfig, UniverseConfig
from src.data.curate import MACRO_PATH as CURATED_MACRO
from src.data.curate import PRICES_PATH
from src.data.fetch import write_atomic
from src.features.builder import (
    CROSS_PATH,
    ETF_PATH,
    FEATURES_DIR,
    MACRO_PATH,
    MANIFEST_PATH,
    build_all,
    build_manifest,
    correlation_diagnostic,
    warmup_report,
)
from src.features.folds import build_folds
from src.features.scalers import SCALERS_DIR, Scaler


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fold", default=None,
                   help="fit the scaler for one test year only (e.g. 2012)")
    p.add_argument("--allow-warmup-mismatch", action="store_true",
                   help="report NaN warm-up mismatches instead of failing")


@stage(
    name="s03_build_features",
    config_default="config/features.yaml",
    config_cls=FeaturesConfig,
    inputs=["data/curated/prices.parquet", "data/curated/macro.parquet"],
    outputs=[
        "data/features/etf.parquet",
        "data/features/cross_sectional.parquet",
        "data/features/macro.parquet",
        "data/features/feature_manifest.json",
    ],
    upstream="s02_curate_data",
    add_args=_add_args,
)
def main(cfg: FeaturesConfig, ctx: StageContext) -> None:
    """Build the per-ETF, cross-sectional and macro blocks, plus per-fold scalers."""
    ucfg, _ = load_typed(cfg.universe_config, UniverseConfig)

    prices = pd.read_parquet(PRICES_PATH)
    macro = pd.read_parquet(CURATED_MACRO)
    ctx.log(f"curated prices {prices.shape}, macro {macro.shape}")

    etf, cross, macro_features = build_all(prices, macro, ucfg, cfg)
    ctx.log(f"etf block {etf.shape}, cross-sectional {cross.shape}, macro {macro_features.shape}")

    # ------------------------------------------------------------- warm-up check
    warm = warmup_report(etf, ucfg, cfg)
    ctx.log(f"warm-up: {warm['n_mismatches']} (ticker, feature) mismatch(es)")
    if warm["n_mismatches"] and not ctx.args.allow_warmup_mismatch:
        sample = warm["mismatches"][:5]
        raise StageError(
            f"{warm['n_mismatches']} NaN warm-up mismatch(es); the NaN pattern must match "
            f"the manifest exactly, and any other NaN is a bug rather than something to "
            f"fill. Examples: {sample}"
        )

    # -------------------------------------------------------------- correlations
    corr = correlation_diagnostic(etf, cfg.correlation_report_threshold)
    ctx.log(f"correlation diagnostic: {corr['n_pairs_above_threshold']} pair(s) at "
            f"|r| >= {cfg.correlation_report_threshold}")
    for pair in corr["pairs"][:10]:
        ctx.log(f"  {pair['a']:<22} {pair['b']:<22} r={pair['corr']:+.4f}")

    # -------------------------------------------------------------- write blocks
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    write_atomic(etf, ETF_PATH)
    write_atomic(cross, CROSS_PATH)
    write_atomic(macro_features, MACRO_PATH)

    manifest = build_manifest(etf, cross, macro_features, ucfg, cfg)
    manifest["warmup"] = warm
    manifest["correlation_diagnostic"] = corr
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # -------------------------------------------------------------- fold scalers
    folds = build_folds(cfg.folds.first_test_year, cfg.folds.last_test_year)
    if ctx.args.fold:
        folds = [f for f in folds if str(f.test_year) == str(ctx.args.fold)]
        if not folds:
            raise StageError(f"--fold {ctx.args.fold} is not in the configured fold range")

    # One flat frame per session: the per-asset block is scaled with the same statistics
    # across assets, which is what the shared per-asset encoder assumes.
    joined = etf.join(cross, how="left")
    fitted = []
    for fold in folds:
        train_start, train_end = fold.train_range(ucfg.study_start)
        sess = joined.index.get_level_values("session")
        train_rows = joined[(sess >= pd.Timestamp(train_start)) & (sess <= pd.Timestamp(train_end))]
        macro_rows = macro_features.loc[train_start:train_end]
        if train_rows.empty:
            raise StageError(f"{fold.fold_id}: empty training window {train_start}..{train_end}")

        scaler = Scaler.fit(
            pd.concat([train_rows.reset_index(drop=True),
                       macro_rows.reset_index(drop=True)], axis=1),
            fold_id=fold.fold_id, fit_start=train_start, fit_end=train_end,
            spec=cfg.scaling,
        )
        path = scaler.save(SCALERS_DIR)
        fitted.append(fold.as_dict(ucfg.study_start))
        ctx.log(f"{fold.fold_id}: fitted on {scaler.n_rows_fitted:,} rows "
                f"{train_start}..{train_end} "
                f"(val {fold.val_range[0][:4]}, test {fold.test_range[0][:4]}) -> {path.name}")

        # The prohibition, asserted rather than trusted: a scaler's fitted range must not
        # reach into the fold's evaluation period. T10 tests the same property.
        if pd.Timestamp(train_end) >= pd.Timestamp(fold.val_range[0]):
            raise StageError(f"{fold.fold_id}: scaler range overlaps the validation window")

    (FEATURES_DIR / "folds.json").write_text(json.dumps(fitted, indent=2), encoding="utf-8")

    ctx.record(n_etf_features=int(etf.shape[1]),
               n_cross_features=int(cross.shape[1]),
               n_macro_features=int(macro_features.shape[1]),
               n_folds=len(fitted),
               n_correlated_pairs=corr["n_pairs_above_threshold"],
               n_warmup_mismatches=warm["n_mismatches"])
    ctx.log(f"wrote {len(fitted)} fold scaler(s) -> {SCALERS_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())
