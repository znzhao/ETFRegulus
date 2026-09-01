"""Stage 3 gate: feature manifest completeness.

The manifest is what the environment validates the live observation against, so a feature
added upstream without updating the environment fails loudly instead of shifting every
index in the observation vector by one. That only works if the manifest is actually
complete and actually matches the artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.builder import (
    CROSS_PATH,
    ETF_PATH,
    MACRO_PATH,
    MANIFEST_PATH,
    build_manifest,
    warmup_report,
)

pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists(), reason="run Stage 3 first"
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def blocks():
    return (pd.read_parquet(ETF_PATH), pd.read_parquet(CROSS_PATH),
            pd.read_parquet(MACRO_PATH))


def test_manifest_lists_every_column_of_every_block(manifest, blocks):
    etf, cross, macro = blocks
    listed = {(c["block"], c["name"]) for c in manifest["columns"]}
    actual = ({("etf", c) for c in etf.columns}
              | {("cross_sectional", c) for c in cross.columns}
              | {("macro", c) for c in macro.columns})
    assert listed == actual, (
        f"manifest/artifact mismatch: missing {sorted(actual - listed)}, "
        f"extra {sorted(listed - actual)}"
    )


def test_every_column_declares_source_kind_and_scaling(manifest):
    for col in manifest["columns"]:
        assert col["source"], f"{col['name']} has no source"
        assert col["kind"] in ("level", "return", "rank"), col
        assert col["scaling"], f"{col['name']} has no scaling treatment"


def test_per_etf_columns_declare_a_lookback(manifest):
    for col in manifest["columns"]:
        if col["block"] == "etf":
            assert col["lookback_sessions"] is not None, f"{col['name']} has no lookback"
            assert col["expected_warmup_sessions"] is not None


def test_canonical_ticker_order_is_recorded_and_matches_the_config(manifest, universe_cfg):
    """Appending a ticker is a breaking change that invalidates trained policies, so the
    order is pinned in the manifest and the manifest hash is what catches a change."""
    assert manifest["canonical_tickers"] == universe_cfg.tradable_tickers
    assert manifest["synthetic_asset"] == universe_cfg.synthetic_asset
    assert manifest["synthetic_asset"] not in manifest["canonical_tickers"]


def test_the_manifest_is_reproducible_from_the_artifacts(blocks, universe_cfg, features_cfg):
    etf, cross, macro = blocks
    rebuilt = build_manifest(etf, cross, macro, universe_cfg, features_cfg)
    stored = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert rebuilt["columns"] == stored["columns"]
    assert rebuilt["n_columns"] == stored["n_columns"]


def test_no_interior_nan_and_warmup_matches_exactly(blocks, universe_cfg, features_cfg):
    """NaN counts must match the expected warm-up pattern exactly. Any other NaN is a
    bug, not something to fill."""
    etf, _, _ = blocks
    report = warmup_report(etf, universe_cfg, features_cfg)
    assert report["n_mismatches"] == 0, report["mismatches"][:5]


def test_no_infinities_reached_the_artifacts(blocks):
    """Every division guards its denominator; an inf means one did not."""
    for name, block in zip(("etf", "cross_sectional", "macro"), blocks):
        numeric = block.select_dtypes(include=[np.number])
        n_inf = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
        assert n_inf == 0, f"{name} block contains {n_inf} infinite value(s)"


def test_correlation_diagnostic_is_recorded(manifest):
    """High correlation is not automatically wrong -- but it must be a decision, not an
    accident, and that requires it to be written down."""
    diag = manifest["correlation_diagnostic"]
    assert "threshold" in diag and "pairs" in diag
    assert diag["n_pairs_above_threshold"] >= len(diag["pairs"]) or diag["pairs"]


def test_folds_artifact_is_consistent_with_the_scalers():
    folds = json.loads(Path("data/features/folds.json").read_text(encoding="utf-8"))
    assert folds, "Stage 3 wrote no folds"
    for spec in folds:
        assert Path(f"data/features/scalers/{spec['fold_id']}.json").exists()
        assert spec["train_end"] < spec["val_start"] < spec["val_end"] < spec["test_start"]
