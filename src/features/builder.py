"""Assembly of the three feature blocks, and the feature manifest.

Portfolio features are deliberately NOT here: they depend on the live portfolio and are
computed in the environment at every step (reference/features.md section 4).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.schema import FeaturesConfig, UniverseConfig
from src.features.cross_sectional import build_cross_sectional
from src.features.etf import build_etf_features, expected_warmup
from src.features.macro import build_macro_features
from src.features.scalers import is_bounded

FEATURES_DIR = Path("data/features")
ETF_PATH = FEATURES_DIR / "etf.parquet"
CROSS_PATH = FEATURES_DIR / "cross_sectional.parquet"
MACRO_PATH = FEATURES_DIR / "macro.parquet"
MANIFEST_PATH = FEATURES_DIR / "feature_manifest.json"


def _kind(name: str) -> str:
    if is_bounded(name):
        return "rank"
    if name.startswith(("ret_", "logret_")) or "_ret_" in name or name.endswith("_yoy"):
        return "return"
    if name.startswith("rank_"):
        return "rank"
    return "level"


def build_all(
    prices: pd.DataFrame, macro: pd.DataFrame, ucfg: UniverseConfig, fcfg: FeaturesConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the per-ETF, cross-sectional, and macro blocks.

    `prices` is the curated long frame indexed by (session, ticker).
    """
    tradable = ucfg.tradable_tickers

    blocks = []
    for ticker in tradable:
        try:
            g = prices.xs(ticker, level="ticker").sort_index()
        except KeyError:
            continue
        f = build_etf_features(g, fcfg)
        f["ticker"] = ticker
        blocks.append(f.reset_index().set_index(["session", "ticker"]))
    etf = pd.concat(blocks).sort_index()

    cross = build_cross_sectional(etf, ucfg.group_of)

    feature_prices = (
        prices[~prices["is_tradable"]]["close_raw"]
        .unstack("ticker")
        .reindex(columns=ucfg.feature_symbols)
    )
    spy = prices.xs("SPY", level="ticker")["close_adj"].sort_index()
    macro_features = build_macro_features(macro, feature_prices, spy, ucfg, fcfg)

    return etf, cross, macro_features


def build_manifest(
    etf: pd.DataFrame, cross: pd.DataFrame, macro: pd.DataFrame,
    ucfg: UniverseConfig, fcfg: FeaturesConfig,
) -> dict:
    """Every column, with lookback, source, kind, scaling, and expected warm-up.

    The observation builder validates the live observation against this manifest, so a
    feature added upstream without updating the environment fails loudly instead of
    shifting every index in the observation vector by one.
    """
    columns = []
    for block_name, block, source in (
        ("etf", etf, "data/curated/prices.parquet"),
        ("cross_sectional", cross, "data/features/etf.parquet"),
        ("macro", macro, "data/curated/macro.parquet"),
    ):
        for col in block.columns:
            columns.append({
                "name": col,
                "block": block_name,
                "source": source,
                "kind": _kind(col),
                "lookback_sessions": expected_warmup(col, fcfg) if block_name == "etf" else None,
                "scaling": "none (bounded)" if is_bounded(col) else fcfg.scaling.method,
                "expected_warmup_sessions": expected_warmup(col, fcfg) if block_name == "etf" else None,
            })

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "study_start": ucfg.study_start,
        "canonical_tickers": ucfg.tradable_tickers,
        "synthetic_asset": ucfg.synthetic_asset,
        "n_columns": {"etf": etf.shape[1], "cross_sectional": cross.shape[1],
                      "macro": macro.shape[1]},
        "columns": columns,
        # Recorded so a later change to the ticker set is detectable: appending a ticker
        # is a breaking change that invalidates trained policies.
        "per_asset_feature_count": etf.shape[1] + cross.shape[1],
        "note": (
            "Portfolio features are NOT in this manifest -- they depend on the live "
            "portfolio and are built by src/env/state_builder.py. See "
            "reference/features.md section 4. The observation may select a SUBSET of "
            "these columns; the selection is declared by the environment and validated "
            "against this manifest."
        ),
    }


def warmup_report(etf: pd.DataFrame, ucfg: UniverseConfig, fcfg: FeaturesConfig) -> dict:
    """Observed vs expected leading-NaN count, per (ticker, feature).

    NaN counts must match the expected warm-up pattern exactly. Any other NaN is a bug,
    not something to fill.
    """
    mismatches = []
    for ticker in ucfg.tradable_tickers:
        if ticker not in etf.index.get_level_values("ticker"):
            continue
        g = etf.xs(ticker, level="ticker").sort_index()
        for col in g.columns:
            s = g[col]
            first_valid = s.first_valid_index()
            observed = len(s) if first_valid is None else int(s.index.get_loc(first_valid))
            expected = expected_warmup(col, fcfg)
            # Interior NaN is the real bug; a longer-than-expected lead-in can also come
            # from a genuine data gap, so both are reported rather than one being assumed.
            interior = int(s.iloc[observed:].isna().sum()) if first_valid is not None else 0
            if observed != expected or interior:
                mismatches.append({
                    "ticker": ticker, "feature": col,
                    "observed_warmup": observed, "expected_warmup": expected,
                    "interior_nan": interior,
                })
    return {"n_mismatches": len(mismatches), "mismatches": mismatches[:200]}


def correlation_diagnostic(etf: pd.DataFrame, threshold: float) -> dict:
    """Pairwise correlation of the per-ETF block, so redundancy is visible.

    Stacking many highly correlated indicators is a known trap. High correlation is not
    automatically wrong -- but it should be a decision, not an accident.
    """
    sample = etf.select_dtypes(include=[np.number])
    corr = sample.corr(numeric_only=True)
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            c = corr.at[a, b]
            if pd.notna(c) and abs(c) >= threshold:
                pairs.append({"a": a, "b": b, "corr": round(float(c), 4)})
    pairs.sort(key=lambda p: -abs(p["corr"]))
    return {"threshold": threshold, "n_pairs_above_threshold": len(pairs),
            "pairs": pairs[:100]}
