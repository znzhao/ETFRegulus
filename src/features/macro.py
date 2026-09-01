"""Macro / regime features.

**The feature builder never calls FRED.** It consumes the already point-in-time-aligned
curated series, and the FRED client is deliberately kept out of `src/features/` entirely
so a lag cannot be forgotten at this layer (reference/features.md section 3).

Levels are used as levels and as level *differences*. A log return on a yield is a bug,
and `kind: level` in `config/universe.yaml` is what marks the series it would be a bug on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.schema import FeaturesConfig, UniverseConfig

LEVEL_DIFF_WINDOWS = (1, 5, 21, 63)
VOL_REGIME_WINDOWS = (21, 252)


def build_macro_features(
    macro: pd.DataFrame,
    feature_prices: pd.DataFrame,
    spy_close_adj: pd.Series,
    ucfg: UniverseConfig,
    fcfg: FeaturesConfig,
) -> pd.DataFrame:
    """Assemble the macro block.

    `macro` is the curated FRED frame (already lagged). `feature_prices` is wide,
    session-indexed, one column per feature-only symbol, carrying `close_raw`.
    """
    eps = fcfg.epsilon
    out: dict[str, pd.Series] = {}

    # -- FRED: levels and level differences -------------------------------------
    for sid in macro.columns:
        s = macro[sid]
        out[f"fred_{sid}"] = s
        for n in LEVEL_DIFF_WINDOWS:
            out[f"fred_{sid}_d{n}"] = s.diff(n)

    # CPI and INDPRO are index levels, not rates: a level difference is not comparable
    # across decades, so a year-over-year growth rate is the meaningful transform.
    for sid in ("CPIAUCSL", "INDPRO"):
        if sid in macro.columns:
            out[f"fred_{sid}_yoy"] = macro[sid].pct_change(252)

    # -- yfinance feature-only symbols ------------------------------------------
    for symbol, spec in ucfg.feature_only.items():
        if symbol not in feature_prices.columns:
            continue
        s = feature_prices[symbol]
        key = symbol.replace("^", "").replace("-", "_").replace(".", "_")
        if spec.kind == "level":
            out[f"{key}_level"] = s
            for n in LEVEL_DIFF_WINDOWS:
                out[f"{key}_d{n}"] = s.diff(n)
            # A ratio to the trailing level says "elevated relative to recently", which
            # is the regime question, and it is scale-free.
            mean_21 = s.rolling(21, min_periods=21).mean()
            out[f"{key}_rel_21"] = s / mean_21.where(mean_21.abs() > eps, np.nan)
        else:
            out[f"{key}_ret_1"] = s.pct_change()
            for n in (5, 21, 63):
                out[f"{key}_ret_{n}"] = s.pct_change(n)

    # -- derived spreads --------------------------------------------------------
    # Built from the caret yield series, which are not revised and need only same-day
    # alignment (unlike the monthly FRED series).
    if {"^TNX", "^IRX"} <= set(feature_prices.columns):
        out["term_spread_10y_3m"] = feature_prices["^TNX"] - feature_prices["^IRX"]
    if {"^TNX", "^FVX"} <= set(feature_prices.columns):
        out["term_spread_10y_5y"] = feature_prices["^TNX"] - feature_prices["^FVX"]
    if {"BAA10Y", "AAA10Y"} <= set(macro.columns):
        # Baa minus Aaa strips the common duration leg and leaves pure credit quality.
        out["credit_quality_spread"] = macro["BAA10Y"] - macro["AAA10Y"]
    if {"T10YIE", "T5YIE"} <= set(macro.columns):
        out["breakeven_slope"] = macro["T10YIE"] - macro["T5YIE"]

    # -- market regime ----------------------------------------------------------
    lr = np.log(spy_close_adj.where(spy_close_adj > eps)).diff()
    for n in VOL_REGIME_WINDOWS:
        out[f"market_vol_{n}"] = lr.rolling(n, min_periods=n).std() * np.sqrt(252.0)
    short, long = out["market_vol_21"], out["market_vol_252"]
    out["vol_regime_ratio"] = short / long.where(long.abs() > eps, np.nan)

    sma_200 = spy_close_adj.rolling(200, min_periods=200).mean()
    out["market_trend_regime"] = spy_close_adj / sma_200.where(sma_200.abs() > eps, np.nan)
    roll_max = spy_close_adj.rolling(252, min_periods=252).max()
    out["market_drawdown_252"] = 1.0 - spy_close_adj / roll_max.where(roll_max.abs() > eps, np.nan)

    df = pd.DataFrame(out, index=macro.index)
    df.index.name = "session"
    return df
