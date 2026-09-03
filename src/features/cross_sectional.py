"""Cross-sectional features: where each asset sits relative to the others, that day.

The subtlety that makes or breaks this block: **the universe expands over time**. XLRE
launches in 2015, XLC in 2018, so the available set grows from ~20 names to 24. Ranks are
therefore computed over the assets available *that session* and normalized to a
percentile in [0, 1], so a rank means the same thing in 2004 as in 2020. A raw ordinal
rank would drift in meaning across the sample and the agent would learn the drift.

Pre-inception assets are **excluded**, never zero-filled -- zero-filling would place a
non-existent ETF at a meaningful rank (reference/features.md rule 5).
"""

from __future__ import annotations

import pandas as pd

RANK_SOURCES = {
    "rank_ret_21": "ret_21",
    "rank_ret_63": "ret_63",
    "rank_momentum_252": "ret_252",
    "rank_vol_21": "vol_21",
    "rank_vol_63": "vol_63",
    "rank_drawdown_63": "drawdown_63",
}

BENCHMARK = "SPY"


def build_cross_sectional(
    etf_features: pd.DataFrame, group_of: dict[str, str]
) -> pd.DataFrame:
    """Per-session percentile ranks over the available universe.

    `etf_features` is indexed by (session, ticker). Rows absent for a ticker on a session
    are exactly the pre-inception rows, so "available that day" needs no extra masking:
    it is the set of rows that exist.
    """
    out = pd.DataFrame(index=etf_features.index)

    for name, source in RANK_SOURCES.items():
        if source not in etf_features.columns:
            continue
        s = etf_features[source]
        # pct=True gives the percentile directly, over the non-NaN values in that
        # session -- which is precisely the available set.
        out[name] = s.groupby(level="session").rank(pct=True)

    # Relative strength: against SPY, and against the mean of the available universe.
    for horizon in (21, 63, 252):
        col = f"ret_{horizon}"
        if col not in etf_features.columns:
            continue
        s = etf_features[col]
        universe_mean = s.groupby(level="session").transform("mean")
        out[f"rs_vs_universe_{horizon}"] = s - universe_mean

        spy = s.xs(BENCHMARK, level="ticker") if BENCHMARK in s.index.get_level_values("ticker") else None
        if spy is not None:
            aligned = s.index.get_level_values("session").map(spy)
            out[f"rs_vs_spy_{horizon}"] = s.to_numpy() - aligned.to_numpy()

    # Rank within the asset-class group, so "the strongest sector ETF" is expressible
    # separately from "the strongest thing in the whole universe".
    groups = etf_features.index.get_level_values("ticker").map(group_of)
    if "ret_63" in etf_features.columns:
        out["rank_ret_63_in_group"] = (
            etf_features["ret_63"]
            .groupby([etf_features.index.get_level_values("session"), groups])
            .rank(pct=True)
            .to_numpy()
        )

    # The size of the available universe is itself a model input: the agent must be able
    # to tell a 20-name universe from a 24-name one.
    n_available = out.groupby(level="session").transform("size")
    out["universe_size"] = n_available.iloc[:, 0] if isinstance(n_available, pd.DataFrame) else n_available

    return out
