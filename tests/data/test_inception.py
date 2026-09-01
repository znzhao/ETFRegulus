"""I5 -- the inception invariant.

Before a ticker's `first_session`: the position is 0, the target is 0, and the asset is
unavailable. Inception is a hard constraint and the most common source of lookahead in an
ETF backtest, so it gets a dedicated test rather than a line in another one.

The companion prohibition is on *synthesizing* pre-inception history. It is tempting
(proxy XLRE with a REIT index) and it manufactures a track record that did not exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.curate import curate_prices


def availability_mask(inception: pd.DataFrame, tickers: list[str],
                      session: pd.Timestamp) -> np.ndarray:
    """The availability mask, derived from the inception table and nowhere else."""
    first = inception.set_index("ticker")["first_session"]
    return np.array([bool(t in first.index and session >= pd.Timestamp(first[t]))
                     for t in tickers])


def test_xlre_is_unavailable_before_its_first_bar(price_slice):
    """XLRE launched 2015-10-08. Everything before that must be absent, not zero."""
    xlre = price_slice.xs("XLRE", level="ticker").sort_index()
    first = xlre.index.min()
    assert first == pd.Timestamp("2015-10-08")

    inception = pd.DataFrame({"ticker": ["SPY", "XLRE"],
                              "first_session": [pd.Timestamp("2015-09-01"), first]})
    tickers = ["SPY", "XLRE"]

    before = availability_mask(inception, tickers, pd.Timestamp("2015-09-15"))
    assert before.tolist() == [True, False]
    on = availability_mask(inception, tickers, first)
    assert on.tolist() == [True, True]


def test_pre_inception_rows_are_absent_never_backfilled(universe_cfg, price_slice):
    """Absent, not zero and not back-filled.

    Zero-filling would place a non-existent ETF at a meaningful cross-sectional rank;
    back-filling would manufacture a price history that never existed.
    """
    sessions = price_slice.index.get_level_values("session").unique().sort_values()
    raw = {t: price_slice.xs(t, level="ticker").sort_index()
           for t in ("SPY", "XLRE")}
    curated, inception, _ = curate_prices(universe_cfg, raw, pd.DatetimeIndex(sessions))

    xlre = curated[curated["ticker"] == "XLRE"]
    first = pd.Timestamp(inception.set_index("ticker").loc["XLRE", "first_session"])
    assert xlre["session"].min() == first, "a pre-inception row was materialized"
    assert not (xlre["close_raw"] == 0).any(), "a price was zero-filled"

    spy = curated[curated["ticker"] == "SPY"]
    assert spy["session"].min() < first, "the control ticker was truncated too"


def test_cross_sectional_ranks_exclude_pre_inception_assets(price_slice, features_cfg):
    """A non-existent ETF must not receive a rank, not even a neutral one."""
    from src.features.cross_sectional import build_cross_sectional
    from src.features.etf import build_etf_features

    etf_rows = []
    for ticker in ("SPY", "TLT", "XLRE"):
        g = price_slice.xs(ticker, level="ticker").sort_index()
        f = build_etf_features(g, features_cfg)
        f["ticker"] = ticker
        etf_rows.append(f.reset_index().set_index(["session", "ticker"]))
    etf = pd.concat(etf_rows).sort_index()

    cross = build_cross_sectional(etf, {"SPY": "broad", "TLT": "treasury", "XLRE": "sector"})

    early = cross.xs(pd.Timestamp("2015-09-15"), level="session")
    assert "XLRE" not in early.index, "XLRE was ranked before it existed"
    assert "SPY" in early.index


def test_ranks_are_percentiles_so_a_growing_universe_stays_comparable(price_slice,
                                                                     features_cfg):
    """The universe grows from ~20 names to 24 over the sample.

    A raw ordinal rank would drift in meaning across the sample and the agent would learn
    the drift. A percentile in [0, 1] does not.
    """
    from src.features.cross_sectional import build_cross_sectional
    from src.features.etf import build_etf_features

    rows = []
    for ticker in ("SPY", "TLT", "HYG", "GLD", "XLRE"):
        g = price_slice.xs(ticker, level="ticker").sort_index()
        f = build_etf_features(g, features_cfg)
        f["ticker"] = ticker
        rows.append(f.reset_index().set_index(["session", "ticker"]))
    etf = pd.concat(rows).sort_index()
    cross = build_cross_sectional(etf, {t: "g" for t in
                                        ("SPY", "TLT", "HYG", "GLD", "XLRE")})

    ranks = cross["rank_ret_21"].dropna()
    assert ranks.min() >= 0.0 and ranks.max() <= 1.0, "ranks left the [0, 1] percentile scale"

    # The universe size must actually change across the sample, or this proves nothing.
    sizes = cross["universe_size"]
    assert sizes.nunique() > 1, "the fixture does not exercise a growing universe"
