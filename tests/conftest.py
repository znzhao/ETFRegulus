"""Shared fixtures.

Real slices over the 2008, 2020 and XLRE-inception windows, checked into
`tests/fixtures/` as small parquet files so the suite runs offline and deterministically
(reference/testing.md section 4). Full history appears only in the stage acceptance runs,
never in the unit suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.loader import load_typed
from src.config.schema import FeaturesConfig, UniverseConfig

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def universe_cfg() -> UniverseConfig:
    cfg, _ = load_typed("config/universe.yaml", UniverseConfig)
    return cfg


@pytest.fixture(scope="session")
def features_cfg() -> FeaturesConfig:
    cfg, _ = load_typed("config/features.yaml", FeaturesConfig)
    return cfg


@pytest.fixture(scope="session")
def price_slice() -> pd.DataFrame:
    """SPY/TLT/HYG/GLD/XLRE, 2015-09 .. 2020-12 -- CONTIGUOUS.

    Contiguity is load-bearing: a fixture stitched from disjoint year-windows manufactures
    enormous fake returns at the seams and breaks every return-based check applied to it.
    This window is chosen because one contiguous range covers both XLRE's inception
    (2015-10-08) and the COVID crash.
    """
    df = pd.read_parquet(FIXTURES / "prices_slice.parquet")
    return df.set_index(["session", "ticker"]).sort_index()


@pytest.fixture(scope="session")
def gfc_slice() -> pd.DataFrame:
    """SPY/TLT/GLD across the GFC, 2007-10 .. 2009-06. Also contiguous."""
    df = pd.read_parquet(FIXTURES / "prices_gfc.parquet")
    return df.set_index(["session", "ticker"]).sort_index()


@pytest.fixture(scope="session")
def macro_slice() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "macro_slice.parquet")


@pytest.fixture(scope="session")
def synthetic_prices() -> pd.DataFrame:
    """A deterministic 800-session price path where the correct answer is computable.

    Used wherever a real slice would be slower without being more diagnostic -- notably
    the lookahead tests, which need more history than a crisis window contains.
    """
    rng = np.random.default_rng(20260901)
    n = 800
    idx = pd.bdate_range("2010-01-04", periods=n, name="session")
    steps = rng.normal(0.0004, 0.011, n)
    close_raw = pd.Series(100.0 * np.exp(np.cumsum(steps)), index=idx)

    # A quarterly distribution, so the adjusted series genuinely diverges from the raw.
    div = pd.Series(0.0, index=idx)
    div.iloc[::63] = 0.35
    div.iloc[0] = 0.0
    growth = (1.0 + close_raw.pct_change().fillna(0.0) + div / close_raw.shift(1).fillna(close_raw.iloc[0]))
    close_adj = 100.0 * growth.cumprod() / growth.iloc[0]

    spread = 0.004 * close_raw
    return pd.DataFrame({
        "close_adj": close_adj,
        "close_raw": close_raw,
        "open_raw": close_raw.shift(1).fillna(close_raw.iloc[0]),
        "high_raw": np.maximum(close_raw, close_raw.shift(1).fillna(close_raw.iloc[0])) + spread,
        "low_raw": np.minimum(close_raw, close_raw.shift(1).fillna(close_raw.iloc[0])) - spread,
        "volume": pd.Series(rng.integers(1_000_000, 5_000_000, n), index=idx, dtype=float),
        "div_per_share": div,
        "is_tradable": True,
    }, index=idx)
