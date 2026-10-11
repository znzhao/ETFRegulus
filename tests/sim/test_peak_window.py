"""The rolling drawdown peak (`SimulationConfig.peak_window`), used by the web app.

With a window of W sessions, the peak on every row is the highest close NAV over the last W
rows (the current one included), and the drawdown follows from it. With the default 0 the
simulator is unchanged: the peak is the running maximum since the start.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baselines.strategies import BASELINES
from src.config.loader import load_typed
from src.config.schema import UniverseConfig
from src.sim.simulator import MarketData, SimulationConfig, simulate

CURATED = Path("data/curated/prices.parquet")
pytestmark = pytest.mark.skipif(not CURATED.exists(), reason="run Stage 2 first")


@pytest.fixture(scope="module")
def market() -> MarketData:
    ucfg, _ = load_typed("config/universe.yaml", UniverseConfig)
    prices = pd.read_parquet(CURATED)
    prices = prices[prices["is_tradable"]]
    return MarketData.from_curated(prices, ucfg.tradable_tickers, start="2019-01-02", end="2021-12-31")


def run(market, window, **kw):
    cfg = SimulationConfig(hold_days=30, max_drawdown=0.15, risk_enabled=False, peak_window=window)
    return simulate(market, BASELINES["spy_buy_hold"].build(market, {}), cfg, **kw).trajectory


@pytest.mark.parametrize("window", [1, 5, 63, 252])
def test_peak_is_the_rolling_max_of_close_nav(market, window):
    t = run(market, window)
    # Row k's window: the previous closes on rows k-W+1..k, plus the initial NAV while the
    # window still reaches back before the first row.
    navs = np.concatenate([[1_000_000.0], t["nav"].to_numpy()])
    expect = [navs[max(0, k + 2 - window):k + 2].max() for k in range(len(t))]
    np.testing.assert_allclose(t["peak_nav"].to_numpy(), expect, rtol=1e-12)
    np.testing.assert_allclose(t["drawdown"], 1 - t["nav"] / t["peak_nav"], atol=1e-12)


def test_default_is_the_all_time_peak(market):
    t = run(market, 0)
    np.testing.assert_allclose(t["peak_nav"], np.maximum.accumulate(
        np.maximum(t["nav"].to_numpy(), 1_000_000.0)), rtol=1e-12)


def test_rolling_peak_recovers_after_a_crash(market):
    """COVID: in June-July 2020 SPY was still below its February high, which a 63-session
    peak has already forgotten."""
    all_time, rolling = run(market, 0), run(market, 63)
    window = slice("2020-06", "2020-07")
    assert rolling.loc[window, "drawdown"].mean() < 0.5 * all_time.loc[window, "drawdown"].mean()
    assert (rolling["peak_nav"] <= all_time["peak_nav"] + 1e-6).all()


def test_rolling_window_rejects_an_inherited_peak(market):
    with pytest.raises(ValueError, match="rolling"):
        run(market, 252, initial_peak=2_000_000.0)
