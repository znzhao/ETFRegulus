"""The holdings limit: at most `max_holdings` ETFs (cash aside) held at once."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baselines.strategies import BASELINES
from src.config.loader import load_typed
from src.config.schema import ConstraintsConfig, UniverseConfig
from src.constraints.risk_envelope import RiskEnvelope
from src.sim.engine import cap_holdings
from src.sim.runner import make_projector
from src.sim.simulator import MarketData, SimulationConfig, simulate


def test_cap_keeps_locked_then_largest_and_preserves_invested_share():
    a = np.array([0.1, 0.05, 0.3, 0.2, 0.25, 0.1])
    lower = np.array([0, 0.04, 0, 0, 0, 0])          # ETF 0 is locked
    cur = np.zeros(6)
    out = cap_holdings(a, lower, cur, 3)
    assert np.count_nonzero(out[1:]) == 3
    assert out[1] > 0 and out[2] > 0 and out[4] > 0   # locked, then the two largest
    assert out[0] == pytest.approx(0.1) and out.sum() == pytest.approx(1.0)


def test_cap_prefers_a_held_etf_on_a_tie_and_is_off_at_zero():
    a = np.array([0.4, 0.2, 0.2, 0.2])
    cur = np.array([0.5, 0.0, 0.0, 0.5])
    out = cap_holdings(a, np.zeros(4), cur, 1)
    assert out[3] == pytest.approx(0.6) and out[1] == out[2] == 0
    np.testing.assert_array_equal(cap_holdings(a, np.zeros(4), cur, 0), a)


CURATED = Path("data/curated/prices.parquet")


@pytest.mark.skipif(not CURATED.exists(), reason="run Stage 2 first")
@pytest.mark.parametrize("name", ["equal_weight", "momentum", "classical_optimizer"])
@pytest.mark.parametrize("k", [3, 5])
def test_simulation_never_holds_more_than_k(name, k):
    ucfg, _ = load_typed("config/universe.yaml", UniverseConfig)
    cons, _ = load_typed("config/constraints.yaml", ConstraintsConfig)
    prices = pd.read_parquet(CURATED)
    market = MarketData.from_curated(prices[prices["is_tradable"]], ucfg.tradable_tickers,
                                     start="2019-01-02", end="2021-06-30")
    r = cons.risk
    env = RiskEnvelope(quantile=r.quantile, horizon_days=r.horizon_days, block_length=r.block_length,
                       aggregation=r.aggregation, estimators=tuple(r.estimators),
                       crisis_windows=dict(r.crisis_windows), n_bootstrap_paths=64)
    cfg = SimulationConfig(hold_days=30, max_drawdown=0.10, no_trade_band=0.01, peak_window=252,
                           max_holdings=k)
    res = simulate(market, BASELINES[name].build(market, {}), cfg,
                   projector=make_projector(over_budget_rule="no_risk_increase", derisk_step=0.2),
                   envelope=env)
    t = res.trajectory
    held = (t[[f"shares_{x}" for x in market.universe]] > 1e-9).sum(axis=1)
    assert held.max() <= k
    assert held.max() == k or name == "momentum"
    assert res.diagnostics["lock_violations"] == 0
    assert res.diagnostics["feasibility_violations"] == 0
