"""Shared wiring between a resolved config and a simulation run.

Stage 4 and Stage 5 both need to turn `config/sim/*.yaml` into a `MarketData`, a
`RiskEnvelope` and a weight source. Keeping that in one place is what stops the two stages
drifting into running subtly different simulators -- which would defeat the point of
running the baselines through the agent's machinery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.baselines.strategies import BASELINES
from src.config.loader import load_typed
from src.config.schema import ConstraintsConfig, UniverseConfig
from src.constraints.projector import make_projector
from src.constraints.risk_envelope import RiskEnvelope
from src.data.curate import PRICES_PATH
from src.sim.simulator import MarketData, SimulationConfig, StepContext


def load_market(resolved: dict, *, start=None, end=None) -> tuple[MarketData, UniverseConfig]:
    ucfg, _ = load_typed(resolved["universe_config"], UniverseConfig)
    prices = pd.read_parquet(PRICES_PATH)
    prices = prices[prices["is_tradable"]]
    sim = resolved.get("simulation", {})
    market = MarketData.from_curated(
        prices, ucfg.tradable_tickers,
        start=start if start is not None else sim.get("start"),
        end=end if end is not None else sim.get("end"),
    )
    return market, ucfg


def build_constraints(resolved: dict) -> ConstraintsConfig:
    from src.config.loader import strict_from_dict

    keys = ("lock", "drawdown", "projection", "risk", "execution")
    return strict_from_dict(ConstraintsConfig, {k: resolved[k] for k in keys if k in resolved})


def build_envelope(constraints: ConstraintsConfig) -> RiskEnvelope:
    r = constraints.risk
    return RiskEnvelope(
        quantile=r.quantile, horizon_days=r.horizon_days, block_length=r.block_length,
        aggregation=r.aggregation, measure=r.measure, estimators=tuple(r.estimators),
        crisis_windows=dict(r.crisis_windows),
    )


def build_sim_config(resolved: dict, constraints: ConstraintsConfig, *,
                     hold_days=None, max_drawdown=None, seed=42) -> SimulationConfig:
    sim = resolved.get("simulation", {})
    return SimulationConfig(
        hold_days=int(hold_days if hold_days is not None
                      else sim.get("hold_days", constraints.lock.hold_days.primary)),
        max_drawdown=float(max_drawdown if max_drawdown is not None
                           else sim.get("max_drawdown",
                                        constraints.drawdown.max_drawdown.primary)),
        cost_bps=constraints.execution.cost_bps,
        initial_cash=float(sim.get("initial_cash", 1_000_000.0)),
        lock_scope=constraints.lock.scope,
        risk_enabled=bool(sim.get("risk_enabled", True)),
        seed=seed,
        risk_lookback=int(sim.get("risk_lookback", 756)),
    )


def weight_source(name: str, market: MarketData, params: dict | None = None) -> Callable:
    if name not in BASELINES:
        raise ValueError(
            f"unknown strategy {name!r}; available: {sorted(BASELINES)} "
            f"(or set simulation.weights_file for an explicit sequence)"
        )
    return BASELINES[name].build(market, params or {})


def weights_from_file(path: str | Path, market: MarketData) -> Callable:
    """Replay an explicit weight sequence: one row per session, one column per ticker.

    Missing sessions hold the previous target; missing tickers are zero. Whatever is not
    allocated becomes cash, so a row need not sum to one.
    """
    frame = pd.read_parquet(path)
    if "session" in frame.columns:
        frame = frame.set_index("session")
    frame.index = pd.DatetimeIndex(frame.index)
    frame = frame.reindex(columns=market.universe).reindex(market.sessions).ffill()
    matrix = np.nan_to_num(frame.to_numpy(dtype=float))
    n = market.n_assets

    def weights(session, ctx: StepContext) -> np.ndarray:
        w = np.zeros(n + 1)
        row = np.clip(matrix[ctx.step], 0.0, None)
        row = np.where(ctx.available, row, 0.0)
        total = row.sum()
        if total > 1.0:
            row = row / total
            total = 1.0
        w[1:] = row
        w[0] = 1.0 - total
        return w

    return weights


def make_weight_fn(resolved: dict, market: MarketData, *, strategy=None) -> tuple[Callable, str]:
    sim = resolved.get("simulation", {})
    path = sim.get("weights_file")
    if strategy is None and path:
        return weights_from_file(path, market), f"file:{path}"
    name = strategy or sim.get("strategy", "equal_weight")
    return weight_source(name, market, sim.get("strategy_params")), name


def make_projector_from(constraints: ConstraintsConfig):
    return make_projector(constraints.projection.backend,
                          constraints.projection.alpha_tolerance)
