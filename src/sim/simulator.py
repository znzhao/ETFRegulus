"""The deterministic simulator.

Given a legal initial state, a sequence of target weights, and the price history, produce
the complete portfolio trajectory. **No RL anywhere in this module** -- it is the shared
engine that Stage 4, the Stage 5 baselines, and later the environment all run through, so
that a baseline and a policy are compared on exactly the same machinery.

The timeline is fixed and mechanical, because acting on information you would not have had
is the single most common backtest bug:

    session t, CLOSE   observe prices/features through t, portfolio, locks, NAV, peak, DD
                       decide raw target weights a_raw
                       project -> a_proj          (uses only t-close information)
                       (no trading happens on session t)
    session t+1, OPEN  execute at open_raw[t+1] toward a_proj; update shares, cash, locks
    session t+1, CLOSE reinvest distributions; value NAV; update peak, drawdown
                       reward r_t = log(NAV_{t+1} / NAV_t)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.constraints.projector import CASH, AnalyticProjector, FeasibilityProjector
from src.constraints.risk_envelope import RiskEnvelope
from src.portfolio.ledger import Ledger
from src.portfolio.lock_manager import LockManager

#: Signature of a weight source: (session, context) -> raw weights over [CASH, *universe],
#: or None to HOLD the book without deciding (an event-driven agent between decisions).
WeightFn = Callable[[pd.Timestamp, "StepContext"], "np.ndarray | None"]


@dataclass
class StepContext:
    """What a weight source may look at. Deliberately only t-close information."""

    session: pd.Timestamp
    universe: list[str]
    ledger: Ledger
    lock_manager: LockManager
    nav: float
    peak: float
    drawdown: float
    available: np.ndarray
    current_weights: np.ndarray
    lock_remaining_days: np.ndarray
    hold_days: int
    max_drawdown: float
    step: int


@dataclass
class MarketData:
    """Prices and the derived arrays the simulator needs, pre-pivoted once.

    Built from the curated long frame. Pivoting per step would dominate the step cost.
    """

    sessions: pd.DatetimeIndex
    universe: list[str]
    close_raw: np.ndarray        # (T, K)
    open_raw: np.ndarray
    high_raw: np.ndarray
    low_raw: np.ndarray
    div_per_share: np.ndarray
    available: np.ndarray        # (T, K) bool, from inception only
    returns: np.ndarray          # (T, K+1) simple total returns; column 0 is cash = 0

    @property
    def n_assets(self) -> int:
        return len(self.universe)

    @classmethod
    def from_curated(
        cls, prices: pd.DataFrame, universe: Sequence[str],
        start: str | dt.date | None = None, end: str | dt.date | None = None,
    ) -> "MarketData":
        universe = list(universe)
        wide = {}
        for col in ("close_raw", "open_raw", "high_raw", "low_raw", "div_per_share",
                    "close_adj"):
            wide[col] = (prices[col].unstack("ticker")
                         .reindex(columns=universe).sort_index())

        idx = wide["close_raw"].index
        if start is not None:
            idx = idx[idx >= pd.Timestamp(start)]
        if end is not None:
            idx = idx[idx <= pd.Timestamp(end)]

        # Availability comes from the presence of a price, which is derived from
        # inception and nothing else: pre-inception rows were never materialized.
        full_close = wide["close_raw"]
        available = full_close.notna().reindex(idx).to_numpy(dtype=bool)

        # Returns for the risk envelope use close_adj (total return). The cash column is
        # identically zero: zero risk, zero return, the agent's outside option.
        adj = wide["close_adj"].reindex(idx)
        rets = adj.pct_change().to_numpy(dtype=float)
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
        returns = np.zeros((len(idx), len(universe) + 1))
        returns[:, 1:] = rets

        take = lambda c: wide[c].reindex(idx).to_numpy(dtype=float)  # noqa: E731
        return cls(
            sessions=pd.DatetimeIndex(idx), universe=universe,
            close_raw=take("close_raw"), open_raw=take("open_raw"),
            high_raw=take("high_raw"), low_raw=take("low_raw"),
            div_per_share=np.nan_to_num(take("div_per_share")),
            available=available, returns=returns,
        )

    def prices_at(self, i: int, field: str = "close_raw") -> dict[str, float]:
        arr = getattr(self, field)[i]
        return {t: float(v) for t, v in zip(self.universe, arr) if np.isfinite(v)}

    def bars_at(self, i: int) -> dict[str, dict[str, float]]:
        return {
            t: {"low_raw": float(self.low_raw[i, j]), "high_raw": float(self.high_raw[i, j])}
            for j, t in enumerate(self.universe)
            if np.isfinite(self.low_raw[i, j]) and np.isfinite(self.high_raw[i, j])
        }


@dataclass
class SimulationConfig:
    hold_days: int = 30
    max_drawdown: float = 0.05
    cost_bps: float = 0.0
    initial_cash: float = 1_000_000.0
    lock_scope: str = "per_etf"
    risk_enabled: bool = True
    seed: int = 42
    #: Trailing window the stress estimators may see. Visible history only, always.
    risk_lookback: int = 756
    #: Snapshot a reachable state every N steps into the initial-state reservoir.
    #: 0 disables. See reference/env-mdp.md section 5, Approach A.
    reservoir_every: int = 0


@dataclass
class SimulationResult:
    trajectory: pd.DataFrame
    final_ledger: Ledger
    final_lock_manager: LockManager
    diagnostics: dict = field(default_factory=dict)
    #: Reachable-BY-CONSTRUCTION initial states, for the Stage 5 reset sampler. Every
    #: entry was produced by a legal trajectory through this very simulator, which is
    #: what makes it reachable without any verification step.
    reservoir: list[dict] = field(default_factory=list)


def simulate(
    market: MarketData,
    weight_fn: WeightFn,
    cfg: SimulationConfig,
    *,
    projector: FeasibilityProjector | None = None,
    envelope: RiskEnvelope | None = None,
    initial_ledger: Ledger | None = None,
    initial_lock_manager: LockManager | None = None,
    initial_peak: float | None = None,
    start_row: int = 0,
    end_row: int | None = None,
) -> SimulationResult:
    """Run the full trajectory. Deterministic given `cfg.seed`.

    The per-step body lives in `src/sim/engine.py` and is shared verbatim with the
    Gymnasium environment, so a baseline and a policy cannot be run on different
    machinery. Imported inside the function because `engine` imports this module.
    """
    from src.sim.engine import advance, hold, initial_state, observe, reservoir_entry

    projector = projector or AnalyticProjector()
    T = len(market.sessions)
    if T < 2:
        raise ValueError("a simulation needs at least two sessions")

    # `start_row`/`end_row` bound the DECISIONS, not the market. The strategy still sees
    # the whole of `market`, so a 252-day lookback works on the first session of an
    # evaluation window instead of silently returning nothing.
    last = (T - 1) if end_row is None else min(int(end_row), T - 1)
    if not 0 <= start_row < last:
        raise ValueError(
            f"start_row={start_row} and end_row={end_row} leave no sessions to simulate "
            f"(market has {T})")

    st = initial_state(market, cfg, ledger=initial_ledger,
                       lock_manager=initial_lock_manager, peak=initial_peak,
                       row=start_row)

    rows: list[dict] = []
    reservoir: list[dict] = []
    n_lock_violations = 0
    n_feasibility_violations = 0

    for i in range(start_row, last):
        dec = observe(market, st, cfg, i, envelope=envelope)

        if cfg.reservoir_every and i % cfg.reservoir_every == 0 and i > 0:
            reservoir.append(reservoir_entry(market, st, cfg, i, dec))

        proposal = weight_fn(market.sessions[i], dec.ctx)
        if proposal is None:
            out = hold(market, st, cfg, i, dec)
            rows.append(out.row)
            continue
        a_raw = np.asarray(proposal, dtype=float)
        try:
            out = advance(market, st, cfg, i, dec, a_raw, projector)
        except Exception:
            n_feasibility_violations += 1
            raise
        n_lock_violations += out.lock_violations
        rows.append(out.row)

    trajectory = pd.DataFrame(rows).set_index("session")
    return SimulationResult(
        trajectory=trajectory, final_ledger=st.ledger, final_lock_manager=st.lock_manager,
        reservoir=reservoir,
        diagnostics={
            "n_steps": len(rows),
            "lock_violations": n_lock_violations,
            "feasibility_violations": n_feasibility_violations,
            "hold_days": cfg.hold_days,
            "max_drawdown": cfg.max_drawdown,
            "cost_bps": cfg.cost_bps,
            "risk_enabled": cfg.risk_enabled,
            "final_nav": float(trajectory["nav"].iloc[-1]),
            "safety_intervention_rate": float(trajectory["safety_intervened"].mean()),
            "capital_preservation_rate": float(trajectory["capital_preservation"].mean()),
            "mean_proj_distance": float(trajectory["proj_distance"].mean()),
        },
    )
