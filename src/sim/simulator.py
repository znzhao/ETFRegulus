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
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.constraints.projector import (
    CASH,
    AnalyticProjector,
    FeasibilityProjector,
    ProjectionDiagnostics,
)
from src.constraints.risk_envelope import RiskEnvelope, headroom, risk_budget
from src.portfolio.execution import apply_and_lock, execute
from src.portfolio.ledger import Ledger
from src.portfolio.lock_manager import LockManager
from src.portfolio.valuation import value

#: Signature of a weight source: (session, context) -> raw weights over [CASH, *universe].
WeightFn = Callable[[pd.Timestamp, "StepContext"], np.ndarray]


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
    max_drawdown: float = 0.15
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
) -> SimulationResult:
    """Run the full trajectory. Deterministic given `cfg.seed`."""
    projector = projector or AnalyticProjector()
    universe = market.universe
    K = len(universe)
    T = len(market.sessions)
    if T < 2:
        raise ValueError("a simulation needs at least two sessions")

    ledger = (initial_ledger or Ledger(cash=cfg.initial_cash)).copy()
    lm = (initial_lock_manager or LockManager(universe=list(universe),
                                              scope=cfg.lock_scope)).copy()

    first_prices = market.prices_at(0)
    nav_now = ledger.cash + sum(ledger.get(t) * first_prices.get(t, 0.0) for t in universe)
    # The peak is inherited, never reset to the current NAV: setting `peak = nav` would
    # hand the run a fresh zero drawdown and hide any drawdown it started inside.
    peak = float(initial_peak if initial_peak is not None else nav_now)
    peak = max(peak, nav_now)

    rows: list[dict] = []
    reservoir: list[dict] = []
    n_lock_violations = 0
    n_feasibility_violations = 0

    for i in range(T - 1):
        session = market.sessions[i]
        session_date = session.date()
        closes = market.prices_at(i, "close_raw")

        nav_now = ledger.cash + sum(ledger.get(t) * closes.get(t, 0.0) for t in universe)
        if nav_now <= 0:
            raise ValueError(f"NAV went non-positive at {session_date}")
        peak = max(peak, nav_now)
        drawdown = 1.0 - nav_now / peak

        available = market.available[i].copy()
        share_vec = ledger.share_vector(universe)
        cur_w = np.zeros(K + 1)
        for j, t in enumerate(universe):
            cur_w[j + 1] = share_vec[j] * closes.get(t, 0.0) / nav_now
        cur_w[CASH] = ledger.cash / nav_now

        if cfg.reservoir_every and i % cfg.reservoir_every == 0 and i > 0:
            reservoir.append({
                "session": str(session.date()),
                "ledger": ledger.to_dict(),
                "lock_manager": lm.to_dict(),
                "peak": float(peak), "nav": float(nav_now), "drawdown": float(drawdown),
                "n_param": cfg.hold_days, "dmax_param": cfg.max_drawdown,
            })

        # -- the decision, from t-close information only ------------------------
        ctx = StepContext(
            session=session, universe=universe, ledger=ledger, lock_manager=lm,
            nav=nav_now, peak=peak, drawdown=drawdown,
            available=available, current_weights=cur_w,
            lock_remaining_days=lm.remaining_days(session_date),
            hold_days=cfg.hold_days, max_drawdown=cfg.max_drawdown, step=i,
        )
        a_raw = np.asarray(weight_fn(session, ctx), dtype=float)
        if a_raw.shape != (K + 1,):
            raise ValueError(f"weight_fn returned shape {a_raw.shape}, expected {(K + 1,)}")

        # -- constraints --------------------------------------------------------
        lock_floor_shares = lm.lower_bounds(share_vec, session_date)
        lower_w = np.zeros(K + 1)
        for j, t in enumerate(universe):
            if lock_floor_shares[j] > 0:
                lower_w[j + 1] = lock_floor_shares[j] * closes.get(t, 0.0) / nav_now
        avail_full = np.concatenate([[True], available])

        # Layer one: if the drawdown has already happened, the agent cannot undo it. The
        # action set shrinks; the episode does not end.
        capital_preservation = drawdown > cfg.max_drawdown

        risk_model = None
        if cfg.risk_enabled and envelope is not None:
            lo = max(0, i - cfg.risk_lookback)
            envelope.prepare(
                market.returns[lo:i + 1], market.sessions[lo:i + 1].to_numpy(),
                as_of=session_date, seed=cfg.seed + i, n_assets=K + 1,
            )
            envelope.set_budget(nav_now, peak, cfg.max_drawdown)
            risk_model = envelope

        projected = projector.project(
            a_raw, lower_bounds=lower_w, available=avail_full, risk=risk_model,
            capital_preservation=capital_preservation, current_weights=cur_w,
        )
        diag: ProjectionDiagnostics = projected.diagnostics
        target = {t: float(projected.weights[j + 1]) for j, t in enumerate(universe)}

        # -- execution at the NEXT open -----------------------------------------
        nxt = i + 1
        opens = market.prices_at(nxt, "open_raw")
        exec_session = market.sessions[nxt].date()
        avail_next = {t: bool(market.available[nxt, j]) for j, t in enumerate(universe)}

        # In capital preservation the cap is on SHARE COUNTS, not weights. A weight cap
        # would read as "restore yesterday's weight", which in a falling market means
        # buying the dip every session -- the exact opposite of what it is for.
        max_shares = None
        if capital_preservation:
            max_shares = {t: ledger.get(t) for t in universe}

        try:
            result = execute(
                ledger, target, opens, exec_session,
                available=avail_next, lock_manager=lm, bars=market.bars_at(nxt),
                cost_bps=cfg.cost_bps, max_shares=max_shares,
            )
        except Exception:
            n_feasibility_violations += 1
            raise

        # A locked position that shrank is a lock violation, checked against what
        # actually executed rather than against what was intended.
        for j, t in enumerate(universe):
            if lm.is_locked(t, exec_session) and result.ledger.get(t) < share_vec[j] - 1e-9:
                n_lock_violations += 1

        apply_and_lock(result, lm, exec_session, cfg.hold_days)
        ledger = result.ledger

        # -- value at the next close --------------------------------------------
        next_closes = market.prices_at(nxt, "close_raw")
        divs = {t: float(market.div_per_share[nxt, j]) for j, t in enumerate(universe)}
        val = value(ledger, next_closes, peak=peak, div_per_share=divs)
        peak = val.peak
        reward = float(np.log(val.nav / nav_now))

        row = {
            "session": market.sessions[nxt],
            "nav": val.nav, "peak_nav": val.peak, "drawdown": val.drawdown,
            "cash": ledger.cash,
            "reward": reward,
            "turnover": result.turnover,
            "cost_paid": result.cost_paid,
            "proj_distance": diag.l1_distance,
            "safety_intervened": bool(diag.risk_binding),
            "capital_preservation": bool(diag.capital_preservation),
            "infeasible_fallback": bool(diag.infeasible_fallback),
            "budget_degenerate": bool(diag.budget_degenerate),
            "de_risk_alpha": diag.de_risk_alpha,
            "availability_clipped": diag.availability_clipped,
            "lock_bound_active": diag.lock_bound_active,
            "share_floor_binding": result.share_floor_binding,
            "preservation_cap_binding": result.preservation_cap_binding,
            "n_param": cfg.hold_days, "dmax_param": cfg.max_drawdown,
            "drawdown_budget": headroom(val.nav, val.peak, cfg.max_drawdown),
            "risk_budget": risk_budget(nav_now, peak, cfg.max_drawdown),
        }
        for j, t in enumerate(universe):
            shares = ledger.get(t)
            row[f"w_{t}"] = shares * next_closes.get(t, 0.0) / val.nav
            row[f"shares_{t}"] = shares
            row[f"locked_{t}"] = lm.is_locked(t, exec_session)
            unlock = lm.unlock_dates.get(t)
            row[f"unlock_date_{t}"] = pd.Timestamp(unlock) if unlock else pd.NaT
        rows.append(row)

    trajectory = pd.DataFrame(rows).set_index("session")
    return SimulationResult(
        trajectory=trajectory, final_ledger=ledger, final_lock_manager=lm,
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
