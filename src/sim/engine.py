"""The single step of the simulation, factored out so nothing can run a different one.

`simulate()` is a `for` loop; a Gymnasium environment is the same loop turned inside out,
driven one call at a time by an agent. If each owned its own copy of the step body they
would drift, and the moment they drift the Stage 5 baselines stop being a valid reference
point for the Stage 7 policy -- which is the entire reason the baselines were run through
the simulator in the first place.

So the body lives here, exactly once:

    observe(...)  -> everything decidable from session t's close, including the risk model
                     prepared on visible history only
    advance(...)  -> project, execute at the t+1 open, value at the t+1 close

`src/sim/simulator.py` calls these in a loop. `src/env/etf_env.py` calls them from `step`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.constraints.projector import CASH, FeasibilityProjector, ProjectionDiagnostics
from src.constraints.risk_envelope import RiskEnvelope, headroom, risk_budget
from src.portfolio.execution import apply_and_lock, execute
from src.portfolio.ledger import Ledger
from src.portfolio.lock_manager import LockManager
from src.portfolio.valuation import value
from src.sim.simulator import MarketData, SimulationConfig, StepContext


@dataclass
class EngineState:
    """The whole mutable portfolio state between two decisions.

    The peak is part of the state and is *inherited*, never recomputed from the current
    NAV -- see reference/env-mdp.md section 6, step 5.
    """

    ledger: Ledger
    lock_manager: LockManager
    peak: float

    def copy(self) -> "EngineState":
        return EngineState(self.ledger.copy(), self.lock_manager.copy(), float(self.peak))


@dataclass
class Decision:
    """Everything session `i`'s close determines, before the agent has acted."""

    ctx: StepContext
    nav: float
    drawdown: float
    share_vec: np.ndarray
    closes: dict
    lower_w: np.ndarray
    avail_full: np.ndarray
    capital_preservation: bool
    risk_model: RiskEnvelope | None


@dataclass
class Outcome:
    """What one executed step produced."""

    row: dict
    reward: float
    nav: float
    lock_violations: int = 0
    diagnostics: ProjectionDiagnostics | None = None
    projected: np.ndarray = field(default_factory=lambda: np.zeros(0))


def observe(
    market: MarketData, st: EngineState, cfg: SimulationConfig, i: int,
    *, envelope: RiskEnvelope | None = None,
) -> Decision:
    """Build the decision state at session `i`'s close. t-close information only."""
    universe = market.universe
    K = len(universe)
    session = market.sessions[i]
    session_date = session.date()
    closes = market.prices_at(i, "close_raw")

    nav_now = st.ledger.cash + sum(st.ledger.get(t) * closes.get(t, 0.0) for t in universe)
    if nav_now <= 0:
        raise ValueError(f"NAV went non-positive at {session_date}")
    st.peak = max(st.peak, nav_now)
    drawdown = 1.0 - nav_now / st.peak

    available = market.available[i].copy()
    share_vec = st.ledger.share_vector(universe)
    cur_w = np.zeros(K + 1)
    for j, t in enumerate(universe):
        cur_w[j + 1] = share_vec[j] * closes.get(t, 0.0) / nav_now
    cur_w[CASH] = st.ledger.cash / nav_now

    ctx = StepContext(
        session=session, universe=universe, ledger=st.ledger,
        lock_manager=st.lock_manager, nav=nav_now, peak=st.peak, drawdown=drawdown,
        available=available, current_weights=cur_w,
        lock_remaining_days=st.lock_manager.remaining_days(session_date),
        hold_days=cfg.hold_days, max_drawdown=cfg.max_drawdown, step=i,
    )

    lock_floor_shares = st.lock_manager.lower_bounds(share_vec, session_date)
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
        envelope.set_budget(nav_now, st.peak, cfg.max_drawdown)
        risk_model = envelope

    return Decision(
        ctx=ctx, nav=nav_now, drawdown=drawdown, share_vec=share_vec, closes=closes,
        lower_w=lower_w, avail_full=avail_full,
        capital_preservation=capital_preservation, risk_model=risk_model,
    )


def advance(
    market: MarketData, st: EngineState, cfg: SimulationConfig, i: int,
    dec: Decision, a_raw: np.ndarray, projector: FeasibilityProjector,
) -> Outcome:
    """Project `a_raw`, execute at the `i+1` open, and value at the `i+1` close.

    Mutates `st` in place: this is the transition.
    """
    universe = market.universe
    K = len(universe)
    a_raw = np.asarray(a_raw, dtype=float)
    if a_raw.shape != (K + 1,):
        raise ValueError(f"action has shape {a_raw.shape}, expected {(K + 1,)}")

    projected = projector.project(
        a_raw, lower_bounds=dec.lower_w, available=dec.avail_full, risk=dec.risk_model,
        capital_preservation=dec.capital_preservation,
        current_weights=dec.ctx.current_weights,
    )
    diag: ProjectionDiagnostics = projected.diagnostics
    target = {t: float(projected.weights[j + 1]) for j, t in enumerate(universe)}

    nxt = i + 1
    opens = market.prices_at(nxt, "open_raw")
    exec_session = market.sessions[nxt].date()
    avail_next = {t: bool(market.available[nxt, j]) for j, t in enumerate(universe)}

    # In capital preservation the cap is on SHARE COUNTS, not weights. A weight cap would
    # read as "restore yesterday's weight", which in a falling market means buying the dip
    # every session -- the exact opposite of what it is for.
    max_shares = None
    if dec.capital_preservation:
        max_shares = {t: st.ledger.get(t) for t in universe}

    result = execute(
        st.ledger, target, opens, exec_session,
        available=avail_next, lock_manager=st.lock_manager, bars=market.bars_at(nxt),
        cost_bps=cfg.cost_bps, max_shares=max_shares,
    )

    # A locked position that shrank is a lock violation, checked against what actually
    # executed rather than against what was intended.
    n_lock_violations = 0
    for j, t in enumerate(universe):
        if (st.lock_manager.is_locked(t, exec_session)
                and result.ledger.get(t) < dec.share_vec[j] - 1e-9):
            n_lock_violations += 1

    apply_and_lock(result, st.lock_manager, exec_session, cfg.hold_days)
    st.ledger = result.ledger

    next_closes = market.prices_at(nxt, "close_raw")
    divs = {t: float(market.div_per_share[nxt, j]) for j, t in enumerate(universe)}
    val = value(st.ledger, next_closes, peak=st.peak, div_per_share=divs)
    st.peak = val.peak
    reward = float(np.log(val.nav / dec.nav))

    row = {
        "session": market.sessions[nxt],
        "nav": val.nav, "peak_nav": val.peak, "drawdown": val.drawdown,
        "cash": st.ledger.cash,
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
        "risk_budget": risk_budget(dec.nav, dec.ctx.peak, cfg.max_drawdown),
        # The action that was actually PROJECTED, over [CASH, *universe], recorded so the
        # preventable-violation detector can replay the decision rather than infer it
        # from the position that resulted. Without this, "the executed action was in the
        # feasible set" is an assertion the trajectory cannot support.
        "proj_weights": projected.weights.astype(float).copy(),
        "raw_weights": a_raw.astype(float).copy(),
        # The decision-time state the replay needs, before the trade moved anything.
        "decision_nav": float(dec.nav),
        "decision_peak": float(dec.ctx.peak),
        "decision_drawdown": float(dec.drawdown),
    }
    for j, t in enumerate(universe):
        shares = st.ledger.get(t)
        row[f"w_{t}"] = shares * next_closes.get(t, 0.0) / val.nav
        row[f"shares_{t}"] = shares
        row[f"locked_{t}"] = st.lock_manager.is_locked(t, exec_session)
        unlock = st.lock_manager.unlock_dates.get(t)
        row[f"unlock_date_{t}"] = pd.Timestamp(unlock) if unlock else pd.NaT

    return Outcome(row=row, reward=reward, nav=val.nav,
                   lock_violations=n_lock_violations, diagnostics=diag,
                   projected=projected.weights)


def reservoir_entry(market: MarketData, st: EngineState, cfg: SimulationConfig,
                    i: int, dec: Decision) -> dict:
    """One reachable-by-construction state, in the reservoir's on-disk shape."""
    return {
        "session": str(market.sessions[i].date()),
        "ledger": st.ledger.to_dict(),
        "lock_manager": st.lock_manager.to_dict(),
        "peak": float(st.peak), "nav": float(dec.nav), "drawdown": float(dec.drawdown),
        "n_param": cfg.hold_days, "dmax_param": cfg.max_drawdown,
    }


def initial_state(market: MarketData, cfg: SimulationConfig, *,
                  ledger: Ledger | None = None,
                  lock_manager: LockManager | None = None,
                  peak: float | None = None,
                  row: int = 0) -> EngineState:
    """Seed an `EngineState`, inheriting the peak rather than resetting it to the NAV.

    `row` is where the run begins. It is not always 0: an evaluation window that needs
    history behind it -- a walk-forward test year, or one year of a comparison report --
    builds `market` over the full span so lookbacks work, and starts the portfolio at the
    first session of the window.
    """
    led = (ledger or Ledger(cash=cfg.initial_cash)).copy()
    lm = (lock_manager or LockManager(universe=list(market.universe),
                                      scope=cfg.lock_scope)).copy()
    first = market.prices_at(row)
    nav0 = led.cash + sum(led.get(t) * first.get(t, 0.0) for t in market.universe)
    return EngineState(ledger=led, lock_manager=lm,
                       peak=max(float(peak if peak is not None else nav0), nav0))
