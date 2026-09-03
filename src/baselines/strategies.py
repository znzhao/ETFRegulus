"""The six baselines, as weight sources for the shared simulator.

**The non-negotiable rule:** every baseline runs through the *identical* simulator, lock
manager, projection and risk envelope as the agent. A baseline is a weight sequence fed to
Stage 4; nothing else about it is special. Two reasons that matters:

1. The comparison is apples to apples -- the agent is not credited for constraints the
   baselines dodge.
2. A baseline that violates the lock or leverage is a **bug in the constraint layer**, and
   finding it here, against a strategy whose correct behaviour is obvious, is far cheaper
   than finding it during training.

Several of these double as simulator correctness tests with obvious expected answers
(reference/baselines.md section 7), which is why they come before any RL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.sim.simulator import MarketData, StepContext

CASH = 0


@dataclass
class BaselineSpec:
    name: str
    build: Callable[[MarketData, dict], Callable]
    description: str
    params: dict = field(default_factory=dict)


def _empty(n_assets: int) -> np.ndarray:
    w = np.zeros(n_assets + 1)
    w[CASH] = 1.0
    return w


def _month_ends(sessions: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """Last session of each calendar month."""
    s = pd.Series(sessions, index=sessions)
    return set(s.groupby([sessions.year, sessions.month]).max().to_numpy())


def _hold(ctx: StepContext) -> np.ndarray:
    """Ask for exactly what is already held, so the step is a genuine no-op.

    `rebalance: monthly` means **trade monthly and hold in between**. Re-asserting a fixed
    target weight vector on non-rebalance sessions does not mean that: prices drift, so
    the fixed vector becomes a daily instruction to trade back to it. That turns every
    monthly strategy into a daily one and -- far worse here -- relocks every position it
    touches, every session, making the lock bind at values of `N` where it should not.
    Returning the current weights instead produces no trade legs and therefore no relock.
    """
    return ctx.current_weights.copy()


# ------------------------------------------------------------------ B4: cash


def build_cash(market: MarketData, params: dict):
    """100% cash, always. The floor, and a constraint-layer sanity check.

    Two exact expectations: **zero drawdown and zero turnover**, at every `N` and `D_max`.
    Any deviation is a ledger bug. It also proves the lock never *forces* risk-taking.
    """
    n = market.n_assets

    def weights(session, ctx: StepContext) -> np.ndarray:
        return _empty(n)

    return weights


# ------------------------------------------------------- B1: spy buy and hold


def build_spy_buy_hold(market: MarketData, params: dict):
    """Buy SPY with 100% of NAV on the first session. Never trade again.

    **A control.** One buy, no sells ever, so it is legal at every `N` and unaffected by
    `D_max` (the envelope only constrains increases). Its results must therefore be
    *identical* across every value of `N`. If they are not, the lock manager is corrupting
    state it should not touch.
    """
    ticker = params.get("ticker", "SPY")
    n = market.n_assets
    idx = market.universe.index(ticker)

    def weights(session, ctx: StepContext) -> np.ndarray:
        w = np.zeros(n + 1)
        if ctx.available[idx]:
            w[idx + 1] = 1.0
        else:
            w[CASH] = 1.0
        return w

    return weights


# ------------------------------------------------------------ B5: equal weight


def build_equal_weight(market: MarketData, params: dict):
    """Equal weight across all available tradables, rebalanced monthly with a drift band.

    Moves with the expanding universe, so it exercises the availability mask continuously.
    The band matters for the same reason it does in B3: every rebalancing buy relocks.
    """
    n = market.n_assets
    band = float(params.get("drift_band", 0.05))
    month_ends = _month_ends(market.sessions)
    state = {"target": None, "n_available": 0}

    def weights(session, ctx: StepContext) -> np.ndarray:
        avail = ctx.available
        target = np.zeros(n + 1)
        if avail.any():
            target[1:][avail] = 1.0 / avail.sum()
        else:
            target[CASH] = 1.0

        if state["target"] is None:
            state["target"] = target
            state["n_available"] = int(avail.sum())
            return target

        if session in month_ends:
            # A CHANGE IN THE AVAILABLE SET is a structural change, not drift, and it
            # rebalances regardless of the band. Without this the band silently swallows
            # every new listing: with 24 names at ~4.2% each, an ETF entering at weight 0
            # is only a 4.2pp deviation and never trips a 5pp band -- so B5 would quietly
            # stop tracking the expanding universe, which is the one thing it is here to
            # exercise (reference/baselines.md section 5).
            universe_changed = int(avail.sum()) != state["n_available"]
            drifted = np.abs(ctx.current_weights - target).max() > band
            if universe_changed or drifted:
                state["n_available"] = int(avail.sum())
                return target
        return _hold(ctx)

    return weights


# ----------------------------------------------------------- B3: 60/40 SPY/TLT


def build_spy_tlt_60_40(market: MarketData, params: dict):
    """60% SPY / 40% TLT, monthly, with a drift band.

    The band matters more here than in an unconstrained backtest: **every rebalancing buy
    relocks the bought asset for `N` days**, so an unbanded monthly rebalance would keep
    the whole book permanently locked at `N >= 30`.

    The sharpest test in the set. 60/40 relies on the negative stock/bond correlation that
    held for most of the sample and **broke in 2022**, when SPY and TLT fell together and
    TLT alone drew down 31%.
    """
    n = market.n_assets
    band = float(params.get("drift_band", 0.05))
    legs = params.get("weights", {"SPY": 0.60, "TLT": 0.40})
    month_ends = _month_ends(market.sessions)

    target = np.zeros(n + 1)
    for ticker, weight in legs.items():
        target[market.universe.index(ticker) + 1] = float(weight)

    state = {"held": False}

    def weights(session, ctx: StepContext) -> np.ndarray:
        avail_full = np.concatenate([[True], ctx.available])
        want = np.where(avail_full, target, 0.0)
        if want[1:].sum() <= 0:
            return _empty(n)
        want = want / want.sum()

        if not state["held"]:
            state["held"] = True
            return want
        if session in month_ends and np.abs(ctx.current_weights - want).max() > band:
            return want
        return _hold(ctx)

    return weights


# --------------------------------------------------------------- B2: momentum


def build_momentum(market: MarketData, params: dict):
    """12-1 cross-sectional momentum, top-5 equal weight, monthly, absolute filter.

    The standard construction: a 21-session skip omits the most recent month to avoid the
    well-documented short-term reversal effect that would otherwise contaminate the signal.

    **The most informative baseline for `N` sensitivity.** Monthly rebalancing means
    selling last month's losers, but every buy relocks that ETF for `N` calendar days. At
    the D16 primary of `N = 30` (~21 sessions) the unlock lands almost exactly on the next
    rebalance, so B2 sits right on the boundary where the lock starts to bind. Its
    performance should degrade visibly and monotonically as `N` grows; if it does not, the
    lock is not actually binding.
    """
    n = market.n_assets
    lookback = int(params.get("lookback_days", 252))
    skip = int(params.get("skip_days", 21))
    top_n = int(params.get("top_n", 5))
    absolute_filter = bool(params.get("absolute_filter", True))
    month_ends = _month_ends(market.sessions)

    # Signals are precomputed from close_adj, using only data at or before each session.
    adj = np.cumprod(1.0 + market.returns[:, 1:], axis=0)
    adj[~market.available] = np.nan
    state = {"target": _empty(n), "held": False}

    def weights(session, ctx: StepContext) -> np.ndarray:
        i = ctx.step
        if session not in month_ends:
            return _hold(ctx) if state["held"] else state["target"]
        if i < lookback + skip:
            return _hold(ctx) if state["held"] else state["target"]

        end, start = i - skip, i - skip - lookback
        signal = adj[end] / adj[start] - 1.0
        signal[~ctx.available] = np.nan
        if absolute_filter:
            # In a broad bear market the strategy moves to cash rather than holding the
            # least-bad loser. That is what makes B2 a different RISK profile from B1/B3,
            # not merely a different return stream.
            signal[signal <= 0] = np.nan

        valid = np.flatnonzero(np.isfinite(signal))
        if valid.size == 0:
            state["target"] = _empty(n)
            state["held"] = True
            return state["target"]

        chosen = valid[np.argsort(signal[valid])[::-1][:top_n]]
        w = np.zeros(n + 1)
        w[chosen + 1] = 1.0 / top_n
        w[CASH] = 1.0 - w[1:].sum()     # unfilled slots go to cash
        state["target"] = w
        state["held"] = True
        return w

    return weights


# --------------------------------------------------- B6: classical optimizer


def build_classical_optimizer(market: MarketData, params: dict):
    """Rolling risk parity on trailing **visible** data only, re-solved at each rebalance.

    The honest representation of what a competent quant would do without RL. Inverse-
    volatility weighting is the risk-parity form that needs no matrix inversion, so it
    stays well-conditioned on a universe whose size changes over the sample.
    """
    n = market.n_assets
    window = int(params.get("window", 252))
    month_ends = _month_ends(market.sessions)
    max_weight = float(params.get("max_weight", 0.25))
    state = {"target": _empty(n), "held": False}

    def weights(session, ctx: StepContext) -> np.ndarray:
        i = ctx.step
        if session not in month_ends:
            return _hold(ctx) if state["held"] else state["target"]
        lo = max(0, i - window)
        if i - lo < 60:
            return _hold(ctx) if state["held"] else state["target"]

        hist = market.returns[lo:i + 1, 1:]      # visible history only
        vol = np.nanstd(hist, axis=0)
        usable = ctx.available & np.isfinite(vol) & (vol > 1e-8)
        if not usable.any():
            state["target"] = _empty(n)
            state["held"] = True
            return state["target"]

        inv = np.zeros(n)
        inv[usable] = 1.0 / vol[usable]
        raw = inv / inv.sum()
        # A concentration cap, so a single low-vol asset does not become the whole book.
        raw = np.minimum(raw, max_weight)
        w = np.zeros(n + 1)
        w[1:] = raw
        w[CASH] = max(0.0, 1.0 - raw.sum())
        state["target"] = w
        state["held"] = True
        return w

    return weights


BASELINES: dict[str, BaselineSpec] = {
    "spy_buy_hold": BaselineSpec(
        "spy_buy_hold", build_spy_buy_hold,
        "Buy SPY on session 1, never trade again. PRIMARY, and the lock-isolation control."),
    "momentum": BaselineSpec(
        "momentum", build_momentum,
        "12-1 cross-sectional momentum, top-5 equal weight, monthly, absolute filter. PRIMARY."),
    "spy_tlt_60_40": BaselineSpec(
        "spy_tlt_60_40", build_spy_tlt_60_40,
        "60/40 SPY/TLT, monthly with a 5pp drift band. PRIMARY, and the 2022 test."),
    "cash": BaselineSpec(
        "cash", build_cash,
        "100% cash. The floor, and a constraint-layer sanity check."),
    "equal_weight": BaselineSpec(
        "equal_weight", build_equal_weight,
        "Equal weight across available tradables, monthly with a drift band."),
    "classical_optimizer": BaselineSpec(
        "classical_optimizer", build_classical_optimizer,
        "Rolling risk parity on trailing visible data. The classical constrained baseline."),
}

PRIMARY = ("spy_buy_hold", "momentum", "spy_tlt_60_40")
