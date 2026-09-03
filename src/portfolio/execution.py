"""Next-open execution: sells settle before buys are funded.

The ordering is what makes the no-margin constraint *structural* rather than a check
performed after the fact. Buys draw strictly from cash on hand once sells have settled,
so there is no intermediate state in which the portfolio is levered.

Two authoritative checks live here rather than upstream:

* **The share floor is the law.** The projection reasons in weight space, but the lock
  bound is a bound on *shares*, and the weight-to-share mapping depends on the NAV at
  execution, which is not known at decision time. So execution re-derives the floors and
  enforces them; the weight-space bound was only the optimizer's guide
  (reference/feasibility-projection.md section 3.4).
* **T15: the fill price must lie inside the day's range.** A fill outside `[low, high]`
  is a data bug and raises rather than being silently trusted.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from src.portfolio.ledger import EPS, Ledger, LedgerError
from src.portfolio.lock_manager import LockManager, TradeLeg


class ExecutionError(AssertionError):
    """The execution layer was asked to do something illegal, or a fill was impossible."""


@dataclass
class ExecutionResult:
    ledger: Ledger
    legs: list[TradeLeg]
    nav_at_open: float
    cost_paid: float = 0.0
    turnover: float = 0.0
    #: Weight actually achieved minus weight intended. Recorded, never hidden: an
    #: overnight gap can make a weight target infeasible in share terms, and the residual
    #: goes to cash.
    weight_residual: dict[str, float] = field(default_factory=dict)
    share_floor_binding: int = 0
    preservation_cap_binding: int = 0


def _check_price_in_range(
    ticker: str, price: float, bar: Mapping[str, float] | None, session: dt.date
) -> None:
    """T15. Always on -- it costs a comparison and catches a whole class of data bug."""
    if bar is None:
        return
    lo, hi = bar.get("low_raw"), bar.get("high_raw")
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return
    if price < lo - 1e-6 or price > hi + 1e-6:
        raise ExecutionError(
            f"{ticker} on {session}: fill at {price} is outside the day's range "
            f"[{lo}, {hi}]. This is a data bug, not a fill."
        )


def execute(
    ledger: Ledger,
    target_weights: Mapping[str, float],
    open_prices: Mapping[str, float],
    session: dt.date,
    *,
    available: Mapping[str, bool] | None = None,
    lock_manager: LockManager | None = None,
    bars: Mapping[str, Mapping[str, float]] | None = None,
    cost_bps: float = 0.0,
    max_shares: Mapping[str, float] | None = None,
) -> ExecutionResult:
    """Trade toward `target_weights` at `session`'s open.

    `target_weights` covers tradable tickers only; whatever is not allocated stays in cash.

    `max_shares` is the capital-preservation cap: no ETF may end above that share count.
    Like the lock floor, it is a bound on **shares** and is therefore enforced here rather
    than in weight space. Capping weights instead would mean "restore yesterday\'s weight",
    which in a falling market is an instruction to buy the dip every session -- precisely
    the increase in risky exposure that capital preservation exists to forbid.
    """
    ledger = ledger.copy()
    available = available or {}
    tradables = set(open_prices)

    # 1. NAV at the open. Every target is a fraction of this, not of last night's close.
    nav_at_open = ledger.cash
    for ticker, n in ledger.shares.items():
        price = float(open_prices[ticker])
        if not np.isfinite(price) or price <= 0.0:
            raise ExecutionError(f"{ticker}: unusable open price {price!r} on {session}")
        nav_at_open += n * price
    if nav_at_open <= 0.0:
        raise ExecutionError(f"NAV at open must be positive, got {nav_at_open!r}")

    # 2-3. Target share counts. An unavailable ticker gets zero: it cannot be bought, and
    # by construction is never held, so it never appears in a lock floor either.
    target_shares: dict[str, float] = {}
    for ticker in tradables:
        if available.get(ticker, True) is False:
            target_shares[ticker] = 0.0
            continue
        w = float(target_weights.get(ticker, 0.0) or 0.0)
        price = float(open_prices[ticker])
        target_shares[ticker] = max(0.0, nav_at_open * w / price)

    # The share floor, re-derived here and enforced as the authoritative bound.
    floor_binding = 0
    if lock_manager is not None:
        for ticker in tradables:
            held = ledger.get(ticker)
            if held > EPS and lock_manager.is_locked(ticker, session):
                if target_shares[ticker] < held - EPS:
                    target_shares[ticker] = held
                    floor_binding += 1

    # The capital-preservation cap, in share space and authoritative. Applied AFTER the
    # lock floor so that a locked position is never cut below its floor to satisfy it --
    # the lock outranks capital preservation.
    cap_binding = 0
    if max_shares is not None:
        for ticker in tradables:
            cap = float(max_shares.get(ticker, np.inf))
            if np.isfinite(cap) and target_shares[ticker] > cap + EPS:
                floor = ledger.get(ticker) if (
                    lock_manager is not None and lock_manager.is_locked(ticker, session)
                ) else 0.0
                target_shares[ticker] = max(cap, floor)
                cap_binding += 1

    # A sale of a locked position must never reach this point.
    if lock_manager is not None:
        for ticker in tradables:
            if target_shares[ticker] < ledger.get(ticker) - EPS:
                lock_manager.assert_sale_legal(ticker, session)

    rate = float(cost_bps) / 10_000.0
    legs: list[TradeLeg] = []
    cost_paid = 0.0
    turnover = 0.0

    # 4a. SELLS FIRST. Proceeds land in cash before any buy is funded.
    for ticker in sorted(tradables):
        delta = target_shares[ticker] - ledger.get(ticker)
        if delta >= -EPS:
            continue
        price = float(open_prices[ticker])
        _check_price_in_range(ticker, price, (bars or {}).get(ticker), session)
        proceeds = -delta * price
        fee = proceeds * rate
        ledger.add_shares(ticker, delta)
        ledger.cash += proceeds - fee
        cost_paid += fee
        turnover += proceeds
        legs.append(TradeLeg(ticker, delta, price))

    # 4b. BUYS, funded strictly from cash on hand. If the requested buys exceed available
    # cash -- which an overnight gap against a binding share floor can cause -- they are
    # scaled down proportionally rather than borrowed against.
    wanted: dict[str, float] = {}
    for ticker in sorted(tradables):
        delta = target_shares[ticker] - ledger.get(ticker)
        if delta > EPS:
            wanted[ticker] = delta

    gross = sum(d * float(open_prices[t]) for t, d in wanted.items())
    budget = ledger.cash / (1.0 + rate) if rate else ledger.cash
    scale = 1.0
    if gross > budget + EPS:
        scale = max(0.0, budget / gross) if gross > 0 else 0.0

    for ticker, delta in wanted.items():
        delta *= scale
        if delta <= EPS:
            continue
        price = float(open_prices[ticker])
        _check_price_in_range(ticker, price, (bars or {}).get(ticker), session)
        notional = delta * price
        fee = notional * rate
        if notional + fee > ledger.cash + EPS:      # last-resort guard against float drift
            notional = max(0.0, ledger.cash - fee)
            delta = notional / price
            if delta <= EPS:
                continue
            fee = notional * rate
        ledger.add_shares(ticker, delta)
        ledger.cash -= notional + fee
        cost_paid += fee
        turnover += notional
        legs.append(TradeLeg(ticker, delta, price))

    # 6. Post-conditions.
    if ledger.cash < -EPS:
        raise ExecutionError(f"execution left negative cash: {ledger.cash!r}")
    ledger.check()
    ledger.as_of = session

    nav_after = ledger.cash + sum(n * float(open_prices[t]) for t, n in ledger.shares.items())
    residual = {}
    for ticker in tradables:
        intended = float(target_weights.get(ticker, 0.0) or 0.0)
        achieved = ledger.get(ticker) * float(open_prices[ticker]) / nav_after
        if abs(achieved - intended) > 1e-6:
            residual[ticker] = achieved - intended

    return ExecutionResult(
        ledger=ledger, legs=legs, nav_at_open=nav_at_open, cost_paid=cost_paid,
        turnover=turnover / nav_at_open if nav_at_open else 0.0,
        weight_residual=residual, share_floor_binding=floor_binding,
        preservation_cap_binding=cap_binding,
    )


def apply_and_lock(
    result: ExecutionResult, lock_manager: LockManager, session: dt.date, hold_days: int
) -> None:
    """Hand the executed legs to the lock manager. Locks are set AFTER the buys."""
    lock_manager.apply_execution(result.legs, session, hold_days,
                                 shares_after=result.ledger.shares)
