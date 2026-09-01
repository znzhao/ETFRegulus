"""NAV, dividend reinvestment, running peak, drawdown, weights.

Section 3 of reference/portfolio-ledger.md is the load-bearing part: the ledger stays
entirely in raw quoted prices, and total return arrives through **share accretion** when
distributions are reinvested. Both obvious alternatives are wrong -- valuing at
`close_raw` alone systematically understates every income-paying ETF, and `close_adj` is
back-adjusted, so a share count times an adjusted price is not a dollar amount and
changes retroactively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.portfolio.ledger import EPS, Ledger, LedgerError

#: Below this the implied distribution is float noise in the provider's rounding, not a
#: payment. Clipped rather than reinvested.
DIVIDEND_EPS = 1e-10


@dataclass
class Valuation:
    nav: float
    cash_weight: float
    weights: dict[str, float]
    peak: float
    drawdown: float


def nav(ledger: Ledger, prices: Mapping[str, float]) -> float:
    """`cash + sum(shares * close_raw)`. Raw prices, always."""
    total = ledger.cash
    for ticker, n in ledger.shares.items():
        price = prices.get(ticker)
        if price is None or not np.isfinite(price):
            raise LedgerError(f"no usable price for held ticker {ticker}: {price!r}")
        total += n * float(price)
    return total


def reinvest_dividends(
    ledger: Ledger, prices: Mapping[str, float], div_per_share: Mapping[str, float]
) -> float:
    """Credit each holding's distribution and immediately buy more of the same ticker.

    Returns the total cash value distributed, for diagnostics.

    Routed through `Ledger.accrue_shares`, which the lock manager does not observe: a
    reinvestment increases a share count but is **not** a discretionary buy and must not
    reset any unlock date (reference/lock-state-machine.md section 3.5).
    """
    distributed = 0.0
    for ticker, n in list(ledger.shares.items()):
        if n <= EPS:
            continue
        dps = float(div_per_share.get(ticker, 0.0) or 0.0)
        if not np.isfinite(dps) or dps <= DIVIDEND_EPS:
            continue
        price = float(prices[ticker])
        if not np.isfinite(price) or price <= 0.0:
            continue
        gross = n * dps
        ledger.accrue_shares(ticker, gross / price)
        distributed += gross
    return distributed


def weights(ledger: Ledger, prices: Mapping[str, float], nav_value: float | None = None
            ) -> tuple[dict[str, float], float]:
    """Market-value weights plus the cash weight. `sum(w) + w_cash == 1` by construction."""
    total = nav(ledger, prices) if nav_value is None else nav_value
    if total <= 0.0:
        raise LedgerError(f"NAV must be positive to form weights, got {total!r}")
    w = {t: n * float(prices[t]) / total for t, n in ledger.shares.items() if n > EPS}
    return w, ledger.cash / total


def value(
    ledger: Ledger, prices: Mapping[str, float], peak: float,
    div_per_share: Mapping[str, float] | None = None,
) -> Valuation:
    """Mark the book at the close, after reinvesting any distribution paid that session."""
    if div_per_share:
        reinvest_dividends(ledger, prices, div_per_share)

    total = nav(ledger, prices)
    if total <= 0.0:
        raise LedgerError(f"NAV must stay positive, got {total!r}")

    # The peak is a running maximum and is monotone non-decreasing. It is carried across
    # resets rather than set to the current NAV -- setting `peak = nav` at reset would
    # hand every episode a fresh zero drawdown and teach the agent that drawdown resets
    # for free (reference/env-mdp.md section 6).
    new_peak = max(float(peak), total)
    w, w_cash = weights(ledger, prices, total)
    return Valuation(
        nav=total, cash_weight=w_cash, weights=w,
        peak=new_peak, drawdown=1.0 - total / new_peak,
    )
