"""The share/cash ledger.

Fractional shares, long-only, no leverage, and a lossless round-trip to a plain dict
(D4 -- the live job needs durable state, and retrofitting serialization later is
expensive). Invariants are asserted at runtime, not merely tested, so a violation
surfaces at the step that caused it rather than in an aggregate at the end.

The ledger holds **raw quoted prices and real share counts**. That is what makes it
correspond to something you could actually execute. Total return arrives through share
accretion when distributions are reinvested (see `valuation.py`), not by valuing the
book at a back-adjusted price.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

#: Slack for float comparisons on money and share counts. Tighter than any real
#: rounding, loose enough that accumulated float error is not mistaken for a bug.
EPS = 1e-9


class LedgerError(AssertionError):
    """An invariant was violated. Always a bug -- never something to work around."""


@dataclass
class Ledger:
    """Portfolio state at the close of `as_of`."""

    cash: float
    shares: dict[str, float] = field(default_factory=dict)
    as_of: dt.date | None = None

    def __post_init__(self) -> None:
        self.shares = {k: float(v) for k, v in self.shares.items() if abs(v) > EPS}
        self.cash = float(self.cash)
        self.check()

    # ------------------------------------------------------------------ invariants

    def check(self) -> None:
        """I1: `cash >= 0`, `shares_i >= 0`. Cheap, so it stays on unconditionally."""
        if self.cash < -EPS:
            raise LedgerError(f"negative cash: {self.cash!r}")
        for ticker, n in self.shares.items():
            if n < -EPS:
                raise LedgerError(f"negative shares in {ticker}: {n!r}")
            if not np.isfinite(n):
                raise LedgerError(f"non-finite shares in {ticker}: {n!r}")
        if not np.isfinite(self.cash):
            raise LedgerError(f"non-finite cash: {self.cash!r}")

    # --------------------------------------------------------------------- access

    def get(self, ticker: str) -> float:
        return self.shares.get(ticker, 0.0)

    def holdings(self) -> list[str]:
        """Tickers with a non-zero position, in canonical dict order."""
        return [t for t, n in self.shares.items() if n > EPS]

    def share_vector(self, universe: Iterable[str]) -> np.ndarray:
        return np.array([self.get(t) for t in universe], dtype=float)

    # ---------------------------------------------------------------- mutation

    def set_shares(self, ticker: str, n: float) -> None:
        n = float(n)
        if n < -EPS:
            raise LedgerError(f"cannot set negative shares in {ticker}: {n!r}")
        if abs(n) <= EPS:
            self.shares.pop(ticker, None)
        else:
            self.shares[ticker] = n

    def add_shares(self, ticker: str, delta: float) -> None:
        self.set_shares(ticker, self.get(ticker) + float(delta))

    def accrue_shares(self, ticker: str, delta: float) -> None:
        """Share accretion from a reinvested distribution.

        A **separate entry point from `add_shares`** on purpose. The lock manager never
        observes this path, which is what implements the carve-out in
        reference/lock-state-machine.md section 3.5: a corporate action must not be able
        to silently freeze the portfolio.
        """
        self.set_shares(ticker, self.get(ticker) + float(delta))

    def copy(self) -> "Ledger":
        return Ledger(cash=self.cash, shares=dict(self.shares), as_of=self.as_of)

    # ------------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "shares": dict(self.shares),
            "as_of": self.as_of.isoformat() if self.as_of is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Ledger":
        as_of = d.get("as_of")
        return cls(
            cash=float(d["cash"]),
            shares={str(k): float(v) for k, v in dict(d.get("shares", {})).items()},
            as_of=dt.date.fromisoformat(as_of) if as_of else None,
        )
