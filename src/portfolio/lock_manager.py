"""The resettable N-calendar-day holding lock.

Per-ETF (D13): buying ETF `i` resets only `unlock_date_i`; every other holding keeps its
own independent clock.

Two things here are easy to get wrong and are handled explicitly:

* **Absolute unlock dates, never countdowns.** A countdown has to be decremented, and
  anything that has to be decremented eventually gets decremented twice, or not at all
  across a reset. The observation exposes the derived countdown; the state stores the date.
* **`N` is in calendar days.** Not trading days. `unlock = execution_date + timedelta(N)`,
  and a sale is legal on the first session on or after that. Adding `N` *business* days is
  a different and wrong rule.

The lock is triggered by an **executed buy leg**, not by a net share increase. Selling a
position out and rebuying it the same day relocks, which closes the "sell to unlock,
immediately rebuy" loophole.

Tested independently of the environment (reference/lock-state-machine.md).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

Scope = Literal["per_etf", "portfolio"]


@dataclass(frozen=True)
class TradeLeg:
    """One executed leg. `delta_shares > 0` is a buy, `< 0` a sell."""

    ticker: str
    delta_shares: float
    price: float

    @property
    def is_buy(self) -> bool:
        return self.delta_shares > 0.0

    @property
    def notional(self) -> float:
        return abs(self.delta_shares) * self.price


class LockViolation(AssertionError):
    """An illegal sale reached execution. A hard stop, never a warning."""


@dataclass
class LockManager:
    """Absolute unlock dates per tradable ticker."""

    universe: list[str]
    unlock_dates: dict[str, dt.date] = field(default_factory=dict)
    scope: Scope = "per_etf"

    # --------------------------------------------------------------------- query

    def is_locked(self, ticker: str, session: dt.date) -> bool:
        unlock = self.unlock_dates.get(ticker)
        return unlock is not None and session < unlock

    def is_sellable(self, ticker: str, session: dt.date) -> bool:
        """`session` is by construction a session, so comparing dates directly resolves
        "the first session on or after the unlock date" implicitly."""
        return not self.is_locked(ticker, session)

    def locked_mask(self, session: dt.date) -> np.ndarray:
        return np.array([self.is_locked(t, session) for t in self.universe], dtype=bool)

    def remaining_days(self, session: dt.date) -> np.ndarray:
        """Calendar days until unlock, 0 where unlocked. This is the observation form."""
        out = np.zeros(len(self.universe), dtype=float)
        for i, t in enumerate(self.universe):
            unlock = self.unlock_dates.get(t)
            if unlock is not None:
                out[i] = max(0, (unlock - session).days)
        return out

    def lower_bounds(self, share_vector: np.ndarray, session: dt.date) -> np.ndarray:
        """Minimum share count the projection must respect: current shares if locked.

        A locked position may be increased but never decreased
        (reference/feasibility-projection.md section 3.4).
        """
        return np.where(self.locked_mask(session), share_vector, 0.0)

    # ------------------------------------------------------------------- mutate

    def apply_execution(
        self, legs: Sequence[TradeLeg], session: dt.date, hold_days: int,
        shares_after: Mapping[str, float] | None = None,
    ) -> None:
        """Update unlock dates from the legs that actually executed at `session`'s open.

        Order matters and follows reference/lock-state-machine.md section 3.4: positions
        that went to zero are cleared first, then buys set locks. So a same-day
        sell-then-rebuy ends locked, and the intermediate `None` is never observed.
        """
        if shares_after is not None:
            # 3.3 -- a position that reached zero has a clean slot.
            for ticker in list(self.unlock_dates):
                if abs(float(shares_after.get(ticker, 0.0))) <= 1e-9:
                    self.unlock_dates.pop(ticker, None)

        buys = [leg for leg in legs if leg.is_buy]
        if not buys:
            return  # 3.2 / 3.4 -- a sell-only or hold execution changes no unlock date.

        unlock = session + dt.timedelta(days=int(hold_days))
        if self.scope == "portfolio":
            # The rejected variant, retained only as an ablation: any purchase relocks
            # the entire book.
            held = set(self.unlock_dates)
            if shares_after is not None:
                held |= {t for t, n in shares_after.items() if abs(float(n)) > 1e-9}
            for ticker in held:
                self.unlock_dates[ticker] = unlock
            return

        for leg in buys:
            # 3.1 -- the reset applies to the ENTIRE position in i, not just the newly
            # bought portion. Shares are fungible; there are no tax lots in this model.
            self.unlock_dates[leg.ticker] = unlock

    def assert_sale_legal(self, ticker: str, session: dt.date) -> None:
        if not self.is_sellable(ticker, session):
            raise LockViolation(
                f"illegal sale of {ticker} on {session}: locked until "
                f"{self.unlock_dates.get(ticker)}. The projection should have removed "
                f"this; reaching execution means the constraint layer is broken."
            )

    # ------------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        return {
            "universe": list(self.universe),
            "scope": self.scope,
            "unlock_dates": {k: v.isoformat() for k, v in self.unlock_dates.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "LockManager":
        return cls(
            universe=list(d["universe"]),
            scope=d.get("scope", "per_etf"),
            unlock_dates={str(k): dt.date.fromisoformat(v)
                          for k, v in dict(d.get("unlock_dates", {})).items()},
        )

    def copy(self) -> "LockManager":
        return LockManager(universe=list(self.universe),
                           unlock_dates=dict(self.unlock_dates), scope=self.scope)


def sync_unlock_dates(lm: LockManager, shares: Mapping[str, float]) -> None:
    """L2: `unlock_date_i is None` if and only if `shares_i == 0`.

    Called after any path that can zero a position outside `apply_execution`.
    """
    for ticker in list(lm.unlock_dates):
        if abs(float(shares.get(ticker, 0.0))) <= 1e-9:
            lm.unlock_dates.pop(ticker, None)
