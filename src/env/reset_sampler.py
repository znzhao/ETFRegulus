"""Reachable initial states for `env.reset`.

The requirement (reference/env-mdp.md section 5) is a random *legal* portfolio state.
Naively randomizing weights and lock clocks produces states no legal trajectory could
reach -- a position locked for 200 days under `N = 30`, or a holding that predates its
ETF's inception -- and training on those wastes capacity and distorts the value function.

Two generators, both implemented:

* **Approach A, the default** -- replay. Stage 5 snapshotted `(ledger, lock, peak, NAV)`
  tuples out of real trajectories through this very simulator, so every state is reachable
  *by construction*. There is no verification step because there is nothing to verify.
  What does need checking is the *parameter* the state is replayed under: a state produced
  at `N = 90` carries lock clocks that `N = 30` could never have created, so admissibility
  is re-checked against the episode's own `(N, D_max)`.

* **Approach B** -- constructive, with rejection. Slower, used to cover what the reservoir
  under-samples (high lock fractions, near-ceiling drawdowns). Everything it produces is
  checked by `is_reachable` before it is returned.

Both obey `D_t <= D_max` in the normal mode. Deliberately-breached states come only from
`stress=True`, which is a separate environment setting and never a default.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.portfolio.ledger import Ledger
from src.portfolio.lock_manager import LockManager
from src.sim.simulator import MarketData


@dataclass
class InitialState:
    """What a reset hands the engine."""

    session: pd.Timestamp
    ledger: Ledger
    lock_manager: LockManager
    peak: float
    nav: float
    drawdown: float
    source: str

    def describe(self) -> dict:
        return {"session": str(self.session.date()), "nav": self.nav, "peak": self.peak,
                "drawdown": self.drawdown, "source": self.source,
                "n_positions": len(self.ledger.holdings()),
                "n_locked": len(self.lock_manager.unlock_dates)}


class ReachabilityError(AssertionError):
    """A generator produced a state no legal trajectory could have reached."""


# --------------------------------------------------------------------- the check


def is_reachable(
    state: InitialState, market: MarketData, *, hold_days: int, max_drawdown: float,
    stress: bool = False,
) -> list[str]:
    """Return the reasons `state` is not reachable. Empty list means it is.

    Returns reasons rather than a bool so a rejection can be counted by cause -- a
    sampler that rejects 99% of draws for one reason is a bug, not bad luck.
    """
    reasons: list[str] = []
    session = pd.Timestamp(state.session)
    try:
        row = market.sessions.get_indexer([session])[0]
    except Exception:
        row = -1
    if row < 0:
        reasons.append(f"session {session.date()} is not a trading session")
        return reasons

    avail = market.available[row]
    for j, ticker in enumerate(market.universe):
        held = state.ledger.get(ticker)
        if held > 1e-9 and not avail[j]:
            reasons.append(f"{ticker} held on {session.date()}, before its inception")

    # L2: an unlock date without a position is a state the lock manager clears on every
    # execution, so it cannot persist in a legal trajectory.
    for ticker, unlock in state.lock_manager.unlock_dates.items():
        if state.ledger.get(ticker) <= 1e-9:
            reasons.append(f"{ticker} carries an unlock date with no position")
        remaining = (unlock - session.date()).days
        if remaining > hold_days:
            # The lock is set to exactly `exec_date + N` and only ever counts down, so a
            # remaining clock above N could not have been produced at this N.
            reasons.append(
                f"{ticker} has {remaining}d of lock remaining under N={hold_days}"
            )
        if remaining < 0:
            reasons.append(f"{ticker} carries an expired unlock date")

    if state.ledger.cash < -1e-9:
        reasons.append("negative cash")
    if state.nav <= 0:
        reasons.append("non-positive NAV")
    if state.peak < state.nav - 1e-6:
        reasons.append("peak below the NAV it caps")
    if not stress and state.drawdown > max_drawdown + 1e-9:
        reasons.append(
            f"drawdown {state.drawdown:.4f} exceeds D_max {max_drawdown:.4f} at reset"
        )
    return reasons


# ------------------------------------------------------------------- Approach A


class ReservoirSampler:
    """Approach A. Samples states Stage 5 recorded out of real trajectories."""

    def __init__(self, states: Sequence[dict], market: MarketData):
        self.market = market
        self.states = [s for s in states if s.get("ledger") is not None]
        if not self.states:
            raise ValueError("the reset reservoir is empty; run Stage 5 first")
        self._sessions = np.array([pd.Timestamp(s["session"]) for s in self.states])

    @classmethod
    def from_file(cls, path: str | Path, market: MarketData) -> "ReservoirSampler":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), market)

    @classmethod
    def latest(cls, market: MarketData,
               runs: Path = Path("artifacts/runs")) -> "ReservoirSampler":
        found = sorted(runs.glob("s05_run_baselines_*/reset_reservoir.json"))
        if not found:
            raise FileNotFoundError(
                "no reset reservoir found; run "
                "`python -m scripts.s05_run_baselines --config config/evaluation.yaml`"
            )
        return cls.from_file(found[-1], market)

    def candidates(self, *, hold_days: int, max_drawdown: float,
                   window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
                   stress: bool = False) -> list[int]:
        """Indices admissible under this episode's parameters."""
        out: list[int] = []
        for k, raw in enumerate(self.states):
            session = self._sessions[k]
            if window is not None and not (window[0] <= session <= window[1]):
                continue
            if not stress and raw["drawdown"] > max_drawdown + 1e-9:
                continue
            state = self._materialize(raw)
            if is_reachable(state, self.market, hold_days=hold_days,
                            max_drawdown=max_drawdown, stress=stress):
                continue
            out.append(k)
        return out

    def _materialize(self, raw: dict) -> InitialState:
        return InitialState(
            session=pd.Timestamp(raw["session"]),
            ledger=Ledger.from_dict(raw["ledger"]),
            lock_manager=LockManager.from_dict(raw["lock_manager"]),
            peak=float(raw["peak"]), nav=float(raw["nav"]),
            drawdown=float(raw["drawdown"]), source="reservoir",
        )

    def sample(self, rng: np.random.Generator, *, hold_days: int, max_drawdown: float,
               window=None, stress: bool = False) -> InitialState | None:
        idx = self.candidates(hold_days=hold_days, max_drawdown=max_drawdown,
                              window=window, stress=stress)
        if not idx:
            return None
        return self._materialize(self.states[int(rng.choice(idx))])


# ------------------------------------------------------------------- Approach B


class ConstructiveSampler:
    """Approach B. Builds a state, then rejects it if it is not reachable.

    Deliberately covers the corners the reservoir under-samples: it can be asked for a
    minimum locked fraction or a minimum drawdown, which a replayed baseline trajectory
    almost never supplies.
    """

    def __init__(self, market: MarketData, *, initial_cash: float = 1_000_000.0,
                 max_positions: int = 8, max_attempts: int = 200):
        self.market = market
        self.initial_cash = float(initial_cash)
        self.max_positions = int(max_positions)
        self.max_attempts = int(max_attempts)
        self.rejections: dict[str, int] = {}

    def sample(self, rng: np.random.Generator, *, hold_days: int, max_drawdown: float,
               window=None, stress: bool = False,
               min_locked_fraction: float = 0.0,
               min_drawdown: float = 0.0) -> InitialState | None:
        rows = np.arange(len(self.market.sessions))
        if window is not None:
            lo, hi = window
            keep = (self.market.sessions >= lo) & (self.market.sessions <= hi)
            rows = rows[keep]
        # The last session cannot start an episode: there would be no t+1 to execute at.
        rows = rows[rows < len(self.market.sessions) - 1]
        if rows.size == 0:
            return None

        for _ in range(self.max_attempts):
            state = self._draw(rng, rows, hold_days=hold_days,
                               max_drawdown=max_drawdown, stress=stress,
                               min_locked_fraction=min_locked_fraction,
                               min_drawdown=min_drawdown)
            if state is None:
                continue
            reasons = is_reachable(state, self.market, hold_days=hold_days,
                                   max_drawdown=max_drawdown, stress=stress)
            if not reasons:
                return state
            for r in reasons:
                key = r.split(" ", 1)[-1][:40]
                self.rejections[key] = self.rejections.get(key, 0) + 1
        return None

    def _draw(self, rng, rows, *, hold_days, max_drawdown, stress,
              min_locked_fraction, min_drawdown) -> InitialState | None:
        row = int(rng.choice(rows))
        session = self.market.sessions[row]
        closes = self.market.close_raw[row]
        avail = np.flatnonzero(self.market.available[row] & np.isfinite(closes)
                               & (closes > 0))
        if avail.size == 0:
            return None

        n_pos = int(rng.integers(0, min(self.max_positions, avail.size) + 1))
        chosen = rng.choice(avail, size=n_pos, replace=False) if n_pos else np.array([], int)

        # Dirichlet over the chosen assets plus cash gives a uniform draw from the
        # simplex rather than the biased one that normalizing uniforms produces.
        parts = rng.dirichlet(np.ones(n_pos + 1))
        cash_w, asset_w = float(parts[0]), parts[1:]

        nav = self.initial_cash
        ledger = Ledger(cash=cash_w * nav)
        for w, j in zip(asset_w, chosen):
            ticker = self.market.universe[j]
            ledger.set_shares(ticker, w * nav / float(closes[j]))

        lm = LockManager(universe=list(self.market.universe))
        want_locked = max(min_locked_fraction, float(rng.random()) * 0.8)
        for w, j in zip(asset_w, chosen):
            if hold_days > 0 and rng.random() < want_locked:
                # An unlock date is always `some past execution + N`, so the remaining
                # clock is drawn from [0, N] and never beyond it.
                remaining = int(rng.integers(1, hold_days + 1))
                lm.unlock_dates[self.market.universe[j]] = (
                    session.date() + dt.timedelta(days=remaining))

        # The peak is drawn to produce a target drawdown, then the state carries it --
        # an episode should be able to begin partway into a drawdown.
        hi = max_drawdown if not stress else min(max_drawdown * 2.0 + 0.05, 0.9)
        lo = min(min_drawdown, hi)
        drawdown = float(rng.uniform(lo, hi))
        peak = nav / max(1.0 - drawdown, 1e-6)

        return InitialState(session=session, ledger=ledger, lock_manager=lm, peak=peak,
                            nav=nav, drawdown=drawdown, source="constructive")


def flat_state(market: MarketData, rng: np.random.Generator, *,
               initial_cash: float = 1_000_000.0, window=None) -> InitialState:
    """An all-cash start on a random session. Curriculum stages 1-4 use this.

    Trivially reachable -- it is the state every simulation begins in -- and deliberately
    boring: the early curriculum is about learning portfolio mechanics, and a randomized
    inherited portfolio would confound that with learning to read a position it did not
    take. The *session* is still randomized, so the policy sees the whole training window.
    """
    rows = np.arange(len(market.sessions) - 1)
    if window is not None:
        keep = ((market.sessions >= window[0]) & (market.sessions <= window[1]))[:-1]
        rows = rows[keep]
    if rows.size == 0:
        raise ReachabilityError(f"no sessions in window {window}")
    row = int(rng.choice(rows))
    return InitialState(
        session=market.sessions[row], ledger=Ledger(cash=float(initial_cash)),
        lock_manager=LockManager(universe=list(market.universe)),
        peak=float(initial_cash), nav=float(initial_cash), drawdown=0.0,
        source="flat",
    )


# ------------------------------------------------------------------- composition


class ResetSampler:
    """Approach A with Approach B as the fallback, which is also the coverage path.

    The reservoir is finite and was built from six baselines, so it cannot cover every
    `(session, N, D_max)` cell -- an early-2004 session under `N = 60` and `D_max = 0.05`
    may have no admissible entry at all. Falling through to the constructive generator
    keeps the parameter grid uniformly reachable instead of quietly biasing the episode
    distribution toward the cells the baselines happened to visit.
    """

    def __init__(self, reservoir: ReservoirSampler | None,
                 constructive: ConstructiveSampler,
                 *, constructive_fraction: float = 0.25):
        self.reservoir = reservoir
        self.constructive = constructive
        self.constructive_fraction = float(constructive_fraction)
        self.counts = {"reservoir": 0, "constructive": 0, "fallback": 0}

    def sample(self, rng: np.random.Generator, *, hold_days: int, max_drawdown: float,
               window=None, stress: bool = False) -> InitialState:
        use_constructive = (self.reservoir is None
                            or rng.random() < self.constructive_fraction)
        if not use_constructive:
            state = self.reservoir.sample(rng, hold_days=hold_days,
                                          max_drawdown=max_drawdown, window=window,
                                          stress=stress)
            if state is not None:
                self.counts["reservoir"] += 1
                return state
            self.counts["fallback"] += 1
        state = self.constructive.sample(rng, hold_days=hold_days,
                                         max_drawdown=max_drawdown, window=window,
                                         stress=stress)
        if state is None:
            raise ReachabilityError(
                f"no reachable initial state for N={hold_days}, D_max={max_drawdown} "
                f"in window {window}; constructive rejections: {self.constructive.rejections}"
            )
        if not use_constructive:
            return state
        self.counts["constructive"] += 1
        return state


def build_sampler(market: MarketData, *, reservoir_path: str | Path | None = None,
                  initial_cash: float = 1_000_000.0,
                  constructive_fraction: float = 0.25,
                  require_reservoir: bool = True) -> ResetSampler:
    reservoir: ReservoirSampler | None = None
    try:
        reservoir = (ReservoirSampler.from_file(reservoir_path, market)
                     if reservoir_path else ReservoirSampler.latest(market))
    except (FileNotFoundError, ValueError):
        if require_reservoir:
            raise
    return ResetSampler(reservoir,
                        ConstructiveSampler(market, initial_cash=initial_cash),
                        constructive_fraction=constructive_fraction)
