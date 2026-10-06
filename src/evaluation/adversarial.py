"""Stage 11 -- adversarial historical scenarios.

reference/robustness.md section 3. Unfavourable combinations constructed **within the
support of the historical empirical distribution** -- not synthetic shocks, not
fitted-distribution tail draws.

    Equity shock + credit widening   worst equity block beside the worst credit block
    Duration loss                    worst TLT/IEF blocks with a non-rallying equity block
    Correlation spike                blocks with maximum realized cross-asset correlation
    Diversification breakdown        blocks where stock/bond correlation flipped positive

**How the splice avoids inventing returns.** Blocks are concatenated by *chaining returns*,
never by concatenating price levels. A price series stitched from disjoint windows
manufactures an enormous fake return at every seam -- the exact bug the project's own test
fixtures are contiguous to avoid. Here each block's real per-session returns are applied in
sequence to a running price, so every return in an adversarial path is one that actually
happened; only the ordering and combination are adversarial.

Features come from the same source sessions as the returns, so within a block they are
real and self-consistent. At a seam the features jump, which is the honest analogue of the
returns coming from a different date -- the policy sees an abrupt regime change, which is
what an adversarial ordering *is*.

**The disclaimer is part of the deliverable.** A scenario built by picking the worst
historical blocks is, by construction, not a probability statement about the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.sim.simulator import MarketData

DISCLAIMER = (
    "This is a stronger HISTORICAL ROBUSTNESS test, not a forward-looking worst-case "
    "guarantee. Every path is assembled from real historical blocks, so every return "
    "actually happened -- but the ordering was chosen adversarially, which makes these "
    "paths a stress construction and NOT a probability statement about the future."
)

#: Sessions per block. Long enough to carry a regime, short enough that several fit.
DEFAULT_BLOCK = 21


@dataclass
class Block:
    """A contiguous run of real sessions, and why it was selected."""

    start_row: int
    end_row: int
    score: float
    reason: str

    @property
    def length(self) -> int:
        return self.end_row - self.start_row + 1

    def to_dict(self, market: MarketData) -> dict:
        return {"start": str(market.sessions[self.start_row].date()),
                "end": str(market.sessions[self.end_row].date()),
                "sessions": self.length, "score": float(self.score),
                "reason": self.reason}


@dataclass
class Scenario:
    name: str
    description: str
    blocks: list[Block] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Blocks are SELECTED by score but REPLAYED in chronological order, and the
        # difference matters for correctness rather than neatness. Selection order can
        # run backwards through time, and an asset that exists in a 2010 block does not
        # exist in a 2004 one -- so a portfolio holding HYG would suddenly be holding an
        # ETF that had not launched, which is not an adversarial state but an illegal one
        # (invariant I5). Chronological order makes availability monotone, exactly as in
        # real history, and the scenario stays adversarial because the adversarial content
        # is WHICH blocks are combined, not the direction time runs in.
        self.blocks = sorted(self.blocks, key=lambda b: b.start_row)

    def rows(self) -> np.ndarray:
        return np.concatenate([np.arange(b.start_row, b.end_row + 1)
                               for b in self.blocks]) if self.blocks else np.zeros(0, int)


# ------------------------------------------------------------- block selection


def _windows(n_rows: int, block: int, step: int = 5) -> list[tuple[int, int]]:
    return [(i, i + block - 1) for i in range(0, n_rows - block, step)]


def _col(market: MarketData, ticker: str) -> int | None:
    return market.universe.index(ticker) if ticker in market.universe else None


def _block_returns(market: MarketData, lo: int, hi: int, cols) -> np.ndarray:
    """Total returns for `cols` over rows [lo, hi]. Column 0 of `returns` is cash."""
    return market.returns[lo:hi + 1][:, [c + 1 for c in cols]]


def worst_blocks(market: MarketData, tickers: list[str], *, block: int, n: int,
                 reason: str, rows: np.ndarray | None = None,
                 exclude: list[Block] | None = None) -> list[Block]:
    """The blocks over which the named assets lost the most, jointly."""
    cols = [c for c in (_col(market, t) for t in tickers) if c is not None]
    if not cols:
        return []
    lo_bound = int(rows[0]) if rows is not None and rows.size else 0
    hi_bound = int(rows[-1]) if rows is not None and rows.size else len(market.sessions) - 1
    scored = []
    for lo, hi in _windows(hi_bound - lo_bound, block):
        lo, hi = lo + lo_bound, hi + lo_bound
        r = _block_returns(market, lo, hi, cols)
        if not np.isfinite(r).all():
            continue
        scored.append(Block(lo, hi, float(np.nanmean(np.nansum(np.log1p(r), axis=0))),
                            reason))
    scored.sort(key=lambda b: b.score)
    return _disjoint(scored, n, exclude)


def most_correlated_blocks(market: MarketData, *, block: int, n: int, reason: str,
                           rows: np.ndarray | None = None,
                           exclude: list[Block] | None = None) -> list[Block]:
    """Blocks where every diversifier moved together -- the correlation spike."""
    lo_bound = int(rows[0]) if rows is not None and rows.size else 0
    hi_bound = int(rows[-1]) if rows is not None and rows.size else len(market.sessions) - 1
    scored = []
    for lo, hi in _windows(hi_bound - lo_bound, block):
        lo, hi = lo + lo_bound, hi + lo_bound
        r = market.returns[lo:hi + 1, 1:]
        keep = np.isfinite(r).all(axis=0) & (r.std(axis=0) > 1e-12)
        if keep.sum() < 4:
            continue
        corr = np.corrcoef(r[:, keep], rowvar=False)
        off = corr[~np.eye(corr.shape[0], dtype=bool)]
        scored.append(Block(lo, hi, float(np.nanmean(off)), reason))
    scored.sort(key=lambda b: -b.score)
    return _disjoint(scored, n, exclude)


def stock_bond_flip_blocks(market: MarketData, *, block: int, n: int, reason: str,
                           rows: np.ndarray | None = None,
                           exclude: list[Block] | None = None) -> list[Block]:
    """Blocks where the historically negative stock/bond correlation went positive.

    This is the 2022 failure mode generalized: the diversifier stops diversifying, and a
    policy that learned "bonds are the safe asset" has nowhere to hide.
    """
    spy, tlt = _col(market, "SPY"), _col(market, "TLT")
    if spy is None or tlt is None:
        return []
    lo_bound = int(rows[0]) if rows is not None and rows.size else 0
    hi_bound = int(rows[-1]) if rows is not None and rows.size else len(market.sessions) - 1
    scored = []
    for lo, hi in _windows(hi_bound - lo_bound, block):
        lo, hi = lo + lo_bound, hi + lo_bound
        a = market.returns[lo:hi + 1, spy + 1]
        b = market.returns[lo:hi + 1, tlt + 1]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        # Both falling AND correlated: a positive correlation in a rally is not the risk.
        corr = float(np.corrcoef(a, b)[0, 1])
        drift = float(np.log1p(a).sum() + np.log1p(b).sum())
        scored.append(Block(lo, hi, corr - drift, reason))
    scored.sort(key=lambda b: -b.score)
    return _disjoint(scored, n, exclude)


def _disjoint(blocks: list[Block], n: int,
              exclude: list[Block] | None = None) -> list[Block]:
    """Take the top `n` non-overlapping blocks, so a scenario is not one window repeated.

    `exclude` carries the blocks an earlier group already claimed. Without it the two
    halves of a combination scenario collapse: the worst equity window and the worst
    credit window are usually the SAME window -- 2008 hit both -- so "equity shock plus
    credit widening" would silently become "the 2008 crash, twice". Excluding what is
    already taken is what makes it a genuine combination of distinct episodes.
    """
    chosen: list[Block] = list(exclude or [])
    n += len(chosen)
    for b in blocks:
        if all(b.end_row < c.start_row or b.start_row > c.end_row for c in chosen):
            chosen.append(b)
        if len(chosen) >= n:
            break
    return chosen[len(exclude or []):]


# ------------------------------------------------------------- the scenarios


def build_scenarios(market: MarketData, *, block: int = DEFAULT_BLOCK, n_blocks: int = 4,
                    rows: np.ndarray | None = None) -> list[Scenario]:
    """The four scenarios from robustness.md section 3."""
    equity = ["SPY", "QQQ", "IWM"]
    credit = ["HYG", "LQD"]
    duration = ["TLT", "IEF"]

    half = max(1, n_blocks // 2)

    equity_worst = worst_blocks(market, equity, block=block, n=half,
                                reason="worst joint equity block", rows=rows)
    credit_worst = worst_blocks(market, credit, block=block, n=half,
                                reason="worst joint credit block", rows=rows,
                                exclude=equity_worst)
    duration_worst = worst_blocks(market, duration, block=block, n=half,
                                  reason="worst duration block", rows=rows)
    joint_worst = worst_blocks(market, equity + duration, block=block, n=half,
                               reason="worst joint equity+duration block", rows=rows,
                               exclude=duration_worst)

    return [
        Scenario(
            "equity_shock_credit_widening",
            "The worst joint equity blocks concatenated with the worst credit blocks, "
            "chosen DISJOINT from them: two distinct risk premia failing, rather than "
            "one crash counted twice.",
            equity_worst + credit_worst),
        Scenario(
            "duration_loss",
            "The worst observed long-duration blocks placed alongside a disjoint "
            "non-rallying equity block -- the 2022 shape, where the hedge is the loss.",
            duration_worst + joint_worst),
        Scenario(
            "correlation_spike",
            "Blocks selected for maximum realized cross-asset correlation: every "
            "diversifier moving together, so the allocation decision buys nothing.",
            most_correlated_blocks(market, block=block, n=n_blocks,
                                   reason="max mean pairwise correlation", rows=rows)),
        Scenario(
            "diversification_breakdown",
            "Blocks where the historically negative stock/bond correlation flipped "
            "positive while both fell -- a policy that learned 'bonds are safe' has "
            "nowhere to hide.",
            stock_bond_flip_blocks(market, block=block, n=n_blocks,
                                   reason="positive stock/bond correlation, both falling",
                                   rows=rows)),
    ]


# --------------------------------------------------------------- the splice


def splice(market: MarketData, rows: np.ndarray) -> MarketData:
    """Build a MarketData whose sessions are `rows`, chaining returns across seams.

    Prices are rebuilt by compounding each source session's own real return onto a running
    level, so no seam invents a return. Everything derived from a price -- open, high, low,
    dividends -- is scaled by the same per-session factor, which keeps `open <= high` and
    the rest of the OHLC relationships intact.

    The synthetic calendar reuses the ORIGINAL sessions of the first block and then
    continues with consecutive trading days, so the result is a legal, strictly increasing
    calendar that the lock's calendar-day arithmetic can operate on.
    """
    rows = np.asarray(rows, dtype=int)
    if rows.size < 2:
        raise ValueError("a spliced path needs at least two sessions")
    K = market.n_assets

    close = np.full((rows.size, K), np.nan)
    open_ = np.full((rows.size, K), np.nan)
    high = np.full((rows.size, K), np.nan)
    low = np.full((rows.size, K), np.nan)
    div = np.zeros((rows.size, K))
    available = np.zeros((rows.size, K), dtype=bool)
    returns = np.zeros((rows.size, K + 1))

    # Each asset's level is seeded LAZILY, the first session it is available. Seeding
    # them all from the first row would leave anything not yet born -- GLD launched in
    # November 2004, XLRE in 2015 -- at NaN forever, and the availability mask would then
    # claim an asset the price grid does not have. That mismatch is not cosmetic: the
    # executor looks a ticker up by name and raises when the mask and the prices disagree.
    level = np.full(K, np.nan)
    for i, r in enumerate(rows):
        src_close = market.close_raw[r]
        prev = r - 1 if r > 0 else r
        prev_close = market.close_raw[prev]
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where((prev_close > 0) & np.isfinite(prev_close)
                            & np.isfinite(src_close), src_close / prev_close, 1.0)
        step = np.where(np.isfinite(step) & (step > 0), step, 1.0)

        # A usable session needs a usable OPEN as well as a close: execution fills at the
        # open, so a finite close with a missing open would put a ticker in the
        # availability mask that `prices_at` cannot price.
        src_open = market.open_raw[r]
        live = (np.isfinite(src_close) & (src_close > 0)
                & np.isfinite(src_open) & (src_open > 0))
        # Compound where the asset already has a level; seed where it just appeared.
        level = np.where(np.isfinite(level), level * step, level)
        level = np.where(live & ~np.isfinite(level), src_close, level)
        level = np.where(live, level, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(live, level / src_close, np.nan)
        close[i] = level
        open_[i] = market.open_raw[r] * factor
        high[i] = market.high_raw[r] * factor
        low[i] = market.low_raw[r] * factor
        div[i] = np.nan_to_num(market.div_per_share[r]) * factor
        # Availability and the price grid must agree, or the executor raises on a ticker
        # the mask promised and `prices_at` dropped.
        available[i] = market.available[r] & live
        returns[i, 1:] = np.where(live, market.returns[r, 1:], 0.0)

    # A consecutive calendar taken from the original sessions, so the lock's calendar-day
    # arithmetic sees a real trading calendar rather than an invented one.
    start = int(rows[0])
    end = min(start + rows.size, len(market.sessions))
    sessions = market.sessions[start:end]
    if len(sessions) < rows.size:                       # ran off the end; take a tail
        sessions = market.sessions[-rows.size:]

    return MarketData(
        sessions=pd.DatetimeIndex(sessions), universe=list(market.universe),
        close_raw=close, open_raw=open_, high_raw=high, low_raw=low,
        div_per_share=div, available=available, returns=returns,
    )
