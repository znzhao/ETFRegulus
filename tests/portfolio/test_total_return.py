"""T1 -- total-return reconstruction, against the real ledger.

Buy and hold 100% of one ticker with no rebalancing, and the resulting NAV series must
match that ticker's `close_adj` total-return series. This is what proves the
dividend-reinvestment ledger. **If TLT does not match, everything downstream is wrong** --
an agent trained on a raw-close NAV would rationally never hold a bond ETF, purely as a
data artifact.

Unlike the Stage 2 version of this check, which works on the derived series directly, this
one runs the actual `Ledger` / `valuation` code path the simulator uses.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.curate import TOTAL_RETURN_TOLERANCE
from src.portfolio.ledger import Ledger
from src.portfolio.valuation import value

CURATED = Path("data/curated/prices.parquet")


def buy_and_hold_nav(g: pd.DataFrame, ticker: str) -> pd.Series:
    """Run the real ledger: buy at the first close, then never trade again."""
    first = g.iloc[0]
    shares = 1.0 / float(first["close_raw"])
    ledger = Ledger(cash=0.0, shares={ticker: shares})

    peak, navs = 0.0, []
    for session, row in g.iterrows():
        prices = {ticker: float(row["close_raw"])}
        # The distribution on the very first session was already reflected in the price
        # we bought at, so it is not re-credited.
        div = {} if session == g.index[0] else {ticker: float(row["div_per_share"] or 0.0)}
        v = value(ledger, prices, peak=peak, div_per_share=div)
        peak = v.peak
        navs.append(v.nav)
    return pd.Series(navs, index=g.index)


def relative_error(nav_series: pd.Series, close_adj: pd.Series) -> float:
    a = nav_series.to_numpy(float) / float(nav_series.iloc[0])
    b = close_adj.to_numpy(float) / float(close_adj.iloc[0])
    return float(np.nanmax(np.abs(a / b - 1.0)))


# ------------------------------------------------------------------- synthetic


def test_reconstruction_on_a_constructed_dividend_payer(synthetic_prices):
    """A path where the answer is computable: quarterly distributions, known adjustment."""
    err = relative_error(buy_and_hold_nav(synthetic_prices, "X"),
                         synthetic_prices["close_adj"])
    assert err < 1e-9, f"the ledger does not reproduce a constructed series: {err:.3e}"


def test_ignoring_the_distribution_visibly_breaks_it(synthetic_prices, price_slice):
    """The test must be capable of failing, or it proves nothing.

    With reinvestment switched off the NAV is a price-return series, and it must diverge
    from total return by a wide margin over the tolerance. Measured on the real income
    payers, the margin is enormous -- which is the whole reason the ledger exists.
    """
    def price_return_error(g):
        shares = 1.0 / float(g.iloc[0]["close_raw"])
        return relative_error(g["close_raw"] * shares, g["close_adj"])

    err = price_return_error(synthetic_prices)
    assert err > 5 * TOTAL_RETURN_TOLERANCE, (
        f"price-return and total-return are indistinguishable in the synthetic fixture "
        f"({err:.3e}); it cannot detect a missing dividend"
    )

    # The real cases, and the ordering that matters: a missing distribution costs an
    # income payer far more than it costs gold, which is exactly the bias that would make
    # an agent refuse to hold bonds.
    errs = {t: price_return_error(price_slice.xs(t, level="ticker").sort_index())
            for t in ("TLT", "HYG", "GLD")}
    assert errs["TLT"] > 20 * TOTAL_RETURN_TOLERANCE, errs
    assert errs["HYG"] > 20 * TOTAL_RETURN_TOLERANCE, errs
    assert errs["TLT"] > errs["GLD"] and errs["HYG"] > errs["GLD"], (
        f"dropping dividends should penalize the income payers most, got {errs}"
    )


# ----------------------------------------------------------------- real slices


@pytest.mark.parametrize("ticker", ["SPY", "TLT", "HYG", "GLD", "XLRE"])
def test_reconstruction_on_the_real_slice(price_slice, ticker):
    g = price_slice.xs(ticker, level="ticker").sort_index()
    err = relative_error(buy_and_hold_nav(g, ticker), g["close_adj"])
    assert err < TOTAL_RETURN_TOLERANCE, f"{ticker}: reconstruction error {err:.3e}"


@pytest.mark.parametrize("ticker", ["SPY", "TLT", "GLD"])
def test_reconstruction_across_the_gfc(gfc_slice, ticker):
    """A window with large moves in both directions, where a sign error would show."""
    g = gfc_slice.xs(ticker, level="ticker").sort_index()
    err = relative_error(buy_and_hold_nav(g, ticker), g["close_adj"])
    assert err < TOTAL_RETURN_TOLERANCE, f"{ticker}: reconstruction error {err:.3e}"


def test_the_income_payers_are_the_ones_that_matter(price_slice):
    """TLT and HYG are where a missing distribution would be largest.

    Stated as an explicit ordering so the test records *why* those two are the sharp cases
    rather than leaving it to a comment.
    """
    income_gap = {}
    for ticker in ("TLT", "HYG", "GLD"):
        g = price_slice.xs(ticker, level="ticker").sort_index()
        price_ret = float(g["close_raw"].iloc[-1] / g["close_raw"].iloc[0])
        total_ret = float(g["close_adj"].iloc[-1] / g["close_adj"].iloc[0])
        income_gap[ticker] = total_ret - price_ret

    assert income_gap["TLT"] > income_gap["GLD"], "TLT should out-yield GLD"
    assert income_gap["HYG"] > income_gap["GLD"], "HYG should out-yield GLD"
    assert abs(income_gap["GLD"]) < 1e-9, "GLD pays no distribution and must show no gap"


# -------------------------------------------------------- full history, all tickers


@pytest.mark.skipif(not CURATED.exists(), reason="run Stage 2 first")
def test_reconstruction_over_the_full_history_for_every_tradable(universe_cfg):
    """The acceptance form of T1: every tradable, from inception to today.

    Slow enough to be worth its own test and fast enough to keep in the suite.
    """
    prices = pd.read_parquet(CURATED)
    worst: dict[str, float] = {}
    for ticker in universe_cfg.tradable_tickers:
        g = prices.xs(ticker, level="ticker").sort_index()
        worst[ticker] = relative_error(buy_and_hold_nav(g, ticker), g["close_adj"])

    failures = {t: e for t, e in worst.items() if not (e < TOTAL_RETURN_TOLERANCE)}
    assert not failures, f"total-return reconstruction failed for {failures}"
    assert worst["GLD"] == pytest.approx(0.0, abs=1e-12), (
        "GLD pays no distribution, so its reconstruction must be exact; a non-zero error "
        "means the derivation is inventing income"
    )
