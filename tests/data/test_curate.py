"""Stage 2 gate tests, plus the data-side halves of T1 and T15.

The gate's value is entirely in whether it *fires*. A check that has never been observed
to fail is untested infrastructure, so every check here is exercised against a
deliberately corrupted frame, not only against clean data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.calendar import sessions
from src.data.curate import (
    TOTAL_RETURN_TOLERANCE,
    implied_dividend_per_share,
    run_quality_gate,
    total_return_error,
)


def _long(df: pd.DataFrame, ticker: str = "SPY") -> pd.DataFrame:
    out = df.reset_index().rename(columns={"index": "session"})
    out["ticker"] = ticker
    return out


@pytest.fixture
def session_index():
    return sessions("2008-01-01", "2020-12-31")


# ------------------------------------------------------- dividend / total return


def test_implied_dividend_is_zero_when_there_is_no_distribution():
    raw = pd.Series([100.0, 101.0, 99.5, 102.0])
    adj = raw.copy()
    div = implied_dividend_per_share(adj, raw)
    assert np.allclose(div.fillna(0.0), 0.0)


def test_implied_dividend_recovers_a_known_distribution():
    """A hand-computable case: a $1.00 distribution on a $100 stock that closes flat."""
    raw = pd.Series([100.0, 100.0])
    # Total return that day is +1% (the dividend), price return is 0%.
    adj = pd.Series([100.0, 101.0])
    div = implied_dividend_per_share(adj, raw)
    assert div.iloc[1] == pytest.approx(1.0, rel=1e-9)


def test_dividend_derivation_is_split_neutral():
    """A 2:1 split halves both series, so the derived distribution must not move."""
    raw = pd.Series([100.0, 102.0, 51.5, 52.0])   # split between t1 and t2
    adj = pd.Series([100.0, 102.0, 51.5, 52.0])
    assert np.allclose(implied_dividend_per_share(adj, raw).fillna(0.0), 0.0)


def test_total_return_reconstruction_on_a_dividend_payer(synthetic_prices):
    """T1, data-side: a reinvesting share ledger must reproduce close_adj."""
    err = total_return_error(synthetic_prices)
    assert err < TOTAL_RETURN_TOLERANCE, f"reconstruction error {err:.2e}"


def test_total_return_reconstruction_on_the_real_slice(price_slice):
    for ticker in ("SPY", "TLT", "HYG"):
        g = price_slice.xs(ticker, level="ticker").sort_index()
        err = total_return_error(g)
        assert err < TOTAL_RETURN_TOLERANCE, f"{ticker}: {err:.2e}"


def test_ignoring_dividends_visibly_understates_an_income_payer(price_slice):
    """The bug this whole mechanism exists to prevent, demonstrated.

    Valuing the ledger at close_raw alone makes TLT look permanently worse than it was,
    and an agent trained on that NAV would rationally never hold a bond ETF.
    """
    g = price_slice.xs("TLT", level="ticker").sort_index()
    price_only = g["close_raw"].iloc[-1] / g["close_raw"].iloc[0]
    total = g["close_adj"].iloc[-1] / g["close_adj"].iloc[0]
    assert total > price_only, "TLT's total return must exceed its price return"


# ------------------------------------------------------------------ gate fires


def test_gate_passes_clean_data(universe_cfg, price_slice, session_index):
    """Real data, with the configured waivers applied exactly as the stage applies them.

    `inception_pin` and `spy_vti_corr` are excluded only because this fixture is a
    windowed slice: its first session is not any ticker's inception, and it has no VTI.
    """
    prices = price_slice.reset_index()
    prices["ticker"] = prices["ticker"].astype("category")
    inception = (prices.groupby("ticker", observed=True)["session"].min()
                 .rename("first_session").reset_index())

    waived = {(w.ticker, w.check) for w in universe_cfg.known_exceptions}
    hard = [v for v in run_quality_gate(universe_cfg, prices, inception, session_index)
            if v.hard and v.key() not in waived
            and v.check not in ("inception_pin", "spy_vti_corr")]
    assert hard == [], f"clean data produced hard violations: {hard}"


def test_the_xlre_volume_waiver_is_still_needed_and_still_narrow(universe_cfg, price_slice):
    """A waiver that silently stops applying to anything is a waiver that should be gone.

    This pins what is actually being waived, so the exception cannot quietly widen to
    cover a defect nobody reviewed.
    """
    waiver = next(w for w in universe_cfg.known_exceptions
                  if (w.ticker, w.check) == ("XLRE", "zero_volume"))
    g = price_slice.xs("XLRE", level="ticker").sort_index()
    offending = [str(d.date()) for d in g.index[(g["volume"] <= 0).fillna(False)]]
    assert offending, "the XLRE zero-volume waiver no longer covers anything; remove it"
    assert set(offending) <= set(waiver.sessions), (
        f"zero-volume sessions {sorted(set(offending) - set(waiver.sessions))} are not "
        f"listed in the waiver"
    )
    assert max(pd.Timestamp(d) for d in offending) < pd.Timestamp("2016-04-01"), (
        "the waiver is scoped to XLRE's first months; a later zero-volume print is a new "
        "defect and needs its own review"
    )


@pytest.mark.parametrize(
    "corrupt, expect",
    [
        (lambda d: d.assign(close_raw=d["close_raw"].mask(d.index == 5, -1.0)), "non_positive_price"),
        (lambda d: d.assign(low_raw=d["low_raw"].mask(d.index == 5, 1e6)), "ohlc_range"),
        (lambda d: d.assign(high_raw=d["high_raw"].mask(d.index == 5, 0.0)), "ohlc_range"),
        (lambda d: d.assign(close_adj=d["close_adj"].mask(d.index == 5, d["close_adj"].iloc[5] * 3)), "split_jump"),
        (lambda d: d.assign(div_per_share=d["div_per_share"].mask(d.index == 5, -5.0)), "negative_dividend"),
    ],
)
def test_gate_catches_injected_corruption(universe_cfg, price_slice, session_index,
                                          corrupt, expect):
    """Each check is shown to fire. A detector that has never fired is not known to work."""
    prices = price_slice.xs("SPY", level="ticker").sort_index().reset_index()
    prices["ticker"] = "SPY"
    prices = corrupt(prices)
    prices["ticker"] = prices["ticker"].astype("category")
    inception = pd.DataFrame({"ticker": ["SPY"], "first_session": [prices["session"].min()]})

    checks = {v.check for v in run_quality_gate(universe_cfg, prices, inception, session_index)
              if v.hard}
    assert expect in checks, f"expected {expect}, got {checks}"


def test_level_series_are_exempt_from_price_and_return_checks(universe_cfg, session_index):
    """^IRX genuinely closed negative in March 2020 and VIX routinely doubles in a day.

    Applying a price check to a level is the same class of error as computing a log
    return on a yield.
    """
    idx = session_index[:60]
    irx = pd.DataFrame({
        "session": idx, "ticker": "^IRX",
        "close_raw": np.linspace(0.5, -0.105, len(idx)),
        "close_adj": np.linspace(0.5, -0.105, len(idx)),
        "open_raw": np.nan, "high_raw": np.nan, "low_raw": np.nan, "volume": np.nan,
        "div_per_share": 0.0, "is_tradable": False,
    })
    irx["ticker"] = irx["ticker"].astype("category")
    inception = pd.DataFrame({"ticker": ["^IRX"], "first_session": [idx[0]]})
    hard = {v.check for v in run_quality_gate(universe_cfg, irx, inception, session_index)
            if v.hard}
    assert "non_positive_price" not in hard
    assert "split_jump" not in hard


def test_level_range_check_catches_a_provider_scale_change(universe_cfg, session_index):
    """The classic: ^TNX quoted as 42.5 instead of 4.25. It must fail, not divide by ten."""
    idx = session_index[:40]
    tnx = pd.DataFrame({
        "session": idx, "ticker": "^TNX",
        "close_raw": 42.5, "close_adj": 42.5,
        "open_raw": np.nan, "high_raw": np.nan, "low_raw": np.nan, "volume": np.nan,
        "div_per_share": 0.0, "is_tradable": False,
    })
    tnx["ticker"] = tnx["ticker"].astype("category")
    inception = pd.DataFrame({"ticker": ["^TNX"], "first_session": [idx[0]]})
    hard = {v.check for v in run_quality_gate(universe_cfg, tnx, inception, session_index)
            if v.hard}
    assert "level_range" in hard
