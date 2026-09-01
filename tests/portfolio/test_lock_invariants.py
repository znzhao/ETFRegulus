"""I3, I4 and the L1-L8 lock invariants -- property-based.

These are `hypothesis` tests over random buy/sell/hold sequences with random `N`, random
session gaps and random inception dates, because this is the test that finds the
sell-then-rebuy-same-day case and fixed fixtures will not.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.portfolio.ledger import Ledger
from src.portfolio.lock_manager import LockManager, LockViolation, TradeLeg

UNIVERSE = ["SPY", "TLT", "GLD"]
START = dt.date(2020, 1, 2)

# The D16 operating range plus its out-of-distribution stress points, so the property
# tests cover the extremes the sweep will visit.
hold_days_st = st.sampled_from([0, 7, 15, 21, 30, 42, 60, 90, 180])


def _lm(scope: str = "per_etf") -> LockManager:
    return LockManager(universe=list(UNIVERSE), scope=scope)


# ------------------------------------------------------------------ basic algebra


def test_a_buy_sets_the_unlock_date_exactly():
    """L3: an executed buy sets `unlock == execution_date + timedelta(N)`, exactly."""
    for n in (0, 7, 30, 180):
        lm = _lm()
        lm.apply_execution([TradeLeg("SPY", 10.0, 100.0)], START, n,
                           shares_after={"SPY": 10.0})
        assert lm.unlock_dates["SPY"] == START + dt.timedelta(days=n)


def test_the_lock_is_in_calendar_days_not_trading_days():
    """30 calendar days from a Thursday is a Saturday; the position becomes sellable on
    the next session, with no extra delay beyond the calendar."""
    lm = _lm()
    thursday = dt.date(2020, 1, 2)
    lm.apply_execution([TradeLeg("SPY", 1.0, 100.0)], thursday, 30,
                       shares_after={"SPY": 1.0})
    unlock = lm.unlock_dates["SPY"]
    assert unlock == dt.date(2020, 2, 1)          # a Saturday
    assert unlock.weekday() == 5
    assert lm.is_locked("SPY", dt.date(2020, 1, 31))
    # The first session on or after is Monday 2020-02-03, and it is sellable.
    assert not lm.is_locked("SPY", dt.date(2020, 2, 3))
    # 30 BUSINESS days would have been 2020-02-13 -- a different and wrong rule.
    assert not lm.is_locked("SPY", dt.date(2020, 2, 5))


def test_n_zero_is_sellable_the_next_session():
    """L7: `N = 0` behaves as a no-lock portfolio at every decision point."""
    lm = _lm()
    lm.apply_execution([TradeLeg("SPY", 1.0, 100.0)], START, 0, shares_after={"SPY": 1.0})
    assert lm.unlock_dates["SPY"] == START
    assert not lm.is_locked("SPY", START)
    assert lm.locked_mask(START).sum() == 0


def test_adding_to_a_position_relocks_the_whole_holding():
    """The point of the constraint: adding to a winner costs the optionality to sell what
    you already hold in that name."""
    lm = _lm()
    day1 = dt.date(2020, 1, 1)
    lm.apply_execution([TradeLeg("SPY", 10.0, 100.0)], day1, 30, shares_after={"SPY": 10.0})
    assert lm.unlock_dates["SPY"] == dt.date(2020, 1, 31)

    day10 = dt.date(2020, 1, 10)
    lm.apply_execution([TradeLeg("SPY", 5.0, 100.0)], day10, 30, shares_after={"SPY": 15.0})
    assert lm.unlock_dates["SPY"] == dt.date(2020, 2, 9)
    # The shares bought on day 1 are now locked until day 40.
    assert lm.is_locked("SPY", dt.date(2020, 2, 1))


def test_the_lock_is_per_etf_not_portfolio_wide():
    """D13. Buying SPY must leave TLT's clock alone."""
    lm = _lm()
    lm.apply_execution([TradeLeg("TLT", 5.0, 100.0)], dt.date(2020, 1, 1), 30,
                       shares_after={"TLT": 5.0})
    tlt_unlock = lm.unlock_dates["TLT"]
    lm.apply_execution([TradeLeg("SPY", 5.0, 100.0)], dt.date(2020, 1, 10), 30,
                       shares_after={"TLT": 5.0, "SPY": 5.0})
    assert lm.unlock_dates["TLT"] == tlt_unlock, "buying SPY moved TLT's clock"
    assert lm.unlock_dates["SPY"] == dt.date(2020, 2, 9)


def test_the_portfolio_scope_ablation_relocks_everything():
    """The rejected variant, kept available for ablation -- and shown to differ."""
    lm = _lm(scope="portfolio")
    lm.apply_execution([TradeLeg("TLT", 5.0, 100.0)], dt.date(2020, 1, 1), 30,
                       shares_after={"TLT": 5.0})
    lm.apply_execution([TradeLeg("SPY", 5.0, 100.0)], dt.date(2020, 1, 10), 30,
                       shares_after={"TLT": 5.0, "SPY": 5.0})
    assert lm.unlock_dates["TLT"] == lm.unlock_dates["SPY"] == dt.date(2020, 2, 9)


def test_a_sell_does_not_move_the_unlock_date():
    """L4: selling does not extend, shorten, or reset the lock. A partial sale of an
    unlocked position leaves the remainder unlocked."""
    lm = _lm()
    lm.apply_execution([TradeLeg("SPY", 10.0, 100.0)], dt.date(2020, 1, 1), 30,
                       shares_after={"SPY": 10.0})
    before = lm.unlock_dates["SPY"]
    lm.apply_execution([TradeLeg("SPY", -4.0, 100.0)], dt.date(2020, 3, 1), 30,
                       shares_after={"SPY": 6.0})
    assert lm.unlock_dates["SPY"] == before


def test_selling_out_entirely_clears_the_slot():
    """L2: `unlock_date_i is None` iff `shares_i == 0`."""
    lm = _lm()
    lm.apply_execution([TradeLeg("SPY", 10.0, 100.0)], dt.date(2020, 1, 1), 30,
                       shares_after={"SPY": 10.0})
    lm.apply_execution([TradeLeg("SPY", -10.0, 100.0)], dt.date(2020, 3, 1), 30,
                       shares_after={"SPY": 0.0})
    assert "SPY" not in lm.unlock_dates


def test_sell_then_rebuy_on_the_same_day_relocks():
    """3.4 -- the genuinely subtle case.

    Net-flat still relocks, because a buy occurred. This closes the loophole of "sell to
    unlock, immediately rebuy, keep the lock clock frozen". The lock manager inspects the
    executed legs, not the start-to-end share delta.
    """
    lm = _lm()
    lm.apply_execution([TradeLeg("SPY", 10.0, 100.0)], dt.date(2020, 1, 1), 30,
                       shares_after={"SPY": 10.0})
    day = dt.date(2020, 3, 1)
    lm.apply_execution(
        [TradeLeg("SPY", -10.0, 100.0), TradeLeg("SPY", 10.0, 100.0)], day, 30,
        shares_after={"SPY": 10.0},
    )
    assert lm.unlock_dates["SPY"] == day + dt.timedelta(days=30)
    assert lm.is_locked("SPY", day + dt.timedelta(days=29))


def test_dividend_reinvestment_never_moves_an_unlock_date():
    """L5 / 3.5 -- the carve-out. A corporate action must not freeze the portfolio.

    Implemented structurally: reinvestment goes through `Ledger.accrue_shares`, which
    produces no trade leg, so the lock manager cannot observe it.
    """
    from src.portfolio.valuation import reinvest_dividends

    lm = _lm()
    lm.apply_execution([TradeLeg("TLT", 100.0, 90.0)], dt.date(2020, 1, 1), 30,
                       shares_after={"TLT": 100.0})
    before = lm.unlock_dates["TLT"]

    ledger = Ledger(cash=0.0, shares={"TLT": 100.0})
    paid = reinvest_dividends(ledger, {"TLT": 90.0}, {"TLT": 0.25})
    assert paid == pytest.approx(25.0)
    assert ledger.get("TLT") > 100.0, "the distribution did not accrete shares"
    assert lm.unlock_dates["TLT"] == before, "reinvestment reset the lock"


def test_an_illegal_sale_raises_rather_than_warning():
    lm = _lm()
    lm.apply_execution([TradeLeg("SPY", 10.0, 100.0)], START, 30, shares_after={"SPY": 10.0})
    with pytest.raises(LockViolation):
        lm.assert_sale_legal("SPY", START + dt.timedelta(days=5))
    lm.assert_sale_legal("SPY", START + dt.timedelta(days=30))   # legal, must not raise


def test_round_trip_is_lossless():
    """L8 / T2: `from_dict(to_dict(lm))` reproduces every mask and bound bit-identically."""
    lm = _lm()
    lm.apply_execution([TradeLeg("SPY", 1.0, 1.0), TradeLeg("TLT", 1.0, 1.0)],
                       START, 42, shares_after={"SPY": 1.0, "TLT": 1.0})
    back = LockManager.from_dict(lm.to_dict())
    for offset in range(0, 60, 3):
        day = START + dt.timedelta(days=offset)
        assert np.array_equal(lm.locked_mask(day), back.locked_mask(day))
        assert np.array_equal(lm.remaining_days(day), back.remaining_days(day))
    assert back.unlock_dates == lm.unlock_dates
    assert back.scope == lm.scope


# ------------------------------------------------------------- property-based


action_st = st.tuples(
    st.sampled_from(UNIVERSE),
    st.sampled_from(["buy", "sell_partial", "sell_all", "hold", "sell_rebuy"]),
    st.integers(min_value=0, max_value=9),          # session gap in calendar days
)


@settings(max_examples=250, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(actions=st.lists(action_st, min_size=1, max_size=25), hold_days=hold_days_st,
       inception_offset=st.integers(min_value=0, max_value=20))
def test_lock_invariants_hold_over_random_sequences(actions, hold_days, inception_offset):
    """I3 + I4 + L1-L6 over arbitrary buy/sell/hold sequences.

    The simulated executor below mirrors the real ordering (legal sells, then buys, then
    locks) and asserts the invariants after every step.
    """
    lm = _lm()
    shares = dict.fromkeys(UNIVERSE, 0.0)
    inception = {t: START + dt.timedelta(days=inception_offset * i)
                 for i, t in enumerate(UNIVERSE)}
    session = START

    for ticker, kind, gap in actions:
        session = session + dt.timedelta(days=gap)
        if session < inception[ticker]:
            continue                                # I5: cannot trade before inception

        before = dict(shares)
        locked_before = lm.is_locked(ticker, session)
        unlock_before = dict(lm.unlock_dates)
        legs: list[TradeLeg] = []

        if kind == "buy":
            shares[ticker] += 5.0
            legs.append(TradeLeg(ticker, 5.0, 100.0))
        elif kind in ("sell_partial", "sell_all") and shares[ticker] > 0 and not locked_before:
            delta = -shares[ticker] / 2 if kind == "sell_partial" else -shares[ticker]
            shares[ticker] += delta
            legs.append(TradeLeg(ticker, delta, 100.0))
        elif kind == "sell_rebuy" and shares[ticker] > 0 and not locked_before:
            held = shares[ticker]
            legs.append(TradeLeg(ticker, -held, 100.0))
            legs.append(TradeLeg(ticker, held, 100.0))
            # net flat

        lm.apply_execution(legs, session, hold_days, shares_after=shares)

        # L1: a locked position never shrank.
        for t in UNIVERSE:
            if t in unlock_before and session < unlock_before[t]:
                assert shares[t] >= before[t] - 1e-9, (
                    f"{t} shrank while locked until {unlock_before[t]} on {session}"
                )

        # I4 / L3: an executed buy set the unlock date exactly.
        if any(leg.is_buy for leg in legs):
            assert lm.unlock_dates[ticker] == session + dt.timedelta(days=hold_days)

        # L4: a sell-only or hold execution left every unlock date alone, except that a
        # position reaching zero clears its own slot.
        if legs and not any(leg.is_buy for leg in legs):
            for t, d in unlock_before.items():
                if shares[t] > 1e-9:
                    assert lm.unlock_dates.get(t) == d

        # L2: unlock date present iff a position is held.
        for t in UNIVERSE:
            if shares[t] <= 1e-9:
                assert t not in lm.unlock_dates, f"{t} has an unlock date with no position"
            elif t in lm.unlock_dates:
                assert lm.unlock_dates[t] is not None

        # L7: N = 0 is a no-lock portfolio.
        if hold_days == 0:
            assert not lm.locked_mask(session).any()

        # L8: round-trip stays exact at every step.
        assert LockManager.from_dict(lm.to_dict()).unlock_dates == lm.unlock_dates


@settings(max_examples=100, deadline=None)
@given(hold_days=hold_days_st, offset=st.integers(min_value=0, max_value=200))
def test_remaining_days_matches_the_stored_unlock_date(hold_days, offset):
    """The observation exposes a countdown derived from the date; it must never disagree
    with the date, and never go negative."""
    lm = _lm()
    lm.apply_execution([TradeLeg("SPY", 1.0, 1.0)], START, hold_days,
                       shares_after={"SPY": 1.0})
    session = START + dt.timedelta(days=offset)
    remaining = lm.remaining_days(session)[UNIVERSE.index("SPY")]
    assert remaining >= 0
    assert remaining == max(0, hold_days - offset)
    assert lm.is_locked("SPY", session) == (remaining > 0)


@settings(max_examples=100, deadline=None)
@given(hold_days=hold_days_st)
def test_lower_bounds_pin_locked_positions_and_free_the_rest(hold_days):
    lm = _lm()
    lm.apply_execution([TradeLeg("SPY", 7.0, 100.0)], START, hold_days,
                       shares_after={"SPY": 7.0})
    held = np.array([7.0, 3.0, 0.0])
    mid = START + dt.timedelta(days=max(0, hold_days - 1))
    bounds = lm.lower_bounds(held, mid)
    if hold_days > 0:
        assert bounds[0] == 7.0, "a locked position was not pinned"
    else:
        assert bounds[0] == 0.0
    assert bounds[1] == 0.0 and bounds[2] == 0.0
    assert np.all(bounds <= held + 1e-12), "a floor exceeded the position it bounds"
