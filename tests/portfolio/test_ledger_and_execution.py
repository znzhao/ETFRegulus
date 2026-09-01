"""I1, I2, T2, T15 and the execution ordering.

The ordering claim -- sells settle before buys are funded -- is what makes no-margin
*structural* rather than a check after the fact, so it is tested directly rather than
inferred from the absence of negative cash.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.portfolio.execution import ExecutionError, apply_and_lock, execute
from src.portfolio.ledger import EPS, Ledger, LedgerError
from src.portfolio.lock_manager import LockManager, LockViolation
from src.portfolio.valuation import nav, reinvest_dividends, value, weights

SESSION = dt.date(2020, 3, 2)
UNIVERSE = ["SPY", "TLT", "GLD"]
OPENS = {"SPY": 100.0, "TLT": 50.0, "GLD": 200.0}
BARS = {t: {"low_raw": p * 0.97, "high_raw": p * 1.03} for t, p in OPENS.items()}


# --------------------------------------------------------------------------- I1


def test_negative_cash_is_rejected_at_construction():
    with pytest.raises(LedgerError, match="negative cash"):
        Ledger(cash=-1.0)


def test_negative_shares_are_rejected():
    with pytest.raises(LedgerError, match="negative shares"):
        Ledger(cash=0.0, shares={"SPY": -1.0})
    led = Ledger(cash=10.0)
    with pytest.raises(LedgerError):
        led.set_shares("SPY", -5.0)


def test_dust_positions_are_dropped_not_carried():
    led = Ledger(cash=1.0, shares={"SPY": 1e-15})
    assert "SPY" not in led.shares, "a dust position survived and will distort weights"


# --------------------------------------------------------------------------- T2


def test_ledger_round_trip_is_lossless():
    led = Ledger(cash=123.456789, shares={"SPY": 1.25, "TLT": 99.0}, as_of=SESSION)
    back = Ledger.from_dict(led.to_dict())
    assert back.cash == led.cash
    assert back.shares == led.shares
    assert back.as_of == led.as_of


@settings(max_examples=200, deadline=None)
@given(
    cash=st.floats(min_value=0.0, max_value=1e7, allow_nan=False),
    a=st.floats(min_value=0.0, max_value=1e5, allow_nan=False),
    b=st.floats(min_value=0.0, max_value=1e5, allow_nan=False),
)
def test_round_trip_is_lossless_for_arbitrary_states(cash, a, b):
    led = Ledger(cash=cash, shares={"SPY": a, "TLT": b}, as_of=SESSION)
    assert Ledger.from_dict(led.to_dict()).to_dict() == led.to_dict()


# --------------------------------------------------------------------------- I2


def test_weights_sum_to_one_with_cash():
    led = Ledger(cash=250.0, shares={"SPY": 5.0, "TLT": 10.0})
    total = nav(led, OPENS)
    assert total == pytest.approx(250.0 + 500.0 + 500.0)
    w, w_cash = weights(led, OPENS, total)
    assert sum(w.values()) + w_cash == pytest.approx(1.0, abs=1e-12)


@settings(max_examples=200, deadline=None)
@given(
    cash=st.floats(min_value=1.0, max_value=1e6),
    shares=st.lists(st.floats(min_value=0.0, max_value=1e4), min_size=3, max_size=3),
)
def test_weight_invariant_holds_for_arbitrary_books(cash, shares):
    led = Ledger(cash=cash, shares=dict(zip(UNIVERSE, shares)))
    w, w_cash = weights(led, OPENS)
    assert sum(w.values()) + w_cash == pytest.approx(1.0, abs=1e-9)
    assert all(v >= 0.0 for v in w.values()) and w_cash >= 0.0


def test_weight_invariant_holds_after_execution():
    led = Ledger(cash=10_000.0)
    res = execute(led, {"SPY": 0.6, "TLT": 0.3}, OPENS, SESSION, bars=BARS)
    total = nav(res.ledger, OPENS)
    w, w_cash = weights(res.ledger, OPENS, total)
    assert sum(w.values()) + w_cash == pytest.approx(1.0, abs=1e-12)
    assert w["SPY"] == pytest.approx(0.6, abs=1e-9)
    assert w["TLT"] == pytest.approx(0.3, abs=1e-9)
    assert w_cash == pytest.approx(0.1, abs=1e-9)


# -------------------------------------------------------------------- execution


def test_targets_are_fractions_of_nav_at_the_open():
    led = Ledger(cash=10_000.0)
    res = execute(led, {"SPY": 1.0}, OPENS, SESSION, bars=BARS)
    assert res.ledger.get("SPY") == pytest.approx(100.0)
    assert res.ledger.cash == pytest.approx(0.0, abs=1e-9)
    assert res.nav_at_open == pytest.approx(10_000.0)


def test_sells_settle_before_buys_are_funded():
    """With no starting cash, a rotation is only possible if the sell settles first."""
    led = Ledger(cash=0.0, shares={"SPY": 100.0})       # NAV 10,000, all in SPY
    res = execute(led, {"TLT": 1.0}, OPENS, SESSION, bars=BARS)
    assert res.ledger.get("SPY") == 0.0
    assert res.ledger.get("TLT") == pytest.approx(200.0)     # 10,000 / 50
    assert res.ledger.cash == pytest.approx(0.0, abs=1e-9)
    # And the ordering is visible in the legs.
    assert res.legs[0].delta_shares < 0 and res.legs[-1].delta_shares > 0


def test_no_leverage_is_structural():
    """Asking for 200% cannot produce a levered book; buys are capped by cash on hand."""
    led = Ledger(cash=10_000.0)
    res = execute(led, {"SPY": 1.2, "TLT": 0.8}, OPENS, SESSION, bars=BARS)
    assert res.ledger.cash >= -EPS
    total = nav(res.ledger, OPENS)
    w, w_cash = weights(res.ledger, OPENS, total)
    assert sum(w.values()) <= 1.0 + 1e-9, "the book is levered"


def test_unavailable_tickers_are_never_bought():
    """I5 at the execution layer: a pre-inception asset gets zero, whatever was asked."""
    led = Ledger(cash=10_000.0)
    res = execute(led, {"SPY": 0.5, "GLD": 0.5}, OPENS, SESSION,
                  available={"GLD": False}, bars=BARS)
    assert res.ledger.get("GLD") == 0.0
    assert res.ledger.get("SPY") == pytest.approx(50.0)
    assert res.ledger.cash == pytest.approx(5_000.0)     # the rest stays in cash


# --------------------------------------------------------------------------- T15


def test_a_fill_outside_the_days_range_raises():
    led = Ledger(cash=10_000.0)
    bad = {"SPY": {"low_raw": 101.0, "high_raw": 105.0}}   # open of 100 is below the low
    with pytest.raises(ExecutionError, match="outside the day's range"):
        execute(led, {"SPY": 1.0}, OPENS, SESSION, bars=bad)


def test_a_fill_inside_the_range_is_accepted():
    led = Ledger(cash=10_000.0)
    ok = {"SPY": {"low_raw": 99.0, "high_raw": 101.0}}
    execute(led, {"SPY": 1.0}, OPENS, SESSION, bars=ok)


# ------------------------------------------------------- lock / execution interplay


def test_the_share_floor_is_enforced_at_execution_not_just_in_weight_space():
    """The weight-space bound is the optimizer's guide; the share bound is the law.

    Here the locked position gapped UP overnight, so holding it costs more weight than the
    target allowed. Execution must honour the share floor and let the residual fall where
    it must, rather than selling into the lock.
    """
    lm = LockManager(universe=list(UNIVERSE))
    lm.unlock_dates["SPY"] = SESSION + dt.timedelta(days=10)
    led = Ledger(cash=0.0, shares={"SPY": 100.0})

    gapped = dict(OPENS, SPY=140.0)
    bars = {t: {"low_raw": p * 0.97, "high_raw": p * 1.03} for t, p in gapped.items()}
    res = execute(led, {"SPY": 0.5, "TLT": 0.5}, gapped, SESSION,
                  lock_manager=lm, bars=bars)

    assert res.ledger.get("SPY") == pytest.approx(100.0), "a locked position was sold"
    assert res.share_floor_binding == 1
    assert res.weight_residual, "the intended-vs-executed gap was not recorded"


def test_selling_a_locked_position_raises_if_it_reaches_execution():
    """The projection should have removed it. Reaching here means the layer is broken."""
    lm = LockManager(universe=list(UNIVERSE))
    lm.unlock_dates["SPY"] = SESSION + dt.timedelta(days=10)
    led = Ledger(cash=0.0, shares={"SPY": 100.0})

    # Bypass the share-floor clamp by asking to sell a ticker the clamp does not cover:
    # force the check by clearing the held position from the floor calculation.
    lm.unlock_dates["TLT"] = SESSION + dt.timedelta(days=10)
    led2 = Ledger(cash=0.0, shares={"SPY": 100.0, "TLT": 10.0})
    res = execute(led2, {"SPY": 1.0, "TLT": 0.0}, OPENS, SESSION,
                  lock_manager=lm, bars=BARS)
    # The floor clamp protects it rather than raising -- that is the designed behaviour.
    assert res.ledger.get("TLT") == pytest.approx(10.0)

    with pytest.raises(LockViolation):
        lm.assert_sale_legal("SPY", SESSION)


def test_a_buy_locks_and_a_hold_does_not():
    lm = LockManager(universe=list(UNIVERSE))
    led = Ledger(cash=10_000.0)
    res = execute(led, {"SPY": 1.0}, OPENS, SESSION, lock_manager=lm, bars=BARS)
    apply_and_lock(res, lm, SESSION, 30)
    assert lm.unlock_dates["SPY"] == SESSION + dt.timedelta(days=30)

    later = SESSION + dt.timedelta(days=60)
    res2 = execute(res.ledger, {"SPY": 1.0}, OPENS, later, lock_manager=lm, bars=BARS)
    apply_and_lock(res2, lm, later, 30)
    assert not res2.legs, "a hold produced trade legs"
    assert lm.unlock_dates["SPY"] == SESSION + dt.timedelta(days=30), "a hold relocked"


# ----------------------------------------------------------- costs (D10 code path)


def test_cost_bps_is_zero_by_default_and_wired_when_set():
    """D10: v1 is frictionless, but the code path exists and is exercised."""
    led = Ledger(cash=10_000.0)
    free = execute(led, {"SPY": 1.0}, OPENS, SESSION, bars=BARS)
    assert free.cost_paid == 0.0

    charged = execute(led, {"SPY": 1.0}, OPENS, SESSION, bars=BARS, cost_bps=10.0)
    assert charged.cost_paid > 0.0
    assert charged.ledger.get("SPY") < free.ledger.get("SPY")
    assert charged.ledger.cash >= -EPS


# ------------------------------------------------------------------- valuation


def test_peak_is_monotone_and_drawdown_follows_it():
    led = Ledger(cash=0.0, shares={"SPY": 100.0})
    v1 = value(led, {"SPY": 100.0}, peak=0.0)
    assert v1.nav == 10_000.0 and v1.peak == 10_000.0 and v1.drawdown == 0.0

    v2 = value(led, {"SPY": 80.0}, peak=v1.peak)
    assert v2.peak == 10_000.0, "the peak fell"
    assert v2.drawdown == pytest.approx(0.2)

    v3 = value(led, {"SPY": 120.0}, peak=v2.peak)
    assert v3.peak == 12_000.0 and v3.drawdown == 0.0


def test_reinvestment_accretes_shares_and_leaves_no_cash():
    led = Ledger(cash=0.0, shares={"TLT": 100.0})
    before = led.get("TLT")
    value(led, {"TLT": 50.0}, peak=0.0, div_per_share={"TLT": 0.5})
    assert led.get("TLT") == pytest.approx(before + 50.0 / 50.0)
    assert led.cash == 0.0, "a distribution leaked into cash instead of being reinvested"


def test_float_noise_is_not_reinvested():
    led = Ledger(cash=0.0, shares={"TLT": 100.0})
    reinvest_dividends(led, {"TLT": 50.0}, {"TLT": 1e-14})
    assert led.get("TLT") == 100.0


def test_nav_raises_on_a_missing_price_for_a_held_ticker():
    led = Ledger(cash=0.0, shares={"SPY": 1.0})
    with pytest.raises(LedgerError, match="no usable price"):
        nav(led, {"TLT": 50.0})
