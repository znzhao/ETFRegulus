"""The trading calendar -- the single source of truth for "is this a session".

`exchange_calendars` XNYS. Every price and feature frame is reindexed to these sessions:
no weekends, no holidays. Half-days (day after Thanksgiving, Christmas Eve, July 3) are
**normal sessions** and are never dropped.

The N-day hold rule is in *calendar* days, not sessions -- see
reference/lock-state-machine.md section 2. This module deliberately offers no
"add N business days" helper, because that is a different and wrong rule.
"""

from __future__ import annotations

import datetime as dt
import functools
from typing import Iterable

import pandas as pd


#: Calendars are constructed with an EXPLICIT start, always.
#:
#: `xcals.get_calendar("XNYS")` with no bounds returns a rolling ~20-year window --
#: measured 2026-09-01, it began at 2006-09-01. Every session before that raises
#: `DateOutOfBounds`, so a frame reindexed against the default calendar would silently
#: lose 2003-2006: the entire warm-up buffer and the first two study years, including
#: the run-up to the GFC. Caught by Stage 0 on its first real run.
CALENDAR_START = "2002-01-01"


@functools.lru_cache(maxsize=4)
def get_calendar(name: str = "XNYS", start: str = CALENDAR_START):
    import exchange_calendars as xcals

    return xcals.get_calendar(name, start=start)


def sessions(
    start: str | dt.date,
    end: str | dt.date | None = None,
    calendar: str = "XNYS",
) -> pd.DatetimeIndex:
    """Tz-naive session dates in `[start, end]`, inclusive."""
    cal = get_calendar(calendar)
    end = end or dt.date.today()
    idx = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return pd.DatetimeIndex(idx).tz_localize(None).normalize()


def is_session(day: str | dt.date, calendar: str = "XNYS") -> bool:
    return bool(get_calendar(calendar).is_session(pd.Timestamp(day)))


def first_session_on_or_after(day: str | dt.date, calendar: str = "XNYS") -> pd.Timestamp:
    """Resolve a calendar date to the first session at or after it.

    This is the sell-legality rule: an unlock date landing on a weekend or holiday makes
    the position sellable on the next session, with no extra delay beyond the calendar.
    """
    cal = get_calendar(calendar)
    ts = pd.Timestamp(day).normalize()
    if cal.is_session(ts):
        return ts
    nxt = cal.next_session(ts)
    return pd.Timestamp(nxt).tz_localize(None).normalize()


def normalize_index(index: Iterable) -> pd.DatetimeIndex:
    """Force any date-ish index to tz-naive, midnight-normalized timestamps.

    yfinance daily bars are date-indexed but sometimes tz-aware; normalizing at the
    boundary is what keeps every downstream join honest.
    """
    idx = pd.DatetimeIndex(pd.Index(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()
