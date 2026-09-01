"""Raw data acquisition: yfinance prices and FRED macro series.

Nothing here cleans, aligns, or validates -- `data/raw/` is exactly what the provider
returned, normalized only in the two ways that must happen at the boundary: tz-naive
dates, and the provider's column names mapped to ours.

Two rules that are not negotiable:

* **Never write partial data on a failed fetch.** Everything goes to a temp path and is
  renamed into place, so a killed job leaves the previous good file intact.
* **`auto_adjust=False`.** It keeps both `Close` (raw quoted) and `Adj Close` (total
  return). We need both: execution happens at raw prices, features come from adjusted
  ones (reference/portfolio-ledger.md section 3).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from src.data.calendar import normalize_index

RAW = Path("data/raw")
PRICES_DIR = RAW / "prices"
FRED_DIR = RAW / "fred"

#: Provider column -> ours. `Adj Close` is the only total-return column; OHLC and
#: Volume are never adjusted.
_PRICE_COLUMNS = {
    "Open": "open_raw",
    "High": "high_raw",
    "Low": "low_raw",
    "Close": "close_raw",
    "Adj Close": "close_adj",
    "Volume": "volume",
}

FEATURE_ONLY_COLUMNS = ["close_raw", "close_adj"]
TRADABLE_COLUMNS = ["open_raw", "high_raw", "low_raw", "close_raw", "close_adj", "volume"]


class FetchError(RuntimeError):
    """A fetch failed after every retry, or returned data that cannot be trusted."""


def symbol_to_filename(symbol: str) -> str:
    """`^VIX` -> `_VIX_`, `DX-Y.NYB` -> `DX-Y_NYB`. Reversible enough to be readable."""
    return symbol.replace("^", "_").replace(".", "_")


def write_atomic(df: pd.DataFrame, path: Path) -> Path:
    """Write a parquet via temp + rename, so a killed job never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        df.to_parquet(tmp, index=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def retrying(fn: Callable[[], object], *, retries: int, backoff: float, what: str,
             log: Callable[[str], None] = print) -> object:
    """Call `fn`, retrying with exponential backoff. Raises `FetchError` if all fail."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if attempt < retries:
                wait = backoff * (2 ** (attempt - 1))
                log(f"{what}: attempt {attempt}/{retries} failed ({exc!r}); retrying in {wait:.1f}s")
                time.sleep(wait)
    raise FetchError(f"{what}: failed after {retries} attempts: {last!r}") from last


# ----------------------------------------------------------------------- yfinance


def _extract_symbol(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Pull one symbol's columns out of a (possibly grouped) yfinance frame."""
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        df = raw[symbol].copy()
    else:
        df = raw.copy()

    df = df.rename(columns=_PRICE_COLUMNS)
    keep = [c for c in _PRICE_COLUMNS.values() if c in df.columns]
    df = df[keep]
    # yfinance daily bars are date-indexed but sometimes tz-aware. Normalize here, once.
    df.index = normalize_index(df.index)
    df.index.name = "session"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(how="all")


def fetch_prices(
    symbols: Sequence[str],
    start: str,
    end: str | None = None,
    *,
    batch_size: int = 20,
    pause: float = 2.0,
    retries: int = 3,
    backoff: float = 5.0,
    log: Callable[[str], None] = print,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for `symbols` in rate-limit-friendly batches.

    Returns symbol -> frame. A symbol that comes back empty is *reported*, not silently
    dropped -- yfinance returns empty frames on transient failure, and treating one as
    "this ticker has no data" is how a hole gets into the dataset.
    """
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    batches = [list(symbols[i:i + batch_size]) for i in range(0, len(symbols), batch_size)]

    for n, batch in enumerate(batches, start=1):
        log(f"batch {n}/{len(batches)}: {len(batch)} symbols")

        def _call(batch=batch):
            df = yf.download(
                batch, start=start, end=end,
                auto_adjust=False,   # keeps BOTH Close (raw) and Adj Close (total return)
                actions=False, group_by="ticker", threads=True, progress=False,
            )
            if df is None or df.empty:
                raise FetchError(f"empty frame for batch {batch}")
            return df

        raw = retrying(_call, retries=retries, backoff=backoff,
                       what=f"yfinance batch {n}", log=log)
        for symbol in batch:
            out[symbol] = _extract_symbol(raw, symbol)  # type: ignore[arg-type]

        if n < len(batches):
            time.sleep(pause)
    return out


# --------------------------------------------------------------------------- FRED


def fetch_fred_series(
    series_ids: Iterable[str],
    start: str,
    *,
    retries: int = 3,
    backoff: float = 5.0,
    api_key: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, pd.DataFrame]:
    """Fetch FRED series as `(date, value)` frames.

    **No lag is applied here.** The observation date is the period the value describes,
    which is not when it became public. Applying the publication lag is Stage 2's job
    (reference/data-pipeline.md section 3) -- doing it at fetch time would make
    `data/raw/` no longer a faithful record of what the provider returned.
    """
    from fredapi import Fred

    key = api_key or os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise FetchError("FRED_API_KEY is unset. Copy .env.example to .env and add a key.")

    fred = Fred(api_key=key)
    out: dict[str, pd.DataFrame] = {}
    for sid in series_ids:
        def _call(sid=sid):
            s = fred.get_series(sid, observation_start=start)
            if s is None or len(s) == 0:
                raise FetchError(f"empty series {sid}")
            return s

        series = retrying(_call, retries=retries, backoff=backoff,
                          what=f"FRED {sid}", log=log)
        df = pd.DataFrame({"value": series})  # type: ignore[arg-type]
        df.index = normalize_index(df.index)
        df.index.name = "date"
        out[sid] = df[~df.index.duplicated(keep="last")].sort_index()
        log(f"FRED {sid}: {len(df)} observations {df.index[0].date()} .. {df.index[-1].date()}")
    return out


# -------------------------------------------------------------------- incremental


def merge_incremental(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Overwrite the overlapping tail with freshly fetched rows.

    Append-only is wrong here: an adjusted history is *rewritten* every time a dividend
    is paid, so the cached `close_adj` silently drifts from the true series. The trailing
    re-pull exists precisely so the overlap is overwritten, not merged.
    """
    if existing is None or existing.empty:
        return fresh
    if fresh is None or fresh.empty:
        return existing
    kept = existing.loc[existing.index < fresh.index.min()]
    combined = pd.concat([kept, fresh])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
