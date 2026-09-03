"""Stage 1 gate: the symbol quirks and the write/merge semantics.

The live-provider checks are marked `network` and excluded from the default run, so the
unit suite stays offline and deterministic. Run them explicitly after any change to the
symbol list:

    .venv/Scripts/python.exe -m pytest -m network -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.fetch import (
    FEATURE_ONLY_COLUMNS,
    TRADABLE_COLUMNS,
    _extract_symbol,
    merge_incremental,
    symbol_to_filename,
    write_atomic,
)

FETCH_LOG = Path("data/raw/fetch_log.json")

#: Symbols whose names break naive path or column handling. Pinned because each one has
#: cost somebody an afternoon at least once.
QUIRKY_SYMBOLS = {
    "^VIX": "_VIX",
    "^VVIX": "_VVIX",
    "^TNX": "_TNX",
    "^FVX": "_FVX",
    "^IRX": "_IRX",
    "DX-Y.NYB": "DX-Y_NYB",
}


@pytest.mark.parametrize("symbol,expected", sorted(QUIRKY_SYMBOLS.items()))
def test_quirky_symbols_map_to_safe_filenames(symbol, expected):
    assert symbol_to_filename(symbol) == expected


def test_filename_mapping_is_injective(universe_cfg):
    """Two symbols mapping to one file would silently overwrite each other's history."""
    names = [symbol_to_filename(s) for s in universe_cfg.all_yfinance_symbols]
    assert len(names) == len(set(names))


def test_column_mapping_keeps_both_close_series():
    """`auto_adjust=False` is what keeps Close (raw) and Adj Close (total return).

    Losing either is fatal in a different way: without close_raw there is no price to
    transact at, and without close_adj every income-paying ETF looks permanently terrible.
    """
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    raw = pd.DataFrame({
        "Open": [1.0, 2.0], "High": [3.0, 4.0], "Low": [0.5, 1.5],
        "Close": [2.0, 3.0], "Adj Close": [1.8, 2.7], "Volume": [10.0, 20.0],
    }, index=idx)
    out = _extract_symbol(raw, "SPY")
    assert set(TRADABLE_COLUMNS) <= set(out.columns)
    assert out["close_raw"].tolist() == [2.0, 3.0]
    assert out["close_adj"].tolist() == [1.8, 2.7]
    assert out.index.name == "session"


def test_extraction_normalizes_a_tz_aware_index():
    """yfinance daily bars are date-indexed but sometimes tz-aware."""
    idx = pd.DatetimeIndex(["2024-01-02 00:00:00-05:00", "2024-01-03 00:00:00-05:00"])
    raw = pd.DataFrame({"Close": [1.0, 2.0], "Adj Close": [1.0, 2.0]}, index=idx)
    out = _extract_symbol(raw, "^VIX")
    assert out.index.tz is None
    assert set(FEATURE_ONLY_COLUMNS) <= set(out.columns)


def test_incremental_merge_overwrites_the_overlap_rather_than_appending():
    """An adjusted history is REWRITTEN every time a dividend is paid, so the cached
    series is not append-only. The trailing re-pull must win on the overlap."""
    existing = pd.DataFrame(
        {"close_adj": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"], name="session"),
    )
    fresh = pd.DataFrame(
        {"close_adj": [99.0, 99.0]},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="session"),
    )
    merged = merge_incremental(existing, fresh)
    assert merged["close_adj"].tolist() == [1.0, 99.0, 99.0]
    assert merged.index.is_monotonic_increasing


def test_merge_handles_the_empty_cases():
    df = pd.DataFrame({"a": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"]))
    pd.testing.assert_frame_equal(merge_incremental(pd.DataFrame(), df), df)
    pd.testing.assert_frame_equal(merge_incremental(df, pd.DataFrame()), df)


def test_atomic_write_leaves_no_temp_file(tmp_path):
    """A killed job must never leave a truncated parquet where a good one was."""
    path = tmp_path / "x.parquet"
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]))
    write_atomic(df, path)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp*"))
    pd.testing.assert_frame_equal(pd.read_parquet(path), df)


# ------------------------------------------------------- pins against real output


@pytest.mark.skipif(not FETCH_LOG.exists(), reason="run Stage 1 first")
def test_every_universe_symbol_has_a_file(universe_cfg):
    log = json.loads(FETCH_LOG.read_text(encoding="utf-8"))
    missing = [s for s in universe_cfg.all_yfinance_symbols if s not in log["prices"]]
    assert missing == [], f"no raw file for {missing}"
    missing_fred = [s for s in universe_cfg.fred if s not in log["fred"]]
    assert missing_fred == [], f"no FRED file for {missing_fred}"


@pytest.mark.skipif(not FETCH_LOG.exists(), reason="run Stage 1 first")
def test_row_counts_match_the_session_count_over_each_symbols_own_range(universe_cfg):
    """A symbol short of its sessions has holes, and a hole is a gate failure."""
    log = json.loads(FETCH_LOG.read_text(encoding="utf-8"))
    thin = {s: e["coverage"] for s, e in log["prices"].items() if e["coverage"] < 0.99}
    assert thin == {}, f"coverage below 0.99: {thin}"


@pytest.mark.skipif(not FETCH_LOG.exists(), reason="run Stage 1 first")
def test_quirky_symbols_actually_returned_data(universe_cfg):
    """The caret-prefixed and dotted symbols are the ones that come back empty."""
    log = json.loads(FETCH_LOG.read_text(encoding="utf-8"))
    for symbol in QUIRKY_SYMBOLS:
        if symbol not in universe_cfg.all_yfinance_symbols:
            continue
        entry = log["prices"][symbol]
        assert entry["rows"] > 4000, f"{symbol} returned only {entry['rows']} rows"


@pytest.mark.skipif(not FETCH_LOG.exists(), reason="run Stage 1 first")
def test_inception_pins_hold_against_the_fetched_data(universe_cfg):
    log = json.loads(FETCH_LOG.read_text(encoding="utf-8"))
    history_start = pd.Timestamp(universe_cfg.history_start)
    for ticker, pinned in universe_cfg.inception_pins.items():
        first = pd.Timestamp(log["prices"][ticker]["first_session"])
        want = pd.Timestamp(pinned)
        if want < history_start:
            # SHY launched 2002-07-22 and we fetch from 2003-01-01, so its first bar is
            # the fetch window, not its inception. It must still start at the very top
            # of that window.
            assert (first - history_start).days <= 7, (
                f"{ticker}: pinned inception predates history_start, so the first bar "
                f"should sit at {history_start.date()}; it is {first.date()}"
            )
            continue
        assert abs((first - want).days) <= 7, (
            f"{ticker}: first bar {first.date()} vs pinned {pinned}"
        )


@pytest.mark.network
def test_live_provider_returns_the_quirky_symbols():
    """Excluded by default. Run after any change to the symbol list."""
    import yfinance as yf

    for symbol in QUIRKY_SYMBOLS:
        df = yf.download(symbol, start="2024-01-01", end="2024-03-01",
                         auto_adjust=False, actions=False, progress=False, threads=False)
        assert df is not None and len(df) > 20, f"{symbol} came back empty"
