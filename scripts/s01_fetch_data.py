"""Stage 1 -- fetch raw data.

    python -m scripts.s01_fetch_data --config config/universe.yaml
    python -m scripts.s01_fetch_data --config config/universe.yaml --incremental
    python -m scripts.s01_fetch_data --config config/universe.yaml --skip-fred

Writes `data/raw/prices/<ticker>.parquet`, `data/raw/fred/<series>.parquet`, and
`data/raw/fetch_log.json`. `data/raw/` is exactly what the provider returned; alignment
and validation are Stage 2.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pandas as pd

from src.cli.stage import StageContext, StageError, stage
from src.config.schema import UniverseConfig
from src.data.calendar import sessions
from src.data.fetch import (
    FEATURE_ONLY_COLUMNS,
    FRED_DIR,
    PRICES_DIR,
    RAW,
    TRADABLE_COLUMNS,
    FetchError,
    fetch_fred_series,
    fetch_prices,
    merge_incremental,
    read_existing,
    symbol_to_filename,
    write_atomic,
)


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--incremental", action="store_true",
                   help="re-pull and overwrite only the trailing window (fetch.incremental_days)")
    p.add_argument("--since", default=None,
                   help="explicit fetch start date (YYYY-MM-DD); overrides --incremental")
    p.add_argument("--only", default=None,
                   help="comma-separated symbols to fetch instead of the whole universe")
    p.add_argument("--skip-fred", action="store_true",
                   help="fetch prices only (use when no FRED key is available yet)")


@stage(
    name="s01_fetch_data",
    config_default="config/universe.yaml",
    config_cls=UniverseConfig,
    outputs=["data/raw/fetch_log.json"],
    upstream="s00_check_env",
    add_args=_add_args,
)
def main(cfg: UniverseConfig, ctx: StageContext) -> None:
    """Fetch yfinance OHLCV and FRED macro series into data/raw/."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    args = ctx.args
    today = dt.date.today()

    if args.since:
        start = args.since
    elif args.incremental:
        start = str(today - dt.timedelta(days=cfg.fetch.incremental_days))
    else:
        start = cfg.history_start
    ctx.log(f"fetch window: {start} .. today  (incremental={args.incremental})")

    tradable = set(cfg.tradable_tickers)
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = set(wanted) - set(cfg.all_yfinance_symbols) - set(cfg.fred)
        if unknown:
            raise StageError(f"--only names symbols not in the universe: {sorted(unknown)}")
        price_symbols = [s for s in cfg.all_yfinance_symbols if s in wanted]
        fred_ids = [s for s in cfg.fred if s in wanted]
    else:
        price_symbols = cfg.all_yfinance_symbols
        fred_ids = list(cfg.fred)

    expected = sessions(start, today, calendar=cfg.calendar)
    ctx.log(f"{len(expected)} sessions expected in the window "
            f"({expected[0].date()} .. {expected[-1].date()})" if len(expected)
            else "no sessions in the window")

    # ------------------------------------------------------------------ prices
    log_entries: dict[str, dict] = {}
    frames = fetch_prices(
        price_symbols, start=start,
        batch_size=cfg.fetch.batch_size, pause=cfg.fetch.batch_pause_seconds,
        retries=cfg.fetch.retries, backoff=cfg.fetch.backoff_seconds,
        log=ctx.log,
    )

    empty = [s for s, df in frames.items() if df.empty]
    if empty:
        # Empty is reported, never treated as "this symbol has no data" -- that is how a
        # transient provider failure becomes a permanent hole in the dataset.
        raise StageError(
            f"yfinance returned empty frames for {empty}. This is a transient failure "
            f"mode, not evidence the symbols are dead. Re-run before concluding otherwise."
        )

    for symbol, fresh in frames.items():
        cols = TRADABLE_COLUMNS if symbol in tradable else FEATURE_ONLY_COLUMNS
        missing = [c for c in cols if c not in fresh.columns]
        if missing:
            raise StageError(f"{symbol}: provider did not return {missing}")
        fresh = fresh[cols]

        path = PRICES_DIR / f"{symbol_to_filename(symbol)}.parquet"
        merged = merge_incremental(read_existing(path), fresh) if args.incremental or args.since else fresh
        write_atomic(merged, path)

        in_window = expected[(expected >= merged.index.min()) & (expected <= merged.index.max())]
        coverage = len(merged) / len(in_window) if len(in_window) else float("nan")
        log_entries[symbol] = {
            "path": str(path),
            "rows": int(len(merged)),
            "first_session": str(merged.index[0].date()),
            "last_session": str(merged.index[-1].date()),
            "expected_sessions_in_range": int(len(in_window)),
            "coverage": round(float(coverage), 6),
            "tradable": symbol in tradable,
            "columns": list(merged.columns),
        }
        ctx.log(f"{symbol:<10} {len(merged):>6} rows  "
                f"{merged.index[0].date()} .. {merged.index[-1].date()}  "
                f"coverage {coverage:.4f}")

    # Row counts must match the session count over each symbol's own available range --
    # a symbol short of its sessions has holes, and a hole is a Stage 2 gate failure.
    thin = {s: e for s, e in log_entries.items() if e["coverage"] < 0.98}
    if thin:
        ctx.warn(f"coverage below 0.98 for {sorted(thin)} -- Stage 2's gap check will "
                 f"adjudicate; investigate before curating")

    # -------------------------------------------------------------------- FRED
    fred_entries: dict[str, dict] = {}
    if args.skip_fred:
        ctx.warn("--skip-fred: macro series NOT fetched. Stage 3's macro block will be "
                 "incomplete until this runs without the flag.")
    elif fred_ids:
        try:
            series = fetch_fred_series(
                fred_ids, start=start, retries=cfg.fetch.retries,
                backoff=cfg.fetch.backoff_seconds, log=ctx.log,
            )
        except FetchError as exc:
            raise StageError(f"FRED fetch failed: {exc}") from exc

        for sid, fresh in series.items():
            spec = cfg.fred[sid]
            path = FRED_DIR / f"{sid}.parquet"
            merged = merge_incremental(read_existing(path), fresh) if args.incremental or args.since else fresh
            write_atomic(merged, path)
            fred_entries[sid] = {
                "path": str(path),
                "rows": int(len(merged)),
                "first_observation": str(merged.index[0].date()),
                "last_observation": str(merged.index[-1].date()),
                # Recorded, not applied: the lag is Stage 2's transformation.
                "configured_lag_days": spec.lag_days,
                "freq": spec.freq,
                "kind": spec.kind,
            }

    # ---------------------------------------------------------------- fetch log
    log_path = RAW / "fetch_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # A partial run (--only, --skip-fred) UPDATES the log; it does not replace it.
    # Replacing would erase the record of symbols this invocation never touched, and
    # Stage 2 reads that record to know what exists.
    previous = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    prices_log = {**previous.get("prices", {}), **log_entries}
    fred_log = {**previous.get("fred", {}), **fred_entries}

    payload = {
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": ctx.run_id,
        "start": start,
        "incremental": bool(args.incremental or args.since),
        "partial": bool(args.only or args.skip_fred),
        "calendar": cfg.calendar,
        "prices": prices_log,
        "fred": fred_log,
        "skipped_fred": bool(args.skip_fred),
    }
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ctx.record(n_price_symbols=len(log_entries), n_fred_series=len(fred_entries),
               skipped_fred=bool(args.skip_fred))
    ctx.log(f"wrote {len(log_entries)} price files and {len(fred_entries)} FRED files -> {log_path}")


if __name__ == "__main__":
    raise SystemExit(main())
