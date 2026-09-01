"""Stage 2 -- curate.

    python -m scripts.s02_curate_data --config config/universe.yaml

Reindexes everything to the exchange calendar, derives the inception table (the only
source of the availability mask), derives the implied dividend stream, aligns the macro
series to point-in-time, and runs the quality gate.

**The gate is a gate.** A hard violation exits non-zero and nothing downstream runs.
Waivers go in `config/universe.yaml` under `known_exceptions`, with a reason.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict

import pandas as pd

from src.cli.stage import StageContext, StageError, stage
from src.config.schema import UniverseConfig
from src.data.calendar import sessions
from src.data.curate import (
    CURATED,
    INCEPTION_PATH,
    MACRO_PATH,
    PRICES_PATH,
    QUALITY_PATH,
    Violation,
    curate_macro,
    curate_prices,
    load_raw_prices,
    run_quality_gate,
    total_return_error,
)
from src.data.fetch import FRED_DIR, write_atomic


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--allow-missing-macro", action="store_true",
                   help="downgrade absent FRED series to warnings (no key yet); the "
                        "macro feature block will be incomplete")


@stage(
    name="s02_curate_data",
    config_default="config/universe.yaml",
    config_cls=UniverseConfig,
    inputs=["data/raw/fetch_log.json"],
    outputs=[
        "data/curated/prices.parquet",
        "data/curated/inception.parquet",
        "data/curated/quality_report.json",
    ],
    upstream="s01_fetch_data",
    add_args=_add_args,
)
def main(cfg: UniverseConfig, ctx: StageContext) -> None:
    """Align raw data to the session calendar and gate its quality."""
    session_index = sessions(cfg.history_start, dt.date.today(), calendar=cfg.calendar)
    ctx.log(f"{len(session_index)} sessions {session_index[0].date()} .. {session_index[-1].date()}")

    raw = load_raw_prices(cfg.all_yfinance_symbols)
    prices, inception, violations = curate_prices(cfg, raw, session_index)
    ctx.log(f"curated {len(prices):,} (session, ticker) rows over "
            f"{prices['ticker'].nunique()} symbols")

    violations += run_quality_gate(cfg, prices, inception, session_index)

    # ------------------------------------------------------------------- macro
    raw_fred = {}
    for sid in cfg.fred:
        path = FRED_DIR / f"{sid}.parquet"
        if path.exists():
            raw_fred[sid] = pd.read_parquet(path)
    macro, macro_violations = curate_macro(cfg, raw_fred, session_index)
    if ctx.args.allow_missing_macro:
        for mv in macro_violations:
            if mv.check in ("macro_missing", "macro_empty"):
                mv.hard = False
    violations += macro_violations
    ctx.log(f"macro block: {macro.shape[1]}/{len(cfg.fred)} series aligned point-in-time")

    # ---------------------------------------------------------------- waivers
    waived_keys = {(w.ticker, w.check) for w in cfg.known_exceptions}
    waived = [v for v in violations if v.key() in waived_keys]
    for v in waived:
        v.hard = False

    hard = [v for v in violations if v.hard]
    soft = [v for v in violations if not v.hard]

    for v in violations:
        line = f"{'FAIL' if v.hard else 'warn'} {v.check:<20} {v.ticker:<10} {v.detail}"
        if v.sessions:
            line += f"  e.g. {', '.join(v.sessions)}"
        ctx.log(line, level="ERROR" if v.hard else "WARN")

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": ctx.run_id,
        "n_sessions": len(session_index),
        "n_symbols": int(prices["ticker"].nunique()),
        "n_rows": int(len(prices)),
        "n_hard_violations": len(hard),
        "n_soft_violations": len(soft),
        "n_waived": len(waived),
        "violations": [asdict(v) for v in violations],
        "inception": {r.ticker: str(pd.Timestamp(r.first_session).date())
                      for r in inception.itertuples()},
        "macro_series": list(macro.columns),
        # Recorded every run: T1 (Stage 4) asserts the same property against the real
        # ledger, and a drift here is the earliest possible warning.
        "total_return_error": {
            t: round(total_return_error(
                prices[prices["ticker"] == t].set_index("session").sort_index()), 9)
            for t in cfg.tradable_tickers
        },
    }
    CURATED.mkdir(parents=True, exist_ok=True)
    QUALITY_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    ctx.record(n_hard_violations=len(hard), n_soft_violations=len(soft),
               n_waived=len(waived), n_rows=int(len(prices)))

    if hard:
        names = sorted({f"{v.ticker}:{v.check}" for v in hard})
        raise StageError(
            f"quality gate FAILED with {len(hard)} hard violation(s): {names}. "
            f"Report: {QUALITY_PATH}. Waive one only by adding it to `known_exceptions` "
            f"in config/universe.yaml, with a reason."
        )

    # Written only after the gate passes: a downstream stage must never be able to read
    # curated data that failed validation.
    write_atomic(prices.set_index(["session", "ticker"]), PRICES_PATH)
    write_atomic(inception.set_index("ticker"), INCEPTION_PATH)
    write_atomic(macro, MACRO_PATH)

    ctx.log(f"gate passed ({len(soft)} warning(s)) -> {PRICES_PATH}, {INCEPTION_PATH}, {MACRO_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())
