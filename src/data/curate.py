"""Curation: align raw provider data to the trading calendar, and gate its quality.

Produces the artifacts every later stage reads (reference/architecture.md section 4):

* `data/curated/prices.parquet`   -- long, one row per (session, ticker)
* `data/curated/inception.parquet` -- the ONLY source of the availability mask
* `data/curated/macro.parquet`    -- FRED series, publication-lagged to point-in-time
* `data/curated/quality_report.json`

**The quality gate is a gate, not a report.** Stage 2 exits non-zero on a hard violation.
A missed day is recoverable; a model trained on corrupt data is not.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.schema import UniverseConfig
from src.data.calendar import sessions
from src.data.fetch import PRICES_DIR, symbol_to_filename

CURATED = Path("data/curated")
PRICES_PATH = CURATED / "prices.parquet"
INCEPTION_PATH = CURATED / "inception.parquet"
MACRO_PATH = CURATED / "macro.parquet"
QUALITY_PATH = CURATED / "quality_report.json"

PRICE_COLUMNS = [
    "close_adj", "close_raw", "open_raw", "high_raw", "low_raw", "volume",
    "div_per_share", "is_tradable",
]


@dataclass
class Violation:
    check: str
    ticker: str
    detail: str
    hard: bool = True
    sessions: list[str] = field(default_factory=list)

    def key(self) -> tuple[str, str]:
        return (self.ticker, self.check)


# ------------------------------------------------------------------------- load


def load_raw_prices(symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for symbol in symbols:
        path = PRICES_DIR / f"{symbol_to_filename(symbol)}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"missing raw price file for {symbol}: {path}. "
                f"Run: python -m scripts.s01_fetch_data --config config/universe.yaml"
            )
        out[symbol] = pd.read_parquet(path)
    return out


# ---------------------------------------------------------- dividend derivation


def implied_dividend_per_share(
    close_adj: pd.Series, close_raw: pd.Series, epsilon: float = 1e-10
) -> pd.Series:
    """The cash distribution per share implied by the gap between the two series.

    This is what makes a raw-price ledger produce a total-return NAV: the ledger holds
    real share counts at real quoted prices, and total return arrives through share
    accretion when the distribution is reinvested
    (reference/portfolio-ledger.md section 3).

    Split-neutral by construction -- both series are ratios of consecutive values within
    themselves, and yfinance applies splits to both consistently.
    """
    tr = close_adj / close_adj.shift(1) - 1.0   # total return -- the ground truth
    pr = close_raw / close_raw.shift(1) - 1.0   # price return
    div = (tr - pr) * close_raw.shift(1)
    # ~0 on the overwhelming majority of sessions; positive on ex-dividend dates.
    div = div.where(div.abs() > epsilon, 0.0)
    return div


# --------------------------------------------------------------------- assemble


def curate_prices(
    cfg: UniverseConfig, raw: dict[str, pd.DataFrame], session_index: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame, list[Violation]]:
    """Reindex every symbol onto the session calendar and derive inception.

    Returns (long prices frame, inception frame, violations).
    """
    tradable = set(cfg.tradable_tickers)
    violations: list[Violation] = []
    rows: list[pd.DataFrame] = []
    inception: list[dict] = []

    for symbol, df in raw.items():
        is_tradable = symbol in tradable

        # Provider rows outside the exchange calendar are dropped, not kept: FX and
        # index feeds print on days XNYS is closed, and those rows are not sessions.
        extra = df.index.difference(session_index)
        if len(extra):
            violations.append(Violation(
                "off_calendar_rows", symbol,
                f"{len(extra)} provider rows fell outside {cfg.calendar} sessions; dropped",
                hard=False, sessions=[str(d.date()) for d in extra[:5]],
            ))
        df = df.loc[df.index.intersection(session_index)]

        if df.empty:
            violations.append(Violation("empty", symbol, "no rows on the session calendar"))
            continue

        first_session = df["close_raw"].first_valid_index()
        if first_session is None:
            violations.append(Violation("all_nan", symbol, "close_raw is entirely NaN"))
            continue
        inception.append({"ticker": symbol, "first_session": first_session,
                          "is_tradable": is_tradable})

        # Reindex only over the symbol's own live range. Pre-inception stays ABSENT --
        # never zero-filled, never back-filled (reference/data-pipeline.md section 5).
        live = session_index[session_index >= first_session]
        out = df.reindex(live)

        gaps = out["close_raw"].isna()
        if gaps.any():
            runs = _consecutive_runs(gaps)
            long_runs = [r for r in runs if len(r) > cfg.quality.max_forward_fill_sessions]
            if long_runs:
                violations.append(Violation(
                    "gap", symbol,
                    f"{len(long_runs)} gap(s) longer than "
                    f"{cfg.quality.max_forward_fill_sessions} session(s); longest "
                    f"{max(len(r) for r in long_runs)}",
                    sessions=[str(r[0].date()) for r in long_runs[:5]],
                ))
            else:
                violations.append(Violation(
                    "gap_filled", symbol,
                    f"{int(gaps.sum())} single-session gap(s) forward-filled",
                    hard=False, sessions=[str(d.date()) for d in out.index[gaps][:5]],
                ))
                out = out.ffill(limit=cfg.quality.max_forward_fill_sessions)

        out["div_per_share"] = implied_dividend_per_share(out["close_adj"], out["close_raw"])
        out["is_tradable"] = is_tradable
        for col in PRICE_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
        out = out[PRICE_COLUMNS]
        out.index.name = "session"
        out = out.assign(ticker=symbol).reset_index()
        rows.append(out)

    prices = pd.concat(rows, ignore_index=True)
    prices["ticker"] = prices["ticker"].astype("category")
    prices["is_tradable"] = prices["is_tradable"].astype(bool)
    prices = prices.sort_values(["session", "ticker"]).reset_index(drop=True)

    inception_df = pd.DataFrame(inception).sort_values("ticker").reset_index(drop=True)
    return prices, inception_df, violations


def _consecutive_runs(mask: pd.Series) -> list[pd.DatetimeIndex]:
    """Group a boolean mask into runs of consecutive True positions."""
    if not mask.any():
        return []
    positions = np.flatnonzero(mask.to_numpy())
    runs, current = [], [mask.index[positions[0]]]
    for prev, cur in zip(positions, positions[1:]):
        if cur == prev + 1:
            current.append(mask.index[cur])
        else:
            runs.append(pd.DatetimeIndex(current))
            current = [mask.index[cur]]
    runs.append(pd.DatetimeIndex(current))
    return runs


# ----------------------------------------------------------------- quality gate


def run_quality_gate(
    cfg: UniverseConfig, prices: pd.DataFrame, inception: pd.DataFrame,
    session_index: pd.DatetimeIndex,
) -> list[Violation]:
    """Every check in reference/data-pipeline.md section 8 that the price data can answer."""
    v: list[Violation] = []
    q = cfg.quality
    tradable = set(cfg.tradable_tickers)
    levels = cfg.level_symbols

    for ticker, g in prices.groupby("ticker", observed=True):
        g = g.set_index("session").sort_index()
        sym = str(ticker)
        # A level series is not a price. Positivity and return checks are meaningless on
        # one -- ^IRX closed NEGATIVE for seven sessions in March 2020 (bills traded
        # through zero in the flight to quality) and VIX routinely doubles in a day.
        # Applying a price check to a level is the same class of error as computing a
        # log return on a yield (reference/features.md rule 3).
        is_level = sym in levels

        if g.index.has_duplicates:
            v.append(Violation("duplicate_sessions", sym, "duplicate session index entries"))

        # 5. No zero or negative prices. Prices only.
        if not is_level:
            for col in ("close_raw", "close_adj", "open_raw", "high_raw", "low_raw"):
                if col not in g or g[col].isna().all():
                    continue
                bad = g.index[(g[col] <= 0).fillna(False)]
                if len(bad):
                    v.append(Violation("non_positive_price", sym,
                                       f"{col} <= 0 on {len(bad)} session(s)",
                                       sessions=[str(d.date()) for d in bad[:5]]))

        if sym in tradable:
            # 10. The execution-price sanity check, at the data layer. A fill outside the
            # day's range is a data bug, and T15 asserts the same thing at execution.
            lo, hi = g["low_raw"], g["high_raw"]
            for col in ("open_raw", "close_raw"):
                bad = g.index[((g[col] < lo - 1e-9) | (g[col] > hi + 1e-9)).fillna(False)]
                if len(bad):
                    v.append(Violation("ohlc_range", sym,
                                       f"{col} outside [low_raw, high_raw] on {len(bad)} session(s)",
                                       sessions=[str(d.date()) for d in bad[:5]]))
            bad = g.index[(hi < lo - 1e-9).fillna(False)]
            if len(bad):
                v.append(Violation("ohlc_range", sym, f"high_raw < low_raw on {len(bad)} session(s)",
                                   sessions=[str(d.date()) for d in bad[:5]]))

            # 11. A zero-volume print on a listed ETF means the bar is bad, not that
            # nobody traded it. Stage 2 is the ONLY place volume gaps are adjudicated:
            # an isolated bad print is forward-fillable and reported as a warning, but a
            # run at or beyond the forward-fill limit is a hard failure, exactly as for
            # any other run of missing sessions. The feature layer relies on this having
            # already been decided here.
            if q.require_positive_volume:
                zero = (g["volume"] <= 0).fillna(False)
                if zero.any():
                    runs = _consecutive_runs(zero)
                    longest = max(len(r) for r in runs)
                    bad = g.index[zero]
                    v.append(Violation(
                        "zero_volume", sym,
                        f"volume <= 0 on {int(zero.sum())} session(s); longest run "
                        f"{longest} (forward-fill limit "
                        f"{q.max_forward_fill_sessions})",
                        hard=longest > q.max_forward_fill_sessions,
                        sessions=[str(d.date()) for d in bad[:5]]))

            # A negative implied distribution is not physically meaningful for these
            # ETFs; it means an adjustment or split-handling problem, and it is failed
            # rather than clipped (reference/portfolio-ledger.md section 3).
            tol = -1e-4 * g["close_raw"]
            bad = g.index[(g["div_per_share"] < tol).fillna(False)]
            if len(bad):
                worst = (g["div_per_share"] / g["close_raw"]).min()
                v.append(Violation("negative_dividend", sym,
                                   f"implied dividend negative on {len(bad)} session(s); "
                                   f"worst {worst:.2e} of price",
                                   sessions=[str(d.date()) for d in bad[:5]]))

        # 7. Suspicious returns. Flagged at the ordinary threshold, FAILED at the
        # split-jump threshold -- a 40% one-day move is almost certainly a bad split.
        # Never applied to a level series: a 40% move in VIX is a Tuesday.
        if not is_level and not g["close_adj"].isna().all():
            ret = g["close_adj"].pct_change()
            limit = (q.max_abs_daily_return_volatile if sym in q.volatile_tickers
                     else q.max_abs_daily_return)
            flagged = g.index[(ret.abs() > limit).fillna(False)]
            if len(flagged):
                v.append(Violation("large_return", sym,
                                   f"|return| > {limit:.0%} on {len(flagged)} session(s)",
                                   hard=False, sessions=[str(d.date()) for d in flagged[:5]]))
            jumps = g.index[(ret.abs() > q.split_jump_threshold).fillna(False)]
            if len(jumps):
                v.append(Violation("split_jump", sym,
                                   f"|return| > {q.split_jump_threshold:.0%} on "
                                   f"{len(jumps)} session(s), unexplained by a known split",
                                   sessions=[str(d.date()) for d in jumps[:5]]))

    # The dividend derivation is the load-bearing one, so it is verified here rather
    # than trusted until Stage 4. See `total_return_error`.
    v += _total_return_check(cfg, prices)

    # 3. First non-NaN date matches the configured inception pin, within tolerance.
    first = inception.set_index("ticker")["first_session"]
    for ticker, pinned in cfg.inception_pins.items():
        if ticker not in first.index:
            v.append(Violation("inception_pin", ticker, "pinned ticker absent from the data"))
            continue
        actual = pd.Timestamp(first[ticker])
        want = pd.Timestamp(pinned)

        # A pin EARLIER than history_start means the fetch window, not inception,
        # determines the first bar -- SHY launched 2002-07-22 and we fetch from 2003.
        # Comparing the two directly would either fail spuriously or, worse, pass by
        # accident because the session index has no sessions in between, which would let
        # a genuinely truncated series through unnoticed.
        if want < session_index[0]:
            expected = session_index[0]
            n = len(session_index[(session_index >= expected) & (session_index < actual)])
            if n > cfg.inception_tolerance_sessions:
                v.append(Violation("inception_pin", ticker,
                                   f"pinned inception {want.date()} predates history_start, "
                                   f"so the first bar should be {expected.date()}; it is "
                                   f"{actual.date()}, {n} sessions later"))
            continue

        # Tolerance is in sessions, so measure it in sessions, not calendar days.
        lo, hi = min(actual, want), max(actual, want)
        n = len(session_index[(session_index > lo) & (session_index <= hi)])
        if n > cfg.inception_tolerance_sessions:
            v.append(Violation("inception_pin", ticker,
                               f"first bar {actual.date()} is {n} sessions from the "
                               f"pinned {want.date()} (tolerance "
                               f"{cfg.inception_tolerance_sessions})"))

    # 9. Cross-check: two series tracking the same market must agree, or one is corrupt.
    v += _spy_vti_check(cfg, prices)

    # A provider convention change must fail loudly, not divide every rate signal by ten
    # in silence (reference/data-pipeline.md section 2). The band is per-symbol because
    # a VIX band and a yield band are not the same band.
    for sym in sorted(cfg.level_symbols):
        g = prices[prices["ticker"] == sym]
        if g.empty:
            continue
        s = g["close_raw"].dropna()
        if s.empty:
            continue
        lo_l, hi_l = cfg.feature_only[sym].range or cfg.quality.default_level_range
        if s.min() < lo_l or s.max() > hi_l:
            v.append(Violation("level_range", sym,
                               f"level range [{s.min():.3f}, {s.max():.3f}] outside "
                               f"[{lo_l}, {hi_l}] -- suspect a provider scale change"))
    return v


#: Tolerance for the total-return reconstruction, set from MEASURED provider precision,
#: not from float epsilon. yfinance rounds `Adj Close` to ~6 significant digits, so each
#: of the ~280 distribution events over 24 years injects ~1e-6 of relative noise into
#: the derived dividend. Measured 2026-09-01 across all 24 tradables: worst 9.2e-4 (SHY),
#: best 0.0 (GLD, which pays nothing). 5e-3 leaves headroom over the observed worst case
#: while still failing loudly if the adjustment itself breaks.
TOTAL_RETURN_TOLERANCE = 5e-3


def total_return_error(g: pd.DataFrame) -> float:
    """Max relative gap between a reinvesting share ledger and `close_adj`.

    Buy one share, reinvest every implied distribution at that session's raw close, and
    the resulting NAV must track the total-return series. This is T1
    (reference/testing.md) measured at the data layer: if TLT does not reconstruct, the
    dividend derivation is wrong and everything downstream inherits it.
    """
    raw = g["close_raw"].to_numpy(dtype=float)
    adj = g["close_adj"].to_numpy(dtype=float)
    div = np.nan_to_num(g["div_per_share"].to_numpy(dtype=float))
    if len(raw) < 2 or not np.isfinite(raw[0]) or raw[0] <= 0:
        return float("nan")

    shares = 1.0
    nav = np.empty(len(raw))
    for i in range(len(raw)):
        if i > 0 and div[i] > 0:
            shares += shares * div[i] / raw[i]
        nav[i] = shares * raw[i]
    return float(np.nanmax(np.abs((nav / nav[0]) / (adj / adj[0]) - 1.0)))


def _total_return_check(cfg: UniverseConfig, prices: pd.DataFrame) -> list[Violation]:
    v: list[Violation] = []
    for ticker in cfg.tradable_tickers:
        g = prices[prices["ticker"] == ticker].set_index("session").sort_index()
        if g.empty:
            continue
        err = total_return_error(g)
        if not np.isfinite(err) or err > TOTAL_RETURN_TOLERANCE:
            v.append(Violation(
                "total_return_reconstruction", ticker,
                f"reinvesting ledger diverges from close_adj by {err:.2e} "
                f"(tolerance {TOTAL_RETURN_TOLERANCE:.0e}) -- the dividend derivation is "
                f"wrong, and every downstream NAV inherits it",
            ))
    return v


def _spy_vti_check(cfg: UniverseConfig, prices: pd.DataFrame) -> list[Violation]:
    wide = prices.pivot_table(index="session", columns="ticker", values="close_adj",
                              observed=True)
    if "SPY" not in wide or "VTI" not in wide:
        return [Violation("spy_vti_corr", "SPY/VTI",
                          "VTI is absent; the corroboration check cannot run", hard=False)]
    r = wide[["SPY", "VTI"]].pct_change().dropna()
    if len(r) < cfg.quality.spy_vti_correlation_window:
        return []
    corr = r["SPY"].tail(cfg.quality.spy_vti_correlation_window).corr(
        r["VTI"].tail(cfg.quality.spy_vti_correlation_window))
    if corr < cfg.quality.spy_vti_min_correlation:
        return [Violation("spy_vti_corr", "SPY/VTI",
                          f"trailing-{cfg.quality.spy_vti_correlation_window} correlation "
                          f"{corr:.4f} < {cfg.quality.spy_vti_min_correlation} -- one of "
                          f"the two series is corrupted")]
    return []


# -------------------------------------------------------------------- macro/PIT


def curate_macro(
    cfg: UniverseConfig, raw_fred: dict[str, pd.DataFrame],
    session_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[Violation]]:
    """Shift each FRED series by its publication lag, then align to sessions.

    FRED indexes an observation by the period it *describes*, not by when it was
    published: CPI for March is dated 2024-03-01 and was released 2024-04-10. Using it
    on 2024-03-15 is lookahead, and it is the flavour that makes a backtest look
    brilliant. The lags are measured release maxima plus a buffer -- a nominal release
    calendar describes the median delay and leaks on roughly half of all observations
    (reference/data-pipeline.md section 3).
    """
    v: list[Violation] = []
    cols: dict[str, pd.Series] = {}

    for sid, spec in cfg.fred.items():
        if sid not in raw_fred:
            v.append(Violation("macro_missing", sid,
                               "no raw file; run Stage 1 without --skip-fred"))
            continue
        s = raw_fred[sid]["value"].dropna()
        if s.empty:
            v.append(Violation("macro_empty", sid, "series is empty"))
            continue

        # The value becomes VISIBLE lag_days after the date it is stamped with.
        visible = s.copy()
        visible.index = visible.index + pd.Timedelta(days=spec.lag_days)
        # Reindex onto the union so an observation released between sessions is not lost,
        # then take sessions only. Macro is a step function; forward-fill is correct, but
        # it is capped so a dead feed shows up as NaN rather than a stale constant.
        union = session_index.union(pd.DatetimeIndex(visible.index)).sort_values()
        aligned = visible.reindex(union).ffill(
            limit=None if spec.freq == "monthly" else cfg.quality.max_macro_forward_fill_sessions
        ).reindex(session_index)
        cols[sid] = aligned

        # Coverage, not just freshness. A series can be perfectly current and cover 18%
        # of the study period, which is worse than not having it: early folds would see
        # nothing and late folds a strong signal, making the folds incomparable.
        in_study = aligned.loc[aligned.index >= pd.Timestamp(cfg.study_start)]
        coverage = float(in_study.notna().mean()) if len(in_study) else 0.0
        if coverage < 0.95:
            v.append(Violation("macro_coverage", sid,
                               f"covers {coverage:.1%} of the study period; a series that "
                               f"does not span it makes walk-forward folds incomparable"))

        # 8. Staleness: the last observation must be recent enough to be believable.
        last_obs = s.index[-1].date()
        age = (dt.date.today() - last_obs).days
        allowed = spec.lag_days + cfg.quality.macro_staleness_buffer_days + (
            35 if spec.freq == "monthly" else 5)
        if age > allowed:
            v.append(Violation("macro_stale", sid,
                               f"last observation {last_obs} is {age}d old (allowed {allowed}d)"))

    macro = pd.DataFrame(cols, index=session_index)
    macro.index.name = "session"
    return macro, v
