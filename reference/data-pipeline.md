# Data Pipeline

Coverage: **2003-01-01 → today**, extended automatically as time passes.

## 1. Trading calendar

Use `exchange_calendars` with the `XNYS` calendar. It is the single source of truth for "is today a trading day" and for the index of every price frame.

```python
import exchange_calendars as xcals

nyse = xcals.get_calendar("XNYS")
sessions = nyse.sessions_in_range("2003-01-01", today)
```

Rules:
- All price/feature frames are reindexed to `sessions`. No weekends, no holidays.
- Half-days (day after Thanksgiving, Christmas Eve, July 3) are **normal sessions**. Do not drop them.
- The daily job checks `nyse.is_session(today)`; if false it exits 0 immediately.
- The N-day hold rule is in **calendar days**, not trading days (per spec). Compute unlock dates with `date + timedelta(days=hold_days)`. The first date on which a sale is legal is the first *session* ≥ the unlock date.

## 2. Price data — yfinance

### Fetch

For every **tradable** ETF (see [etf-universe.md](etf-universe.md), `tradable`) we pull the full **OHLCV** bar — `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume` — not just a
close series. `Open`/`High`/`Low` support execution-price sanity checks (a fill outside the day's range is a bug), and `Volume` feeds the liquidity check in
§8 and any future slippage modeling. Feature-only symbols (indices, FX, rate tickers — anything not in the tradable set) only need `Close`; `yf.download`
still returns OHLCV for them but the extra columns are discarded.

We need **both** close series: `close_adj` (total return, drives every return and feature) and `close_raw` (actual quoted price, used only to convert a dollar
recommendation into a share count, and as the reference for the OHLC sanity check). One call yields everything:

```python
yf.download(
    tickers,  # all tradable + feature_only + market indices
    start="2003-01-01",
    end=None,
    auto_adjust=False,  # keeps BOTH "Close" (raw) and "Adj Close" (total return)
    actions=False,
    group_by="ticker",
    threads=True,
    progress=False,
)
```

**Verified equivalence** (2026-08-31, yfinance 1.6.0): the `Close` column from `auto_adjust=True` and the `Adj Close` column from `auto_adjust=False` are bit-identical (max abs diff 0.0 over TLT, Jan–Jun 2024). So `auto_adjust=False` is strictly more informative and halves the request count. Map
`Adj Close → close_adj` and `Close → close_raw`. For tradable tickers also keep `Open → open_raw`, `High → high_raw`, `Low → low_raw`, `Volume → volume`
unadjusted (OHLC and Volume are never total-return adjusted — only `Adj Close` is).

**OHLC sanity check:** `low_raw <= close_raw <= high_raw` and `low_raw <= open_raw <= high_raw` for every session; a violation fails the data-quality gate rather than being silently trusted.

Total-return adjustment is **mandatory** for `close_adj`. Without it, TLT/LQD/HYG/XLU look permanently terrible (their return is mostly income) and
the agent will rationally never hold them — a pure data artifact. Confirmed on TLT over Jan–Jun 2024: raw 98.31 → 94.18, adjusted 88.05 → 84.62.

### Warm-up buffer

Fetch from **2003-01-01** even though the study period starts 2004-01-01. The 252-day features need a full year of prior data, otherwise the first year of the
study is silently NaN-filled or, worse, computed on a short window.

### Known yfinance hazards

| Hazard | Handling |
|---|---|
| Silent empty frames on transient failure | Assert row count ≥ expected sessions; retry with exponential backoff (3 attempts); fail the job loudly, never write partial data |
| Rate limiting / `YFRateLimitError` | Batch ≤ 20 tickers per call, 2s sleep between batches, cache aggressively so daily refresh pulls only the last ~5 sessions |
| Retroactive adjustment changes | An adjusted history is *rewritten* every time a dividend is paid. So the cached series is not append-only. **Refresh strategy: re-pull the trailing 90 days every day and overwrite; append-only beyond that.** Full re-pull monthly. |
| Ticker symbol quirks | `^VIX`, `^VVIX`, `^TNX`, `^FVX`, `^IRX`, `DX-Y.NYB` — verify each returns data on first backfill and pin them in a test |
| Timezone | yfinance daily bars are date-indexed but sometimes tz-aware. Normalize to naive `date` immediately. |

### Yields are not prices

Do not compute "returns" on a yield series. Use **level** and **level differences**, never log returns. Applies to `^TNX`, `^FVX`, `^IRX` and to every
FRED rate/spread series. Enforced by `config/universe.yaml` (`kind: level`) and tested.

**Scaling — verified, and contrary to the usual folklore.** Many references say `^TNX` quotes ten times the percent yield (42.5 → 4.25%). That is *not* what this pipeline receives. Measured 2026-08-22 against yfinance 1.6.0:

| Symbol | Close on 2024-06-03 | Actual yield that day |
|---|---|---|
| `^TNX` | 4.402 | ~4.40% |
| `^FVX` | 4.417 | ~4.42% |
| `^IRX` | 5.243 | ~5.24% |

So `scale: 1.0`. Applying the folklore 0.1 would have divided every rate signal by ten — a silent, plausible-looking corruption. The data-quality gate
range-checks yields into `[0, 20]` so that a provider-side convention change fails the job instead of quietly poisoning the features.

---

## 3. Macro data — FRED

Free API key from https://fred.stlouisfed.org/docs/api/api_key.html. Store as `FRED_API_KEY`.

```python
from fredapi import Fred

fred = Fred(api_key=os.environ["FRED_API_KEY"])
s = fred.get_series("T10Y2Y", observation_start="2003-01-01")
```

### Publication lag — the lookahead trap

FRED indexes an observation by the **period it describes**, not by when it was published. `CPIAUCSL` for March is dated 2024-03-01 but was published
2024-04-10. Using it on 2024-03-15 is lookahead, and it is the kind that makes a backtest look brilliant.

Two acceptable handling strategies:

**Fixed lags, set from MEASURED delays.** Shift each series forward by a lag that exceeds its true publication delay, then forward-fill onto sessions.

> ### Lags must be set from measured maxima, not release schedules
>
> This was got wrong once and it matters. The original lags (35/45/45 days) were derived from nominal BLS/Fed release calendars. Those calendars describe the *typical* release, which lands near the **median** delay — so the lags leaked on 34–73% of observations.
> 
> Measured 2026-08-22 from ALFRED vintages, 199 monthly observations each (2010–2026), delay from an observation's stamped date to its **first** release:
> 
>| Series | Old lag | Median | p95 | **Max** | Observations leaking |
> |---|---|---|---|---|---|
> | `UNRATE` | 35d | 34d | 38d | **80d** | 98 / 199 |
>| `CPIAUCSL` | 45d | 43d | 50d | **78d** | 68 / 199 |
> | `INDPRO` | 45d | 45d | 48d | **93d** | 145 / 199 |
> 
> The long tails are government-shutdown episodes (late 2025) that suspended statistical releases for weeks. A fixed lag cannot anticipate the next one, so the configured values carry a further buffer beyond the observed maximum.

Applied lags:

| Series | Freq | Applied lag | Basis |
|---|---|---|---|
| `T10Y2Y`, `T10Y3M`, `DGS3MO`, `T5YIE`, `T10YIE` | daily | 1 day | not revised |
| `BAA10Y`, `AAA10Y` | daily | 1 day | not revised |
| `UNRATE` | monthly | **95 days** | measured max 80d + buffer |
| `CPIAUCSL` | monthly | **95 days** | measured max 78d + buffer |
| `INDPRO` | monthly | **110 days** | measured max 93d + buffer |

Daily market-based series are **not revised** — verified against ALFRED, which shows exactly one release per observation for `BAMLH0A0HYM2`. A 1-day lag is
sufficient and honest for those.

> ### Screen every candidate series for revision before adopting it
>
> A publication lag corrects *when* a value becomes visible. It does nothing about *which vintage* is stored. A continuously re-estimated series is not
> made point-in-time correct by any lag, because the problem is revision, not timing.
> 
> Before adding any macro series, measure its revision count from ALFRED:
>
> ```python
>releases = fred.get_series_all_releases(series_id)
> revisions_per_obs = releases.groupby("date").size()
> ```
> 
> One release per observation means the series is safe to lag (all the daily market series here behave this way). A high median revision count means the whole history is being rewritten and the series must be reconstructed from vintages or rejected. Prefer the non-revised market series — VIX, term spreads, HY/IG OAS, breakevens, DXY — which carry the regime signal without this hazard.

> ### Check coverage, not just freshness
>
> A series can be perfectly current and still nearly useless. FRED serves only a rolling **~3-year window** of the licensed ICE BofA credit-spread series:
> measured 2026-08-22, `BAMLH0A0HYM2` returned 787 observations starting 2023-08-22 *even with no start date*. It passed the freshness check while
> covering 18% of the study period. That is worse than having no credit feature: walk-forward folds 1–7 would see nothing and folds 8–9 would suddenly see a strong signal, making the folds incomparable and the aggregate result meaningless.
> 
> Replaced with Moody's `BAA10Y` / `AAA10Y` — genuine credit spreads over the 10-year Treasury, daily since 1990, unrestricted. Over the overlap with the ICE HY OAS they track it at +0.74 on 21-day changes and spike correctly in stress (COVID max 4.31 vs 1.70 in calm markets). The `macro_coverage` quality check now fails any series that does not span the study period.

### Verifying this

`tests/test_providers_live.py::TestPublicationLagAgainstRealData` fetches ALFRED vintages and asserts **zero** observations where the real release delay meets or exceeds the configured lag. Run it after any change to the macro series list:

```powershell
.\.venv\Scripts\python.exe -m pytest -m network -q
```

### Rate limit

FRED allows 120 requests/minute. ~14 series is nothing. Cache to parquet; daily refresh pulls only the trailing 90 days.

---

## 4. Total-return construction

`auto_adjust=True` handles this. But verify it:

- Compute SPY's cumulative return 2010-01-01 → 2024-12-31 from the adjusted series. It should land within ~0.5% of the published S&P 500 **Total Return** index over the same window (~+13.0% annualized), *not* the price index (~+10.9%). If you get the price-index number, adjustment is off.
- Do the same for TLT against its published NAV total return. TLT is the sharpest test because its income share is largest.

Store both `close_adj` (total return, used for all returns/features) and `close_raw` (unadjusted, used only for translating dollars→shares on the
recommendation screen, since you buy at actual quoted prices).

---

## 5. Inception masking

First bar actually available from the provider (verified empirically — a fund's prospectus inception may precede Yahoo's first bar, but data we cannot fetch is data we cannot use):

```yaml
XLRE: 2015-10-08
XLC:  2018-06-19
SHY:  2002-07-22
DBC:  2006-02-03
```

Before a ticker's inception:
- price = `NaN` (never 0, never forward-filled backwards)
- all its features = 0 **and** a companion `is_available_<ticker>` flag = 0
- it is removed from the action mask

Never synthesize pre-inception history. It is tempting (proxy XLRE with a REIT index) and it manufactures a track record that did not exist.

**Consequence to accept:** the model sees a growing action space over time. This is handled by the action mask, and it means early walk-forward folds train on a smaller universe than late ones. That is correct, not a bug.

---

## 6. Missing data policy

| Situation | Policy |
|---|---|
| Single missing session, ticker traded that day elsewhere | Forward-fill **at most 1 session**, flag it |
| ≥2 consecutive missing sessions | Fail the data-quality gate; do not train, do not recommend. Investigate. |
| Missing before inception | Expected; masked |
| Missing macro on a session | Forward-fill (macro is step-function by nature); cap at 10 sessions |
| Suspicious return (\|r\| > 25% for a broad ETF in one day) | Flag for manual review. Legitimate for 2020-03-16 (SPY −12%); a 40% jump is almost certainly a bad split adjustment. |

The daily job **must fail loudly** on a gate violation rather than emit a recommendation from corrupt data. A missed day is recoverable; a recommendation based on garbage is not.

---

## 7. Caching and storage

```
data/
├── raw/ohlcv_{ticker}.csv        # tradable tickers: open/high/low/close_raw/close_adj/volume
├── raw/prices_{ticker}.csv       # feature-only tickers/indices: close_raw/close_adj only
├── raw/macro_{series}.csv
├── curated/ohlcv.csv
├── curated/prices.csv
├── curated/macro.csv
└── curated/features_{version}.csv
```

## 8. Data-quality gate

Runs after every refresh and before every train/recommend. Emits `artifacts/data_quality_{date}.json` and fails the job on any hard violation.

Checks:
1. Every tradable ticker has data through the last completed session.
2. No gaps > 1 session post-inception.
3. First non-NaN date matches the configured inception date (±3 sessions).
4. No duplicate index entries; index is a subset of NYSE sessions.
5. No zero or negative prices.
6. Adjusted series is strictly positive and monotone in the absence of returns.
7. `|daily return| < 0.25` for all broad-market tickers, `< 0.40` for DBC/VWO; violations flagged not failed.
8. Macro series present, and their last observation is not older than
   `lag + 10 days`.
9. Cross-check: correlation(SPY, VTI) over trailing 252 sessions > 0.95. If it is not, one of the two series is corrupted.
10. For every tradable ticker: `low_raw <= {open_raw, close_raw} <= high_raw` on every session post-inception; hard failure on violation.
11. For every tradable ticker: `volume > 0` on every session post-inception (a zero-volume print on a listed ETF means the bar is bad, not that nobody traded it).

---

## 9. Refresh schedule

| Job | When | Scope |
|---|---|---|
| Daily refresh | 22:00 UTC weekdays (GH Action) | trailing 90 sessions, overwrite; then quality gate |
| Weekly deep refresh | Sunday 06:00 UTC | trailing 2 years, overwrite |
| Full rebuild | Monthly, local, before retrain | 2003-01-01 → today |

The trailing-90-day overwrite is what handles yfinance's retroactive dividend re-adjustment. Append-only would silently drift from the true adjusted series.

---

## 10. Lookahead test

```python
def test_no_lookahead():
    full = build_features(prices, macro, as_of=D)
    truncated_prices = prices.loc[:D]  # nothing after D exists
    partial = build_features(truncated_prices, macro.loc[:D], as_of=D)
    assert_frame_equal(full.loc[:D], partial)
```

Plus the stronger version: randomize all data strictly after `D`, rebuild, and assert the feature row at `D` is bit-identical. Any feature that changes is
leaking. Run this over 20 random dates in CI.
