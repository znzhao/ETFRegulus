# Feature Engineering

Per-ETF OHLCV, cross-sectional, macro/regime, and portfolio features. Built by [Stage 3](stages.md) into
`data/features/`, except the portfolio block (§4), which is computed live in the environment.

---

## 0. Rules that override every individual feature

1. **Point-in-time only.** A feature at session `t` uses data through `t`'s close and nothing after. Tested by
   perturbing future data and asserting no past feature changes.
2. **Returns come from `close_adj`; execution comes from `close_raw`.** Never mix. Total-return adjustment is
   mandatory or the bond ETFs look permanently terrible ([data-pipeline.md](data-pipeline.md)).
3. **Yields are levels, not prices.** `^TNX`, `^FVX`, `^IRX` and every FRED rate/spread use levels and level
   *differences*. Computing a log return on a yield is a bug, guarded by `kind: level` in
   `config/universe.yaml` and by a test.
4. **No full-sample scaling.** It is explicitly prohibited. Scalers are fitted on a fold's training window only
   and stored per fold ([architecture.md](architecture.md)).
5. **Pre-inception assets are excluded, not zero-filled.** They are absent from cross-sectional ranks and
   masked in the observation. Zero-filling would place a non-existent ETF at a meaningful rank.
6. **Every division guards its denominator.** Most acute for the OHLC ratios, but the rule applies
   everywhere. Denominators are clipped at an epsilon and the guard is tested.
7. **Warm-up is real.** History starts 2003-01-01, the study starts 2004-01-01, so a 252-day feature has a
   full window from the first study session. NaN counts must match the expected warm-up pattern exactly —
   any other NaN is a bug, not something to fill.
8. **Never forward-fill across an inception boundary.**

---

## 1. Per-ETF OHLCV features

Computed per tradable ticker, indexed `(session, ticker)`, written to `data/features/etf.parquet`.

| Group | Features | Windows |
|---|---|---|
| Return | arithmetic, log, cumulative | 1, 2, 5, 10, 21, 63, 126, 252 |
| Volatility | rolling std of log returns, annualized | 5, 10, 21, 63, 126, 252 |
| Drawdown | drawdown from rolling high, distance from rolling max | 21, 63, 252 |
| OHLC | `(H-L)/C`, `(C-O)/O`, `(C-L)/(H-L)` | daily |
| Volume | `vol / rolling_mean(vol)`, volume z-score, volume change | 21, 63 |
| Trend | `close / SMA(w)`, SMA slope | 20, 50, 100, 200 |
| Technical | RSI(14), MACD line/signal/histogram, ATR-normalized range, Bollinger %B | standard |

Stacking many highly correlated indicators at the start is a known trap. The list above is deliberately a
*limited* set. Adding indicators is an experiment with a config change and a recorded ablation,
not a default.

**Diagnostic to run in Stage 3:** the pairwise correlation matrix of the feature block, reported so redundancy
is visible. High correlation is not automatically wrong, but it should be a decision, not an accident.

---

## 2. Cross-sectional features

Per session, across the assets **available that day**. Written to `data/features/cross_sectional.parquet`.

- Return rank, volatility rank, momentum rank, drawdown rank
- Relative strength vs. SPY and vs. the available-universe mean
- Asset-class group rank (groups from [etf-universe.md](etf-universe.md): US broad, sector, treasury, credit,
  commodity, international)

**The subtlety that makes or breaks this block:** the universe expands over time as inception dates pass — 
XLRE in 2015, XLC in 2018. So ranks must be computed over the available set only, and normalized to a
comparable scale (percentile in `[0,1]`) so that a rank means the same thing in 2004 with 20 assets as in 2020
with 25. A raw ordinal rank would drift in meaning across the sample and the agent would learn the drift.

The availability mask is itself a model input, so the agent can tell a small universe from a large one.

---

## 3. Macro / regime features

From the feature-only symbols and FRED. Written to `data/features/macro.parquet`.

| Source | Features |
|---|---|
| yfinance | VIX level and term structure, VVIX, `^TNX`/`^FVX`/`^IRX` levels and differences, DXY |
| Derived | Term spreads (10y-2y, 10y-3m), credit spreads (HYG-IEF, LQD-IEF proxies), breakeven inflation |
| FRED monthly | UNRATE, CPIAUCSL, INDPRO |

**Publication lag is the whole game for the monthly series.** UNRATE, CPIAUCSL and INDPRO are published weeks
after the month they describe. Giving the agent a value on its *observation* date rather than its *release*
date is a lookahead leak that would be nearly invisible in results and would flatter every regime-detection
claim in the project.

Therefore:

- The curated layer applies the conservative lags already specified in the data pipeline, based on measured
  maximum release delays.
- **The feature builder consumes the already point-in-time-aligned curated series. It never calls FRED
  directly.** This is enforced by keeping the FRED client out of `src/features/` entirely.
- A dedicated test asserts that each monthly series' value at session `t` was publicly available at `t`.

Regime features (rolling volatility regime, trend regime, correlation regime) are derived from the above and
are subject to the same point-in-time discipline.

---

## 4. Portfolio features — computed live, not in Stage 3

These depend on the current portfolio, so they are built by `src/env/state_builder.py` at every step. They are
listed here because they are part of the observation and of the Markov sufficiency requirement.

**Per asset:**

- current weight
- position indicator (`shares > 0`)
- locked indicator
- lock remaining calendar days
- unlock proximity (a normalized `1 - remaining/N` style signal, 0 when unlocked)
- current unrealized value fraction

**Portfolio level:**

- cash weight
- NAV, and normalized NAV (relative to episode start)
- running peak, normalized
- current drawdown `D_t`
- **remaining drawdown budget `B_t = D_max - D_t`** — mandatory, not optional
- number of locked positions
- fraction of NAV locked
- weighted average lock remaining days

**Plus the conditioning parameters:** `N` and `D_max` themselves, normalized.

The rule that nothing required for the Markov property may be missing from state makes this list a floor, not
a menu. The observation must be sufficient to reconstruct transition feasibility, which means the availability
mask, the lock mask, and the headroom are all mandatory. See [env-mdp.md](env-mdp.md) for the assembled
observation vector and its layout.

---

## 5. Scaling

- Per-feature scaling statistics are fitted on the **training window of a fold only** and stored as
  `data/features/scalers/<fold_id>.json`.
- Default: robust scaling (median / IQR), which handles the fat tails in daily financial features far better
  than mean/std. Configurable.
- Bounded features already in `[0,1]` (percentile ranks, `%B`, masks) are left alone.
- Clip scaled values to a configured range (default `±10`) so a single crisis observation cannot dominate a
  batch. Clipping events are counted and reported rather than silent.
- Observations are asserted finite before entering the network. A NaN reaching the policy is a hard error, not
  something to `nan_to_num` away.

---

## 6. Feature manifest

Stage 3 emits `data/features/feature_manifest.json`: for every column, its name, source, lookback in sessions,
kind (`level` / `return` / `rank` / `mask`), scaling treatment, and expected NaN warm-up count. The observation
builder validates the live observation against this manifest at construction, so a feature added upstream
without updating the environment fails loudly instead of shifting every index in the observation vector by one.
