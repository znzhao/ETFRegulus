# ETF Universe

## Instructions

**Inception dates are hard constraints.** A ticker is masked out of the action set on every date before its first trading day.

## Tradeable set

### 1. US Broad Equity (3)

| Ticker | Name | Inception | Role |
|---|---|---|---|
| SPY | SPDR S&P 500 ETF Trust | 1993-01 | Core US large-cap beta; default benchmark |
| QQQ | Invesco QQQ Trust | 1999-03 | Growth / long-duration equity; Nasdaq-100 |
| IWM | iShares Russell 2000 ETF | 2000-05 | Small-cap; domestic cycle and credit sensitivity |

### 2. US Sector Equity (11) — Select Sector SPDR family

| Ticker | Sector | Inception | Primary driver |
|---|---|---|---|
| XLK | Technology | 1998-12 | Growth, semis/AI capex, valuation multiples |
| XLC | Communication Services | 2018-06 | Digital platforms, advertising cycle |
| XLY | Consumer Discretionary | 1998-12 | Consumer spending, cyclical demand |
| XLP | Consumer Staples | 1998-12 | Defensive; risk-appetite gauge vs XLY |
| XLE | Energy | 1998-12 | Oil & gas, commodity cycle, inflation |
| XLF | Financials | 1998-12 | Yield curve, credit cycle, capital markets |
| XLI | Industrials | 1998-12 | Capex, manufacturing, transportation |
| XLB | Materials | 1998-12 | Metals/chemicals, global growth, inflation |
| XLV | Health Care | 1998-12 | Defensive growth; pharma/biotech |
| XLU | Utilities | 1998-12 | Defensive + rate-sensitive (bond proxy) |
| XLRE | Real Estate | 2015-10 | REITs; rates + financing conditions |

### 3. Treasury / Interest Rate (4)

| Ticker | Name | Inception | Role |
|---|---|---|---|
| SHY | iShares 1-3Y Treasury | 2002-07 | Short end of the curve; cash-like but rate-sensitive |
| IEF | iShares 7-10Y Treasury | 2002-07 | Belly of the curve; growth/inflation expectations |
| TLT | iShares 20+Y Treasury | 2002-07 | Long duration. **Not a "safe asset"** — 2022 drawdown was −31% |
| TIP | iShares TIPS Bond | 2003-12 | Real rates; separates nominal from inflation |

### 4. Credit (2)

| Ticker | Name | Inception | Role |
|---|---|---|---|
| LQD | iShares IG Corporate Bond | 2002-07 | IG spread + duration |
| HYG | iShares High Yield Corporate | 2007-04 | Default risk, liquidity, risk-on/off |

### 5. Commodities / Real Assets (2)

| Ticker | Name | Inception | Role |
|---|---|---|---|
| GLD | SPDR Gold Shares | 2004-11 | Real rates, USD, safe-haven, geopolitical |
| DBC | Invesco DB Commodity Index Tracking Fund | 2006-02 | Broad energy/ag/metals commodity beta; inflation regime |

### 6. International Equity (2)

| Ticker | Name | Inception | Role |
|---|---|---|---|
| VEA | Vanguard FTSE Developed Markets | 2007-07 | Europe/Japan/Canada; USD sensitivity |
| VWO | Vanguard FTSE Emerging Markets | 2005-03 | China/Asia, global liquidity, commodities |

### 7. CASH (synthetic, always available)

- Return: **0.00% per day** (zero risk, zero return).
- Volatility 0; correlation with everything 0.
- **No holding-period lock.** Deployable or receivable on any day.
- This is the agent's outside option.

## Non-ETF exogenous features

Free daily/weekly sources. Full spec in `data-pipeline.md`.

| Symbol | Source | Freq | Signal |
|---|---|---|---|
| `^VIX` | yfinance | daily | Equity implied vol / fear |
| `^VVIX` | yfinance | daily | Vol-of-vol; tail stress |
| `^TNX` `^FVX` `^IRX` | yfinance | daily | 10Y / 5Y / 13W yields |
| `DX-Y.NYB` | yfinance | daily | US dollar index |
| `T10Y2Y` | FRED | daily | Term spread |
| `T10Y3M` | FRED | daily | Term spread (better recession predictor) |
| `BAA10Y` | FRED | daily | Moody's Baa credit spread over 10y Treasury |
| `AAA10Y` | FRED | daily | Moody's Aaa spread over 10y Treasury (quality control) |
| `DGS3MO` | FRED | daily | 3-month T-bill |
| `T5YIE` `T10YIE` | FRED | daily | Breakeven inflation |
| `UNRATE` `CPIAUCSL` `INDPRO` | FRED | monthly | Macro state |

> **Critical:** monthly macro series are published with a lag and are **revised**. They must be lagged by their real publication delay (or pulled from ALFRED
> vintages) or they inject lookahead bias. Daily market series (VIX, yields, spreads) are not revised and need only same-day-close alignment.

---

## Machine-readable form

Canonical definition lives in `config/universe.yaml`; this document is the rationale. Keep them in sync — CI asserts equality of the ticker sets.

```yaml
tradable:
  broad_equity:   [SPY, QQQ, IWM]
  sector_equity:  [XLK, XLC, XLY, XLP, XLE, XLF, XLI, XLB, XLV, XLU, XLRE]
  treasury:       [SHY, IEF, TLT, TIP]
  credit:         [LQD, HYG]
  real_assets:    [GLD, DBC]
  international:  [VEA, VWO]
  synthetic:      [CASH]
```
