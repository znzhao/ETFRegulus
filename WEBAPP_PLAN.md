# Web App Plan: Daily ETF Recommendations

A step-by-step guide to building a public, multi-page web app. The app shows, every trading day,
which ETFs to hold under each baseline strategy, and the backtest history behind each one. It is
hosted on GitHub Pages and updated by a scheduled GitHub Actions job.

Status: planning, written 2026-10-10. Nothing below is built yet. RL training is paused at
`MODEL_REDESIGN_PLAN.md` §14 and is independent of this plan.

---

## 0. Decisions already made

| # | Question | Decision |
|---|---|---|
| W-1 | Hosting | A **new public repository** for the site (working name `etfregulus-site`). ETFRegulus keeps the research code. The site's own workflow checks out ETFRegulus to run the strategies. Keeping the two apart stops the daily data commits from cluttering the research history |
| W-2 | Which strategy is "my plan" | **The user chooses it in the app.** The choice is stored only in the browser |
| W-3 | Units | **Weights plus share counts.** The user enters an amount in the browser; nothing is uploaded |
| W-4 | Language | **Chinese and English, switchable** |
| W-5 | Rules | **Only the constrained versions** of each strategy are shown: the 30-calendar-day per-ETF lock plus the drawdown-ceiling risk layer, i.e. the same simulator and constraint layer as the research project |
| W-6 | Drawdown ceiling | **Selectable in the app: 5%, 10% or 15%.** All three are computed daily |
| W-7 | Holdings source | **Switchable.** The default is the strategy's own simulated holdings. Alternatively, the user enters real holdings in the browser (ticker, shares, date of the last purchase) and the app says how to rebalance |
| W-8 | Strategies | **All six baselines:** cash, SPY buy-and-hold, equal weight, 60/40 SPY/TLT, momentum, classical optimizer. There is a slot for an RL policy later |
| W-9 | Accounts | None. No login, no server, no user data stored anywhere except the user's own browser |

### Meaning of the amount the user enters

- **Simulated mode:** the amount is how much the user puts into the strategy. Shares are
  `floor(amount × weight / last close)`.
- **Real-holdings mode:** the amount is the **investable cash on hand** on the day of entry.
  The portfolio value the strategy works with is `cash + Σ shares × last close`. The
  recommendation spends or raises cash only within the lock rule.

---

## 1. Architecture

```
            ETFRegulus (public, research code)
                         │  checked out at a pinned tag
                         ▼
 etfregulus-site ── GitHub Actions, daily cron after the US close ───────────────┐
   1. fetch new prices (yfinance)            ── data cache on the `data` branch  │
   2. run the 6 baselines × 3 ceilings through the ETFRegulus simulator          │
   3. write the site's JSON files                                                │
   4. append today's recommendations to the live log (never rewritten)           │
   5. build the front end, deploy to GitHub Pages ───────────────────────────────┘
                         ▼
          https://znzhao.github.io/etfregulus-site/   (static: HTML + JS + JSON)
                         ▼
          any device's browser: reads the JSON; settings and holdings in localStorage
```

**Hosting.** GitHub Pages serves static files, with no server. The site is HTML, JavaScript and
JSON data files. Pages is set to deploy from GitHub Actions (Settings → Pages → Source: GitHub
Actions). The same workflow that computes the data also publishes it, so a deploy never shows
half-updated data.

**Computing.** Everything that needs Python, such as data, strategies and the constraint layer,
runs in Actions once a day. The browser does only light work: scaling weights to shares, and the
real-holdings rebalance (§5, Step 6).

**Storage.**

| What | Where | Growth |
|---|---|---|
| Raw and curated price data (parquet) | an orphan `data` branch, force-pushed each day (one commit) | none |
| Site JSON (latest backtest, today's holdings) | deployed with the site; not committed | none |
| Live recommendation log (what the app said each day) | `log/` on `main`, one small file per month, append-only | ~2 KB/day |

Recomputing the backtests every day is acceptable because they are deterministic. The live log
is the honest track record: it is never recomputed, so a later code or data change cannot
rewrite what the app actually recommended.

---

## 2. Pages

| Page | Content |
|---|---|
| **Today** (home) | The chosen strategy and ceiling. Target holdings as weights and shares, the change since the last trading day (buy, sell, hold), and which positions are locked and until when. Data date and freshness badge. In real-holdings mode: the trade list to reach the target |
| **Strategies** | All six strategies side by side for the selected ceiling: today's holdings, and recent returns (1 day, 1 month, YTD). Sharpe and max drawdown over the full backtest. A tap opens the detail page |
| **Strategy detail** | Equity curve (log scale, optional benchmark overlay), drawdown chart with the ceiling line, bar chart of daily returns with a zoomable range, calendar heatmap of daily returns, monthly and annual return table, holdings-over-time stacked area chart, metrics table, and the live record since launch |
| **Compare** | Pick 2–6 strategies: overlaid equity curves, drawdowns, rolling 1-year Sharpe, and a metrics table |
| **Settings** | Strategy for "my plan", ceiling, holdings mode, amount, real-holdings editor (ticker, shares, last purchase date), language, light/dark/system theme, and export/import of settings as a JSON file (the only way to move them between devices, since nothing is stored on a server) |
| **About / data** | What each strategy does, the rules (lock, ceiling), data sources, last update and pipeline status, and a disclaimer that this is not investment advice |

Navigation: a bottom tab bar on phones and a top bar on desktop. The layout is phone-first.

---

## 3. Data contract (site JSON)

The front end reads only these files. A JSON Schema for each lives in `schema/` in the site
repository, and the pipeline validates its output against it before deploying.

```
data/
  manifest.json            # data date, generated_at, code version (ETFRegulus tag + commit),
                           # pipeline status, list of strategies and ceilings
  universe.json            # tickers, names (zh/en), asset class, last close, inception date
  today.json               # per strategy × ceiling: target weights, shares per 10,000 USD,
                           # changes vs the previous session, lock expiry per held ETF,
                           # drawdown state, whether capital preservation is active
  backtest/<strategy>_D<5|10|15>.json
                           # daily: date, equity, daily return, drawdown, weights (compact
                           # column arrays); summary metrics; monthly and annual returns
  risk.json                # the stress-scenario return matrix and the risk-layer parameters
                           # the browser needs for the real-holdings rebalance (Step 6)
  live/index.json          # the live log since launch, per strategy × ceiling
```

Size estimate: about 5,700 sessions × 18 series. With column arrays and 4-decimal rounding, each
backtest file is about 150–300 KB (gzip by Pages); it is loaded only when a detail page opens.

---

## 4. Daily schedule

- **Cron:** `30 22 * * 1-5` (22:30 UTC = 6:30 pm New York in summer, 5:30 pm in winter), plus a
  second attempt at `30 1 * * 2-6` in case of an upstream delay. Each run first checks the
  XNYS calendar and exits early on a holiday or if that session is already published.
- **Failure handling:** if the fetch fails or the data quality gate (Stage 2 checks) fails, the
  run deploys nothing new. The site keeps the last good data and shows a "data as of <date>,
  update failed" badge, computed in the browser from `manifest.json`. A GitHub Actions failure
  email is the alert.
- **Manual trigger:** `workflow_dispatch`, with an optional "rebuild everything" flag.

---

## 5. Steps

Each step ends with a check. Commit and push before starting each step (standing project rule).

### Step 1 — Mockups with fake data (decide the look first)

Goal: decide what the finished app looks like before writing any real pipeline.

1. Generate a **fake data set** in the exact shape of §3, using a small Python script:
   random-walk prices for the real tickers, six fake strategies, three ceilings, holdings that
   change monthly, some positions locked, and one "update failed" variant.
2. Build **clickable mockups of every page in §2**, wired to the fake JSON: phone and desktop
   layouts, light and dark themes, and Chinese/English text. Plain HTML + a small amount of
   JavaScript + a chart library, with no build step, so changes are quick.
3. Offer **two visual directions** for the Today page and the Strategy detail page (for example
   "dense dashboard" vs "card-based, simple") so a choice can be made by looking.
4. Publish the mockups so they can be opened on a phone (a private preview link, or a
   temporary Pages deploy of the new repository).
5. Review together and record the decisions in §7 of this file: page list, chart types, what
   goes on Today, wording, colours, which visual direction.

Check: every page opens on a phone and a desktop; the decisions list in §7 is filled in.

### Step 2 — Freeze the data contract

1. Adjust §3 to what the mockups actually needed; write the JSON Schemas in `schema/`.
2. Make the fake-data generator validate its output against the schemas. From here on, it is
   the test fixture for the front end.

Check: the fake data passes the schemas; the mockups still load it.

### Step 3 — The publish pipeline in ETFRegulus

New code in ETFRegulus, so it is tested with the research code and pinned by a tag.

1. `scripts/s15_publish.py`: takes the curated data and an output directory, runs the six
   baselines × ceilings {5, 10, 15}% with lock 30 through the existing simulator
   (`src/sim/simulator.py`, `src/baselines/strategies.py`) and writes the files of §3.
2. Rule set: the main-line constraint layer, `config/experiments/redesign_df.yaml` (D-F, under
   which the constrained baselines did best). The config file is a parameter, so the O-2 rule
   (§14 of the redesign plan) can be switched in later. Every output records the config hash.
3. Incremental fetch: `s01`/`s02` gain a mode that starts from the cached curated data and
   adds new sessions only. The full Stage 2 quality gate still runs over the new rows.
4. Measure the running time of a full rebuild on the CPU. If it is over ~15 minutes (the
   classical optimizer and the risk layer are the likely cost), resume each strategy from its
   saved end state instead of replaying from 2004, with a weekly full rebuild as a check that
   the two agree.
5. Tests: output passes the schemas; a strategy's last-day weights equal what `s05` produces for
   the same data; a holiday run is a no-op; a failed fetch produces no output.

Check: one command turns the current local data into a full site data set, within the time
budget, with all tests passing.

### Step 4 — Live log and day-over-day changes

1. Define the launch date. From then on, each run appends that day's target weights and
   changes, per strategy × ceiling, to `log/YYYY-MM.jsonl`, and never edits earlier lines.
2. The live record shown on the detail page is computed from the log, separate from the
   backtest. If the two disagree after a code change, the page shows both and says why.

Check: a re-run of the same day appends nothing; a code change does not alter past lines.

### Step 5 — The site repository and workflows

1. The user creates `etfregulus-site` (public). Settings → Pages → Source: GitHub Actions.
2. `.github/workflows/daily.yml`: checks out ETFRegulus at the pinned tag, restores the
   `data` branch, sets up Python with pip caching, runs fetch → curate → `s15_publish` →
   validate schemas → append log → build the front end → `actions/upload-pages-artifact` →
   `actions/deploy-pages`. Also has concurrency limited to one run, and a timeout.
3. `.github/workflows/ci.yml`: on every push, front-end build, schema checks against the
   fake data, and front-end tests.
4. Bumping the ETFRegulus tag is a deliberate pull request in the site repository, so a
   research change never reaches the live app by accident.

Check: a manual run deploys the site with real data; a scheduled run happens the next evening.

### Step 6 — Real-holdings mode in the browser

The user enters investable cash and holdings (ticker, shares, date of the last purchase of that
ticker; a purchase relocks the whole position, so only the last date matters).

1. Lock state: an ETF is locked until the first XNYS session on or after
   `last purchase date + 30 calendar days`, the same rule as `config/constraints.yaml`. The site
   ships the XNYS session list in `universe.json` for this.
2. Rebalance: start from the chosen strategy's target weights for today, over
   `cash + holdings value`. Locked positions are not reduced. The capital they would have
   released is not spent, and the remaining targets are scaled to fit the free capital. Then the
   same no-risk-increase rule as the backtest (D-F) is applied with the stress scenarios in
   `risk.json`. Shares are rounded down; the remainder stays in cash.
3. Drawdown in this mode: the app cannot know the user's own peak value, so it asks for it
   (optional). Without it, the drawdown is taken as zero and the full ceiling budget is used.
   The page says this plainly.
4. A **parity test**: a Python script produces cases (holdings, prices, targets → expected trades)
   from the real constraint layer, and the JavaScript port must match them, in CI.
5. A starting tip in the UI: "starting fresh? choose real-holdings mode with no holdings and
   enter your cash", which gives a fresh-start plan instead of the strategy's long-running path.

Check: the parity tests pass; on a phone, entering holdings gives a trade list that respects
every lock.

### Step 7 — The real front end

1. Turn the chosen mockup into the app: Vite + TypeScript, a hash router (works on Pages
   without server rewrites), Apache ECharts for charts, and a small i18n layer with
   `zh.json` / `en.json`.
2. `localStorage` only for settings and holdings, every access wrapped in try/catch, and the app
   works with it empty. Export/import of settings as a file.
3. Load `manifest.json` and `today.json` first, and backtest files only when needed.
4. Accessibility and phone checks: 16 px side gutter, no horizontal scrolling, readable charts at
   360 px width.
5. Front-end tests on the fake data (rendering, settings, rebalance).

Check: Lighthouse on a phone profile is acceptable; every page works on the real data.

### Step 8 — Launch

1. Run the daily job for a week alongside, checking each day's output against a local `s15`
   run.
2. Then set the launch date (Step 4) and bookmark the URL on each device (or "add to home
   screen").

### Step 9 — Later

- Add an RL policy as a seventh strategy once one beats the constrained baselines on validation
  (`MODEL_REDESIGN_PLAN.md`). The front end needs no change; the pipeline gains a strategy.
- Switch the rule set to D-F + O-2 when that is built, via the config parameter in Step 3.
- Optional: an RSS feed or e-mail digest of daily changes, generated by the same workflow.

---

## 6. Open items to settle during Step 1 or Step 3

- Backtest start: 2004 (full history), or the start of the first test year (2012) to match the
  research reports. Both are cheap; the mockups can show either.
- Whether the Today page should show only "my plan", or also a compact row for every strategy.
- The precise yfinance price convention to show (adjusted close for returns; last close for share
  counts). The research uses mid/open prices for execution; the app will state which price its
  share counts use.

## 7. Decisions from the mockup review

Reviewed 2026-10-10 on the first mockup (`webapp/mockups/index.html`, fake data).

- **Visual style: B (cards)** on every page, phone and desktop alike. Style A (compact tables) is
  dropped.
- **Pages and content: accepted as mocked.** Today, Strategies, Strategy detail, Compare,
  Settings and About, with the content listed in §2. No page is added or removed.
- **Polish is deferred.** The mockup is rough on purpose; spacing, wording, colours and chart
  details are refined during Step 7 as real data shows what needs work, not in another mockup
  round.
- Kept from the mockup: gain/loss colours switchable, red-up by default; settings backup by
  copy-and-paste text (file downloads are not relied on); the update-failed banner; the
  lock-until badge on every locked holding.
