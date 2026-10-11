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

**Frozen in Step 2 (2026-10-10).** The canonical definition is the JSON Schemas in
`src/publish/schema/` plus the cross-file checks in `src/publish/contract.py`
(`python -m src.publish.contract <dir>`). The pipeline runs that check before every deploy, and
nothing that fails it is published. The site repository's CI uses the same files from the pinned
ETFRegulus tag. `schema_version` is 1 in every file. A breaking change bumps it, and the front end
refuses a version it does not know.

A portfolio is one strategy × ceiling, with the id `<strategy>_D<ceiling>` (e.g. `momentum_D5`).
Weights are fractions of portfolio value with `CASH` included, nonzero entries only, summing to 1.

| File | Content |
|---|---|
| `manifest.json` | `data_date`, `generated_at`, `status` (ok / failed) and `last_attempt` (time, status, message), `next_run`, code version (repo, tag, commit), config name and hash, `backtest_start`, `lock_days`, `ceilings`, the strategies (id, zh/en name and description, colour), the portfolio ids, and the path of every other file |
| `universe.json` | Per ETF: ticker, zh/en name, asset class, inception, `close` and `prev_close` on `data_date`. Also the next ≥ 60 XNYS sessions from `data_date`, which the browser uses to compute lock expiry dates |
| `today.json` | Per portfolio: weights at the close, drawdown, `risk_state` (normal / over_budget / capital_preservation), locks still in force (ticker → first session it may be sold), returns over 1 day / 1 month / YTD / 1 year, the last rebalance (date and per-ticker from → to), `traded_today`. Share counts are computed in the browser from weights × amount ÷ close |
| `backtest/<portfolio>.json` | Daily `dates` and `equity` (growth of 1) as column arrays. Daily returns and drawdowns are derived in the browser. Weights sampled at each week's last session plus `data_date`, every rebalance, metrics (CAGR, volatility, Sharpe, max drawdown, Calmar, best and worst day, share of up days), and monthly and annual returns |
| `risk.json` | The risk envelope's stress samples at `data_date` (rolling, block bootstrap, crisis windows; each a matrix of 5-day returns, CASH first) and its parameters (CVaR quantile, horizon, aggregation, over-budget rule). For the real-holdings rebalance (Step 6); provisional until then |
| `live/index.json` | `launch_date` (null before launch) and, per portfolio, one entry per session since launch: date, that session's return, recommended weights, changes |

**Failure handling in the data:** a failed run republishes the last good data files with a new
`manifest.json` whose `status` is `failed` and whose `last_attempt` says what went wrong. The
browser shows the update-failed banner from that alone.

**Measured size** (fake data, full 2004–2026 history): about 280 KB per backtest file, 47 KB
gzipped. That is 5.5 MB for all 18 portfolios, of which a page loads at most the few it shows.
`today.json` is about 10 KB and `risk.json` about 150 KB.

**Fake data:** `python -m scripts.fake_site_data --out webapp/fake_data` writes a complete,
valid data set. It uses the real tickers and XNYS calendar, random prices, and simplified stand-in
strategies. `--failed` writes the update-failed variant, and `--live-days 0` writes the
before-launch state. The output is git-ignored; regenerate it when needed.

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

### Step 2 — Freeze the data contract ✅ (2026-10-10)

1. Adjust §3 to what the mockups actually needed; write the JSON Schemas. Done:
   `src/publish/schema/*.schema.json`, plus the cross-file checks in `src/publish/contract.py`.
2. Make the fake-data generator validate its output against the schemas. Done:
   `scripts/fake_site_data.py`. It and the real pipeline share `src/publish/build.py`, which turns
   a simulated path into the documents, so derived numbers are computed one way only.

Check: the fake data passes the contract (both variants). 20 tests in
`tests/publish/test_contract.py` cover the validator rejecting 13 kinds of broken data, and the
build helpers. The full suite passes (488). The mockup keeps its own in-page generator; wiring a
page to the JSON files is part of Step 7, which starts from this fake data.

### Step 3 — The publish pipeline in ETFRegulus ✅ (2026-10-10, except item 3)

New code in ETFRegulus, so it is tested with the research code and pinned by a tag.

1. **Done.** `scripts/s15_publish.py --out <dir>` (logic in `src/publish/pipeline.py`) runs the
   six baselines × ceilings {5, 10, 15}% with the 30-day lock through the research simulator,
   writes the files of §3 into a temporary directory, checks them against the contract, and
   only then swaps them in. A failure leaves the previous data untouched.
2. **Done, with two decisions made along the way.** The rule set is
   `config/experiments/webapp.yaml` (hash `d2a9a04c`): the main-line D-F rule, plus:
   - **Drawdown measured from the highest NAV of the past 12 months** (252 sessions; decided by
     the user 2026-10-10). On one continuous 2004–2026 path, an all-time peak that never resets
     leaves a strategy in capital preservation for good after 2008: 60/40 at 5% was 100% cash
     from 2009 to 2026 (Sharpe 0.27). The rolling peak is a new opt-in simulator setting,
     `drawdown.peak_window_sessions` (`SimulationConfig.peak_window`); the default 0 keeps the
     research behaviour, and every research config hash is unchanged.
   - **A strategy's "hold" is a genuine no-trade session** (`pipeline.true_hold`). Between
     rebalances the baselines ask for the current weights; the engine executes weights at the
     next open, so overnight price moves turned every hold into small trades, each buy relocking
     the ETF. Momentum traded on 3,256 of 5,702 sessions. The user flagged the daily trading as
     unintended. Holds now go through `engine.hold`; momentum trades on 214 sessions (~9 a
     year) and its Sharpe at 5% rose from 0.38 to 0.57. **The research simulator still has this
     behaviour** (it affects the research baselines and their comparisons); see the
     redesign plan for whether to change it there.
   - "Today's plan" is the decision taken at the final close, projected through the lock and the
     risk rule but not executed (`engine.plan_next`), so `today.json` carries both the book at
     the close (`current_weights`) and the plan for the next open (`weights`, `next_trades`).
   - Published trades leave out weight changes under 0.1 percentage point.
3. **Not done: incremental fetch.** `s01_fetch_data` already has `--incremental`. It was not run
   here because it would overwrite `data/curated/`, which the research campaigns depend on
   (see `CONTINUAL_TRAINING_PLAN.md` on data versioning). The daily job will keep its own data
   copy on the `data` branch (§1); wiring s01/s02 to that directory is done in Step 5.
4. **Done.** A full rebuild takes **3 min 8 s** on the local CPU (8–12 s per portfolio), far under
   the 15-minute budget, so no resume-from-saved-state is needed. The output is 5.1 MB.
5. **Done.** `tests/publish/test_pipeline.py`: valid output, a monthly strategy trades at most
   about monthly, the path matches the simulation, and a failed check keeps the old data;
   `tests/sim/test_peak_window.py`: the rolling peak. Suite: 500 passed.

Check: met, except item 3.

**Known limitation, carried to the redesign plan (O-2):** under the current D-F rule the
ceiling does not force any selling, so realized drawdowns go far past it in a crash. At 5%,
measured from the 12-month peak, the worst drawdowns were 21–35% (all in 2008–09). It was
on 14–21% of days for most strategies (6% for the classical optimizer). The stepwise de-risking
the user decided on (O-2, `MODEL_REDESIGN_PLAN.md` §14 step 1) is the fix. When it is built,
`true_hold` must let a hold past the ceiling go through the risk rule instead of skipping it.

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

## 8. Launch status (2026-10-11)

The site is live at **https://znzhao.github.io/etfregulus-site/** (repository
`znzhao/etfregulus-site`, public). It is built from ETFRegulus tag `webapp-v1`.

| Step | Status |
|---|---|
| 4 — Live log | **Done.** `src/publish/live.py`; `s15_publish --log-dir` appends one line per new session, only after the output passes the contract. The site workflow commits `log/YYYY-MM.jsonl`. The first entry (launch date) is 2026-10-09 |
| 5 — Site repository and workflow | **Done.** `.github/workflows/daily.yml`: cron 22:30 UTC Mon–Fri plus a 02:30 UTC retry, manual trigger, and a redeploy on app changes. It runs front-end unit tests, checks out ETFRegulus at `ETFREGULUS_REF`, installs `requirements-publish.txt` (no torch), runs s01 `--skip-fred` and s02 `--allow-missing-macro` (prices only; no secrets needed), s15, then commits the live log and deploys to Pages. A full run takes ~7 minutes on GitHub's runner; nothing runs on the local machine. The daily job uses its own fresh data, so `data/curated/` here is untouched (this closes Step 3 item 3) |
| 6 — Real-holdings mode | **Partly done.** Lock-aware rebalance in the browser (`app/core.js` `realPlan`, unit-tested). The D-F risk rule is not yet ported to the browser; the page says so |
| 7 — Front end | **Done (first version).** Static, no build step: `app/core.js` (pure logic, 17 Node unit tests), `app.js`, `i18n.js` (zh/en, key parity tested), `styles.css`; ECharts 5.5.0 from jsDelivr; style B cards; add-to-home-screen icons. An automated browser check (headless Edge, phone and desktop × zh/en × light/dark, every page, tab, range and control, plus failure states) reported no errors after the fixes from its screenshots |
| 8 — Launch | **Live.** Data as of 2026-10-09; the next scheduled update is Monday 2026-10-12 after the close |

Open items: port the risk rule to the browser (Step 6); O-2 stepwise de-risking (also fixes the
drawdown overshoot); decide whether the research simulator adopts true holds
(`MODEL_REDESIGN_PLAN.md` §14 item 5); move the workflow actions off Node 20 before GitHub removes it.
