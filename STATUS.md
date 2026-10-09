# STATUS — Task Tracker

**The single file to update as work proceeds.** Plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
Stage runbook: [reference/stages.md](reference/stages.md).

Legend: `[ ]` not started · `[~]` in progress · `[x]` done (tests green) · `[!]` blocked · `[-]` deliberately deferred

---

## Current position

| | |
|---|---|
| **Current stage** | **Phases 0–3 all complete.** Stages 0–12 green; hard acceptance passed at every stage. Every result to date is at ~2% of `training.total_timesteps`. **The full-budget run is now an incremental campaign** ([INCREMENTAL_TRAINING_PLAN.md](INCREMENTAL_TRAINING_PLAN.md)): built and tested, campaign `budget_v1` is at **37.84%: C0-C4 reported** (2, 4, 8, 12, 16, 22, 32%); a 40% checkpoint (C4b) is part-way. 32% full Stages 9-12: 23/28, no blocking failures. Validation has shown no clear improvement at two consecutive checkpoints, so budget is on hold pending a look at training dynamics, data versioning and a fine-tuning design. |
| **Run this** | Idle-time sessions: `python -m scripts.s14_incremental --status`, `--plan <hours>`, `--hours <hours>`, `--stop`. The first session begins by evaluating C0 (~30 min) |
| **Next gate** | None open — Phase 3 passed. Campaign paused by decision; see the 2026-10-06 log entries. |
| **Lock period** | `N = 30` calendar days (D16); operating range `[15, 18, 21, 25, 30, 36, 42, 50, 60]` (2^(1/4) ladder) in `config/constraints.yaml` |
| **Drawdown ceiling** | `D_max = 0.05` primary; operating range `[0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]` — revised from the original `[0.05..0.25]` down to `[0.001..0.10]` and finally to this range, each time by explicit request, each time retrained |
| **Risk envelope** | **Calibrated** (Q1 closed): `quantile 0.05, horizon 5, aggregation max, measure cvar` |
| **Latest walk-forward run** | `s08_walk_forward_20260903T224114Z_a207776c` — **14/14 folds (2012–2025)**, hard acceptance PASSED, on the current grids |
| **Latest acceptance (Stage 12, unconstrained baselines only, N=30 / D_max=5%, 2012–2025)** | **22/22** — `rl_policy` Sharpe 0.92 leads all six baselines (`spy_tlt_60_40` 0.90, `spy_buy_hold` 0.89, `equal_weight` 0.83, `classical_optimizer` 0.78, `momentum` 0.74, `cash` 0.00). Bootstrap Sharpe band 0.47–1.41 (median 0.95), fully above zero. |
| **Test suite** | 393 passed, 5 deselected (`network`, `slow`), ~190s — `.venv/Scripts/python.exe -m pytest -q` |
| **Last updated** | 2026-10-05 |

---

## Finished

**Phases 0 through 3 are all complete.** Stages 0–12 are `[x]`: every gating test
named for them in [reference/testing.md](reference/testing.md) §3 passes, and each stage runs end
to end from a clean checkout.

| | Stage | Entry point | Gate | State |
|---|---|---|---|---|
| **Phase 0** | 0 — Preflight | `s00_check_env` | T16 | `[x]` 35/35 checks |
| | 1 — Fetch | `s01_fetch_data` | symbol pins, coverage | `[x]` 31 symbols + 10 FRED series |
| | 2 — Curate | `s02_curate_data` | I5, T15 data-side, gate at zero | `[x]` 172,202 rows, 0 hard violations |
| | 3 — Features | `s03_build_features` | I6, T10, manifest | `[x]` 163 columns, 14 fold scalers |
| **Phase 1** | **4 — Simulator** | `s04_simulate` | **I1–I6, T1–T7, T15** | `[x]` **THE GATE — GREEN** |
| | 5 — Baselines | `s05_run_baselines` | zero violations + 5 checks | `[x]` all six, all checks pass |
| **Phase 2** | 6 — Env smoke test | `s06_smoke_env` | T8, T9, T14 + zero violations | `[x]` 7/7 checks, 31,500 steps, 0 violations |
| | 7 — Train PPO | `s07_train_ppo` | per-rung: beats `cash`, zero violations | `[x]` 5/5 rungs green (demonstration budget) |
| | **8 — Walk-forward** | `s08_walk_forward` | **T11, T12, T13 + hard acceptance** | `[x]` **13/13 folds, hard acceptance PASSED** |
| **Phase 3** | 9 — Stress | `s09_stress` | `D_max` monotonicity | `[x]` monotone at every `N`; 29 cells |
| | 10 — Bootstrap | `s10_bootstrap` | bands + block-length sensitivity | `[x]` 1,000 replicates, sensitivity stable |
| | 11 — Adversarial | `s11_adversarial` | 4 scenarios, disclaimer present | `[x]` zero violations on all 4 |
| | 12 — Report | `s12_report` | acceptance table, explicit pass/fail | `[x]` **22/22 at `D_max=5%`**, 0 blocking failures at every ceiling tried |

**Invariants and tests proven** — the full list from [reference/testing.md](reference/testing.md):

| | Covers | Where |
|---|---|---|
| I1 | cash ≥ 0, shares ≥ 0, NAV > 0 | `tests/portfolio/test_ledger_and_execution.py`, asserted at runtime in `Ledger` |
| I2 | weights + cash sum to 1 | same, plus every real trajectory |
| I3 | a locked position never shrinks | `tests/portfolio/test_lock_invariants.py` (**property-based**) |
| I4 | an executed buy sets `unlock = exec + N` exactly | same |
| I5 | nothing exists before inception | `tests/data/test_inception.py`, `tests/sim/test_simulator.py` |
| I6 | no lookahead, incl. future-mutation | `tests/features/test_lookahead.py` |
| T1 | **total-return reconstruction** | `tests/portfolio/test_total_return.py` — ≤9.6e-4 over all 24 tradables |
| T2 | ledger + lock dict round-trip | `tests/portfolio/*` — also satisfies Stage 13's only requirement on current work |
| T3 | projection idempotent on feasible points | `tests/constraints/test_projection.py` |
| T4 | analytic vs **CVXPY oracle** | same |
| T5 | risk **convexity** (replaces monotonicity) | `tests/constraints/test_risk_envelope.py` |
| T6 | fallback always feasible, never raises | same |
| T7 | no silent repair | `tests/constraints/test_projection.py` |
| T10 | scaler fold isolation | `tests/features/test_scalers.py` |
| T15 | fill price inside the day's range | `tests/portfolio/test_ledger_and_execution.py` |
| T8 | reset sampler reachable, `D_t ≤ D_max` | `tests/env/test_reset_sampler.py` — both generators, whole grid |
| T9 | determinism, incl. across worker counts | `tests/env/test_determinism.py` |
| T11 | walk-forward integrity: train < val < test | `tests/evaluation/test_walk_forward.py` |
| T12 | selection is lexicographic; risk before return | `tests/evaluation/test_selection.py` |
| T13 | the preventable-violation detector **fires** | `tests/evaluation/test_violation_taxonomy.py` — four injected bugs |
| T14 | observation matches the manifest | `tests/env/test_observation.py` — asserted by NAME at index |
| T16 | stage harness | `tests/test_stage_harness.py` |

Every test in [reference/testing.md](reference/testing.md) §3 now has a home, and all of them pass.

**Code built and committed:**

```text
src/cli/stage.py            the harness every entry point wraps
src/config/                 layered YAML -> typed dataclasses, unknown key is an error
src/data/                   calendar, fetch, curate (+ the quality gate)
src/features/               etf, cross_sectional, macro, folds, scalers, builder
src/portfolio/              ledger, lock_manager, valuation, execution
src/constraints/            projector (analytic + cvxpy), risk_envelope
src/sim/                    simulator, runner
src/baselines/              the six strategies
src/evaluation/             metrics
scripts/s00..s05            six runnable stages
config/                     universe, features, constraints, sim/default, evaluation
tests/                      217 passing in 52s (budget 60s), 1 network test deselected
```

**Decisions locked and questions closed since planning:** D16 (lock centred on 30 calendar
days) and Q1 (risk-envelope calibration). Three specification errors corrected from
measurement — see *What Phase 1 caught* below.

---

## Rules for whoever is working this file

1. Work stages in order. Do not start a stage whose predecessor is not `[x]`.
2. A stage is `[x]` only when **its gating tests pass** ([reference/testing.md](reference/testing.md) §3).
3. **Never begin Phase 2 before Stage 4 is `[x]`.** This is the project's central sequencing constraint.
4. **Never begin Phase 3 before Stage 8 is `[x]`.**
5. Update the "Current position" table and the decision log at the end of every working session.
6. If a decision in [reference/decisions.md](reference/decisions.md) needs to change, change that file and
   note it in the log below — do not work around it silently.

---

## Phase 0 — Foundation

### `[x]` Stage 0 — Preflight · `python -m scripts.s00_check_env` — **35/35 checks pass**

- [x] **GPU resolved** — RTX 3060 Ti (8 GB, sm_86), driver 596.49/CUDA 13.2, `torch==2.13.0+cu126`,
      `cuda.is_available() == True`, matmul verified against CPU. See [reference/gpu-setup.md](reference/gpu-setup.md)
- [x] **Device benchmarked** — end-to-end PPO: CPU **2.35s** vs CUDA 6.44s over 5120 steps → **D14: `device: cpu`**
- [x] Core RL stack installed: torch 2.13.0+cu126, SB3 2.9.0, gymnasium 1.3.0, numpy 2.4.6, pandas 3.0.5, tensorboard 2.21.0
- [x] `requirements.txt` pinned from the verified versions ([reference/gpu-setup.md](reference/gpu-setup.md) §5)
- [x] Remaining deps installed: exchange-calendars 4.13.2, yfinance 1.7.0, fredapi 0.5.2, pyarrow 25.0.1,
      hypothesis 6.167.1, pytest 9.1.1, cvxpy 1.9.2, scipy 1.17.1, PyYAML 6.0.3, python-dotenv 1.2.3
- [x] `src/cli/stage.py` harness: `--config`, `--dry-run`, `--force`, `--seed`, staleness, manifest
- [x] Typed config layer `src/config/` — layered `extends:`, dataclass validation, **an unknown key is an error**
- [x] Preflight checks incl. **CUDA hard-fail** and the **numpy-less-torch** check (both real failure modes)
- [x] `FRED_API_KEY` + yfinance + `XNYS` calendar reachable
- [x] `data/`, `artifacts/` writable and git-ignored
- [x] **Gate:** T16 passes (15 tests, `tests/test_stage_harness.py`)

### `[x]` Stage 1 — Fetch raw data · `python -m scripts.s01_fetch_data --config config/universe.yaml`

**31 price symbols + 10 FRED series, 2003-01-02 .. 2026-08-31.**

- [x] `config/universe.yaml` from [reference/etf-universe.md](reference/etf-universe.md) — 24 tradable + CASH,
      7 feature-only, 10 FRED, kinds/groups/lags/quality thresholds
- [x] yfinance fetch: `auto_adjust=False`, batches ≤20 with a 2s pause, 3 retries with exponential backoff, from 2003-01-01
- [x] FRED fetch; the lag is **recorded, not applied** here, so `data/raw/` stays a faithful record of the provider
- [x] Atomic writes (temp + rename); tz normalized to naive dates at the boundary
- [x] Incremental mode: trailing-90-day **overwrite** (append-only would drift from the rewritten adjusted series)
- [x] `--only` / `--skip-fred` **merge** into the fetch log rather than replacing it
- [x] **Gate:** symbol-quirk pins, coverage ≥0.99 for every symbol, inception pins hold (16 tests)

> Coverage 1.0000 for most symbols; lowest 0.9980 (`^VVIX`). `DX-Y.NYB` prints on 17 non-XNYS days — dropped at curation.

### `[x]` Stage 2 — Curate · `python -m scripts.s02_curate_data --config config/universe.yaml`

**172,202 (session, ticker) rows · 31 symbols · 5,954 sessions · 0 hard violations, 34 warnings, 1 waived.**

- [x] Reindex to `XNYS` sessions; derive `inception.parquet` — the only source of the availability mask
- [x] Quality gate (OHLC sanity, gaps, non-positive prices, split-unexplained jumps, level ranges,
      SPY/VTI corroboration, macro coverage and staleness) — **exits non-zero**
- [x] `close_adj` / `close_raw` both present and consistent
- [x] **`div_per_share` derived here** — the implied distribution that lets a raw-price ledger produce a total-return NAV
- [x] Macro block publication-lagged to point-in-time
- [x] **Gate:** zero hard violations; I5, T15 data-side — 15 tests, every check exercised against injected corruption

> **Total-return reconstruction, measured across all 24 tradables:** worst **9.57e-4** (SHY), then TLT 8.8e-4,
> IEF 8.7e-4, LQD 8.3e-4; GLD exactly **0.0** (it pays no distribution). The residual is yfinance's `Adj Close`
> rounding over ~280 distribution events, not a modelling error. Recomputed on every Stage 2 run. **This
> pre-validates T1** and gives Stage 4 a measured baseline to hold itself to.

### `[x]` Stage 3 — Features · `python -m scripts.s03_build_features --config config/features.yaml`

**etf 50 cols × 131,531 rows · cross-sectional 14 · macro 99 · 14 fold scalers · 0 warm-up mismatches.**

- [x] Per-ETF OHLCV block — returns, volatility, drawdown, intraday shape, volume, trend, RSI/MACD/ATR/Bollinger
- [x] Cross-sectional block, availability-aware **percentile** ranks, plus `universe_size`
- [x] Macro block from **point-in-time-aligned curated series only** — the FRED client is kept out of `src/features/` entirely
- [x] Per-fold scalers (robust median/IQR, clip ±10, clipping counted not silent); feature manifest
- [x] Feature correlation diagnostic reported — 11 pairs at |r| ≥ 0.95
- [x] **Gate:** I6, T10, manifest completeness (34 tests)

> **The NaN warm-up matches the manifest exactly for every (ticker, feature).** The check found three real
> bugs on its first run — an ATR off-by-one, zero-range bars, and zero-volume prints — all fixed at the
> source rather than by loosening the check.

---

## What Phase 0 caught

Three defects that would each have been invisible downstream. Recorded because catching this
class of thing is the entire justification for Stages 0–3 existing, and because each one is a
standing hazard rather than a one-off.

| # | Defect | Why it mattered | Fix |
|---|---|---|---|
| 1 | `xcals.get_calendar("XNYS")` with no bounds returns a **rolling ~20-year window** — measured, it began 2006-09-01 | Every frame reindexed against it would have silently lost 2003–2006: the whole warm-up buffer and the run-up to the GFC | `src/data/calendar.py` pins `CALENDAR_START = "2002-01-01"`; Stage 0 asserts the first session lands at `history_start` |
| 2 | Price and return checks were being applied to **level** series | `^IRX` closed **negative** for 7 sessions in March 2020 (bills traded through zero) and `^VIX` routinely moves >40% in a day. Both were flagged as corruption. Same class of error as computing a log return on a yield | Gate checks are taxonomized by `kind`; levels get a per-symbol `range` band instead — VIX [0, 200], VVIX [0, 400], yields [-1, 20] |
| 3 | The inception-pin check passed for **SHY by accident** | SHY launched 2002-07-22, before `history_start`, so its first bar is fetch truncation rather than inception. The session-distance arithmetic happened to return 1, so any genuinely truncated series would have slipped through | Pins earlier than `history_start` are handled explicitly: the first bar must sit at the top of the fetch window |

The Stage 3 warm-up assertion found three more (an ATR off-by-one, zero-range bars, zero-volume
prints). All were fixed at the source; none were fixed by loosening the check.

---

## Phase 1 — Deterministic core  ← the critical path

### `[x]` Stage 4 — Deterministic simulator · `python -m scripts.s04_simulate --config config/sim/default.yaml`

**THE GATE — GREEN.** 5,702 sessions, 24 tradables + CASH, zero lock and zero feasibility
violations. ~0.5 ms/step without the envelope, ~1.6 ms/step with it (relevant to Q2).

- [x] `Ledger`: fractional shares, cash, long-only, dict round-trip
- [x] Execution: next-open, sells-then-buys, `cost_bps` parameter set to 0.0, price-in-range check
- [x] **Total-return NAV via dividend reinvestment** ([reference/portfolio-ledger.md](reference/portfolio-ledger.md) §3)
- [x] NAV, running peak, drawdown — peak inherited across reset, never set to current NAV
- [x] `LockManager`: all four transitions + dividend carve-out (structural: reinvestment
      routes through `accrue_shares`, which emits no trade leg, so the lock cannot see it)
- [x] Availability / inception masking
- [x] Analytic projection: **box-constrained** simplex (lower *and* upper bounds) + `alpha` de-risk scan
- [x] CVXPY backend, used as the correctness oracle from day one
- [x] Risk envelope: rolling stress, stationary block bootstrap, date-filtered crisis library
- [x] Capital-preservation mode and the infeasibility fallback
- [x] Reward `log(V_{t+1}/V_t)`
- [x] `trajectory.parquet` writer matching the documented schema
- [x] **Gate:** I1–I6, T1–T7, T15 all green

> **T1 against the real ledger:** buy-and-hold each of the 24 tradables from inception to
> today and the NAV reproduces `close_adj` to **≤9.6e-4**; GLD exactly 0.0. The lock
> invariants are property-tested with `hypothesis` over random buy/sell/hold sequences,
> including the sell-then-rebuy-same-day case that fixed fixtures miss.

### `[x]` Stage 5 — Baselines · `python -m scripts.s05_run_baselines --config config/evaluation.yaml`

Specs: [reference/baselines.md](reference/baselines.md). All six through the identical
simulator and constraint layer, **zero violations**, all five acceptance checks pass.

- [x] **B1 `spy_buy_hold`** — buy SPY session 1, never trade again
- [x] **B2 `momentum`** — 12-1 cross-sectional, top-5 equal weight, monthly, absolute filter
- [x] **B3 `spy_tlt_60_40`** — 60/40, monthly, 5pp drift band
- [x] B4 `cash` · B5 `equal_weight` · B6 `classical_optimizer` (required by the acceptance criteria)
- [x] All six through the identical simulator + constraint layer
- [x] Standard metrics computed for each
- [x] **Risk-envelope calibration** ([reference/risk-envelope.md](reference/risk-envelope.md) §7) — **Q1 CLOSED**
- [x] Initial-state reservoir populated — 1,506 states, reachable by construction
- [x] **Gate:** zero lock/feasibility violations across all six, plus the simulator checks

---

## What Phase 1 caught

Five defects, and **three of them are errors in the specification itself** rather than in the
code. Each is corrected in the reference doc it came from, with the measurement that settled it.

| # | Finding | Why it mattered | Resolution |
|---|---|---|---|
| 1 | **Capital preservation capped *weights*, not share counts** | In a falling market "hold yesterday's weight" is an instruction to buy the dip every session — the exact increase in risky exposure the mode forbids. Cash leaked from 30% back to 0% through the 2008 drawdown, and realized drawdown was *higher* with the safety layer on (0.415) than off (0.364) | Capped in **share space at execution**, like the lock floor. Realized drawdown is now monotone in `D_max`, which is Stage 9's gate |
| 2 | **`stress_loss` monotonicity is false** ([risk-envelope.md](reference/risk-envelope.md) §5 required it) | `w_safe` minimizes *exposure*, not *risk*. With a locked position and an anti-correlated hedge available, `w_safe` measured **0.0605 against 0.0088** — seven times riskier than the book it was meant to de-risk toward. Any universe with SPY and TLT has this structure | The bisection actually needs **convexity**, which CVaR has and VaR does not (violation ~1e-3). `measure: cvar` is now the only measure valid with the analytic backend, and the projector re-checks feasibility regardless |
| 3 | **The envelope calibration procedure was wrong** ([risk-envelope.md](reference/risk-envelope.md) §7) | Run on one 21-year path, the intervention rate came out at 0.774 for *every* candidate, with a **0.9984 correlation to "already breached"** — it was measuring time-under-water, not calibration, because the peak never resets. No parameter choice could have changed it | Calibrate over **annual windows**, matching how the policy is trained and evaluated. Same envelope then gives mean intervention **0.070**, spiking at 2008 (0.29), 2020 (0.37), 2022 (0.50) |
| 4 | **Every rebalancing baseline was trading daily** | `rebalance: monthly` was implemented as "re-assert the target weights every session". Prices drift, so that is a daily instruction to trade back — which relocked every position every day and made the lock appear to bind at values of `N` where it should not | Baselines now `_hold()` between rebalance dates, asking for exactly what is held so no trade leg is generated |
| 5 | **`equal_weight` silently ignored new listings** | With 24 names at ~4.2% each, an ETF entering at weight 0 is only a 4.2pp deviation and never trips the 5pp drift band. B5 quietly stopped tracking the expanding universe — the one thing it exists to exercise | A change in the *available set* is a structural change, not drift, and rebalances regardless of the band |

Two further spec claims were corrected from measurement: `momentum` degrades **in trend**
rather than strictly monotonically in `N` (the effect is overwhelmingly the `0 → N>0`
transition; beyond it a discrete rebalance calendar makes it path-dependent), and
`spy_buy_hold` is a control for `N` **but not for `D_max`** — the action-level budget
constrains the proposed portfolio itself, not merely increases in it.

2004-01-02 .. 2024-12-31 at `N=30`, `D_max=0.15`, frictionless:

| baseline | final NAV | ann. | maxDD | Sharpe | cash |
|---|---|---|---|---|---|
| `classical_optimizer` | 4,124,908 | +6.99% | 0.175 | +0.95 | 0.03 |
| `spy_buy_hold` | 1,270,697 | +1.12% | 0.153 | +0.20 | 0.80 |
| `spy_tlt_60_40` | 1,133,649 | +0.58% | 0.262 | +0.10 | 0.77 |
| `equal_weight` | 1,123,152 | +0.54% | 0.284 | +0.10 | 0.77 |
| `momentum` | 1,025,056 | +0.12% | 0.295 | +0.02 | 0.83 |
| `cash` | 1,000,000 | 0.00% | 0.000 | 0.00 | 1.00 |

> **Read these with the peak caveat below.** Over a single 21-year path the peak never
> resets, so a 2008 breach of `D_max = 0.15` is never recovered from and most baselines sit
> in capital preservation for most of the sample — hence the ~0.8 cash weights and the poor
> returns. These are *not* comparable to per-fold numbers, and Stage 8 must report which
> convention it used.

**Acceptance checks — all pass:**

| check | result |
|---|---|
| `cash` exactly zero drawdown and turnover | 0.0 / 0.0 |
| `spy_buy_hold` identical across `N ∈ {0,30,90,180}` | max NAV difference **0.0** |
| `momentum` lock binds hard | turnover ratio unlocked:locked **21.7x** |
| `momentum` turnover trends down in `N` | Spearman **−1.0** |
| `spy_tlt_60_40` severe 2022 drawdown | **0.186** |

**Q1 calibration (closed).** 6 candidates × 2 baselines × 5 `D_max` × 20 annual windows.
Chosen **`quantile 0.05, horizon 5, aggregation max`** — by design constraint, with return
only as a tie-break. Realized drawdown responds correctly to the ceiling:

| `D_max` | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 |
|---|---|---|---|---|---|
| realized maxDD | 0.067 | 0.082 | 0.091 | 0.095 | 0.096 |
| intervention rate | 0.389 | 0.157 | 0.056 | 0.034 | 0.017 |

---

## Phase 2 — Environment and RL  `[x]` — Stages 6 and 7 are `[x]`

### `[x]` Stage 6 — Env smoke test · `python -m scripts.s06_smoke_env --config config/training.yaml --episodes 500`

- [x] `gymnasium` + SB3 `check_env` pass — **zero warnings** (see the action-space correction below)
- [x] 500 random episodes across the full `(N, D_max)` grid, invariants asserted every step —
      **31,500 steps, 0 violations**, all 25 cells visited ≥20 times, both reset modes exercised
- [x] Observation bounds/dtype/finiteness — `float32`, inside its declared `Box`, never NaN or inf
- [x] **Throughput benchmark** — Dummy vs Subproc at 1/4/8/16 workers, cpu vs cuda → recorded below
- [x] **Gate:** T8, T9, T14, zero violations — 7/7 checks pass

**Q6 closed — the observation is 656 dimensions.** Declared in the new
[config/observation.yaml](config/observation.yaml) and validated against `feature_manifest.json` at
startup, so a typo is an error rather than a silently zeroed column. From Stage 3's 163 columns:
**20 of 64 per-asset** (the `ret_*` ladder with the near-collinear `logret_*` twins dropped, two
vol scales, two drawdowns, two trend filters, four oscillators, one liquidity, four cross-sectional)
and **21 of 99 macro** (level plus one change horizon per family). Layout
`[22 global | 24 × 26 per-asset | 8 portfolio | 2 params]`.

> **Throughput** (24 CPUs, `obs_dim = 656`, envelope on, `strict` off):
>
> | vec | workers | steps/s |
> |---|---|---|
> | Dummy | 1 | 579 |
> | Dummy | 4 | 563 |
> | Dummy | 8 | 580 |
> | Subproc | 1 | 516 |
> | Subproc | 4 | 1,567 |
> | Subproc | **8** | **2,844** |
> | Subproc | 16 | 2,949 |
>
> **Q2 closed: `n_envs = 8`, `SubprocVecEnv`.** Going 8 → 16 buys **3.7%** for double the processes
> and double the memory; 1 → 8 buys 5.5×. `DummyVecEnv` is flat in the worker count, as it must be —
> it steps serially — which confirms the benchmark is measuring what it claims to.
>
> **Q5 — D14 is _not_ confirmed, and not refuted either. It must be re-measured in Stage 7.**
> On a 2×256 trunk at this observation size, CUDA runs 50 forward+backward passes in **0.116 s**
> against CPU's **0.278 s** — the GPU is **2.4× faster**, where D14 measured CPU **2.7× faster
> end to end**. These do not contradict each other: D14's number was end-to-end PPO, where rollout
> transfer latency dominates, and this one is the network alone. What has changed is D14's premise —
> it was decided at ~300 dims and the observation is now 656. `device: cpu` stays the default
> because it is the measured end-to-end verdict, but Stage 7 must re-run the end-to-end comparison
> before the setting is treated as settled.

### `[x]` Stage 7 — Train PPO · `python -m scripts.s07_train_ppo --config config/training.yaml --curriculum`

- [x] Shared per-asset encoder policy; availability masked in logits — **and a shared per-asset
      *head* too**, a correction: the specified flat `Linear(trunk → K+1)` has separate weights per asset
      and throws away the equivariance the encoder buys. 582,620 parameters
- [x] Normalization fitted on the training window only, frozen at eval, saved with the policy —
      **rewards only**; see the correction below
- [x] D9 wiring: `a_raw` stored, `a_proj` executed, `proj_distance` logged
- [x] Diagnostic callbacks + baseline reference lines on TensorBoard
- [x] Curriculum 1 — `N=0`, mechanics only
- [x] Curriculum 2 — lock, `N ∈ {21, 30, 42}` *(the plan said `{7,30}`; D16 moved the operating range)*
- [x] Curriculum 3 — full parameter range
- [x] Curriculum 4 — risk envelope on
- [x] Curriculum 5 — fully randomized + reservoir resets
- [x] **Gate per stage:** beats `cash`, zero lock/feasibility violations — 5/5 passed

**Demonstration run: 300k timesteps per rung (1.5M total, 18 min), all five gates green.**

| Rung | ep. log return | proj. distance | cash w | steps/s | gate |
|---|---|---|---|---|---|
| 1 mechanics | +0.0667 | 0.073 | 0.039 | 2,306 | PASS |
| 2 lock | +0.0370 | 1.812 | 0.040 | 2,306 | PASS |
| 3 parameters | +0.0728 | 1.077 | 0.064 | 2,306 | PASS |
| 4 envelope | +0.0685 | 1.023 | 0.139 | 1,112 | PASS |
| 5 randomized | +0.0732 | 0.966 | 0.187 | 988 | PASS |

> **This is not the headline run.** `training.total_timesteps` is 2,000,000 per rung; this used 300,000
> to exercise the whole ladder end to end. The configured budget is ~2 h wall on this machine.
> Nothing below should be read as a performance result — only as evidence the machinery works.

Three things the run confirmed, and one it did not:

- **The lock is visible in `proj_distance`.** 0.073 with no lock (rung 1) → 1.812 the moment it is added
  (rung 2). The constraint is binding on the policy's proposals, not decorative.
- **Cash weight rises monotonically with the constraints** — 0.039 → 0.187 as the lock, then the grid, then
  the envelope, then reservoir starts come on. The policy is responding to its parameters.
- **The envelope costs about half the throughput** (2,306 → 1,112 steps/s), matching Stage 6's finding that
  it is roughly two-thirds of a step.
- **Q3 is NOT settled.** `proj_distance` fell *within* rung 3 (1.77 → 1.19) and rung 4 (1.20 → 0.99), which
  is the healthy D9 signal — but it rose within rung 2 (1.35 → 1.78) and rung 5 (0.84 → 0.95). At 300k
  timesteps and 19 logged points per rung this is suggestive, not an answer.

### `[x]` Stage 8 — Walk-forward · `python -m scripts.s08_walk_forward --config config/evaluation.yaml`

- [x] Expanding annual folds, train/val/test with no overlap — **T11**, checked before any compute runs
- [x] Lexicographic model selection; `selection.json` per fold — **T12**
- [x] Preventable-vs-market-forced violation taxonomy incl. the replay detector — **T13**
- [x] Per-fold standard metrics, over a 3×3 `(N, D_max)` grid
- [x] **Gate:** T11, T12, T13; zero lock/feasibility violations on every fold

**13 folds (2012–2024) × 4 candidates × 40k timesteps. Hard acceptance PASSED: zero lock, zero
feasibility, zero preventable `D_max` violations. No fold was marked `constraint validation failure`.**

| Fold | Selected | Test return | Max DD | Preventable | Market-forced |
|---|---|---|---|---|---|
| 2012 | base | +6.11% | 7.20% | 0 | 0 |
| 2013 | slow_lr | −6.48% | 12.99% | 0 | 0 |
| 2014 | high_entropy_slow | +9.81% | 10.88% | 0 | 0 |
| 2015 | high_entropy_slow | −8.62% | 23.22% | 0 | 1 |
| 2016 | high_entropy | +14.83% | 6.26% | 0 | 0 |
| 2017 | high_entropy_slow | +9.74% | 5.21% | 0 | 0 |
| 2018 | base | −4.09% | 14.96% | 0 | 0 |
| 2019 | slow_lr | +22.92% | 8.32% | 0 | 0 |
| 2020 | slow_lr | +8.80% | 32.82% | 0 | 0 |
| 2021 | slow_lr | +10.16% | 7.63% | 0 | 0 |
| 2022 | high_entropy | −10.55% | 17.31% | 0 | 3 |
| 2023 | high_entropy_slow | +18.19% | 14.20% | 0 | 0 |
| 2024 | slow_lr | +8.49% | 6.93% | 0 | 0 |

> **These are not performance results.** 40k timesteps per candidate against a configured 2,000,000 —
> roughly 2% of the budget — chosen to exercise the whole protocol across every fold in ~83 min.
> What the table establishes is that the *protocol* runs and its gates hold, not that the policy is good.
> A proper comparison against the baselines needs the full budget and belongs in Stage 12.
>
> `Max DD` is the **worst cell** of the 3×3 grid, not the primary cell — a model safe on eight cells and
> broken on the ninth is a broken model, so the aggregate takes the worst and sums the violations.

What the run established beyond the gate:

- **Selection is discriminating, not decorative.** All four candidates won at least two folds
  (`slow_lr` 5, `high_entropy_slow` 4, `base` 2, `high_entropy` 2). A rule that always picked the same
  candidate would mean the sweep was pointless.
- **The taxonomy fired where it should.** Only 2015 and 2022 produced `D_max` breaches, both classified
  **market-forced** with the evidence recorded — locked exposure, cash on hand, and the count of actions
  the lock blocked. 2020 shows a 32.8% worst-cell drawdown with zero preventable violations, which is
  the taxonomy doing exactly its job: COVID is not an implementation defect.
- **`--seeds K` now does something.** It retrains the *selected* hyperparameters at fresh seeds and
  reports the test spread; it does not re-run selection, because selecting K times and reporting the
  best is a different and much weaker claim.

---

## Phase 3 — Robustness  `[x]` — complete

### `[x]` Stage 9 — Stress · `python -m scripts.s09_stress --config config/evaluation.yaml`

- [x] Crisis windows (GFC, COVID, 2022, taper, volmageddon, 2018Q4)
- [x] `N` sensitivity sweep — 9 values, `{0, 7, 90, 180}` labelled out-of-distribution
- [x] `D_max` sensitivity sweep — 5 values
- [x] Combined grid (3×3 default; `--grid full` for 9×5)
- [x] **Gate:** realized drawdown monotone non-decreasing in `D_max` — **PASSED at every `N`**

**The gate, which is the single most important robustness check in the project:**

| `D_max` | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 |
|---|---|---|---|---|---|
| realized max DD (`N=30`) | 6.60% | 13.54% | 17.83% | 23.52% | 29.92% |
| mean cash weight | 98.3% | 77.1% | 79.4% | 76.8% | 73.9% |
| intervention rate | 99.6% | 85.3% | 79.7% | 76.7% | 74.3% |

Monotone at `N = 15`, `30` and `60` alike. A tighter ceiling never produced a deeper drawdown.

**The lock is binding, not inert.** Turnover falls **1,151 → 3.1** as `N` goes 0 → 180 (ratio 373×), and
the locked NAV fraction rises from 0% to ~20–27%. robustness.md §1.2 warns that a flat sweep is
indistinguishable from a silently-inert constraint; this is not flat.

**Two findings worth carrying forward:**

- **COVID is where the lock hurts.** Over 2020-02-19 → 2020-03-23 the policy took a **43.5% drawdown with
  95.7% of NAV locked** and 239 actions blocked. That window is precisely the one robustness.md flags —
  a 34% index fall in 23 sessions, faster than any `N` can release — and the answer is that the lock does
  trap the agent. Zero preventable violations: the machinery behaved, the constraint simply bit.
- **`D_max = 0.05` is qualitatively different.** 99.6% intervention and 98.3% cash: at the tightest ceiling
  the envelope is effectively forcing an all-cash portfolio. That is the over-calibration signal from
  risk-envelope.md §7, and it should be reported separately rather than averaged into the grid.

### `[x]` Stage 10 — Bootstrap · `python -m scripts.s10_bootstrap --config config/evaluation.yaml`

- [x] Stationary (Politis–Romano) block bootstrap over the **joint** cross-section
- [x] 1,000 replicates; confidence bands for every standard metric
- [x] Block-length sensitivity reported — `[5, 10, 21, 63]`, band width moves ≤ **1.19×**, so the
      conclusions do not depend on the choice

| Metric | observed | q05 | q50 | q95 |
|---|---|---|---|---|
| annualized return | 6.46% | 1.45% | 6.45% | 12.37% |
| volatility | 11.73% | 10.97% | 11.70% | 12.51% |
| Sharpe | 0.55 | 0.12 | 0.55 | 1.04 |
| max drawdown | 23.22% | 14.68% | 22.61% | 36.70% |

Blocks are drawn over **sessions**, so the same indices apply to every asset at once and the cross-asset
correlation survives by construction. IID resampling is refused by name — it destroys the volatility
clustering the whole risk layer exists to handle.

### `[x]` Stage 11 — Adversarial · `python -m scripts.s11_adversarial --config config/evaluation.yaml`

- [x] Equity shock + credit widening; duration loss; correlation spike; diversification breakdown
- [x] Blocks drawn from real history only, and from the fold's **training** window
- [x] "historical robustness, not a forward-looking guarantee" disclaimer in every output
- [x] **Zero lock, zero feasibility, zero preventable violations on all four paths**

| Scenario | policy return | policy maxDD | `spy_tlt_60_40` | `equal_weight` |
|---|---|---|---|---|
| equity shock + credit widening | −12.89% | 24.03% | −15.05% | −23.18% |
| duration loss | −0.00% | 0.01% | −18.23% | −20.09% |
| correlation spike | −7.38% | 7.79% | −1.46% | +1.28% |
| diversification breakdown | −7.43% | 9.74% | −3.94% | −0.83% |

Read the duration-loss row carefully before celebrating it: the policy sat in cash (0.8% locked) through a
window it had every reason to avoid, so the −0.00% is risk avoidance, not skill — and the correlation-spike
and diversification rows show it **losing to both baselines** on the paths designed to punish exactly the
"bonds are safe" reflex.

### `[x]` Stage 12 — Report · `python -m scripts.s12_report --config config/evaluation.yaml --policy-runs latest`

- [x] Acceptance table: hard engineering / risk reporting / performance, explicit pass-fail
- [x] A vs B violation tables kept separate — never summed
- [x] Baseline comparison incl. `classical_optimizer`, with `rl_policy` as one more column
- [ ] Refuses dirty-git runs unless `--allow-dirty` *(not implemented; the manifest records
      `git_dirty` but the stage does not yet refuse on it)*

**Superseded by later iteration in this same session** (grids revised twice more, Sortino added
then Sharpe restored as the primary judge, constrained baseline variants dropped from the default
report). The figures below are the FIRST acceptance run and are kept for the record; the current
numbers are in the **Current position** table at the top of this file.

Original run: acceptance 18/22 criteria pass, ZERO blocking failures, on the `D_max` 5-25% grid.
All four failures were non-blocking performance criteria — the policy's Sharpe of 0.55 lost to
`spy_buy_hold` (0.97), `spy_tlt_60_40` (0.90), `equal_weight` (0.77) and `momentum` (0.64); it beat
`cash` and `classical_optimizer`.

**Current best result** (`D_max` 1-15% grid, `s12_report --no-constrained`, market benchmarks
only): at the primary cell `D_max = 5%`, acceptance is **22/22** — the policy's Sharpe of 0.95
leads all six baselines (`spy_tlt_60_40` 0.90, `spy_buy_hold` 0.89, `equal_weight` 0.79,
`classical_optimizer` 0.72, `momentum` 0.70, `cash` 0.00). At `D_max` 1%, 10% and 15% it is 17/22,
20/22 and 20/22 — it does not lead at every ceiling, only at the one it is deployed at.

> **What Phase 3 establishes, stated precisely.** Every hard engineering criterion is zero across
> walk-forward, the stress grid and the adversarial paths, on every grid tried. The constraint
> machinery is correct and its limits are measured. Performance is a genuinely different claim: the
> policy was trained at ~2% of the configured budget, the 5% result is a single seed, and the
> ranking has moved with grid choice and reporting convention within this same session. It is
> promising, not settled — the full-budget run is what would settle it.

---

## Phase 4 — Reserved

### `[-]` Stage 13 — Daily inference · **specified, not implemented** (D4)

Spec in [reference/stages.md](reference/stages.md). The only requirement it places on current work is T2
(ledger + lock manager dict round-trip) — **satisfied in Stage 4**: `Ledger.to_dict/from_dict` and
`LockManager.to_dict/from_dict` round-trip losslessly, tested including over `hypothesis`-generated
states and after every step of a random buy/sell/hold sequence. Nothing further is owed to Stage 13
until it is built.

---

## Ablations — run once the primary path is complete

- [ ] Reward variants: + drawdown penalty; differential Sharpe
- [ ] Auxiliary projection penalty `lambda ∈ {0, 0.001, 0.01}`
- [ ] `lock.scope: portfolio` — the portfolio-wide lock variant
- [~] Risk aggregation `max` vs `mean` — **measured for the envelope in Stage 5** (both admissible;
      `mean` returned +7.60% vs `max` +7.35% at the primary `D_max`, inside the noise, and `max` was kept
      as the conservative default for a hard constraint). Still to run as a *policy* ablation
- [ ] Flat MLP vs shared per-asset encoder

---

## Open questions

| # | Question | Blocks | Status |
|---|---|---|---|
| ~~Q1~~ | ~~Risk-envelope calibration values~~ | — | **CLOSED 2026-09-01.** `quantile 0.05, horizon 5, aggregation max, measure cvar`, calibrated over 20 annual windows; written into `config/constraints.yaml` |
| Q5 | **Does `device: cpu` still win end to end?** Stage 6 measured the *network alone* at `obs_dim = 656`: CUDA 0.116 s vs CPU 0.278 s for 50 fwd+bwd — the GPU is **2.4× faster**, against D14's end-to-end CPU win of 2.7×. Not a contradiction (D14 measured rollout-inclusive wall clock) but D14's premise moved | Stage 7 wall clock | **Re-run the end-to-end comparison in Stage 7** at `n_envs = 8`. `cpu` remains the default until then |
| Q3 | **Does `proj_distance` decline without an auxiliary penalty?** Stage 7 gave mixed evidence at a 300k-per-rung budget: it fell within rung 3 (1.77 → 1.19) and rung 4 (1.20 → 0.99), but rose within rung 2 (1.35 → 1.78) and rung 5 (0.84 → 0.95). Sharpened by Stage 6: `infeasible_fallback` fires on **20.8%** of random-policy steps and **28.5%** at `D_max = 0.05`, and on each the action is discarded outright | Whether D9 mitigation 2 or 3 is needed | **Re-read on the full 2M-per-rung run** before concluding. Mitigation 2 (confirm the masks are non-degenerate) is already satisfied — observations are not re-normalized, so the masks reach the network intact |
| Q4 | Is the 3×3 stress grid sufficient, or is the full 7×5 needed? | Stage 9 runtime | Decide after Stage 8 timing is known |
| Q7 | **`gamma = 0.999` is inherited from the superseded parameter range.** It was justified by "`N` up to 180 calendar days"; under D16 the lock is ~30 calendar days (~21 sessions), for which 0.99 (~100 sessions) is already several times the constraint horizon. 0.999 gives ~1000 sessions, ~4 years, far longer than the longest episode (504) | Stage 7 credit assignment and sample efficiency | Settle on a **validation** year, the only place tuning is permitted. Not changed as part of D16, because D16 is a spec change and gamma is a tuned value |

---

## Decision log

| Date | Entry |
|---|---|
| 2026-08-31 | Plan written; D1–D12 locked ([reference/decisions.md](reference/decisions.md)). |
| 2026-08-31 | **D13:** lock scope is **per-ETF**, not portfolio-wide — buying ETF i relocks only ETF i. See [reference/lock-state-machine.md](reference/lock-state-machine.md) §0. |
| 2026-08-31 | Raw-price ledger / total-return NAV resolved as an explicit dividend-reinvestment ledger; T1 proves it. |
| 2026-08-31 | Baselines moved ahead of PPO (draft had them at Milestone 10, after training). |
| 2026-08-31 | **D15:** benchmark set fixed by the user — `spy_buy_hold`, `momentum` (12-1), `spy_tlt_60_40` as primary; `cash`, `equal_weight`, `classical_optimizer` retained because the acceptance criteria name them. Replaces the vague `spy` / `static_risk_aware` placeholders. See [reference/baselines.md](reference/baselines.md). |
| 2026-09-01 | **D16: the lock is centred on 30 calendar days.** Operating range `[15, 21, 30, 42, 60]` — a √2 ladder, weighted `[.15, .20, .30, .20, .15]`, weighted geometric mean 29.9 — with `{0, 7, 90, 180}` retained as **out-of-distribution** stress points for Stage 9 only. Canonical in the new `config/constraints.yaml`; `reference/decisions.md` D16 carries the rationale; 15 tests pin the grid and assert no document still quotes the superseded `[0, 7, 14, 30, 60, 90, 180]` as current. Raises **Q7** (gamma). |
| 2026-09-01 | **`config/constraints.yaml` written.** `architecture.md` had always listed it as a base config; it now exists, typed and validated, carrying the lock, the drawdown ceiling, the projection backend, the risk estimators (starting values, pending the Stage 5 calibration in Q1), the crisis-window library, and `cost_bps: 0.0` (D10). Stage 4 reads it. |
| 2026-09-01 | **Phase 0 complete.** Stages 0-3 green; 84 tests pass in 4.9s (budget: 60s). Pipeline runs end to end from an empty `data/`. |
| 2026-09-01 | **Calendar bounds pinned.** `exchange_calendars` defaults to a rolling 20-year window; unbounded it started 2006-09-01 and would have dropped 2003-2006 from every frame. `CALENDAR_START = "2002-01-01"`. |
| 2026-09-01 | **Quality-gate checks taxonomized by `kind`.** Price checks (positivity, split-jump, OHLC range) apply to prices only; level series get a per-symbol `range` band. Driven by two facts in the real data: `^IRX` closed negative for 7 sessions in March 2020, and `^VIX` closed at 82.69 / `^VVIX` at 207.59 on 2020-03-16. |
| 2026-09-01 | **`TOTAL_RETURN_TOLERANCE = 5e-3`, set from measurement not from float epsilon.** The dividend-reinvestment ledger reconstructs `close_adj` to a worst case of 9.57e-4 (SHY) across all 24 tradables; GLD is exactly 0. The residual is provider rounding over ~280 distribution events. Recomputed every Stage 2 run, so T1 has a measured baseline before Stage 4 starts. |
| 2026-09-01 | **XLRE zero-volume waiver.** Five bad volume prints in XLRE's first quarter (Oct 2015 - Jan 2016), two on consecutive sessions. Prices intact; only the volume features are affected, and only inside every fold's training window. Waived in `known_exceptions` with a reason rather than smoothed over — and a test pins the waiver so it cannot quietly widen. |
| 2026-09-01 | **Volume gaps are adjudicated in Stage 2 only.** The gate hard-fails a zero-volume run longer than the forward-fill policy allows; the feature layer then carries the last good volume forward, inventing nothing the gate has not already accepted. |
| 2026-09-01 | **`ret_w` and `logret_w` are both kept** (|r| ≈ 0.98-1.00). Ranks want arithmetic, volatility wants log. The redundancy is now *visible* in the correlation diagnostic rather than accidental, per [reference/features.md](reference/features.md) §1 — a decision, not an accident. Revisit at observation-selection time (Q6). |
| 2026-09-01 | **Publication lag verified end to end.** April 2020's 14.8% UNRATE print first becomes visible on session 2020-07-06 (obs + 95d), against a real release of 2020-05-08 — conservatively late, which is the safe direction. |
| 2026-08-31 | **D14:** GPU resolved. RTX 3060 Ti verified working with `torch==2.13.0+cu126`. Benchmarked end-to-end PPO at CPU 2.35s vs CUDA 6.44s → **`training.device: cpu`**. GPU wins raw matmul 8.6x but loses the real job 2.7x; rollout transfer latency dominates. See [reference/gpu-setup.md](reference/gpu-setup.md). |
| 2026-09-01 | **Phase 1 complete.** Stage 4 green (the gate) and Stage 5 green; 217 tests pass in 52s. Zero lock and zero feasibility violations across all six baselines and every simulator run. |
| 2026-09-01 | **Capital preservation caps SHARE COUNTS, not weights.** A weight cap reads as "restore yesterday's weight", which in a falling market is an instruction to buy the dip daily. Enforced at execution alongside the lock floor — the weight bound is the optimizer's guide, the share bound is the law. Realized drawdown is now monotone in `D_max`. |
| 2026-09-01 | **Convexity replaces monotonicity as the risk-envelope requirement.** `w_safe` minimizes exposure, not risk (measured 7x riskier than a hedged book), so [risk-envelope.md](reference/risk-envelope.md) §5's monotonicity premise is false. The alpha bisection needs a convex sublevel set; `measure: cvar` provides it, `var` does not. |
| 2026-09-01 | **The envelope is calibrated over ANNUAL windows.** On one 21-year path the intervention rate measures time-under-water (0.9984 correlation to "already breached") because the peak never resets. Corrected in [risk-envelope.md](reference/risk-envelope.md) §7. |
| 2026-09-01 | **Q1 closed:** `quantile 0.05, horizon_days 5, aggregation max, measure cvar`. Selected by design constraint with return as tie-break — a 2-day horizon returned +0.5pp more and was rejected, because tuning the risk layer on return is the failure the envelope exists to prevent. |
| 2026-09-01 | **Q6 closed: the observation is 656 dimensions**, declared in `config/observation.yaml` and validated against the feature manifest at startup. 20 of 64 per-asset columns and 21 of 99 macro; the near-collinear `logret_*` twins are dropped, since Stage 2's diagnostic put them at \|r\| = 0.98–1.00 against `ret_*`. |
| 2026-09-01 | **The fold scaler is not optional for the environment.** Unscaled, **3.7%** of observation entries sit hard against the ±10 clip — one input in twenty-seven degraded to a saturated constant — against **0.001%** with a fold scaler applied. `environment.fold_id` is now required rather than nullable. |
| 2026-09-01 | **The action space is `[-1, 1]`, scaled by `LOGIT_SCALE = 10` inside the env.** [reference/env-mdp.md](reference/env-mdp.md) §8 specified `Box(-inf, inf)`; unbounded defeats SB3's action clipping, and a wide bound leaves PPO's unit-variance Gaussian head sampling only near-uniform allocations. Both checkers now pass with zero warnings. |
| 2026-09-01 | **The lock is what empties the risk envelope's feasible set.** With `N = 0`, `w_safe` is all cash and the fallback fires on 0.7% of steps at `D_max = 0.05`; with `N = 30` it fires on **28.5%**, though the envelope *binds* less often. The lock floors `w_safe` at holdings that carry real risk. Recorded in [reference/risk-envelope.md](reference/risk-envelope.md) §6b; it makes `infeasible_fallback` a first-class Stage 7 diagnostic and `D_max = 0.05` a cell to report separately. |
| 2026-09-01 | **`src/sim/engine.py` extracted.** The per-step body is now shared verbatim between `simulate()` and `env.step`, so a baseline and a policy cannot drift onto different machinery — which was the whole reason for running the baselines through the simulator. Stage 4/5 behaviour is unchanged: all 108 Phase-1 tests still pass. |
| 2026-09-01 | **Q2 closed: `n_envs = 8`, `SubprocVecEnv`** (2,844 steps/s). 8 → 16 workers buys 3.7% for twice the processes. |
| 2026-09-02 | **Phase 3 complete.** Stages 9-12 green; acceptance is 18/22 with **zero blocking failures**. Every hard engineering criterion is zero across walk-forward, the stress grid and the adversarial paths. |
| 2026-09-02 | **The `D_max` monotonicity gate PASSED** at `N = 15, 30, 60`: realized drawdown runs 6.60% → 13.54% → 17.83% → 23.52% → 29.92% as the ceiling loosens. This is the check that, had it failed, would have voided every risk number in the report. |
| 2026-09-02 | **The lock is measurably binding.** Turnover falls 1,151 → 3.1 as `N` goes 0 → 180 (373×), locked NAV rises to ~27%. robustness.md §1.2 warns a flat sweep is indistinguishable from an inert constraint; it is not flat. |
| 2026-09-02 | **COVID traps the agent, as designed to be tested.** 43.5% drawdown with 95.7% of NAV locked over 2020-02-19..03-23 — a fall faster than any `N` can release — with zero preventable violations. The machinery behaved; the constraint bit. |
| 2026-09-02 | **`D_max = 0.05` is over-calibrated:** 99.6% intervention, 98.3% cash. Report it separately rather than averaging it into the grid. |
| 2026-09-02 | **The adversarial splice chains RETURNS, never price levels**, and replays blocks **chronologically**. Two bugs found by doing so: seeding all asset levels at row 0 leaves anything not yet born (GLD, XLRE) at NaN forever, so the availability mask promised tickers the price grid lacked; and score-ordered blocks run backwards through time, letting a held ETF un-exist — an illegal state (I5), not an adversarial one. |
| 2026-09-02 | **Combination scenarios must exclude each other's blocks.** The worst equity window and the worst credit window are usually the same window — 2008 hit both — so "equity shock + credit widening" silently became "the 2008 crash, twice" until the groups were made disjoint. |
| 2026-09-02 | **Suite runtime is ~175 s, not the 60 s recorded earlier — and it is not a regression.** The identical Stage 7 commit, checked out in a worktree, measures the same 104 s for `tests/sim` alone. The cost is `_dual_theta`'s bisection (already converging in ~53 iterations) over many full-history simulations; the machine is simply slower than when 60 s was recorded. |
| 2026-09-01 | **Stage 8 PASSED — the Phase 3 gate is open.** 13 folds, 4 candidates each, 9 evaluation cells per fold: zero lock, zero feasibility, zero preventable `D_max` violations, and no fold marked `constraint validation failure`. T11, T12 and T13 all have homes and pass; every test in [testing.md](reference/testing.md) §3 is now covered. |
| 2026-09-01 | **The preventable-violation detector needs the dividend stream, and this was found the hard way.** Its first real run produced **164 false positives** on `equal_weight` 2020 alone. A trajectory records shares *after* distributions are reinvested as accretion, and accretion is deliberately exempt from both the lock and the capital-preservation cap. The fix is exact — `shares_close = shares_executed × (1 + div/close)` inverts cleanly — and a threshold would not have worked: XLE paid \$0.2637 on 2020-03-23 into a collapsed \$11.79 price, a **2.24% one-day accretion** no fixed band separates from dip-buying. Corrected in [evaluation.md](reference/evaluation.md) §4. |
| 2026-09-01 | **`trajectory.parquet` records `proj_weights` and `raw_weights`.** The taxonomy specifies a *replay*, and a replay needs the action the projection actually produced. Inferring legality from the resulting position is much weaker — prices move between the decision and the close. |
| 2026-09-01 | **`config/evaluation.yaml` now extends `training.yaml`.** Stage 8 trains, so it needs the observation selection, the policy architecture and the PPO block, not just the simulator settings. |
| 2026-09-01 | **`simulate()` gained `start_row`/`end_row`, and the policy is a weight source.** A trained policy wrapped as a `WeightFn` runs through the same `simulate()` as the six baselines and emits an identical `trajectory.parquet` — which is why the Stage 12 report needs no special case for it. |
| 2026-09-01 | **Stage 7 complete.** Five curriculum rungs, warm-started in order, all five gates green with zero lock and zero feasibility violations. 582,620-parameter policy; 1.5M timesteps in 18 min at a demonstration budget. 307 tests pass in 60s. |
| 2026-09-01 | **Observations are NOT re-normalized by `VecNormalize`** — a correction to [rl-training.md](reference/rl-training.md) §2. Measured: the fold scaler already leaves the observation at mean 0.20 / std 0.83, and **19 of the 656 dimensions are constant `pf_available` bits** that `VecNormalize` would map to exactly 0 — destroying the masks the policy reads to mask its own logits, which is the silent failure §3 names. Rewards are still normalized (log returns are ~1e-3). |
| 2026-09-01 | **The actor head is per-asset, not a flat `Linear(trunk → K+1)`** — a second correction to §2. A flat head has separate weights per asset and voids the equivariance the shared encoder buys; `logit_i = head([e_i, context])` restores it, with a separate small head for CASH. Both the equivariance and the flat per-asset parameter count are asserted in tests rather than claimed. |
| 2026-09-01 | **Availability masking uses -5, not `-inf`.** The action space is bounded `[-1, 1]`, so `-inf` is unavailable and would NaN the Gaussian log-prob; after `LOGIT_SCALE` a -5 mean is worth `e^-50` of relative weight. |
| 2026-09-01 | **A `slow` pytest marker now exists**, deselected by default. Real PPO updates cost ~8s against a 60s suite budget that is already at its limit; run `pytest -m slow` before trusting a training run. |
| 2026-09-01 | **The comparison report format is pinned before training**, as `scripts/s12_report.py` + `src/evaluation/{report,render,categories}.py`, with a baseline-only reference report in `artifacts/reports/baselines/`. A format settled after seeing results is a format chosen to flatter them. The RL policy becomes one more column and nothing else changes. |
| 2026-09-01 | **Every report year is an INDEPENDENT evaluation window** (fresh capital, peak reset each January). Forced, not stylistic: on the continuous 2004–2024 run `spy_buy_hold` breached `D_max` in 2009, went 100% cash and stayed there for fifteen years — every annual cell from 2009 on would have read 0.00%. It also matches how walk-forward evaluates the policy, one model per test year, which is what makes the RL column comparable. |
| 2026-09-01 | **Report Sharpe uses `rf = 0`.** CASH returns exactly 0.00%/day and is the agent's outside option, so raw return *is* excess return; a T-bill rate would make CASH a negative-carry asset the simulator does not model. |
| 2026-09-01 | **Report allocation is the TIME AVERAGE of daily weights**, in pp summing to 100 across the seven `universe.yaml` categories, asserted per cell. A year-end snapshot cannot distinguish 60% equity all year from 60% in December only. |
| 2026-09-01 | **`simulate()` takes `start_row`/`end_row`.** The decisions are bounded, the market is not, so a 252-day lookback still works on the first session of an evaluation window. Stage 8's walk-forward needs the same thing. |
| 2026-09-01 | **Stage 6 complete.** 7/7 gate checks, 31,500 random-policy steps, zero invariant violations; 260 tests pass in 58s. |
| 2026-09-01 | **A drawdown ceiling against a never-resetting peak is far harsher than the same ceiling per fold.** Stage 8 and Stage 12 must state which convention a result used; the two are not comparable. |
| 2026-09-03 | **Grids refined again, by request: N to nine rungs (2^(1/4) ladder, 15-60), D_max to seven rungs spanning 1%-15%.** Retrained cleanly across all 13 folds; hard acceptance passed. Stage 9 monotonicity still holds. |
| 2026-09-03 | **At the 5% ceiling the policy now beats three of four constrained baselines on Sortino** (1.33 vs spy_tlt_60_40* 1.23, spy_buy_hold* 0.95, equal_weight* 1.13), losing only to momentum_constrained (1.62). Bootstrap band 0.66-2.08 (median 1.33) corroborates it. Acceptance 27/28 at 5%, its best result yet. At 1%, 10% and 15% the policy still trails most baselines (18/28, 23/28, 23/28). |
| 2026-09-04 | **2025 added to the backtest.** `folds.json` already carried `fold_2025` from Stage 3 (250 sessions, its own scaler); only `--last-year` needed to move. Retrained cleanly, 14/14 folds, hard acceptance passed; Stage 9 monotonicity and Stage 11 zero-violations both still hold. 2025 alone: `high_entropy_slow` selected, +7.37% return, 16.51% max DD, zero preventable violations. |
| 2026-09-04 | **At N=30, D_max=5%, 2012-2025, market benchmarks only: acceptance is 22/22.** `rl_policy` Sharpe 0.92 leads all six baselines, including spy_tlt_60_40 (0.90) and spy_buy_hold (0.89) for the first time on a like-for-like unconstrained comparison. Bootstrap Sharpe band 0.47-1.41 (median 0.95) sits entirely above zero. Still a single seed at ~2% of the configured training budget. |
| 2026-10-05 | **The full-budget run becomes an incremental campaign**, grown in idle-time sessions with an evaluation at each budget doubling (2, 4, 8, 16, 32, 64, 100%), each compared with the baselines and every earlier checkpoint. Plan and record: [INCREMENTAL_TRAINING_PLAN.md](INCREMENTAL_TRAINING_PLAN.md). Built as `scripts/s14_incremental.py` + `src/training/incremental.py` + `src/evaluation/learning_curve.py`; fold scoring moved to `src/evaluation/fold_eval.py`, shared with Stage 8 unchanged. |
| 2026-10-05 | **A resumed run is bit-identical to an uninterrupted one**, proven by a slow test (weights, Adam state, reward normalizer). Achieved by one `learn()` per rollout with a forced env reset and a seed derived from `(candidate seed, rollouts done)`, so the learning curve does not depend on where sessions stopped. Budget is counted in whole 16,384-step rollouts: the 2% run asked for 40,000 timesteps and SB3 ran 49,152 = 3 rollouts, which is exactly C0. |
| 2026-10-05 | **Campaign `budget_v1` created from `s08_walk_forward_20260903T224114Z_a207776c`** — all 56 models verified (hyperparameters, seed, 49,152 timesteps) and copied; the source run is untouched. CPU, by decision. |
| 2026-10-05 | **Observation, not changed: the KL early-stop truncates the lr 3e-4 candidates.** `base` and `high_entropy` stopped inside the first of 10 PPO epochs on every rollout in all 14 folds (`target_kl = 0.02`); the lr 1e-4 candidates ran up to 10. Kept as-is by decision so the 2% models remain a valid C0 and because it is a tuning question for validation years; the learning curve reports epochs per rollout at every checkpoint. |
| 2026-10-06 | **Incremental session 1 (58 min, stopped by request): C0 reproduced exactly** — test Sharpe 0.92, bootstrap band 0.47-1.41, all 14 fold selections identical to the original 2% run, hard acceptance passed. Then 112 rollouts to C1 (4%); C1 evaluation 12/14 folds done when stopped. Measured: 19.5 s per rollout, 49 s per fold evaluation, 51 s for the reports. The lr 3e-4 candidates again ran 1 PPO epoch on every one of their 56 rollouts. |
| 2026-10-06 | **Incremental session 2 (2 h): C1 and C2 reported.** Test Sharpe 0.92 (2%) -> 0.44 (4%) -> 0.64 (8%); validation Sharpe (selected) 1.34 -> 1.12 -> 1.23, all-candidate 0.76 -> 0.61 -> 0.70. Paired verdicts: validation **no clear change** at both C1 and C2; test regressed at C1, no clear change at C2. Worst test year drawdown 10% -> 26%. `infeasible_fallback` 48% -> 61%, cash weight 28% -> 43%; lr 1e-4 candidates now run all 10 epochs, lr 3e-4 still ~1. Hard acceptance passed at every checkpoint. Two flat validation checkpoints in a row -- the plan's signal -- so the campaign is paused, not abandoned (state saved at 9.06%). |
| 2026-10-06 | **Data coverage: no model has trained on anything after 2023.** Data runs to 2026-08-31; fold_2025 trains 2004-2023, validates 2024, tests 2025, and Jan-Aug 2026 is unused. Correct for an honest backtest, and not the cause of the flat curve, but it means no deployable model exists. Plan: [CONTINUAL_TRAINING_PLAN.md](CONTINUAL_TRAINING_PLAN.md). Raised alongside it: periodic fine-tuning instead of retraining from zero, which needs (1) versioned data -- re-running Stages 1-3 in place rewrites adjusted history and would silently change the inputs under a running campaign, which only checks the config hash; (2) a chained walk-forward that tests the fine-tuning procedure itself; (3) a scaler policy for fine-tuning; (4) a champion/challenger gate on a recent holdout. |
| 2026-10-06 | **A 12% checkpoint (C2b) inserted into `budget_v1`** with the new `s14_incremental --add-checkpoint`; the six candidates already past 15 rollouts were rolled back to their C2 snapshot and retrained. **Session 3 (4 h): C2b reported** -- test Sharpe 0.84 (band 0.39-1.37; 0.85 at D_max 10%, 0.91 at 15%, the best so far at the looser ceilings), worst test year drawdown 15.9%, but validation (selected) 1.23 -> 0.94, all-candidate 0.70 -> 0.74: **no clear change** on validation for the third checkpoint running while test recovered, i.e. the two disagree. Rollouts slowed from ~15 s to ~30 s in the session's second half, so the deadline fell 6 rollouts short of C3 (16%). |
| 2026-10-06 | **C3 (16%) reported.** Test Sharpe 0.87 (band 0.42-1.41; 0.89 at D_max 10%, 0.83 at 15%), worst test year drawdown 12.7%, gap to spy_tlt_60_40 -0.03, to momentum_constrained -0.26. Validation (selected) 1.14, all-candidate 0.68; vs C2b no clear change on both. **Over 2% -> 16% (8x the budget) the all-candidate validation Sharpe has not moved (0.76, 0.61, 0.70, 0.74, 0.68)** and test is back near, not above, C0. `infeasible_fallback` ~60% of training steps throughout; lr 3e-4 candidates now ~2.5 epochs per rollout. Hard acceptance passed at every checkpoint. |
| 2026-10-06 | **Full Stages 9-12 on the 16% model (C3): acceptance 24/28, no blocking failures** (8/8 hard, 8/8 risk reporting; performance loses to spy_buy_hold 0.89, spy_tlt_60_40 0.90, its constrained version 0.88 and momentum_constrained 1.13 at Sharpe 0.87). Report: `artifacts/reports/budget_v1_C3_full/`. `s12_report` fixed in passing: with `--policy-runs`, the acceptance table now reads that run's own walk-forward summary instead of silently taking the newest Stage 8 run. |
| 2026-10-06 | **C3b (22%) inserted and reported, then full Stages 9-12: acceptance 17/28 with one BLOCKING failure.** Test Sharpe 0.53 (band 0.06-1.08), **regressed** vs 16% (paired 90% interval -0.63..-0.03); validation 1.23, no clear change. Worst test year drawdown 25.8%; loses to every baseline but cash. **Stage 9 D_max monotonicity FAILED at N=30:** on the continuous 2012-2026 sweep, D_max 10% gave a 24.2% drawdown against 19.6% at 15%. Zero preventable violations (0 lock / 0 feasibility / 0 preventable under stress and adversarial; 42 breaches, all market-forced), so the safety layer did not let through a forbidden action. The mechanism is path-dependence: at 10% the policy breached early, then spent 44% of sessions in capital preservation holding 55% of NAV locked, so it could neither add nor cut and rode the drawdown down; at 15% it never breached, stayed ~99% invested and recovered (+171% vs +41%). The gate is per-path; Stage 9's own message says run more seeds before treating it as a bug. Report: `artifacts/reports/budget_v1_C3b_full/`. |
| 2026-10-07 | **Session 6 (9 h): C4 (32%) reported, then full Stages 9-12 on it: acceptance 23/28, no blocking failures** (Stage 9 monotonicity passes again). Test Sharpe 0.84 (band 0.39-1.30), worst test year drawdown 10.7%, annual return 6.3% at 7.5% vol; loses to spy_buy_hold 0.89, spy_tlt_60_40 0.90, its constrained version 0.88, equal_weight_constrained 0.85 and momentum_constrained 1.13. Validation 1.08 (selected), 0.72 (all candidates); vs 22%: no clear change on both. **Across 2% -> 32% (16x the budget) all-candidate validation Sharpe stays 0.61-0.76 and test swings 0.44-0.92 with no trend.** By ceiling at 32%: 0.85 at D_max 5%, 0.44 at 10%, 0.94 at 15%. A 40% checkpoint (C4b) was added; rollouts slowed to ~34 s (machine busy) and the session ended at 37.84%. Report: `artifacts/reports/budget_v1_C4_full/`. |
| 2026-10-07 | **Model redesign planned: [MODEL_REDESIGN_PLAN.md](MODEL_REDESIGN_PLAN.md).** Decided: stop adding budget to `budget_v1` (kept as the control); no transaction costs anywhere (D10 stands); one hyperparameter setting x three seeds instead of four settings x one seed; momentum used as a starting point (imitate, then free RL), not an anchor; no network or algorithm upgrades until the environment is fixed. Planned: event-driven, lock-aware environment where the agent decides only when it has a choice, with a per-decision holding-period reward and action-effectiveness metrics. Open: whether the risk layer should treat locked positions as sunk. |
| 2026-10-08 | **Decided (redesign D-F): when over the drawdown budget, the risk layer allows only risk-reducing actions.** The reference is "do nothing" (locked positions held, the rest in cash): an executed action's stressed loss must not exceed it. Risk-reducing proposals (e.g. hedging with cash) execute as proposed, with the amount chosen by the agent; risk-increasing ones are scaled back proportionally toward "do nothing". Replaces today's freeze into the safe portfolio, which also forbade hedging. Spec: [MODEL_REDESIGN_PLAN.md](MODEL_REDESIGN_PLAN.md) section 5d. |
| 2026-10-08 | **Correction to the budget_v1 conclusion: with the hyperparameter setting FIXED, budget does help.** Pooled over all 14 validation years, the lr 3e-4 settings improve with budget (high_entropy 0.81 -> 0.95, base 0.69 -> 0.89 validation Sharpe, 2% -> 32%) while the lr 1e-4 settings decline (slow_lr 0.78 -> 0.56). Per-year selection from a single validation year kept picking the slow settings when they got lucky, which flattened the reported curve. Scoring the fixed settings on test (setting chosen on validation, so legitimate): high_entropy 0.41, 0.39, 0.50, 0.50, 0.48, 0.58, 0.72 and base 0.57, 0.41, 0.56, 0.69, 0.80, 0.74, 0.70 (average Sharpe over D_max 5/10/15%), about +0.07 per budget doubling for both. Absolute level still below spy_tlt_60_40 (0.90). Strongly supports redesign D-C (one setting, several seeds); calls D-A (stop adding budget) into question. Data: `artifacts/incremental/budget_v1/analysis_fixed_setting/`. |
| 2026-10-08 | **Redesign Phase 1 built: seed-ensemble campaigns.** `s14_incremental --init-fresh --setting NAME --seeds K` creates a campaign of one fixed setting and K seeds per fold, trained from scratch, with no per-year selection; the reported policy is the ensemble (mean of the seeds' portfolios, `src/evaluation/rollout.policy_weights`). Checkpoints score every seed on validation and test at D_max 5/10/15%, and the learning curve reports per-seed results, the seed range (the noise floor) and the ensemble. Tested (25 incremental tests, 398 total) and run end to end on a toy setup (fresh training is bit-reproducible across runs; toy artifacts deleted). **Campaign `seeds_v1` created: high_entropy x seeds 1001-1003 x 14 folds, 0 rollouts, not yet trained.** Plan D-A revised: keep adding budget to a fixed setting while validation keeps rising. |
