# STATUS — Task Tracker

**The single file to update as work proceeds.** Plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
Stage runbook: [reference/stages.md](reference/stages.md).

Legend: `[ ]` not started · `[~]` in progress · `[x]` done (tests green) · `[!]` blocked · `[-]` deliberately deferred

---

## Current position

| | |
|---|---|
| **Current stage** | **Phase 1 complete — Stage 4 is GREEN.** Next: Stage 6, the env smoke test |
| **Run this** | `python -m scripts.s06_smoke_env --config config/training.yaml --episodes 500` *(not yet written)* |
| **Next gate** | Stage 8 (walk-forward) gates Phase 3 |
| **Lock period** | `N = 30` calendar days (D16); operating range `[15, 21, 30, 42, 60]` in `config/constraints.yaml` |
| **Risk envelope** | **Calibrated** (Q1 closed): `quantile 0.05, horizon 5, aggregation max, measure cvar` |
| **Test suite** | 217 passed, 1 deselected (`network`), 52s — `.venv/Scripts/python.exe -m pytest -q` |
| **Last updated** | 2026-09-01 |

---

## Finished

**Phases 0 and 1 are complete.** Stages 0–5 are `[x]`: every gating test named for them in
[reference/testing.md](reference/testing.md) §3 passes, and each stage runs end to end from a
clean checkout.

| | Stage | Entry point | Gate | State |
|---|---|---|---|---|
| **Phase 0** | 0 — Preflight | `s00_check_env` | T16 | `[x]` 35/35 checks |
| | 1 — Fetch | `s01_fetch_data` | symbol pins, coverage | `[x]` 31 symbols + 10 FRED series |
| | 2 — Curate | `s02_curate_data` | I5, T15 data-side, gate at zero | `[x]` 172,202 rows, 0 hard violations |
| | 3 — Features | `s03_build_features` | I6, T10, manifest | `[x]` 163 columns, 14 fold scalers |
| **Phase 1** | **4 — Simulator** | `s04_simulate` | **I1–I6, T1–T7, T15** | `[x]` **THE GATE — GREEN** |
| | 5 — Baselines | `s05_run_baselines` | zero violations + 5 checks | `[x]` all six, all checks pass |

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
| T16 | stage harness | `tests/test_stage_harness.py` |

Still to prove: **T8, T9, T11, T12, T13, T14** — all belong to Stages 6–8, which are not started.
(T9 determinism is already asserted at the Stage 4 level; the vectorized-worker form is Stage 6's.)

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

## Phase 2 — Environment and RL  `[ ]` — **unblocked** (Stage 4 is `[x]`), not started

### `[ ]` Stage 6 — Env smoke test · `python -m scripts.s06_smoke_env --config config/training.yaml --episodes 500`

- [ ] `gymnasium` + SB3 `check_env` pass
- [ ] 500 random episodes across the full `(N, D_max)` grid, invariants asserted every step
- [ ] Observation bounds/dtype/finiteness
- [ ] **Throughput benchmark** — Dummy vs Subproc at 1/4/8/16 workers, cpu vs cuda → record below
      (D14 set `device: cpu` from a Pendulum proxy; re-confirm on the real env, and re-run whenever the
      policy architecture changes)
- [ ] **Gate:** T8, T9, T14, zero violations

> Throughput results: _(fill in — worker count for Stage 7 is chosen from this, not guessed)_

### `[ ]` Stage 7 — Train PPO · `python -m scripts.s07_train_ppo --config config/experiments/ppo_stageK.yaml`

- [ ] Shared per-asset encoder policy; availability masked in logits
- [ ] `VecNormalize` fitted on training window only, frozen at eval, saved with the policy
- [ ] D9 wiring: `a_raw` stored, `a_proj` executed, `proj_distance` logged
- [ ] Diagnostic callbacks + baseline reference lines on TensorBoard
- [ ] `[ ]` Curriculum 1 — `N=0`, mechanics only
- [ ] `[ ]` Curriculum 2 — lock, `N ∈ {7,30}`
- [ ] `[ ]` Curriculum 3 — full parameter range
- [ ] `[ ]` Curriculum 4 — risk envelope on
- [ ] `[ ]` Curriculum 5 — fully randomized + reservoir resets
- [ ] **Gate per stage:** beats `cash`, zero lock/feasibility violations

### `[ ]` Stage 8 — Walk-forward · `python -m scripts.s08_walk_forward --config config/evaluation.yaml`

- [ ] Expanding annual folds, train/val/test with no overlap
- [ ] Lexicographic model selection; `selection.json` per fold
- [ ] Preventable-vs-market-forced violation taxonomy incl. the replay detector
- [ ] Per-fold standard metrics
- [ ] **Gate:** T11, T12, T13; zero lock/feasibility violations on every fold

---

## Phase 3 — Robustness  `[!]` blocked until Stage 8 is `[x]` (D12)

### `[ ]` Stage 9 — Stress · `python -m scripts.s09_stress --config config/evaluation.yaml --policy <path>`

- [ ] Crisis windows (GFC, COVID, 2022, taper, volmageddon, 2018Q4)
- [ ] `N` sensitivity sweep
- [ ] `D_max` sensitivity sweep
- [ ] Combined grid (3×3 default; `--grid full` for 7×5)
- [ ] **Gate:** realized drawdown monotone non-decreasing in `D_max`

### `[ ]` Stage 10 — Bootstrap · `python -m scripts.s10_bootstrap --config config/evaluation.yaml`

- [ ] Stationary / moving-block over the **joint** cross-section
- [ ] 1000 replicates; confidence bands for every standard metric
- [ ] Block-length sensitivity reported

### `[ ]` Stage 11 — Adversarial · `python -m scripts.s11_adversarial --config config/evaluation.yaml`

- [ ] Equity shock + credit widening; duration loss; correlation spike; diversification breakdown
- [ ] Blocks drawn from real history only
- [ ] "historical robustness, not a forward-looking guarantee" disclaimer in every output

### `[ ]` Stage 12 — Report · `python -m scripts.s12_report --config config/evaluation.yaml --runs <ids>`

- [ ] Acceptance table: hard engineering / risk reporting / performance, explicit pass-fail
- [ ] A vs B violation tables kept separate
- [ ] Baseline comparison incl. `classical constrained`
- [ ] Refuses dirty-git runs unless `--allow-dirty`

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
| Q2 | Worker count / vectorization strategy | Stage 7 throughput | Resolve from the Stage 6 benchmark. **Stage 4 measured ~1.6 ms/step with the envelope on** (~625 steps/s single-threaded), and the envelope is ~two-thirds of it — so env stepping, not the network, is the bottleneck, as D5 assumed |
| Q5 | Does `device: cpu` still win on the real env and at high worker counts? Margin is only **1.2x** | Stage 7 wall clock | Confirm in Stage 6; re-run on any architecture change |
| Q3 | Does `proj_distance` decline without an auxiliary penalty? | Whether D9 mitigation 3 is needed | Observe in Curriculum stage 2–3 |
| Q4 | Is the 3×3 stress grid sufficient, or is the full 7×5 needed? | Stage 9 runtime | Decide after Stage 8 timing is known |
| Q7 | **`gamma = 0.999` is inherited from the superseded parameter range.** It was justified by "`N` up to 180 calendar days"; under D16 the lock is ~30 calendar days (~21 sessions), for which 0.99 (~100 sessions) is already several times the constraint horizon. 0.999 gives ~1000 sessions, ~4 years, far longer than the longest episode (504) | Stage 7 credit assignment and sample efficiency | Settle on a **validation** year, the only place tuning is permitted. Not changed as part of D16, because D16 is a spec change and gamma is a tuned value |
| Q6 | **Which feature columns enter the observation?** Stage 3 emits **64 per-asset** columns (50 etf + 14 cross-sectional); 24 assets × 64 = 1,536 before the macro block, against the ~300-dim policy the D14 benchmark assumed | Stage 6 obs size, Stage 7 wall clock, and whether D14 still holds | Select in Stage 6, validated against `feature_manifest.json`. The manifest exists so the selection is explicit rather than implicit |

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
| 2026-09-01 | **A drawdown ceiling against a never-resetting peak is far harsher than the same ceiling per fold.** Stage 8 and Stage 12 must state which convention a result used; the two are not comparable. |
