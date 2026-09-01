# STATUS — Task Tracker

**The single file to update as work proceeds.** Plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
Stage runbook: [reference/stages.md](reference/stages.md).

Legend: `[ ]` not started · `[~]` in progress · `[x]` done (tests green) · `[!]` blocked · `[-]` deliberately deferred

---

## Current position

| | |
|---|---|
| **Current stage** | **Phase 0 complete.** Next: Stage 4 — the deterministic simulator |
| **Run this** | `python -m scripts.s04_simulate --config config/sim/default.yaml` *(not yet written)* |
| **Next gate** | Stage 4 — no RL code before it is green |
| **Lock period** | `N = 30` calendar days (D16); operating range `[15, 21, 30, 42, 60]` in `config/constraints.yaml` |
| **Test suite** | 99 passed, 1 deselected (`network`), 4.4s — `.venv/Scripts/python.exe -m pytest -q` |
| **Last updated** | 2026-09-01 |

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

### `[ ]` Stage 4 — Deterministic simulator · `python -m scripts.s04_simulate --config config/sim/default.yaml`

**This is the gate. No RL code exists until every box here is checked.**

- [ ] `Ledger`: fractional shares, cash, long-only, dict round-trip
- [ ] Execution: next-open, sells-then-buys, `cost_bps` parameter set to 0.0, price-in-range check
- [ ] **Total-return NAV via dividend reinvestment** ([reference/portfolio-ledger.md](reference/portfolio-ledger.md) §3)
- [ ] NAV, running peak, drawdown
- [ ] `LockManager`: all four transitions + dividend carve-out
- [ ] Availability / inception masking
- [ ] Analytic projection: simplex with lower bounds + `alpha` de-risk scan
- [ ] CVXPY backend (as test oracle at minimum)
- [ ] Risk envelope: rolling stress, block bootstrap, date-filtered crisis library
- [ ] Capital-preservation mode and the infeasibility fallback
- [ ] Reward `log(V_{t+1}/V_t)`
- [ ] `trajectory.parquet` writer matching the documented schema
- [ ] **Gate:** I1–I6, T1–T7, T15 all green

### `[ ]` Stage 5 — Baselines · `python -m scripts.s05_run_baselines --config config/evaluation.yaml`

Specs: [reference/baselines.md](reference/baselines.md)

- [ ] **B1 `spy_buy_hold`** — buy SPY session 1, never trade again
- [ ] **B2 `momentum`** — 12-1 cross-sectional, top-5 equal weight, monthly, absolute filter
- [ ] **B3 `spy_tlt_60_40`** — 60/40, monthly, 5pp drift band
- [ ] B4 `cash` · B5 `equal_weight` · B6 `classical_optimizer` (required by the acceptance criteria)
- [ ] All six through the identical simulator + constraint layer
- [ ] Standard metrics computed for each
- [ ] **Risk-envelope calibration** ([reference/risk-envelope.md](reference/risk-envelope.md) §7) — record the table
- [ ] Initial-state reservoir populated for the Stage 5 reset sampler
- [ ] **Gate:** zero lock/feasibility violations across all six, plus the four simulator checks —
      `cash` zero drawdown/turnover · `spy_buy_hold` identical across all `N` · `momentum` degrades
      monotonically in `N` · `spy_tlt_60_40` shows a severe 2022 drawdown

---

## Phase 2 — Environment and RL  `[!]` blocked until Stage 4 is `[x]`

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
(ledger + lock manager dict round-trip), which is already in Stage 4's scope.

---

## Ablations — run once the primary path is complete

- [ ] Reward variants: + drawdown penalty; differential Sharpe
- [ ] Auxiliary projection penalty `lambda ∈ {0, 0.001, 0.01}`
- [ ] `lock.scope: portfolio` — the portfolio-wide lock variant
- [ ] Risk aggregation `max` vs `mean`
- [ ] Flat MLP vs shared per-asset encoder

---

## Open questions

| # | Question | Blocks | Status |
|---|---|---|---|
| Q1 | Risk-envelope calibration values (`quantile`, `horizon_days`, `block_length`, `aggregation`) | Stage 7 quality | Resolve empirically in Stage 5 |
| Q2 | Worker count / vectorization strategy | Stage 7 throughput | Resolve from the Stage 6 benchmark |
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
