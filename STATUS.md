# STATUS — Task Tracker

**The single file to update as work proceeds.** Plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
Stage runbook: [reference/stages.md](reference/stages.md).

Legend: `[ ]` not started · `[~]` in progress · `[x]` done (tests green) · `[!]` blocked · `[-]` deliberately deferred

---

## Current position

| | |
|---|---|
| **Current stage** | Stage 0 — Preflight (GPU/device resolved; harness + remaining deps outstanding) |
| **Run this** | `python -m scripts.s00_check_env` |
| **Next gate** | Stage 4 (deterministic simulator) — no RL code before it is green |
| **Last updated** | 2026-08-31 |

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

### `[~]` Stage 0 — Preflight · `python -m scripts.s00_check_env`

- [x] **GPU resolved** — RTX 3060 Ti (8 GB, sm_86), driver 596.49/CUDA 13.2, `torch==2.13.0+cu126`,
      `cuda.is_available() == True`, matmul verified against CPU. See [reference/gpu-setup.md](reference/gpu-setup.md)
- [x] **Device benchmarked** — end-to-end PPO: CPU **2.35s** vs CUDA 6.44s over 5120 steps → **D14: `device: cpu`**
- [x] Core RL stack installed: torch 2.13.0+cu126, SB3 2.9.0, gymnasium 1.3.0, numpy 2.4.6, pandas 3.0.5, tensorboard 2.21.0
- [ ] `requirements.txt` pinned from the verified versions ([reference/gpu-setup.md](reference/gpu-setup.md) §5)
- [ ] Remaining deps: `exchange_calendars`, `yfinance`, `fredapi`, `pyarrow`, `hypothesis`, `pytest`, `cvxpy`
- [ ] `src/cli/stage.py` harness: `--config`, `--dry-run`, `--force`, `--seed`, staleness, manifest
- [ ] Preflight checks incl. **CUDA hard-fail** and the **numpy-less-torch** check (both real failure modes)
- [ ] `FRED_API_KEY` + yfinance + `XNYS` calendar reachable
- [ ] `data/`, `artifacts/` writable and git-ignored
- [ ] **Gate:** T16 passes

### `[ ]` Stage 1 — Fetch raw data · `python -m scripts.s01_fetch_data --config config/universe.yaml`

- [ ] `config/universe.yaml` from [reference/etf-universe.md](reference/etf-universe.md) (tickers, kinds, groups)
- [ ] yfinance fetch: `auto_adjust=False`, batches ≤20, 3 retries, from 2003-01-01
- [ ] FRED fetch with publication-lag metadata
- [ ] Atomic writes (temp + rename); tz normalized to naive dates
- [ ] Incremental mode: trailing-90-day overwrite
- [ ] **Gate:** symbol-quirk pins, row counts match sessions

### `[ ]` Stage 2 — Curate · `python -m scripts.s02_curate_data --config config/universe.yaml`

- [ ] Reindex to `XNYS` sessions; derive `inception.parquet`
- [ ] Quality gate (OHLC sanity, gaps, non-positive prices, split-unexplained jumps) — **exits non-zero**
- [ ] `close_adj` / `close_raw` both present and consistent
- [ ] **Gate:** zero violations; I5, T15 data-side

### `[ ]` Stage 3 — Features · `python -m scripts.s03_build_features --config config/features.yaml`

- [ ] Per-ETF OHLCV block
- [ ] Cross-sectional block, availability-aware ranks
- [ ] Macro block from **point-in-time-aligned curated series only**
- [ ] Per-fold scalers; feature manifest
- [ ] Feature correlation diagnostic reported
- [ ] **Gate:** I6, T10, manifest completeness

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

---

## Decision log

| Date | Entry |
|---|---|
| 2026-08-31 | Plan written; D1–D12 locked ([reference/decisions.md](reference/decisions.md)). |
| 2026-08-31 | **D13:** lock scope is **per-ETF**, not portfolio-wide — buying ETF i relocks only ETF i. See [reference/lock-state-machine.md](reference/lock-state-machine.md) §0. |
| 2026-08-31 | Raw-price ledger / total-return NAV resolved as an explicit dividend-reinvestment ledger; T1 proves it. |
| 2026-08-31 | Baselines moved ahead of PPO (draft had them at Milestone 10, after training). |
| 2026-08-31 | **D15:** benchmark set fixed by the user — `spy_buy_hold`, `momentum` (12-1), `spy_tlt_60_40` as primary; `cash`, `equal_weight`, `classical_optimizer` retained because the acceptance criteria name them. Replaces the vague `spy` / `static_risk_aware` placeholders. See [reference/baselines.md](reference/baselines.md). |
| 2026-08-31 | **D14:** GPU resolved. RTX 3060 Ti verified working with `torch==2.13.0+cu126`. Benchmarked end-to-end PPO at CPU 2.35s vs CUDA 6.44s → **`training.device: cpu`**. GPU wins raw matmul 8.6x but loses the real job 2.7x; rollout transfer latency dominates. See [reference/gpu-setup.md](reference/gpu-setup.md). |
