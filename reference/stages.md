# Stage Runbook

One section per stage: **what you run**, what it reads, what it writes, what it must satisfy before it counts
as done. Track completion in [../STATUS.md](../STATUS.md).

Universal contract (see [architecture.md](architecture.md)): every stage accepts `--config`, `--dry-run`,
`--force`, `--seed`, refuses to run on stale inputs, and writes a run manifest.

---

## Phase 0 — Foundation

### Stage 0 — Preflight

```bash
python -m scripts.s00_check_env
```

**Reads:** `.env`, `config/*.yaml`, the installed environment.
**Writes:** `artifacts/runs/<run_id>/preflight.json` and a human-readable table to stdout.

Checks, each pass/fail with a fix hint:

- Python version, and every package in `requirements.txt` importable at the pinned version.
- `FRED_API_KEY` present and a single test series fetches.
- `yfinance` returns a non-empty frame for one tradable and one `^`-prefixed feature symbol.
- `exchange_calendars` `XNYS` loads and `sessions_in_range("2003-01-01", today)` is non-empty.
- CUDA availability and device name — **hard fail** if unavailable, plus the numpy-less-torch check
  ([gpu-setup.md](gpu-setup.md) §6).
- `data/`, `artifacts/` writable; `data/` git-ignored.

**Also builds `src/cli/stage.py`** — the harness everything downstream uses.

**Done when:** all checks pass and `tests/test_stage_harness.py` is green (dry-run does nothing, staleness
detection fires, manifest is written on failure too).

---

### Stage 1 — Fetch raw data

```bash
python -m scripts.s01_fetch_data --config config/universe.yaml
python -m scripts.s01_fetch_data --config config/universe.yaml --since 2026-06-01   # incremental
```

**Reads:** `config/universe.yaml`, network.
**Writes:** `data/raw/prices/<ticker>.parquet`, `data/raw/fred/<series>.parquet`, `data/raw/fetch_log.json`.

Implements [data-pipeline.md](data-pipeline.md) exactly: `auto_adjust=False` to keep both close series,
batches of ≤20 tickers with a 2s pause, 3 retries with exponential backoff, full history from **2003-01-01**
(one year of warm-up before the 2004 study start), trailing-90-day re-pull and overwrite on incremental runs.

**Non-negotiables:** never write partial data on a failed fetch — write to a temp path and rename. Normalize
tz-aware indexes to naive dates immediately. Yields (`^TNX`, `^FVX`, `^IRX`, FRED rates) are stored as
**levels**, and the pipeline must never compute a return on them.

**Done when:** every symbol in `config/universe.yaml` has a file, row counts match the session count within
the expected inception-truncated range, and `tests/test_fetch.py` pins the known symbol quirks.

---

### Stage 2 — Curate

```bash
python -m scripts.s02_curate_data --config config/universe.yaml
```

**Reads:** `data/raw/**`.
**Writes:** `data/curated/prices.parquet`, `data/curated/inception.parquet`,
`data/curated/quality_report.json`.

Work: reindex everything to `XNYS` sessions; derive `first_session` per ticker (this is the *only* source of
the availability mask); run the quality gate.

**The quality gate is a gate, not a report.** It exits non-zero on:

- `low_raw <= min(open_raw, close_raw)` or `max(open_raw, close_raw) <= high_raw` violated on any session;
- a non-positive price;
- a gap: a session present in the calendar but missing for a ticker that is past inception;
- a single-day absolute return above a configured threshold that is not on a known split date;
- a `close_adj` series that is not monotone-consistent with `close_raw` up to dividend adjustments.

**Done when:** the gate passes with zero violations and `tests/test_curate.py` is green. Any waived violation
must be listed explicitly in `config/universe.yaml` under `known_exceptions`, with a reason.

---

### Stage 3 — Features

```bash
python -m scripts.s03_build_features --config config/features.yaml
python -m scripts.s03_build_features --config config/features.yaml --fold 2012   # fit scalers for one fold
```

**Reads:** `data/curated/**`.
**Writes:** `data/features/{etf,cross_sectional,macro}.parquet`, `data/features/scalers/<fold_id>.json`,
`data/features/feature_manifest.json`.

Implements [features.md](features.md) — per-ETF OHLCV, cross-sectional, and macro/regime blocks. Portfolio
features are *not* here: they depend on the live portfolio and are computed in the environment.

**The rule that matters:** every feature at session `t` uses only data up to and including `t`'s close.
Scalers are fitted on a fold's training window only and are never refit on evaluation data. Both are enforced
by tests, not by care.

**Done when:** the feature manifest lists every column with its lookback and source; the lookahead test
(perturb the future, assert no past feature changes) passes; NaN counts match the expected warm-up pattern
exactly, with no forward-fill across a ticker's inception boundary.

---

## Phase 1 — Deterministic core

### Stage 4 — The deterministic simulator  ← **the gate**

```bash
python -m scripts.s04_simulate --config config/sim/default.yaml \
    --weights artifacts/weight_sequences/equal_weight.parquet \
    --start 2004-01-02 --end 2026-08-31 --hold-days 30 --max-drawdown 0.20
```

**Reads:** `data/curated/prices.parquet`, a target-weight sequence, `config/constraints.yaml`.
**Writes:** `artifacts/runs/<run_id>/trajectory.parquet` (schema in [architecture.md](architecture.md)),
`.../diagnostics.json`.

The first real development task: given any legal initial state, any sequence of target weights, and the price
history, produce the complete portfolio trajectory. **No RL anywhere in this stage.**

It must already implement, completely:

| Component | Spec |
|---|---|
| EOD decision → next-open execution | [portfolio-ledger.md](portfolio-ledger.md) §2 |
| Fractional shares, cash ledger | [portfolio-ledger.md](portfolio-ledger.md) §1 |
| Total-return NAV from a raw-price ledger | [portfolio-ledger.md](portfolio-ledger.md) §3 — the subtle one |
| Inception / availability masking | [feasibility-projection.md](feasibility-projection.md) §1 |
| Resettable N-calendar-day lock | [lock-state-machine.md](lock-state-machine.md) |
| Feasibility projection | [feasibility-projection.md](feasibility-projection.md) |
| Risk envelope + capital-preservation mode | [risk-envelope.md](risk-envelope.md) |
| NAV, running peak, drawdown | [portfolio-ledger.md](portfolio-ledger.md) §4 |
| Reward | `log(V_{t+1}/V_t)`, D11 |

**Done when the entire invariant suite passes** — see [testing.md](testing.md). Specifically:
portfolio, weight, lock, reset, inception, and lookahead invariants, plus the property-based lock tests
including the sell-then-rebuy same-day case.

> **Do not begin Phase 2 before this stage is green.** This is the single most important sequencing
> constraint in the project.

---

### Stage 5 — Baselines

```bash
python -m scripts.s05_run_baselines --config config/evaluation.yaml
python -m scripts.s05_run_baselines --config config/evaluation.yaml --only spy,equal_weight
```

**Reads:** curated prices, features, the Stage 4 simulator.
**Writes:** `artifacts/runs/<run_id>/baselines/<name>/trajectory.parquet`, `.../baseline_summary.json`.

Six baselines, all run through the *same* simulator and the *same* constraint layer. Full specifications in
**[baselines.md](baselines.md)**.

| Baseline | Definition | |
|---|---|---|
| `spy_buy_hold` | Buy SPY on session 1, never trade again | **Primary** |
| `momentum` | 12-1 cross-sectional momentum, top-5 equal weight, monthly, absolute filter | **Primary** |
| `spy_tlt_60_40` | 60% SPY / 40% TLT, monthly with a 5pp drift band | **Primary** |
| `cash` | 100% cash always — the floor | Required |
| `equal_weight` | Equal weight across available tradables | Acceptance |
| `classical_optimizer` | Rolling mean-variance / risk parity on trailing visible data | Acceptance |

Baselines are subject to identical constraints. A baseline that violates the lock or leverage is a bug in the
constraint layer, and finding it here is far cheaper than finding it during training.

**Done when** all six produce trajectories with zero lock and zero feasibility violations, the standard metrics
([evaluation.md](evaluation.md) §3) are computed for each, and the four simulator-checking expectations in [baselines.md](baselines.md) §7 hold —
`cash` at exactly zero drawdown/turnover, `spy_buy_hold` identical across every `N`, `momentum` degrading
monotonically in `N`, and `spy_tlt_60_40` showing a severe 2022 drawdown.

---

## Phase 2 — Environment and RL

### Stage 6 — Environment smoke test

```bash
python -m scripts.s06_smoke_env --config config/training.yaml --episodes 500
```

**Reads:** everything from Phase 0–1.
**Writes:** `artifacts/runs/<run_id>/smoke_report.json`.

Not a training run. It:

- runs `gymnasium.utils.env_checker.check_env` and SB3's `check_env`;
- steps 500 random-policy episodes across the full `(N, D_max)` grid and every reset mode, asserting every
  invariant on *every* step;
- verifies observation bounds, dtypes, and that no observation is ever NaN or inf;
- verifies determinism: same seed → identical trajectory, byte for byte;
- confirms the reset sampler only ever produces reachable states with `D_t <= D_max`;
- **benchmarks throughput** — steps/second for `DummyVecEnv` vs `SubprocVecEnv` at 1/4/8/16 workers, with and
  without GPU. D5 depends on knowing this number before any hyperparameter is tuned.

**Done when:** zero invariant violations across all episodes, determinism confirmed, and the throughput table
is recorded in [../STATUS.md](../STATUS.md).

---

### Stage 7 — Train PPO

```bash
python -m scripts.s07_train_ppo --config config/experiments/ppo_stage1.yaml
python -m scripts.s07_train_ppo --config config/experiments/ppo_stage3.yaml --resume-from <run_id>
```

**Reads:** features, curated prices, `config/experiments/*.yaml`.
**Writes:** `artifacts/runs/<run_id>/{policy.zip, checkpoints/, tensorboard/, metrics.json, manifest.json}`.

Implements [rl-training.md](rl-training.md): SB3 PPO (D2), pre-projection action storage (D9), and the
five-stage curriculum with warm-starting between stages.

Run the curriculum in order — one command per stage, each resuming the previous:

| Curriculum stage | Config | What is added |
|---|---|---|
| 1 | `ppo_stage1.yaml` | `N=0`, fixed `D_max`, portfolio mechanics only |
| 2 | `ppo_stage2.yaml` | Lock, `N in {7,30}` |
| 3 | `ppo_stage3.yaml` | Full `(N, D_max)` parameter range |
| 4 | `ppo_stage4.yaml` | Drawdown safety envelope active |
| 5 | `ppo_stage5.yaml` | Everything randomized: params, initial state, episode length |

**Watch during training** (all on TensorBoard): projection distance (D9 — should trend down), safety
intervention rate, cash weight distribution, and episode reward against the Stage 5 baselines drawn as
horizontal reference lines.

**Done when:** each curriculum stage completes, beats the `cash` baseline on the training window, and shows
zero lock/feasibility violations across the whole run — violations here mean the constraint layer is broken,
not that the policy is bad.

---

### Stage 8 — Annual walk-forward

```bash
python -m scripts.s08_walk_forward --config config/evaluation.yaml
python -m scripts.s08_walk_forward --config config/evaluation.yaml --folds 2012,2013 --seeds 3
```

**Reads:** features, prices, `config/evaluation.yaml`.
**Writes:** `artifacts/runs/<run_id>/folds/<year>/{trajectory.parquet, selection.json}`,
`.../walk_forward_summary.json`.

Implements [evaluation.md](evaluation.md): expanding-window annual folds (train `[start..Y-2]`, validate
`Y-1`, test `Y`), lexicographic model selection (risk criteria first, return second), and the
preventable-vs-market-forced drawdown violation taxonomy.

**Hard rules:** never retrain mid-year on test-year data; never fit a scaler outside the training window;
never select on test performance. Each is enforced by a test, not by discipline.

**Done when:** every fold produces a trajectory with zero lock violations and zero feasibility violations
(the hard acceptance criteria), and the selection record for each fold names the criterion that decided it — including
folds marked `constraint validation failure`.

> **This is the Phase 3 gate.** Phase 3 does not start until this passes.

---

## Phase 3 — Robustness *(gated on Stage 8)*

### Stage 9 — Stress suite

```bash
python -m scripts.s09_stress --config config/evaluation.yaml --policy artifacts/runs/<run_id>/policy.zip
python -m scripts.s09_stress ... --grid full     # the complete 7x5 grid, D5
```

Crisis windows (2008, 2020, 2022, and the configured list), plus `N` sensitivity, `D_max` sensitivity, and the
combined grid (default 3x3 per D5). **Writes:** one trajectory per cell plus a grid summary.

**Done when:** every cell completes and the `D_max` sensitivity is *monotone in the right direction* — a
tighter `D_max` must not produce a larger realized drawdown. A violation of that monotonicity is a safety-layer
bug and blocks everything downstream.

---

### Stage 10 — Bootstrap

```bash
python -m scripts.s10_bootstrap --config config/evaluation.yaml --replicates 1000
```

Stationary / moving-block bootstrap over return blocks — never IID resampling, which would destroy the
volatility clustering the whole risk layer depends on. **Writes:** confidence bands for every standard metric.

**Done when:** bands are produced for all metrics and the block-length sensitivity is reported.

---

### Stage 11 — Adversarial scenarios

```bash
python -m scripts.s11_adversarial --config config/evaluation.yaml
```

Unfavorable-but-in-support historical combinations: equity shock + credit widening, duration loss,
correlation spike, diversification breakdown. **Writes:** per-scenario trajectories and worst-case tables.

**Done when:** each scenario runs and the report states plainly that this is historical
robustness evidence, **not** a forward-looking worst-case guarantee.

---

### Stage 12 — Report

```bash
python -m scripts.s12_report --config config/evaluation.yaml --runs <run_id>[,<run_id>...]
```

**Writes:** `artifacts/reports/<name>/{report.md, plots/*.png, acceptance.json}`.

Consolidates Stages 5, 8, 9, 10, 11 and evaluates every acceptance criterion — hard engineering, risk
reporting, and performance — as an explicit pass/fail table. Refuses to include a run whose manifest has
`git_dirty: true` unless `--allow-dirty`.

**Done when:** the acceptance table is complete, with every failed criterion stated rather than omitted.

---

## Phase 4 — Reserved

### Stage 13 — Daily inference *(specified, not implemented — D4)*

```bash
python -m scripts.s13_daily_inference --config config/production.yaml   # NOT IMPLEMENTED
```

Intended shape, recorded now so the design does not preclude it:

1. Exit 0 immediately if `nyse.is_session(today)` is false.
2. Incremental fetch (trailing 90-day overwrite) and re-run the quality gate; abort loudly on failure.
3. Load the **persisted real** portfolio ledger and lock ledger — actual holdings and actual unlock dates, not
   a simulated state.
4. Build the observation with the production `(N, D_max)`.
5. Run the frozen policy for the current deployment year; project; apply the risk envelope.
6. Emit an order sheet for the next open, plus the full projection and safety diagnostics.
7. Persist the updated lock ledger only after fills are confirmed.

Open problems to solve before building it: state durability and idempotent reruns of the same session;
reconciling actual fills against expected ones; behavior when the data gate fails mid-week; who is responsible
for the annual retrain-and-freeze step in the deployment protocol
([evaluation.md](evaluation.md) §1).

**Requirement it imposes on Phase 1 today:** the ledger and lock manager must round-trip losslessly through a
plain dict. That is a test in [testing.md](testing.md) and is already in scope.
