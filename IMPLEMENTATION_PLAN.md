# ETFRegulus — Implementation Plan

**Status tracker:** [STATUS.md](STATUS.md) — the single file to update as work proceeds.
**This file** is the map: phases, stages, entry points, and what "done" means. All depth lives in
[reference/](reference/). This plan is self-contained — it is the specification of record.

---

## 1. What this project is

A daily-frequency, parameter-conditioned RL portfolio allocator over a fixed ETF universe, subject to two
*hard* exogenous constraints supplied at inference time:

- `N` — a resettable, ETF-level holding lock in **calendar days**. Buying or adding to an ETF relocks *that
  ETF's entire holding* for `N` days; other holdings keep their own independent clocks (D13).
- `D_max` — a maximum-drawdown ceiling enforced by a deterministic safety layer, **never** by reward shaping.

The policy proposes weights; a deterministic projection + risk envelope makes them legal; the environment
executes at the next open. The agent is conditioned on `(N, D_max)` so one trained policy serves the whole
parameter grid.

## 2. The core thesis

Worth stating before any of the mechanics, because it determines where the engineering effort goes:

> **The RL policy is only responsible for proposing a portfolio preference. Transition feasibility and risk
> safety are controlled by a deterministic constrained execution layer.**

The hard problem here is not "PPO or SAC". It is that the holding lock makes an action affect far more than
the current allocation. Adding to a position in an ETF relocks that ETF's entire holding for `N` calendar
days, so `a_t` simultaneously determines:

1. the current portfolio;
2. future sell feasibility;
3. future liquidity;
4. future drawdown controllability.

An action changes the *feasible action space* for the next `N` days. This is why the observation must carry
lock remaining days, current holdings, cash, and the drawdown budget — a conventional trading agent fed only
market prices cannot represent this problem, let alone solve it. See
[env-mdp.md](reference/env-mdp.md) for the resulting state definition.

## 3. The rule that governs the ordering

> The top project risk is training PPO on a subtly wrong simulator.

Everything before Phase 2 exists to make the deterministic simulator provably correct. No RL code is written
until Stage 4 is green. This is non-negotiable.

Progression: **correctness → constraints → environment → baselines → RL → walk-forward → robustness**.

Explicitly prohibited, in order of how much damage each does:

| Prohibited | Why |
|---|---|
| Training a PPO first, then going back to patch portfolio accounting | Every result before the fix is void, and the fix silently invalidates tuning done against the broken version |
| Simulating hard constraints with a reward penalty | Converts a hard constraint into a soft preference. The drawdown ceiling and the lock are constraints, not costs |
| Feature scaling using the full historical sample | Lookahead. Contaminates every walk-forward fold at once, invisibly |
| Letting the projection silently repair invalid actions without logging | Hides both policy pathology and constraint-layer bugs |

## 4. Locked decisions

Made with the user before planning; see [reference/decisions.md](reference/decisions.md) for full rationale
and the consequences each one carries.

| # | Decision |
|---|---|
| D1 | Entry points are numbered runnable modules: `python -m scripts.sNN_name --config ...` |
| D2 | RL stack = Gymnasium + Stable-Baselines3 PPO |
| D3 | Projection is analytic at v1; a CVXPY QP path lands later behind a config flag |
| D4 | Research-first; the daily live job is specified but deliberately unimplemented (Stage 13) |
| D5 | Single machine with one GPU; walk-forward/stress grids are sized to that and reduced by default |
| D6 | Parquet under `data/` is the source of truth between stages |
| D7 | Tracking = per-run manifest dirs + TensorBoard, no external service |
| D8 | Invariant tests gate every stage; a stage is not done until its named tests pass |
| D9 | PPO stores the **pre-projection** action; the env executes the projected one |
| D10 | **Zero transaction costs at v1** — frictionless. `cost_bps` parameter exists, set to `0.0` |
| D11 | Reward = `log(NAV_t+1 / NAV_t)`. No turnover/vol/drawdown penalty terms |
| D12 | Full robustness battery planned, gated behind a working walk-forward result |
| D13 | Lock scope is **per-ETF**, not portfolio-wide: buying ETF i relocks only ETF i |
| D14 | `torch==2.13.0+cu126` on the RTX 3060 Ti. **`training.device: cpu`** — GPU verified working but benchmarked slower end-to-end |
| D15 | Primary benchmarks: **SPY buy-and-hold, 12-1 momentum, 60/40 SPY-TLT**; cash, equal-weight and classical also required |

## 5. Phases and stages

Each stage is one runnable entry point with declared inputs and outputs. Full runbook — flags, artifact
schemas, acceptance tests — in [reference/stages.md](reference/stages.md).

### Phase 0 — Foundation *(no modeling)*

| Stage | Entry point | Produces |
|---|---|---|
| 0 | `python -m scripts.s00_check_env` | Dependency / API-key / calendar preflight report |
| 1 | `python -m scripts.s01_fetch_data` | `data/raw/**.parquet` |
| 2 | `python -m scripts.s02_curate_data` | `data/curated/prices.parquet`, `inception.parquet`, quality gate |
| 3 | `python -m scripts.s03_build_features` | `data/features/**.parquet` + per-fold fitted scalers |

Detail: [data-pipeline.md](reference/data-pipeline.md), [etf-universe.md](reference/etf-universe.md), [features.md](reference/features.md)

### Phase 1 — Deterministic core *(the critical path; still no RL)*

| Stage | Entry point | Produces |
|---|---|---|
| 4 | `python -m scripts.s04_simulate` | Full portfolio trajectory from a weight sequence — **the gate** |
| 5 | `python -m scripts.s05_run_baselines` | Six baseline curves: SPY buy-and-hold, momentum, 60/40 SPY-TLT, cash, equal weight, classical optimizer |

Detail: [portfolio-ledger.md](reference/portfolio-ledger.md), [lock-state-machine.md](reference/lock-state-machine.md), [feasibility-projection.md](reference/feasibility-projection.md), [risk-envelope.md](reference/risk-envelope.md), [baselines.md](reference/baselines.md)

### Phase 2 — Environment and RL

| Stage | Entry point | Produces |
|---|---|---|
| 6 | `python -m scripts.s06_smoke_env` | Gym API conformance + random-policy invariant sweep |
| 7 | `python -m scripts.s07_train_ppo` | A trained policy per curriculum stage under `artifacts/runs/<run_id>/` |
| 8 | `python -m scripts.s08_walk_forward` | Annual expanding-window fold results + model selection record |

Detail: [env-mdp.md](reference/env-mdp.md), [rl-training.md](reference/rl-training.md), [evaluation.md](reference/evaluation.md)

### Phase 3 — Robustness *(gated: do not start before Stage 8 produces a passing fold set)*

| Stage | Entry point | Produces |
|---|---|---|
| 9 | `python -m scripts.s09_stress` | Crisis-window + N x D_max sensitivity grid |
| 10 | `python -m scripts.s10_bootstrap` | Stationary / moving-block bootstrap confidence bands |
| 11 | `python -m scripts.s11_adversarial` | Adversarial historical scenario results |
| 12 | `python -m scripts.s12_report` | Consolidated report + plots against the acceptance criteria |

Detail: [robustness.md](reference/robustness.md), [evaluation.md](reference/evaluation.md)

### Phase 4 — Reserved

| Stage | Entry point | Status |
|---|---|---|
| 13 | `python -m scripts.s13_daily_inference` | **Specified, not implemented** (D4). See [stages.md](reference/stages.md) |

## 6. How to run any stage yourself

Every stage obeys the same contract, so nothing is ever a mystery:

```bash
python -m scripts.sNN_name --config config/<file>.yaml [--dry-run] [--force]
```

- `--dry-run` prints the resolved config, the input artifacts it will read, and the outputs it will write —
  then exits without touching anything.
- Every stage refuses to run if an upstream artifact is missing or stale, and names the stage to run first.
- Every stage writes `artifacts/runs/<run_id>/manifest.json` recording config hash, git SHA, seed, input
  artifact hashes, and wall time.

Stage state, ownership, and per-task checkboxes live in **[STATUS.md](STATUS.md)**.

## 7. Reference index

| File | Covers |
|---|---|
| [decisions.md](reference/decisions.md) | The locked decisions, rationale, consequences |
| [gpu-setup.md](reference/gpu-setup.md) | Verified hardware, CUDA stack, pinned versions, device benchmark |
| [architecture.md](reference/architecture.md) | Repo layout, config system, artifact contracts, run manifests |
| [stages.md](reference/stages.md) | Per-stage runbook: entry point, flags, I/O, done-criteria |
| [data-pipeline.md](reference/data-pipeline.md) | Calendar, yfinance, FRED, adjustment, quality gate |
| [etf-universe.md](reference/etf-universe.md) | Tradable set, inception dates, feature-only symbols |
| [features.md](reference/features.md) | Per-ETF, cross-sectional, macro, portfolio features; scaling discipline |
| [baselines.md](reference/baselines.md) | The six benchmarks, specs, and what each one tests |
| [portfolio-ledger.md](reference/portfolio-ledger.md) | Shares/cash ledger, execution timing, total-return NAV, peak, drawdown |
| [lock-state-machine.md](reference/lock-state-machine.md) | The resettable N-day lock, all four transition cases |
| [feasibility-projection.md](reference/feasibility-projection.md) | Availability, long-only, no-leverage, lock bounds, projection algorithm |
| [risk-envelope.md](reference/risk-envelope.md) | Drawdown engine, headroom, stress estimators, capital-preservation mode |
| [env-mdp.md](reference/env-mdp.md) | State, action, reward, reset sampler, episode length, param conditioning |
| [rl-training.md](reference/rl-training.md) | SB3 wiring, policy, credit assignment, curriculum, baselines |
| [evaluation.md](reference/evaluation.md) | Walk-forward, model selection, metrics, drawdown violation taxonomy |
| [robustness.md](reference/robustness.md) | Stress grid, bootstrap, adversarial scenarios |
| [testing.md](reference/testing.md) | The invariant suite that gates each stage |

## 8. Design notes: calls this plan makes

Decisions taken while turning the specification into an executable plan, recorded so they are not
re-litigated later:

1. **Baselines come before RL.** They are Phase 1, not a post-training comparison. Without them the entire RL
   development period has no reference point, and several of them double as simulator correctness tests with
   obvious expected answers (see [baselines.md](reference/baselines.md) §7).
2. **The deterministic simulator is a numbered, gated stage.** It is the single most important implementation
   dependency in the project, so it is Stage 4 with fourteen gating tests rather than an informal preliminary.
3. **Stage 0 and Stage 6 exist.** A preflight check and a random-policy environment sweep. Both are cheap, and
   both catch the class of bug that otherwise surfaces six hours into a training run. Stage 0 has already paid
   for itself once — see [gpu-setup.md](reference/gpu-setup.md) §5.
4. **Artifact contracts between stages are pinned.** Every stage boundary is a file boundary with a declared
   schema ([architecture.md](reference/architecture.md) §4). This is what makes stages independently runnable.
5. **The raw-price ledger produces a total-return NAV via explicit dividend reinvestment.** The obvious
   alternatives are both wrong: valuing at `close_raw` systematically understates income-paying ETFs, and
   `close_adj` is back-adjusted so a share count times an adjusted price is not a dollar amount.
   [portfolio-ledger.md](reference/portfolio-ledger.md) §3 gives the design; test T1 proves it.
6. **Grids are sized to one GPU (D5).** The full walk-forward x curriculum x stress product is combinatorial.
   Defaults are reduced with the full grid available behind an opt-in flag rather than silently dropped.
7. **The lock is per-ETF (D13).** Buying ETF `i` relocks only ETF `i`; other holdings keep independent clocks.
   The portfolio-wide alternative remains available as `lock.scope: portfolio` for ablation.
