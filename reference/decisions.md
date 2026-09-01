# Decisions

Locked before implementation began. Each entry: the decision, why, and what it costs. Changing one of these
is a plan change, not an implementation detail — update this file and [../STATUS.md](../STATUS.md) together.

---

## D1 — Entry points are numbered runnable modules

```bash
python -m scripts.s04_simulate --config config/sim/default.yaml
```

Files are `scripts/s00_check_env.py` … `scripts/s13_daily_inference.py`. The number *is* the dependency order,
visible in a directory listing.

**Why:** the user must be able to run any stage independently and know at a glance which file to run. A CLI
with subcommands hides that behind a dispatch table; a Makefile adds a tool that behaves poorly on Windows.

**Cost:** some argument-parsing duplication. Mitigated by `src/cli/stage.py`, a shared harness every script
calls (see [architecture.md](architecture.md)).

---

## D2 — Gymnasium + Stable-Baselines3 PPO

**Why:** the novel content of this project is the constraint layer, not the RL algorithm. A hand-rolled PPO
adds a second source of bugs on top of a simulator that is already the main risk.

**Cost:** SB3 imposes its vec-env and rollout-buffer shapes. D9 (pre-projection storage) is implemented by
putting the projection *inside* the environment's `step`, which is the clean way to satisfy SB3 anyway. If
D9 is ever revisited toward a differentiable projection, SB3 becomes the wrong tool and this decision reopens.

---

## D3 — Analytic projection at v1, CVXPY later behind a flag

v1 solves the projection in closed form (see [feasibility-projection.md](feasibility-projection.md)):
simplex projection with lower bounds, then a scalar de-risking search toward the safe asset.

**Why:** a QP at every step across millions of steps is the difference between a training run that finishes
overnight and one that does not. The analytic path is also easy to test exhaustively.

**Cost:** the analytic solution is exact for the lock/long-only/no-leverage constraints but only
*approximately* optimal once the risk envelope binds — it finds a feasible point close to the proposal, not
the provably closest one. Both implementations share the `FeasibilityProjector` interface, and
`projection.backend: {analytic, cvxpy}` selects between them. The QP is used as a **correctness oracle** in
tests from day one, on small random instances, even before it is used in training.

---

## D4 — Research first; the daily job is specified, not built

Stage 13 exists in the plan with its inputs, outputs, and failure modes documented, and nothing more.

**Why:** live inference needs durable state (a real portfolio and a real lock ledger that survive process
restarts), idempotent reruns, and failure handling. None of that is worth building against a policy that has
not yet passed walk-forward.

**Cost:** none now. The requirement it imposes today is that the ledger and lock manager are serializable to
and from a plain dict from the start — cheap, and enforced by a round-trip test in Phase 1.

---

## D5 — Single machine, one GPU

**Why:** stated compute budget. Hardware verified 2026-08-31 — RTX 3060 Ti, 8 GB, sm_86. Full results and the
device benchmark in [gpu-setup.md](gpu-setup.md); the operative conclusion is **D14: train on CPU**.

**Cost:** the full walk-forward x curriculum x stress grid is out of reach. Defaults are reduced:

| Grid | Draft (full) | Default here | Full via |
|---|---|---|---|
| Walk-forward folds | every year 2012→now | every year, 1 seed | `--seeds K` |
| Curriculum stages | 5 | 5 (sequential, warm-started) | — |
| Stress grid `N` x `D_max` | 7 x 5 = 35 | 3 x 3 = 9 | `--grid full` |
| Bootstrap replicates | "many" | 1000 | `--replicates` |

Policies are small MLPs; the bottleneck is environment stepping, not the network. Scale with `SubprocVecEnv`
across CPU cores. Per D14 the policy update also runs on CPU — this was measured, not assumed, and Stage 6
re-measures it whenever the architecture changes.

---

## D6 — Parquet under `data/` is the source of truth

`data/raw/` → `data/curated/` → `data/features/`. Git-ignored, partitioned, inspectable with pandas alone.

**Why:** it matches the 90-day-overwrite refresh strategy already specified in
[data-pipeline.md](data-pipeline.md), and keeps every stage boundary a file boundary — which is what makes
stages independently runnable.

**Cost:** no SQL layer for ad-hoc analysis. If that becomes painful, DuckDB can read the parquet in place
without changing anything upstream.

---

## D7 — Local run manifests + TensorBoard

Each run writes `artifacts/runs/<run_id>/` with `manifest.json`, `config.snapshot.yaml`, `metrics.json`, and
SB3 TensorBoard logs. Schema in [architecture.md](architecture.md).

**Why:** the goal is reproducibility, not a dashboard. A manifest with the git SHA, config hash, seed, and input
artifact hashes is what actually makes a result reproducible; the UI is optional.

**Cost:** cross-run comparison is a small script rather than a web UI. `scripts/s12_report.py` provides it.

---

## D8 — Invariant tests gate every stage

A stage is not marked done in [../STATUS.md](../STATUS.md) until the tests named in
[testing.md](testing.md) for that stage pass. Property-based tests (`hypothesis`) cover the ledger, the lock
state machine, and lookahead.

**Why:** the invariants in [testing.md](testing.md) are the only thing standing between this project and a
plausible-looking but wrong backtest. The lock state machine in particular has a genuinely tricky case —
selling and rebuying the same ETF on the same day — that fixed fixtures will not find.

**Cost:** Phase 1 takes noticeably longer. This is the intended trade.

---

## D9 — PPO stores the pre-projection action

```
a_raw  ~ pi(.|s)          # stored in the rollout buffer, log_prob'd
a_proj = project(a_raw, s) # executed by the environment
r      = f(a_proj)
log      ||a_proj - a_raw||_1   # projection distance, per step
```

**Why:** the importance ratio must be computed under the distribution that actually generated the stored
action. Storing `a_proj` and evaluating `log pi(a_proj|s)` is wrong — the projected action frequently lands on
the boundary of the simplex, where a continuous policy assigns it vanishing density. Treating the projection as
part of the environment is both correct and the standard framing.

**Cost:** the policy receives no gradient telling it *why* an action was altered; it only sees the reward of
the projected action. Mitigations, in order of preference:

1. **Diagnostic first.** Log projection distance every step and plot it. If it trends down during training,
   the policy is learning feasibility on its own and nothing further is needed.
2. **Feasibility-aware observation.** The lock mask, availability mask, and drawdown headroom are all already
   in the state, so the policy *can* learn this. Verify the masks are actually reaching the network.
3. **Auxiliary penalty, only if 1 shows no improvement.** `-lambda * ||a_proj - a_raw||`, with `lambda`
   ablated. This is a penalty on *the policy's proposal*, not on risk, so it does not violate the
   prohibition on implementing hard constraints via reward shaping (D11) — record the distinction explicitly
   when reporting.

---

## D10 — Zero transaction costs at v1

The environment is frictionless: no transaction cost, slippage, or commission.

The ledger carries a `cost_bps` parameter **set to `0.0`**, so the code path exists and is unit-tested rather
than being absent. It is a parameter, not a knob anyone is expected to turn: v1 results are frictionless
results, and the reports state that plainly.

**Settled. Not revisited.**

---

## D11 — Reward is log NAV growth, nothing else

`r_t = log(V_{t+1}^close / V_t^close)`, where the trade executes at the `t+1` open. No turnover, volatility, or
drawdown term.

**Why:** risk is a hard constraint enforced by the safety layer. Encoding it as a reward penalty instead would
silently convert a hard constraint into a soft preference the policy is free to trade away.

**Cost:** the learning signal for risk is indirect (the agent experiences the safety layer clamping it, and
the drawdown it avoids). Alternative reward formulations are permitted only as **logged ablations** in
[rl-training.md](rl-training.md), never as the primary run.

---

## D12 — Full robustness battery planned, gated on walk-forward

The stress, bootstrap and adversarial suites are specified at full depth in
[robustness.md](robustness.md), and Phase 3 is marked blocked in
[../STATUS.md](../STATUS.md) until Stage 8 yields a fold set meeting the hard engineering criteria (zero lock
violations, zero feasibility violations).

**Why:** stress-testing a policy that has not passed walk-forward measures nothing, and the temptation to
start is real because it is the interesting part.

**Cost:** none. The specs are written; only the execution waits.

---

## D13 — Lock scope is per-ETF

Buying ETF `i` resets only `unlock_date_i`; every other holding keeps its own independent clock. Full
treatment in [lock-state-machine.md](lock-state-machine.md) §0.

**Why:** it is the intended semantics of an ETF-level lock. The alternative — a portfolio-wide relock, where
any purchase freezes the whole book — is a materially different and far more restrictive problem.

**Cost:** a less restrictive constraint, so the agent has more freedom than the portfolio-wide reading would
allow. `lock.scope: portfolio` remains available as an ablation.

---

## D14 — `torch==2.13.0+cu126`; train on CPU

The GPU is verified working. It is not used for v1 training, because it is slower.

**Measured 2026-08-31** (full results in [gpu-setup.md](gpu-setup.md) §5):

| | CPU | CUDA |
|---|---|---|
| Raw matmul, 200x `(4096x256)@(256x256)` | 0.1347s | **0.0156s** |
| End-to-end PPO, 5120 steps, `[256,256]` | **2.35s** | 6.44s |

The GPU wins the arithmetic 8.6x and loses the actual job. **The cause is per-operation overhead at batch 1,
not data transfer** — a rollout forward pass costs 288 µs on CUDA vs 36 µs on CPU even with the input already
resident on the device, because ~6 kernel launches at ~40-50 µs each dwarf ~5,000 FLOPs of arithmetic. PPO
does 2048 such batch-1 passes per iteration against 40 batched updates, so the phase the GPU is bad at is the
phase that dominates. Full decomposition in [gpu-setup.md](gpu-setup.md) §3.

**The margin is 1.2x, not 2.7x**, once the environment's own step cost is accounted for (it cancels from both
sides). This is a thin win and it tilts with network size.

**Why cu126** over cu128/129/130, all of which support sm_86: it carries the newest stable torch and is the
most-exercised CUDA 12.x line. cu129 is stuck at torch 2.9.0.

**Cost:** none today, and one standing obligation — at a 1.2x margin the crossover is genuinely near, so
Stage 6 re-runs this benchmark whenever the policy architecture changes rather than inheriting the answer. A
wider per-asset encoder or an attention block over assets could flip it, as could batching the rollout forward
pass across many parallel envs.

---

## D16 — The lock is centred on 30 calendar days

`N = 30` is the deployment value. The policy is trained and evaluated over a range that
varies *around* 30, not over an arbitrary wide grid that happens to contain it.

Canonical definition in [../config/constraints.yaml](../config/constraints.yaml):

| | Values | Used by |
|---|---|---|
| **Primary** | `30` | Deployment; every report leads with this |
| **Operating range** | `[15, 21, 30, 42, 60]`, weights `[.15, .20, .30, .20, .15]` | Training and evaluation |
| **Out-of-distribution** | `[0, 7, 90, 180]` | Stage 9 sensitivity sweep only |

**Why a sqrt(2) ladder rather than evenly spaced values.** A holding period is a
multiplicative quantity: 30 -> 60 days is the same size of change as 30 -> 15, whereas
30 -> 45 and 30 -> 15 are not. So the grid steps by a factor of ~1.41 and is centred on
30 *geometrically* — weighted geometric mean 29.9. The superseded grid
`[0, 7, 14, 30, 60, 90, 180]` had 30 as merely the fourth of seven equally likely values,
with a median of 30 but a mass strongly skewed long; a policy trained on it would spend
most of its capacity on lock lengths that will never be deployed.

**Why the tails are still there.** A policy trained at a single `N` is not a
parameter-conditioned policy at all — it cannot be asked what happens at 45 days, and it
has no reason to encode `N` in its value function rather than memorizing one lock length.
The weighting concentrates capacity near 30 while keeping enough spread for the
conditioning to be real.

**Why `0`, `7`, `90` and `180` were moved out rather than deleted.** They remain valuable
as *characterization*, and worthless as training distribution. `N = 0` is the no-lock
control (and curriculum stage 1's mechanics-only setting); `180` is the extreme at which
`momentum` should degenerate toward buy-and-hold. Stage 9 sweeps all of them and labels
every such result **out-of-distribution**, so a graceful-degradation claim cannot be
quietly upgraded into a claim about the operating range.

**Cost, and one consequence to pick up.** The narrower range is a weaker robustness claim:
this system is not evidence about a 180-day lock, and the reports must say so. It also
reopens a hyperparameter — `gamma = 0.999` was justified in
[rl-training.md](rl-training.md) section 5 by "`N` up to 180 calendar days". At `N ~ 30`
(~21 sessions) that argument no longer carries, and 0.999 gives an effective horizon of
~1000 sessions, roughly four years. `gamma` is **not** changed here, because it is a tuned
value and this is a specification change; it is recorded as open question Q7 and settled
on a validation year, which is the only place tuning is permitted.
