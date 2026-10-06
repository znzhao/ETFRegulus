# Incremental Training Plan — growing the budget in idle-time sessions

**Status: BUILT and ready; no campaign training has run yet.** Written 2026-10-05, built the
same day. The campaign `budget_v1` exists at C0 (the 2% models), and C0 has not been evaluated:
the first session starts with that evaluation (~30 min). Entry point:
[scripts/s14_incremental.py](scripts/s14_incremental.py).
Related: [STATUS.md](STATUS.md) (current results), [reference/rl-training.md](reference/rl-training.md),
[reference/evaluation.md](reference/evaluation.md).

---

## 1. Goal

Every result so far was trained at ~2% of the configured budget (`training.total_timesteps =
2,000,000` per candidate; the latest walk-forward used 40,000). A full-budget walk-forward is
~112M timesteps — roughly **two days of continuous CPU time**, which this machine cannot give in
one block because it is also used for other work.

So instead of one blind run:

1. Train in **sessions of whatever idle time is available** ("I have 2 hours").
2. Make it possible to **stop at any moment and continue later** with nothing lost.
3. At fixed **checkpoints** along the budget, run the full performance report and compare it
   against **the baselines and every previous checkpoint**, so it is always visible whether more
   budget is actually making the policy better.

A checkpoint that shows no progress is an acceptable outcome. The requirement is to be *aware* of
it, not to hide it or to stop automatically.

---

## 2. Principles (non-negotiable)

| # | Principle | Why |
|---|---|---|
| P1 | **Resume is an exact continuation**, not a warm start | The current `warm_start` copies network weights only. It drops the Adam optimizer state, the reward normalizer, the timestep counter and the RNG state, so every resume would perturb training. A learning curve built that way measures the restarts, not the budget |
| P2 | **Stopping never loses more than one training chunk** | Ctrl-C, closing the terminal, a crash or a power cut must all be survivable |
| P3 | **Sessions are measured in hours; checkpoints in budget** | Session length is whatever idle time exists. Checkpoints are fixed in advance so every evaluation compares like with like |
| P4 | **All 56 candidates (14 folds × 4) reach a checkpoint together** | Selection compares candidates; a candidate that is further along would win for the wrong reason |
| P5 | **Any decision to stop is based on the VALIDATION years, never the test years** | Choosing the budget where test Sharpe looks best is tuning on the test set. Test numbers are reported at every checkpoint, but only for information |
| P6 | **The campaign's config is frozen** | The config hash is recorded when the campaign starts. A resume with a different hash refuses to run, because a learning curve across two configs means nothing |
| P7 | **The learning rate must not depend on the total budget** | It is constant today (`3e-4` / `1e-4`). A schedule that decays toward `total_timesteps` would make "1% then +1%" different from "2% at once" |

---

## 3. Checkpoint schedule

The budget **grows by doubling between checkpoints**, but sessions are arbitrary length — a
checkpoint can be reached across many short sessions.

Budget is counted in **PPO rollouts**, not raw timesteps: one rollout is `n_steps × n_envs = 2048 × 8
= 16,384` timesteps, and SB3 always finishes a whole rollout, so a timestep target that is not a
multiple of 16,384 silently overshoots. Every checkpoint is rounded **up** to whole rollouts, which is
what SB3 would run for that target anyway: 100% = **123 rollouts** per candidate.

| Checkpoint | Rollouts / candidate | Timesteps / candidate | Training to reach it from the previous one* | Cumulative* |
|---|---|---|---|---|
| C0 — 2% | 3 | 49,152 | — the existing 2% models (exactly 3 rollouts each) | — |
| C1 — 4% | 5 | 81,920 | ~45 min | ~45 min |
| C2 — 8% | 10 | 163,840 | ~1.7 h | ~2.5 h |
| C3 — 16% | 20 | 327,680 | ~3.2 h | ~5.5 h |
| C4 — 32% | 40 | 655,360 | ~6.4 h | ~12 h |
| C5 — 64% | 79 | 1,294,336 | ~12.5 h | ~25 h |
| C6 — 100% | 123 | 2,015,232 | ~14 h | ~39 h |

\* Training only, at the default estimate of ~20 s per rollout (pure training measured ~940
timesteps/s in the 2% run, plus the save). **The first session measures the real rate** and
`--plan` uses it from then on. Each checkpoint evaluation adds **~30 min** on top (14 folds × ~110 s,
measured as ~22 s per candidate validation, plus the reports), so the evaluations alone total ~3.5 h
over the campaign.

---

## 4. A session, step by step

This is the procedure for a future chat. It begins when the user says something like
*"I'm going to be idle for about 2 hours."* All commands run with `.venv/Scripts/python.exe`.

**Claude does:**

1. **Read the state**: `python -m scripts.s14_incremental --status`. Report in a few lines:
   average budget, which checkpoints are reported, whether an evaluation is half-done.
2. **Plan the session**: `python -m scripts.s14_incremental --plan 2`. It uses the measured
   throughput once a session has run. Tell the user what the session will achieve before starting.
3. **Launch it in the background**: `python -m scripts.s14_incremental --hours 2`.
   The script enforces the deadline itself (5-minute margin by default, `--margin-minutes`).
4. **To stop early** (the user needs the machine back): `python -m scripts.s14_incremental --stop`.
   The session finishes its current rollout (~20 s) or fold evaluation (~2 min), saves, and exits.
   Killing the process outright is also safe: at most the rollout in progress is lost, but that
   session is then missing from the session history in `--status`.
5. **On completion**: report how far it got. If a checkpoint was reported, show
   `artifacts/incremental/budget_v1/learning_curve.md` and say in one sentence whether it is
   progress, flat or a regression (§6).
6. **Update [STATUS.md](STATUS.md)**: the current-position table and a decision-log line.

**What the script does in a session**, in a loop until the deadline:

- If every candidate has reached a checkpoint that is not yet reported: evaluate it (§5), fold by
  fold. An evaluation cut off by the deadline resumes at the next unfinished fold.
- Otherwise, train every candidate toward the next checkpoint, fold by fold: one vectorized env
  per fold (the env depends on the fold's training window), its 4 candidates advanced one rollout
  at a time, **saved after every rollout** (write to temp, then rename).
- Checks the clock and the STOP file before every rollout and every fold evaluation, and does
  not start one it cannot finish.
- Keeps Windows awake while running (`SetThreadExecutionState`), and only while running.
- At startup, `reconcile` re-reads every saved state and corrects the campaign file if a kill
  left it one rollout behind.

---

## 5. The checkpoint report

Produced automatically at every checkpoint into
`artifacts/incremental/budget_v1/checkpoints/C<k>/` (Stage 8-format folds, `walk_forward_summary.json`,
`checkpoint.json`), and rendered into one cumulative `learning_curve.md` (also copied to
`artifacts/reports/incremental/budget_v1/`) that grows a row per checkpoint.

**5a. Per checkpoint: the standard report.** The unchanged Stage 12 report is run on the
checkpoint's walk-forward results, at `N = 30`, `D_max = 5%`, 2012–2025, into
`artifacts/reports/incremental/budget_v1/C<k>/`. It runs with `--no-acceptance`: the 22-criterion
acceptance table also needs Stages 9–11, which are not re-run per checkpoint (§8). The hard
engineering criteria (lock, feasibility, preventable violations) are checked at every checkpoint
regardless. **C0 should reproduce the known test Sharpe of 0.92**: same models, same
deterministic evaluation.

**5b. Against the baselines.** Both comparisons, side by side:

- **Unconstrained baselines** (`spy_buy_hold`, `spy_tlt_60_40`, `momentum`, `equal_weight`,
  `classical_optimizer`, `cash`): the current headline convention.
- **Constrained baselines** (the same strategies under the same lock and `D_max` envelope): kept as
  a reference because on that comparison `momentum_constrained` beat the policy (Sortino 1.62 vs
  1.33). A gain that shows up on one comparison and not the other should be noticed.

Baselines are deterministic and take ~15 s, so the report simply recomputes them each time.

**5c. Against previous checkpoints — the learning curve.** One row per checkpoint:

| Checkpoint | Budget | Val Sharpe (selected) | Val Sharpe (all candidates) | Δ val vs previous | Test Sharpe | Test band q05–q95 | Δ test vs previous | Test max DD | Gap to best market baseline | Gap to best constrained baseline | Selections changed | Hard acceptance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

"Val Sharpe (selected)" is the chosen candidate on its validation year, and is biased upward
because it was chosen for being best there. "All candidates" averages every candidate and is
the cleaner measure of whether training itself is improving. Δ columns are §6.

Plus, per checkpoint:
- **Which candidate was selected in each fold**, and how many folds changed selection since the
  previous checkpoint. Frequent switching is itself a signal: the candidates are not separating.
- **The hard-acceptance line**: lock, feasibility and preventable violations. Must stay zero at
  every checkpoint; a non-zero count is a bug, not a performance result.
- **Training diagnostics** averaged over candidates: PPO epochs actually run per rollout (split by
  learning rate, see §9 O5), `proj_distance`, `infeasible_fallback` rate, mean cash weight. These
  answer Q3 in STATUS.md (does `proj_distance` fall without a penalty?) as a by-product.

---

## 6. How "progress" is judged

The verdict column is computed, not eyeballed, and it is **informational** — it never stops the run.

- For a checkpoint and the previous one, compute the Sharpe **difference** with a paired stationary
  block bootstrap: the same resampled session blocks applied to both trajectories, as in Stage 10.
  Pairing matters — the two checkpoints face the same market, so most of the noise cancels.
- **improved** — the 90% interval of ΔSharpe is entirely above 0
- **regressed** — entirely below 0
- **no clear change** — straddles 0

Computed on **validation** (this is the one that may inform a decision) and on **test** (shown, but
never used to decide anything).

Expect "no clear change" often, especially early and with one seed. One flat checkpoint means
little; two or three in a row on validation is the signal worth discussing.

**Stopping is the user's call.** The suggested reading: if validation has shown no clear
improvement for **two consecutive checkpoints**, that is a reasonable point to stop and spend the
remaining compute on extra seeds at the best budget instead (§8).

---

## 7. What was built (2026-10-05)

| Piece | Where | Proven by |
|---|---|---|
| Checkpoint schedule in whole rollouts: `[3, 5, 10, 20, 40, 79, 123]` | `src/training/incremental.py` | `tests/training/test_incremental.py`: C0 = exactly the 49,152 timesteps the 2% run trained |
| **Exact resume**: one `learn()` per rollout, forced env reset, reseed from `(candidate seed, rollouts done)`; optimizer, normalizer and counters from the saved files | same | **slow test: 2 rollouts straight vs 1 + save + reload + 1 give bit-identical weights, Adam state and reward normalizer**; a second slow test checks a resumed rollout actually changes the weights |
| Crash-safe state (temp dir + rename + COMPLETE marker); `recover_state`; `reconcile` | same | tests for a crash mid-save, between the renames, with only `.old`, with nothing usable, and with the campaign file one rollout behind |
| Campaign file `campaign.json`; config-hash freeze (P6) | same | refusal on a hash mismatch |
| Session driver: `--init / --status / --plan / --hours / --stop`, deadline, STOP file, keep-awake | `scripts/s14_incremental.py` | end-to-end on a throwaway toy setup (1 fold, 2 candidates, 64-step rollouts): 7 checkpoints evaluated and reported; `--stop` honoured within a second; a hard kill mid-rollout resumed cleanly. Toy artifacts deleted afterwards |
| Shared fold scoring, so Stage 8 and the campaign use the same code | `src/evaluation/fold_eval.py` (moved out of `s08_walk_forward.py`, behaviour unchanged) | full suite: 393 passed |
| Paired-bootstrap verdict, learning-curve table | `src/evaluation/learning_curve.py` | synthetic series with known improvement / regression / noise |
| Campaign `budget_v1` created from `s08_walk_forward_20260903T224114Z_a207776c` | `artifacts/incremental/budget_v1/` (766 MB) | `--init` checked every one of the 56 models: learning rate, entropy coefficient, seed, `n_steps`, `n_envs`, and exactly 49,152 timesteps |

Nothing about how the existing stages behave was changed; `s08_walk_forward` remains the
one-shot protocol.

---

## 8. After the curve is in

When the user decides to stop growing the budget:

1. **Pick the budget** from the validation curve (P5).
2. **Extra seeds at that budget only** — `--seeds K` retrains the 14 selected models; ~6 h per
   extra seed at 50% budget, ~12 h at 100%. This can run in the same idle-time sessions.
3. **Re-run Stages 9–11** (stress, bootstrap, adversarial) on the final policy, then Stage 12.
4. Only then is a performance claim made in STATUS.md.

---

## 9. Decisions

| # | Question | Decision |
|---|---|---|
| O1 | CPU or GPU? | **CPU** (user, 2026-10-05) |
| O2 | Lower process priority / fewer envs so the machine stays usable while training? | Normal priority, `n_envs = 8`: sessions run only when the machine is idle |
| O3 | Paired-bootstrap interval width (90%) and the "two flat checkpoints" suggestion | As written in §6 |
| O4 | Keep the constrained baselines in the checkpoint report? | Yes (§5b) |
| O5 | **The KL early-stop truncates the fast-learning-rate candidates' updates.** In the 2% models, `base` and `high_entropy` (lr 3e-4) stopped inside the **first** of 10 PPO epochs on every rollout in all 14 folds (`target_kl = 0.02`); the lr 1e-4 candidates ran up to 10. | **Keep the config as-is** (user, 2026-10-05): changing it would invalidate the 2% models as C0, and it is a tuning decision that belongs on validation years. The selection sweep already varies the learning rate. The learning curve reports epochs per rollout by learning rate at every checkpoint, so it stays visible |
