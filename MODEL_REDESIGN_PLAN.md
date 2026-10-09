# Model Redesign Plan — making the agent's decisions count

**Status: Phase 1 COMPLETE (2026-10-09): `seeds_v1` trained to 32%, ensemble test Sharpe 0.24 -> 0.85, seed noise floor measured; see STATUS.md.**
Phases 2–5 are planned, not built. Written 2026-10-07.
Related: [INCREMENTAL_TRAINING_PLAN.md](INCREMENTAL_TRAINING_PLAN.md) (the `budget_v1` campaign this
plan responds to), [CONTINUAL_TRAINING_PLAN.md](CONTINUAL_TRAINING_PLAN.md) (data versioning and
fine-tuning, deferred), [STATUS.md](STATUS.md).

---

## 1. Why: what `budget_v1` showed

The `budget_v1` campaign trained the existing setup from 2% to 32% of the configured budget, with a
full evaluation at each checkpoint:

| Budget | Val Sharpe (all candidates) | Test Sharpe | Worst test year drawdown |
|---|---|---|---|
| 2% | 0.76 | 0.92 | 10.2% |
| 4% | 0.61 | 0.44 | 25.9% |
| 8% | 0.70 | 0.64 | 26.0% |
| 12% | 0.74 | 0.84 | 15.9% |
| 16% | 0.68 | 0.87 | 12.7% |
| 22% | 0.74 | 0.53 | 25.8% |
| 32% | 0.72 | 0.84 | 10.7% |

- **16× the budget produced no detectable improvement.** 32% vs 2% is "no clear change" on both
  validation (−0.26, 90% interval −0.66 to +0.17) and test (−0.09, −0.50 to +0.32). The fitted
  slope per budget doubling is −0.04 on validation and +0.003 on test.
- **Checkpoint-to-checkpoint swings of ±0.4 Sharpe**, larger than any plausible improvement.
- **About 60% of the agent's decisions are discarded.** On those steps the risk layer finds that
  even the safest reachable portfolio (locked holdings, everything else in cash) breaches the
  drawdown budget, so it executes that portfolio and ignores the proposal (`infeasible_fallback`).
  PPO still updates the policy from those steps, although the action had no effect on the outcome.
  This did not improve with training (48% at 2%, ~60% from 4% to 32%).
- **Model selection mostly selects noise.** 3–5 of 14 test years switch candidate at every
  checkpoint; the selected candidate's validation Sharpe (≈1.1–1.3) is well above the candidate
  average (≈0.7) and the advantage does not carry over to test.
- **Two of the four candidates barely train.** The lr 3e-4 candidates were stopped by the KL limit
  (`target_kl = 0.02`) inside their first epoch on most rollouts (1.0 → 3.8 epochs per rollout over
  the campaign, against ~10 for the lr 1e-4 candidates).

The conclusion: the limiting factor is how the environment, action and reward are set up, not the
training budget or the network size.

---

## 2. Decisions already made

| # | Decision | Notes |
|---|---|---|
| D-A | **Stop adding budget to `budget_v1`.** Revised 2026-10-08: **keep adding budget to a FIXED setting** while its validation curve keeps rising. | `budget_v1` stays saved at 37.84% as the record of the original design. The original "budget does not help" reading was an artifact of per-year selection: pooled over all validation years, the lr 3e-4 settings improve with budget (high_entropy 0.81 → 0.95 validation Sharpe, 2% → 32%; test 0.41 → 0.72) while the lr 1e-4 settings decline. See STATUS.md, 2026-10-08 |
| D-B | **No transaction costs, anywhere.** | Fills stay at the next session's open with `cost_bps = 0` (decision D10 stands). With a lock period, costs are negligible. Turnover penalties and cost models are out of scope and will not be proposed again |
| D-C | **One hyperparameter setting plus three seeds, instead of four settings and one seed.** | Same training cost. Rationale in §4 |
| D-D | **Momentum is used as a starting point, not as an anchor.** | The final policy must not be a lightly modified momentum strategy. Method in §7 |
| D-E | **No network or algorithm upgrades until the environment is fixed.** | Temporal encoders, attention, LSTMs, and SAC/TD3 comparisons come after this plan, so any gain can be attributed |
| D-F | **Risk-layer rule when over budget: only risk-reducing actions are allowed** (decided 2026-10-08). | Replaces today's "freeze into the safe portfolio". The agent still decides what to buy and how much; the risk layer only checks the result is no riskier than doing nothing. Details in §5d |

---

## 3. The lock rule (authoritative restatement)

Confirmed against the implementation (decision D13, invariants I3/I4):

- **Each ETF is locked independently.** Buying A has no effect on B.
- **After a purchase of A at session T, no A may be sold for N days.**
- **During the lock, more A may be bought.** Any purchase of A re-locks the **entire** A position
  (old and new shares) for N days from the latest purchase.
- **Other ETFs can be bought and sold freely** according to their own locks.
- **Long-only.** No shorting.
- **Cash is the only asset with no lock.** It is the only fully flexible part of the portfolio.
- Dividend reinvestment does not trigger a lock.

What this means for the design: at any session, each ETF falls into one of two states.

| State | Allowed actions |
|---|---|
| Unlocked | buy, sell, hold |
| Locked | hold, or buy more (which re-locks the whole position for N days) |

Cash funds every purchase and receives every sale.

---

## 4. Phase 1 — Fix the measuring stick (no environment change)

**Built 2026-10-08.** The setting is **`high_entropy`** (lr 3e-4, ent_coef 0.02), chosen on
validation pooled over all 14 folds and every `budget_v1` checkpoint (0.85 average validation
Sharpe at D_max 5/10/15%, the highest of the four, and rising with budget). The KL early-stop
is left as configured: this setting improves with budget under it, and changing it would change
the config. Campaign `seeds_v1`: 3 seeds (1001–1003) × 14 folds, trained from scratch on the
standard checkpoint schedule, run with `scripts/s14_incremental.py`:

- `--init-fresh --setting high_entropy --seeds 3` creates the campaign (mode `ensemble`).
- At each checkpoint, every seed is scored on validation and test at D_max 5/10/15%; the
  **reported policy is the ensemble** (the average of the three seeds' portfolios), evaluated on
  the full Stage 8 grid, so Stage 12 and the hard-acceptance checks read it unchanged.
- The learning curve adds a by-ceiling table and a per-seed table (each seed, seed mean, seed
  range, ensemble), for validation and test. The seed range is the noise floor.
- Per-year selection is gone, so per-fold two-year validation (item 2 below) is no longer
  needed: the one remaining choice, the setting, was made on all validation years pooled.

The items as originally planned:

Without this, no later change can be judged: current checkpoint-to-checkpoint noise (±0.4 Sharpe)
exceeds the improvement any single change is likely to deliver.

1. **One hyperparameter setting, three seeds (D-C).**
   - Choose the setting **once**, from validation performance pooled across all folds, not
     re-selected each year from a single year.
   - Train three seeds of that setting per fold. Report the mean and the spread across seeds.
   - Also report the **seed-average portfolio** (average the three models' target weights), which is
     usually steadier than any single model.
   - Resolve the KL early-stop imbalance as part of choosing the setting, on validation years only.
2. **Selection and reporting on what is actually reported.** Validation uses two years where a fold
   has them, and is scored on Sharpe, not cumulative return averaged over the grid.
3. **Report three ceilings, not one.** The 5% ceiling is where the agent acts least (fallback ~70%
   of test days), so results at 5% largely reflect the risk layer. Report 5%, 10% and 15% side by side.
4. **Measure the noise floor first.** Retrain the existing 2% setup with two extra seeds to see how
   much test and validation Sharpe move by chance alone.

**Done when:** a single configuration can be reported as mean ± spread over three seeds, and the
spread is known.

---

## 5. Phase 2 — An environment where the agent only decides when it has a choice

The core change. Built as a new environment version; the current one stays untouched for `budget_v1`.

### 5a. Event-driven decisions

- The agent observes the market every session, but **makes a decision only when it has a real
  choice**: an ETF has unlocked, there is cash to deploy, or the risk layer leaves room to act.
- On sessions with no real choice, time simply advances and returns accumulate. Those sessions are
  **not** training samples, so the policy is no longer updated from actions that could not matter.

### 5b. Actions follow the lock rule (§3)

- **Unlocked ETFs:** the agent sets a target weight (buy, sell or hold).
- **Locked ETFs:** the agent may only hold or add. It must see that adding re-locks the whole
  position for N days (this is already in the state: lock status, days remaining and unlock
  progress are per-ETF inputs, and the critic sees them).
- **Cash:** whatever is left.
- The agent proposes only what is feasible under the lock rule by construction, instead of
  proposing a full portfolio for the constraint layer to rewrite.

### 5c. Reward per decision

- **The reward for a decision is the cumulative log return from that decision to the next one.**
  This is a holding-period reward by construction, without stitching together fixed 5/20/60-day
  windows (which double-count and require future data at decision time).
- A drawdown penalty is tested later as a separate variant (Phase 4), not built in.
- "Return compared with not trading" is reported as a **diagnostic** of what the decisions added,
  not used as the reward: as a reward it would change what the agent optimizes and could bias it
  toward never trading.

### 5d. The safety layer stays as the last line of defence, with the over-budget rule changed (D-F)

- The constraint layer keeps enforcing the lock and long-only exactly as today.
- **Within the drawdown budget:** unchanged. A proposal whose stressed loss fits the budget is executed.
- **Over the budget** (the stressed loss of even "do nothing" exceeds it, typically because a locked
  position alone is too risky):
  - **Reference point:** "do nothing", i.e. locked positions held, everything else in cash.
  - **Rule:** after the action, the portfolio's stressed loss must be **no larger than** that of
    "do nothing".
  - A proposal that lowers stressed loss (e.g. using cash to buy bonds or gold as a hedge) is
    executed **as proposed**; the amount is the agent's choice.
  - A proposal that raises stressed loss is scaled back **proportionally**, toward "do nothing",
    until the rule holds. The layer never picks which part of a proposal to keep and never
    substitutes assets of its own: it is a floor, not a second strategy.
  - "Lowers or raises risk" is judged with the same stress test as today (5-day horizon, CVaR,
    historical crisis library), so the standard does not change.
- This replaces today's behaviour, where an over-budget state discards the proposal and holds the
  safe portfolio, which also forbids hedging. Breaches in such states continue to be classified as
  market-forced, not preventable.
- The guarantee that remains: **when over budget, no action ever increases risk.**
- In the new design the layer should rarely need to change a decision; any override is a diagnostic
  to investigate, not normal operation.

### 5e. Re-validation

- Every simulator invariant and gating test (I1–I6, T1–T15) is re-run against the new environment
  before any training.
- `gamma` and `target_kl` are re-set on validation years, since one step now spans a variable
  number of sessions.

**Done when:** the override rate (5f) is near zero, and all invariants pass, including new tests for
the over-budget rule: a risk-reducing proposal is executed unchanged, a risk-increasing one is scaled
back until it is no riskier than doing nothing, and no executed action ever raises stressed loss while
over budget.

### 5f. Action-effectiveness metrics (reported at every checkpoint)

- **Fully executed rate:** share of decisions executed exactly as proposed.
- **Modified rate:** share changed by the safety layer, and by how much (proposed vs executed weights).
- **Discarded rate:** share replaced outright (today's `infeasible_fallback`).
- **Decision frequency:** decisions per year, and sessions per decision.

These become core training diagnostics, alongside return and drawdown.

---

## 6. Phase 3 — Comparing the new environment with the old

Same folds, same evaluation (Phase 1), three seeds each, at a fixed small budget (e.g. 8%):

| Run | What it shows |
|---|---|
| Old environment, Phase 1 evaluation | The control |
| New environment | Whether decisions that count make the policy better |

Plus the effectiveness metrics (5f) for both. The question is answered on validation years first,
test second.

---

## 7. Phase 4 — Using momentum without becoming momentum (D-D)

Constrained momentum (Sharpe 1.13) is the strongest baseline. Part of its strength is that it
rebalances monthly, which suits a ~30-day lock; the current agent tries to act daily and keeps
colliding with the lock.

### Option A (primary): imitate first, then let go

1. **Pre-train the policy to imitate constrained momentum's decisions**, using training-window data
   only (no lookahead; momentum signals are computed from past prices).
2. **Then train with RL as normal, with no requirement to stay close to momentum.** The policy starts
   from a good strategy instead of a random one, but what it ends up as is learned.

### Option B (alternative): a strategy allocator

The agent chooses how much to give to a few building blocks (momentum, 60/40, cash, and its own free
allocation), learning when to trust which. Kept as the fallback if Option A ends up too
momentum-like or too unstable.

### Not chosen

Small adjustments on top of momentum's weights: the result would inevitably look like momentum.

### Similarity is measured, not assumed

Every evaluation reports how close the policy is to momentum: correlation of returns and overlap of
holdings. This shows whether the agent has learned something of its own.

---

## 8. Phase 5 — Reward variants

On the new environment, each with three seeds against the plain holding-period reward:

- a drawdown penalty (larger penalty the deeper the drawdown);
- others only if Phase 3 or 4 points to a specific need.

No transaction-cost or turnover terms (D-B).

---

## 9. Out of scope for this plan (later)

- **Network upgrades:** a temporal encoder over the last 60–120 sessions, cross-sectional attention
  between ETFs, a regime encoder. Only once the environment is fixed, and only with multi-seed comparisons.
- **Algorithm comparisons** (PPO vs SAC vs TD3) under identical state, action, reward and constraints.
- **More budget** for any configuration that has not first shown it improves with budget.
- **Data versioning and periodic fine-tuning** ([CONTINUAL_TRAINING_PLAN.md](CONTINUAL_TRAINING_PLAN.md)).
  Data versioning must be in place before any data re-fetch.

---

## 10. Resolved decision

| # | Question | Decision |
|---|---|---|
| O-1 | How should the risk layer behave when a locked position alone already exceeds the drawdown budget (the main cause of the ~60% discard rate)? Options considered: (1) keep freezing into the safe portfolio; (2) treat locked positions as sunk and check only new purchases; (3) when over budget, allow only risk-reducing actions | **Option 3** (2026-10-08), recorded as D-F and specified in §5d. Option 1 leaves the agent frozen and unable to hedge; option 2 would weaken what the drawdown ceiling means |

---

## 11. Order of work

| Step | Phase | Changes the environment? |
|---|---|---|
| 1 | Phase 1: measuring stick (one setting, three seeds, two-year validation, three ceilings, noise floor) | No |
| 2 | ~~Decide O-1~~ — decided: option 3 (D-F) | — |
| 3 | Phase 2: event-driven, lock-aware environment with effectiveness metrics | Yes (new version) |
| 4 | Phase 3: old vs new environment | No |
| 5 | Phase 4: momentum imitation, then free RL | Small addition |
| 6 | Phase 5: reward variants | Small |

Each step is decided and started separately; this document does not authorize any of them.
