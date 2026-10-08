# Continual Training Plan — new data, fine-tuning, and a deployable model

**Status: planned, not built.** Written 2026-10-06. Nothing here exists in code yet.
Related: [INCREMENTAL_TRAINING_PLAN.md](INCREMENTAL_TRAINING_PLAN.md) (the `budget_v1` campaign,
paused at 9.06%), [STATUS.md](STATUS.md).

---

## 1. Why this plan exists

Two facts surfaced on 2026-10-06:

1. **No model has trained on anything after 2023.** The data runs to 2026-08-31. Each walk-forward
   fold trains on `[2004 .. Y-2]`, validates on `Y-1` and tests on `Y`, so the last fold
   (`fold_2025`) trains through 2023. 2024 and 2025 are used only for selection and testing, and
   **January–August 2026 is not used at all**: there is no `fold_2026`.
2. **A useful model will have to be refreshed every few months**, and retraining from zero each
   time is too expensive on this machine. The model in use has to be **fine-tuned** with the new
   data instead.

The first is correct for an honest backtest, but it means **no deployable model exists today**.
The second is a design requirement the project has not had to meet before.

### What adding recent data will and will not do

| | Effect |
|---|---|
| Backtest numbers | **None, by construction.** Every fold already trains on everything available before its test year. Putting later data into training would leak the test set |
| The flat learning curve (C0 → C2) | **Not the cause.** The swings hit early folds too (2013, 2016), which have plenty of data. The diagnostics point at training dynamics instead: `infeasible_fallback` rose 48% → 61%, and the lr 3e-4 candidates run ~1 of 10 PPO epochs per rollout against 10 for lr 1e-4 |
| A model you would actually use | **Large.** It should have seen 2024–2026, the most recent market conditions. None has |
| A `fold_2026` (train to 2024, validate 2025, test Jan–Aug 2026) | One more genuinely untouched test period: partial, but data no decision in this project has ever been tuned on |

---

## 2. Principles

| # | Principle | Why |
|---|---|---|
| C1 | **The backtest must test the procedure that will run live** | Live, the model for the next period is the current model plus a fine-tune. Today every fold is trained independently from scratch, so the backtest measures a procedure nobody will run |
| C2 | **Data is versioned and frozen; every model records the version it used** | Re-running Stages 1–3 in place rewrites history (yfinance revises adjusted prices after distributions; Stage 1 already overwrites a trailing 90-day window), which shifts features across all years. A campaign checks only the config hash, so it would silently mix two datasets on one learning curve |
| C3 | **Fine-tune on the whole extended window, not only the new months** | Training only on recent months makes the model forget older regimes (2008, 2020, 2022). The environment window simply extends to `[2004 .. now]`, optionally weighted toward recent years |
| C4 | **A refreshed model is promoted only if it beats the current one on a recent holdout** | Fine-tuning can make a model worse. The old model is kept for rollback |
| C5 | **No tuning on test data**, as everywhere else in this project | Promotion and scaler decisions are made on validation or holdout periods |

---

## 3. The pieces

### 3a. Data versioning (prerequisite, small)

- Each fetch–curate–features run writes a **new, immutable data version** (e.g. `data/versions/2026-08-31/`)
  instead of overwriting `data/curated` and `data/features`. It is identified by its end date and a
  content hash.
- Every campaign, checkpoint and production model records its data version. A resume or fine-tune
  against a different version **refuses**, just as `s14_incremental` already refuses a config-hash change.
- `budget_v1` is registered against the current data as it stands, so its learning curve stays clean.
- **Until this exists, do not re-run Stages 1–3**: it would change the inputs under `budget_v1`
  without any warning.

### 3b. Chained walk-forward: testing the fine-tuning procedure itself

Instead of 14 independent folds:

```
model_2012  = train from scratch on [2004 .. 2010]
model_2013  = model_2012 fine-tuned once 2011 becomes training data   -> test 2013
model_2014  = model_2013 fine-tuned once 2012 becomes training data   -> test 2014
...
model_2026  = model_2025 fine-tuned on [2004 .. 2024]                  -> test Jan-Aug 2026
production  = latest model fine-tuned on [2004 .. now - holdout]       -> champion/challenger (3d)
```

- **It tests the live procedure** (C1): every test year is scored by a model produced exactly the
  way the next live model will be produced.
- **It is much cheaper** than independent folds: one full training run plus 13–14 small fine-tunes,
  rather than 14 runs from scratch. It could take over much of the `budget_v1` campaign's role.
- **The production model is a by-product**: the end of the chain.
- **Reuses what exists**: `s14_incremental`'s exact resume, rollout-counted budgets, crash-safe
  state and idle-time sessions already do "continue training from a saved model". A fine-tune is
  continued training on a longer environment window.
- **Selection** (the four candidates per fold) needs a decision: carry four lineages through the
  chain and re-select at each step on that step's validation year, or settle on one hyperparameter
  setting first. Four lineages costs four times as much but keeps selection honest.

**The comparison that decides whether to trust fine-tuning:** chained vs independent-from-scratch,
at matched total compute, on validation years. If fine-tuning loses clearly, the refresh procedure
has to include periodic full retrains.

### 3c. Feature-scaler policy during fine-tuning

Features are scaled per fold, with a scaler fitted on that fold's training window (T10). A fine-tune
extends the window, so one of two policies is needed:

| Policy | For | Against |
|---|---|---|
| **(a) Keep the model's original scaler** | The network's inputs mean the same thing before and after | New values outside the old range are clipped at ±10; drift accumulates over many refreshes |
| **(b) Refit the scaler on the extended window** | Inputs stay well-scaled | Every input shifts, and the fine-tune must absorb that shift |

To be **measured** in the chained walk-forward, not chosen by argument. The prior: (a) between
occasional full retrains, (b) at each full retrain. Either way, the scaler stays leak-free: fitted on
training data only.

### 3d. Refresh procedure and champion/challenger gate

Every few months:

1. Fetch, curate and build features into a **new data version** (3a).
2. Hold out the most recent months (e.g. the last 6) as the gate period.
3. Fine-tune the current production model on `[2004 .. holdout start]`, in idle-time sessions,
   with the existing session machinery (`--hours`, `--stop`, exact resume).
4. Score the challenger (fine-tuned) and the champion (current) on the holdout, with the same
   lexicographic rule as model selection: risk criteria first, return second. Hard acceptance
   (zero lock, feasibility and preventable violations) must hold.
5. **Promote the challenger only if it is at least as good.** Keep the champion on disk for rollback.
6. After promotion, fold the holdout into training at the next refresh.

---

## 4. Suggested order

| Step | What | Why in this order |
|---|---|---|
| 1 | **Data versioning** (3a) | Small, and it protects everything else. Must come before anyone re-fetches data |
| 2 | **Fix the training dynamics** behind the flat learning curve: the rising `infeasible_fallback` rate and the KL early-stop imbalance between candidates | Fine-tuning a model that does not improve with more training is pointless. Changing `target_kl` invalidates `budget_v1` anyway, so it starts the next campaign on a sound footing |
| 3 | **Chained walk-forward** (3b) with the fixed training setup, including the scaler comparison (3c) and the chained-vs-independent comparison | Answers whether fine-tuning works as well as retraining, and produces the production model |
| 4 | **Refresh procedure** (3d) as a script plus a runbook | Only once 3 has shown fine-tuning can be trusted |

Each step is decided and started separately; this document does not authorize any of them.

---

## 5. Open decisions

| # | Question | Leaning |
|---|---|---|
| D1 | Refresh cadence: every 3, 6 or 12 months? | Decide after step 3 shows how much a fine-tune moves performance |
| D2 | Carry four candidate lineages through the chain, or one? | Four, if compute allows, to keep selection honest |
| D3 | Recency weighting of episodes during a fine-tune? | Start without; test as an ablation |
| D4 | Holdout length for the champion/challenger gate | 6 months |
| D5 | What happens to `budget_v1`? | Stays paused and saved at 9.06%. Likely superseded by step 2's new campaign, but kept as the record of the original setup |
