# Evaluation

Annual walk-forward, model selection, out-of-sample risk metrics, the drawdown violation taxonomy, and the
acceptance criteria. Implemented in `src/evaluation/` and `src/training/{yearly_cv,
model_selection}.py`; run via [Stage 8](stages.md) and reported by Stage 12.

---

## 1. Annual walk-forward

The requirement is retraining every year, so the split is by year. Ordinary k-fold is prohibited — it would
train on the future.

**Expanding window:**

```
Train: [2004 .. Y-2]    Validate: Y-1    Test: Y
```

```
Train 2004-2010  Val 2011  Test 2012
Train 2004-2011  Val 2012  Test 2013
Train 2004-2012  Val 2013  Test 2014
...
```

The training window expands rather than rolls, because there is only ~20 years of data and the crisis regimes
in it (2008, 2020, 2022) are exactly what the risk layer needs to have seen.

### The deployment protocol

Each year:

```
first trading day:  freeze data; train using only past data
during the year:    parameters frozen; daily inference only
year end:           evaluate; retrain for the next year
```

Three prohibitions, each enforced by a test rather than by discipline:

| Prohibition | Test |
|---|---|
| No mid-year retraining on test-year data | Fold config asserts `train_end < test_start` |
| No full-sample feature normalization | Scaler artifact's fitted range must not overlap the test year |
| No hyperparameter tuning on test performance | Selection record must cite validation metrics only |

The third is the easiest to violate accidentally and the hardest to detect after the fact, which is why the
selection record is a required artifact rather than a convenience.

---

## 2. Model selection

Per retraining year: train on the training years, select hyperparameters on the validation year, freeze,
retrain, test on the next year.

**The rule is lexicographic, not maximum return:**

```
Level 1: keep only models satisfying the validation risk criteria
Level 2: among those, maximize cumulative log return

If NO model satisfies the risk criteria:
    select minimum D_max violation, then highest return
    mark the fold: "constraint validation failure"
```

Selecting a high-return model that violates the risk limit is prohibited outright. A fold marked
`constraint validation failure` stays marked all the way into the final report — Stage 12 will not quietly drop
it, and a report containing such folds must say so in its headline.

Level 1's validation risk criteria (config, `evaluation.selection`):

- realized max drawdown `<= D_max` for each evaluated `(N, D_max)` cell, or a breach classified as
  market-forced (§4 below);
- zero preventable violations;
- zero lock violations, zero feasibility violations.

The selection record `folds/<year>/selection.json` names every candidate, its validation metrics, which level
eliminated it, and the criterion that decided the winner.

---

## 3. Out-of-sample metrics

Computed per fold from `trajectory.parquet`, so every backtest — baseline, RL, stress cell — goes through the
identical metric code:

| Group | Metrics |
|---|---|
| Return | Cumulative return, annualized return |
| Risk | Volatility, maximum drawdown, drawdown duration, time under water |
| Tails | Worst 1-day, worst 5-day, worst 21-day return |
| Behavior | Cash weight distribution, turnover |
| **Constraints** | **Lock violations, action feasibility violations**, safety intervention count |

The last row is not diagnostic colour. It is a hard acceptance criterion:

```
lock violations              == 0
execution feasibility violations == 0
```

A fold failing either is not a weaker result — it is an invalid one, and it invalidates the run.

---

## 4. Drawdown violation taxonomy

When realized drawdown exceeds `D_max`, the report **must** distinguish two categories. Conflating them is how
a broken safety layer gets excused as bad luck.

### A. Preventable violation — an implementation failure

- The agent's action should have been blocked and was not.
- The safety engine did not execute as defined.
- A bug in the risk projection.

**Any preventable violation is a project-blocking defect.** It fails the hard engineering criteria in §5. It is not
tuned away; it is fixed.

Detection: replay the decision through the projection and assert that the executed action was in the feasible
set given the information available at that decision. A mismatch is a preventable violation by definition, and
this replay check runs automatically over every trajectory.

The replay needs the action that was actually projected, so `trajectory.parquet` records `proj_weights`
(and `raw_weights`, and the decision-time NAV/peak/drawdown) alongside the position that resulted. Inferring
legality from the resulting position instead would be much weaker — prices move between the decision and the
close, so a position tells you little about whether the action that produced it was legal.

> **The detector must be given the dividend stream, and this is not optional.** A trajectory records shares at
> the close, *after* distributions have been reinvested as share accretion. Accretion is deliberately exempt
> from both the lock and the capital-preservation cap — it is a corporate action, not a trade, which is why
> `Ledger.accrue_shares` is a separate entry point the lock manager never observes
> ([portfolio-ledger.md](portfolio-ledger.md)). Comparing recorded closes directly therefore reports **every
> distribution paid during a capital-preservation window as a cap breach**, which is exactly what happened the
> first time this ran: 164 false positives on `equal_weight` in 2020 alone.
>
> The fix is exact, not a tolerance. Since `shares_close = shares_executed × (1 + div/close)`, the accretion
> inverts cleanly. A threshold would not have worked: XLE paid \$0.2637 on 2020-03-23 into a collapsed \$11.79
> price, a **2.24% one-day accretion**, which no plausible fixed band separates from real dip-buying.

### B. Market-forced violation — a constraint stress event

- A locked asset could not be sold.
- An overnight gap.
- Actual historical prices simply exceeded the limit despite legal behavior.

Legitimate. It is the honest consequence of the constraint being action-level rather than a path guarantee
([risk-envelope.md](risk-envelope.md) §1).

**Required record per breach**:

```
breach_start
breach_depth
breach_duration
locked_exposure_at_breach
available_cash_at_breach
actions_blocked_by_lock
```

`actions_blocked_by_lock` is the one that carries the argument: it shows what the agent tried to do and could
not. Without it, "market-forced" is an assertion rather than evidence.

Stage 12 reports A and B in separate tables. They are never summed into a single "violations" count.

---

## 5. Acceptance criteria

A model is not accepted because `return > benchmark`. The full table, evaluated by Stage 12 into
`acceptance.json` as explicit pass/fail:

### Hard engineering — all must be exactly zero

```
illegal lock sells
negative cash
negative shares
pre-inception trades
future-data accesses
preventable D_max violations
```

Any non-zero value fails the model. There is no partial credit here.

### Risk reporting — all must be present

```
OOS maximum drawdown
number of D_max breaches
breach depth, breach duration
preventable vs market-forced classification
safety intervention frequency
```

These are reporting requirements: a run that cannot produce them is incomplete, independent of performance.

### Performance — compared against all six baselines

Primary: `spy_buy_hold`, `momentum`, `spy_tlt_60_40`. Also required: `cash`, `equal_weight`,
`classical_optimizer`. All come from Stage 5 and run through the same constraint layer, so the comparison is
meaningful. Specifications in [baselines.md](baselines.md).

Two comparisons the report must call out specifically:

- **vs `spy_tlt_60_40` in the 2022 fold** — the cleanest test of whether the risk envelope learned anything
  beyond "bonds are safe", since that is the year the assumption failed.
- **vs `momentum` as `N` grows** — momentum degrades sharply under a long lock. If the agent does not degrade
  less, parameter conditioning on `N` has not bought anything.

All results are frictionless (D10), and every report states that plainly — not as a caveat, but so a
reader knows what the number is.

---

## 6. Compute sizing (D5)

Single machine, one GPU. Defaults:

| Knob | Default | Full |
|---|---|---|
| Folds | every year from 2012 | same |
| Seeds per fold | 1 | `--seeds K` |
| Hyperparameter candidates per fold | 4 | `--candidates K` |
| Eval `(N, D_max)` cells per fold | 9 (3x3) | `--grid full` (35) |

Report the observed seed-to-seed spread whenever `--seeds > 1`. A single-seed result on a stochastic policy
over one test year is a data point, not a conclusion, and the report should present it as such.
