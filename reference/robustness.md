# Robustness Suite

The stress suite, bootstrap robustness, and adversarial historical scenarios. Implemented in
`src/evaluation/{stress,bootstrap,adversarial}.py`; run via Stages 9–11.

> **Gate (D12): none of this starts until [Stage 8](stages.md) produces a fold set meeting the hard
> engineering criteria in [evaluation.md](evaluation.md) §5.** Stress-testing a policy that has not passed walk-forward measures nothing. The specs
> are complete so that the work is ready when the gate opens — not so it can start early.

---

## 1. Stress suite (Stage 9)

The final model must not report only average OOS return.

### 1.1 Historical crisis windows

The policy is evaluated over each window with a portfolio state sampled as it would plausibly have been on
entry:

| Window | Period | What it tests |
|---|---|---|
| GFC | 2007-10 → 2009-03 | Sustained deep drawdown; credit and equity together |
| COVID crash | 2020-02-19 → 2020-03-23 | Speed — a 34% index drawdown in 23 sessions, faster than any lock |
| 2022 rate shock | 2022-01 → 2022-10 | Stocks *and* bonds fall together; TLT −31% |
| Taper tantrum | 2013-05 → 2013-09 | Duration shock without an equity crisis |
| Volmageddon | 2018-02 | Single-day vol spike |
| 2018 Q4 | 2018-10 → 2018-12 | Fast equity drawdown, quick recovery |

The COVID and 2022 windows are the two that matter most for this specific system. COVID tests whether the lock
traps the agent in a fall faster than `N` can release it. 2022 tests whether the agent learned "bonds are the
safe asset" — a lesson that is true in most of the sample and catastrophically false in 2022, which is exactly
the failure mode the multi-estimator envelope exists to catch.

**Unlike the decision-time envelope, evaluation uses the full crisis library regardless of date** — this is a
stress test, not a decision input. The two code paths are deliberately separate
([risk-envelope.md](risk-envelope.md) §4.3).

### 1.2 `N` sensitivity

Sweep `N ∈ {0, 7, 15, 21, 30, 42, 60, 90, 180}` with `D_max` fixed — the D16 operating range plus its
out-of-distribution stress points. **Every cell at `N ∈ {0, 7, 90, 180}` is labelled out-of-distribution**
in the output: the policy was not trained there, so those cells characterize degradation rather than
measure performance. Report per `N`: realized max drawdown, return,
turnover, fraction of NAV locked, actions blocked by lock, and market-forced breach count.

**Expected shape:** larger `N` → less maneuverability → more market-forced breaches and lower turnover. If
performance is *flat* in `N`, the lock is probably not binding — check that the projection's lock lower bounds
are actually being applied, because a silently-inert constraint looks exactly like this.

### 1.3 `D_max` sensitivity

Sweep `D_max ∈ {0.05, 0.10, 0.15, 0.20, 0.25}` with `N` fixed. Report realized max drawdown, return, cash
weight, and intervention rate.

**This is the single most important robustness check in the project:**

> Realized max drawdown must be **monotone non-decreasing in `D_max`**.

A tighter ceiling must not produce a deeper realized drawdown. If it does, the safety layer is not doing what
it claims and every risk number in the report is void. Stage 9 asserts this and fails the stage on violation
(allowing a small configured tolerance for the stochasticity of a single seed — and if the violation exceeds
tolerance, run more seeds before concluding it is a bug, then treat it as one).

### 1.4 Combined grid

`N × D_max`, default 3×3 per D5 (`N ∈ {15, 30, 60}` — the operating range, so the default grid measures
the system where it is meant to run — with `D_max ∈ {0.05, 0.15, 0.25}`), full 9×5 including the
out-of-distribution `N` via
`--grid full`. This is the direct test of parameter conditioning: a single policy must behave sensibly
across the whole grid, not just near the modal training parameters.

Report as a heatmap of realized drawdown and of return, with the `D_max` ceiling overlaid so violations are
visible at a glance.

---

## 2. Bootstrap robustness (Stage 10)

**Stationary bootstrap (Politis–Romano) or moving-block — never IID resampling.** Independent sampling
destroys volatility clustering and serial dependence, which are the properties the entire risk layer is built
to handle; an IID bootstrap would produce comfortable, meaningless confidence intervals.

```yaml
bootstrap:
  method: stationary          # or moving_block
  mean_block_length: 10       # sessions; geometric for stationary
  replicates: 1000            # D5 default
  block_length_sensitivity: [5, 10, 21, 63]
```

Resample **blocks of the joint cross-section** — the same block indices across all assets simultaneously — so
that cross-asset correlation structure is preserved. Resampling each asset independently would manufacture
diversification that does not exist, and would flatter every drawdown number.

Output: confidence bands for every metric in [evaluation.md](evaluation.md) §3, plus the block-length sensitivity. If conclusions change
materially with block length, say so; do not report the most favorable choice.

Interpretive caution to carry into the report: a bootstrap over historical returns quantifies sampling
uncertainty *within the observed regime distribution*. It is not a statement about regimes that have not
occurred.

---

## 3. Adversarial historical scenarios (Stage 11)

Construct unfavorable combinations **within the support of the historical empirical distribution** — not
synthetic shocks, not fitted-distribution tail draws.

| Scenario | Construction |
|---|---|
| Equity shock + credit widening | Concatenate the worst equity block with a contemporaneous or adjacent worst credit-spread block |
| Duration loss | Worst observed TLT/IEF drawdown blocks, applied alongside a non-rallying equity block |
| Correlation spike | Blocks selected for maximum realized cross-asset correlation — every diversifier moving together |
| Diversification breakdown | Blocks where the historically negative stock/bond correlation flipped positive |

Construction rules:

- Every path is assembled from **real historical blocks**, so every return is one that actually happened.
- Blocks are drawn from dates **within the training/visible range** for a given fold, or the report must say
  otherwise explicitly.
- The joint cross-section is preserved within a block; only the *ordering and combination* of blocks is
  adversarial.

**The disclaimer is part of the deliverable:**

> This is a stronger historical robustness test. It is **not** a forward-looking worst-case guarantee, and the
> first version must not pretend otherwise.

Every report containing adversarial results states this. A scenario built by picking the worst historical
blocks is, by construction, not a probability statement about the future.

At v1 these scenarios are **evaluation-only** and do not feed the decision-time risk envelope
([risk-envelope.md](risk-envelope.md) §4.4).

---

## 4. What a robustness result is allowed to claim

Worth writing down before generating results that will be tempting to over-read:

| Legitimate | Not legitimate |
|---|---|
| "Under `D_max = 0.10`, realized drawdown stayed under 10% in 8 of 9 test years; the exception was market-forced with locked exposure of X%" | "The system guarantees a 10% maximum drawdown" |
| "Across 1000 stationary-bootstrap replicates, annualized return had a 90% band of [a, b]" | "Expected return is the bootstrap mean" |
| "In the constructed correlation-spike scenario the policy lost X%" | "Worst case is X%" |
| "Performance degrades gracefully as `N` increases from 0 to 180, with `N > 60` out-of-distribution; the trend is decreasing but not strictly monotone" | "The system is robust to any holding constraint" |

The core thesis applies here: the hard part of this problem was never PPO versus SAC. It is whether the
constraint machinery is correct and whether its limitations are stated honestly.
