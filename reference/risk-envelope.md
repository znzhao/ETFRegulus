# Drawdown Engine and Risk Safety Envelope

The drawdown constraint engine, current-state feasibility, the action-level risk budget, and the stress
estimators behind it. Implemented in `src/constraints/{risk_envelope,safety_layer}.py`.

---

## 1. What the constraint actually means

```
D_t = 1 - V_t / P_t        must satisfy        D_t <= D_max
```

But `D_{t+1}` depends on market moves nobody can know at decision time. So the honest statement — and the one
this implementation commits to — is:

> The maximum-drawdown constraint is an **action-level safety constraint**, not a guarantee about the realized
> path.

Every report must say this. Claiming a hard guarantee on realized drawdown would be false: historical
robustness evidence must never be dressed up as a forward-looking bound.

What *is* enforced: at every decision, the agent may not take an action whose stressed outcome would breach
`D_max`, under a stress estimate built from data visible at that moment.

---

## 2. Layer one — feasibility of the current state

If `D_t > D_max` on arrival, the drawdown has already happened. The agent cannot undo it.

```python
if drawdown > d_max:
    state.capital_preservation = True
```

In this mode:

- **No net increase in risky exposure.** Every ETF's target share count is capped at its current share count.
- Legal reductions are allowed.
- Moving to cash is always allowed.
- **Locked ETFs still cannot be sold.** The lock outranks capital preservation — this is the case where the
  two constraints genuinely conflict, and the lock wins. That means the drawdown can keep deepening while the
  agent is powerless, which is exactly the *market-forced* violation category in [evaluation.md](evaluation.md).

Prior actions are **not** retroactively invalidated, and the episode does not terminate. The policy still
chooses an action; the set it chooses from is just smaller.

---

## 3. Layer two — the action-level risk budget

State feasibility alone is not enough: at `D_t ≈ D_max` with headroom near zero, the agent could still pile
into high-volatility ETFs and be "legal" right up until the loss lands.

Define headroom:

```
H_t = D_max - D_t
```

and define the risk budget by requiring that a stressed loss keeps the projected drawdown legal:

```
D_projected(w) = 1 - V_t (1 - L_stress(w)) / P_t  <=  D_max
```

which rearranges to the operational form:

```
L_stress(w)  <=  1 - (1 - D_max) * P_t / V_t   =:  RiskBudget_t
```

Notes:

- `RiskBudget_t` shrinks as drawdown deepens and goes to zero (or negative) at the ceiling. Negative budget →
  only `w_safe` is admissible → capital preservation. This is consistent with §2 rather than a separate rule.
- `L_stress` is a **loss fraction over the decision horizon**, positive for losses.
- The horizon is a config parameter (`risk.horizon_days`, default 5). It should exceed one day: the constraint
  must survive the overnight gap plus a few sessions, because the lock may prevent reacting.
- **Binding on the design:** this must not be reduced to `r' = r - lambda * DD`. The budget is a
  constraint consumed by the projection ([feasibility-projection.md](feasibility-projection.md) §4.2), never a
  reward term. D11 says the same thing from the reward side.

---

## 4. Stress estimators

No single estimator. `L_stress(w)` is an aggregate over several, combined by a configurable rule
(`risk.aggregation`, default `max` — the most conservative, and the right default for a hard constraint).

All estimators use **only data visible at session `t`**. This is a lookahead-sensitive surface and is covered
by a dedicated test.

### 4.1 Historical rolling stress

Portfolio return series implied by weights `w` over trailing windows, at 5/10/21 days:

```python
port_ret = hist_returns @ w                     # visible history only
L = -np.quantile(rolling_sum(port_ret, h), q)   # q = risk.quantile, default 0.01
```

Cheap, and the default component. Weakness: it only sees what the trailing window contains, so it
under-reacts entering a regime change.

### 4.2 Block bootstrap

Stationary bootstrap (Politis–Romano) or moving-block, over historical return blocks.

> **Never IID resampling.** It destroys volatility clustering and serial dependence, and would produce a
> comfortable, wrong risk number.

Mean block length is config (`risk.block_length`, default 10 sessions, geometric for the stationary variant).
This is the expensive estimator; at training time it runs on a **cached, precomputed** basis (see §6 below),
not live per step.

### 4.3 Historical crisis windows

An explicit scenario library, applied as a fixed set of return paths:

| Window | Period |
|---|---|
| GFC | 2007-10 → 2009-03 |
| COVID crash | 2020-02-19 → 2020-03-23 |
| 2022 rate shock | 2022-01 → 2022-10 |
| Taper tantrum | 2013-05 → 2013-09 |
| Volmageddon | 2018-02 |
| 2018 Q4 selloff | 2018-10 → 2018-12 |

Exact dates live in `config/constraints.yaml` so they are reviewable and versioned.

**Critical lookahead rule:** at decision time `t`, only crisis windows that ended **before** `t` may be used.
A 2005 decision cannot be stress-tested against 2008. The scenario library is therefore filtered by date on
every call, and this is tested. (For *evaluation* in Stage 9, the full library is used deliberately — that is
a stress test, not a decision input, and the two code paths are kept distinct.)

### 4.4 Adversarial historical scenarios

Unfavorable-but-in-support combinations: equity shock + credit widening, duration loss, correlation spike,
diversification breakdown. Constructed as described in [robustness.md](robustness.md).

At v1 these are **evaluation-only** (Stage 11), not part of the live risk budget — they are expensive and
their construction involves choices that would be hard to defend as a decision-time input. Enabling them in
the envelope is a config flag for later.

---

## 5. Interface

```python
class RiskEstimator(Protocol):
    def stress_loss(self, w: np.ndarray, market_state: MarketState) -> float: ...

class RiskEnvelope:
    estimators: list[RiskEstimator]
    aggregation: Literal["max", "mean", "quantile"]

    def stress_loss(self, w, market_state) -> float: ...
    def budget(self, nav: float, peak: float, d_max: float) -> float: ...
    def is_feasible(self, w, market_state, nav, peak, d_max) -> bool: ...
```

**Required property, asserted in tests:** `stress_loss` must be monotone non-increasing as weight shifts from
risky assets to cash. The `alpha` bisection in the projection depends on it
([feasibility-projection.md](feasibility-projection.md) §4.2, test P5). An estimator that violates monotonicity
cannot be used with the analytic backend, and the test says so explicitly rather than failing mysteriously.

---

## 6. Making this fast enough to train against

Called once per step across millions of steps, so:

- **Precompute per session, not per call.** The trailing return matrix, its covariance, and the bootstrap
  return-path sample are all functions of the session and the visible history — not of `w`. Compute them once
  per session in the environment and reuse across the `alpha` bisection.
- Given a cached set of stress return paths `R` (paths x assets), `L_stress(w)` is a matrix-vector product and
  a quantile: microseconds.
- The bootstrap paths are drawn once per session with a session-derived seed, so the estimator is deterministic
  given the seed — required by the determinism rule in [architecture.md](architecture.md).
- Benchmark this in Stage 6 alongside env throughput. If the risk envelope dominates step time, the training
  budget under D5 is the thing that suffers.

---

## 7. Calibration, and how to read it

The envelope has real parameters (`quantile`, `horizon_days`, `block_length`, `aggregation`) and they trade
off directly:

- Too conservative → the agent sits in cash, safety intervention rate near 100%, no learning signal about
  anything else.
- Too loose → realized drawdowns exceed `D_max` routinely and the constraint is decorative.

**Calibrate before Stage 7, using Stage 5.** Run the `equal_weight` and `spy_tlt_60_40` baselines through
the envelope across the `D_max` grid and record, per setting: realized max drawdown, intervention rate, and
average cash weight. Pick parameters where a tighter `D_max` measurably reduces realized drawdown while the
intervention rate stays under a configured ceiling (default 50%).

This calibration is cheap, uses no RL, and is the difference between a safety layer that works and one that is
either ornamental or paralyzing. Record the chosen values and the table that justified them in
[../STATUS.md](../STATUS.md).
