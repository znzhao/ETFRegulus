# Feasibility Projection

The action representation, the hard portfolio constraints, and the deterministic projection that enforces
them. Implemented in `src/constraints/`.

The risk-envelope half of the safety layer is in [risk-envelope.md](risk-envelope.md); this file covers the
hard portfolio constraints and the projection algorithm that enforces them.

---

## 1. The pipeline

```
policy
  -> a_raw               (K+1 logits -> softmax -> weights over CASH + K ETFs)
  -> availability mask        section 3.1
  -> long-only / no-leverage  section 3.2, 3.3
  -> lock lower bounds        section 3.4
  -> risk envelope        risk-envelope.md
  -> a_proj               executable target weights
  -> next-open execution      portfolio-ledger.md section 5
```

Every layer is deterministic and uses only information available at session `t`'s close. The whole pipeline is
therefore part of the environment, which is exactly what makes D9 (store `a_raw`, execute `a_proj`) correct.

**Nothing is silently fixed.** Letting the projection quietly repair invalid actions is explicitly prohibited
(see the plan's prohibitions table).
Every layer records what it changed into a `ProjectionDiagnostics` object, which is written to the trajectory
and logged to TensorBoard.

---

## 2. Action representation

The policy outputs `a_raw ∈ R^(K+1)`, passed through a softmax:

```
a = softmax(logits)        a_i >= 0,  sum(a) = 1
index 0 = CASH, indices 1..K = the tradable ETFs
```

This gives a valid simplex point but says nothing about the lock, availability, or risk. The projection is
what turns it into something executable.

Design notes:

- Softmax rather than a raw-weight head, so long-only and no-leverage hold *before* projection and the
  projection only has to handle the interesting constraints.
- Unavailable assets are masked in the **logits** (set to `-inf`) as well as clipped after, so the policy never
  spends probability mass on an asset that does not exist yet. This materially reduces projection distance
  early in the sample when only a fraction of the universe has launched.
- `a_raw` stored for PPO is the **pre-softmax sample from the policy distribution** in SB3's own
  parameterization — the softmax is part of the action transformation, not part of the projection. What
  matters for D9 is that the stored action is the one whose `log_prob` PPO computes.

---

## 3. Hard constraints

### 3.1 Availability

Derived solely from `data/curated/inception.parquet`:

```python
available[i] = session >= first_session[i]
```

If unavailable: `w_i = 0`, masked out of the action set entirely. An unavailable asset can never be bought,
and by construction is never held, so it never appears in the lock lower bounds.

**Inception is a hard constraint and the most common source of lookahead in ETF backtests.** There is a
dedicated inception invariant test.

### 3.2 Long-only

`shares_i >= 0`, equivalently `w_i >= 0`. Guaranteed by the softmax and re-asserted after every projection
step.

### 3.3 No leverage / no margin

`sum_i MV_i <= NAV`, equivalently `sum_i w_i <= 1` with the remainder in cash. Guaranteed by the simplex
constraint and made structural at execution time by settling sells before funding buys.

### 3.4 Lock lower bounds

If `i` is locked at the decision session:

```
shares_i^target >= shares_i^current
```

The position may be increased but never decreased. Expressed in weight space, the bound is:

```
w_i >= w_i^lower,   w_i^lower = shares_i^current * price_i / NAV_projected
```

**A subtlety worth stating plainly.** The bound is a *share* bound, but the projection works in *weight*
space, and the mapping between them depends on the NAV at execution, which is not known at decision time
(it depends on the overnight gap). The plan handles this by:

1. projecting in weight space using `w_i^lower` computed at the **decision close** NAV;
2. re-deriving the binding share floors at execution and enforcing `shares_i^target >= shares_i^current`
   there as the authoritative check.

So the weight-space bound is the optimizer's guide; the share-space bound is the law. If an overnight gap
makes the weight target infeasible in share terms, execution honors the share floor and the residual goes to
cash. The gap between intended and executed weights is recorded in the diagnostics rather than hidden.

**Consequence to internalize:** the sum of lock lower bounds can exceed what the policy wanted to allocate to
those names. Then the free weight budget is `1 - sum(w_lower)` and only that portion is distributed across
cash, unlocked ETFs, and *additional* buys of locked ETFs. If `sum(w_lower) > 1` after a gap — possible if
locked positions gapped up hard — everything else goes to zero and there is nothing to redistribute. That is
a legal state, not an error.

---

## 4. The projection problem

```
minimize    ||w - w_policy||_2^2
subject to  w_i >= 0
            sum_i w_i = 1                (index 0 is cash, unbounded above)
            w_i >= w_i^lower             for locked i
            w_i = 0                      for unavailable i
            Risk(w) <= RiskBudget_t      see risk-envelope.md
```

### 4.1 Interface

Shared by both backends, so they are interchangeable:

```python
class FeasibilityProjector(Protocol):
    def project(
        self,
        raw_action: np.ndarray,
        portfolio_state: PortfolioState,
        market_state: MarketState,
        constraints: Constraints,
    ) -> ProjectedAction: ...
```

```python
@dataclass
class ProjectedAction:
    weights: np.ndarray
    diagnostics: ProjectionDiagnostics
```

```python
@dataclass
class ProjectionDiagnostics:
    l1_distance: float              # ||a_proj - a_raw||_1
    l2_distance: float
    availability_clipped: int       # count of assets zeroed by inception
    lock_bound_active: int          # count of locked assets at their floor
    risk_binding: bool              # the risk envelope bound
    de_risk_alpha: float            # 1.0 = untouched, 0.0 = fully to safe asset
    capital_preservation: bool      # drawdown-breached mode
    infeasible_fallback: bool       # no feasible point found -> fallback engaged
```

### 4.2 Backend `analytic` (v1, D3)

Two steps, both closed-form and fast.

**Step 1 — simplex projection with lower bounds.** Substitute `v_i = w_i - w_i^lower` for locked assets and
`v_i = w_i` otherwise. Then

```
min ||v - (w_policy - w_lower)||^2   s.t.  v >= 0,  sum v = 1 - sum(w_lower)
```

is the standard Euclidean projection onto a scaled simplex, solvable exactly with a sort and a threshold
search in `O(K log K)`. Unavailable assets are removed from the problem before solving, not clipped after.
If `sum(w_lower) > 1`, the free budget is zero and `w = w_lower`, renormalized — the degenerate case above.

**Step 2 — risk de-risking scan.** If `Risk(w) > RiskBudget_t`, blend toward the safe portfolio:

```
w(alpha) = alpha * w  +  (1 - alpha) * w_safe
```

where `w_safe` holds every locked position at its floor and puts everything else in cash — the **minimum-
exposure** portfolio still reachable under the lock. Bisect on `alpha ∈ [0, 1]` for the largest feasible value,
to a configured tolerance (default 1e-3, ~10 risk evaluations).

What makes the bisection valid is that `Risk` is **convex** in `w`, so the feasible set along the segment is an
interval, and `w_safe` being feasible puts `alpha = 0` inside it. It is *not* validated by monotonicity: see
the correction in [risk-envelope.md](risk-envelope.md) §5 — `w_safe` minimizes exposure, not risk, so moving
toward it can genuinely increase risk when a hedge is available. Convexity is asserted in a test, and the
projector re-checks feasibility after the bisection regardless.

This is a feasible point close to the proposal, not the provably closest one under the risk constraint. That
approximation is the acknowledged cost of D3.

### 4.3 Backend `cvxpy` (later, and a test oracle now)

The same problem as a QP. Selected by `projection.backend: cvxpy`. Used from day one in tests: on small random
instances the analytic result must be feasible, and its objective within a configured tolerance of the QP
optimum. This catches analytic-projection bugs long before they can be mistaken for policy behavior.

---

## 5. Fallback when nothing is feasible

Do **not** terminate the episode. Terminating would teach the policy that entering a
risky state ends the game, which is precisely the wrong lesson.

Order of preference when the risk envelope rejects every candidate:

1. Find the feasible portfolio closest to the raw proposal (the `alpha` scan above).
2. If none exists, preserve existing locked positions at their floors.
3. Reduce all newly added risky exposure to zero.
4. Put the remainder in cash.

That is exactly `w_safe`, which is **always feasible by construction** — a portfolio of forced holdings plus
cash is the minimum-risk reachable point. So step 4 always terminates. If even `w_safe` violates the risk
budget, the state is a *market-forced* violation, not an agent choice: set `capital_preservation = True`,
record it, and continue. The taxonomy for reporting this is in [evaluation.md](evaluation.md) §30.

`infeasible_fallback` is flagged in the diagnostics and counted in the trajectory. A high fallback rate is a
signal that the risk budget is mis-calibrated, not that the agent is misbehaving — read it that way in
Stage 7.

---

## 6. Tests

| # | Test |
|---|---|
| P1 | Output is always on the simplex, non-negative, availability-respecting |
| P2 | Locked lower bounds are never violated, in share space, at execution |
| P3 | If `a_raw` is already feasible, `a_proj == a_raw` exactly (projection is idempotent on feasible points) |
| P4 | Analytic vs CVXPY: feasible, and objective within tolerance, over 10k random instances |
| P5 | `Risk` is **convex** along `w(alpha)` (the bisection's actual requirement); the non-monotone counter-example is pinned, and `var` is shown non-convex and therefore barred from the analytic backend |
| P6 | The fallback always returns a feasible point, or flags a market-forced violation — never raises |
| P7 | Diagnostics are non-empty whenever `a_proj != a_raw` (no silent repair) |
