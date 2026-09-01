# Testing

Per D8, **a stage is not done until the tests named for it pass.** Tests live in `tests/`,
mirroring `src/`.

Stack: `pytest`, `hypothesis` for property-based tests, `pytest-benchmark` for the throughput checks that
Stages 6 and 9 depend on.

---

## 1. The six invariants

These are the backbone. They are asserted both as tests and — where cheap — as runtime assertions inside the
simulator, so a violation surfaces at the step that caused it rather than in an aggregate at the end.

### I1 — Portfolio invariants

At every step: `cash >= 0`, `shares_i >= 0`, `NAV > 0`.

`tests/portfolio/test_ledger_invariants.py`. Runtime-asserted in `Ledger` (behind `-O`-disableable asserts, on
by default; leave them on — the step cost is negligible next to the risk envelope).

### I2 — Weight invariant

`sum_i w_i + w_cash == 1` to floating tolerance, before and after projection, and after execution.

`tests/portfolio/test_weights.py`.

### I3 — Lock invariant

For any `session_date < unlock_date_i`: `shares_i^after >= shares_i^before`. A locked position may grow, never
shrink.

`tests/portfolio/test_lock_invariants.py`, **property-based** over `hypothesis`-generated buy/sell/hold
sequences with random `N`, random session gaps, and random inception dates. This is the test that finds the
sell-then-rebuy same-day case; fixed fixtures will not.

### I4 — Reset invariant

If `shares_i^after > shares_i^before` (an executed buy), then `unlock_date_i == execution_date + timedelta(N)`
exactly.

Same file, same property generator. Includes the carve-out check: dividend reinvestment increases shares and
must **not** trigger a reset ([lock-state-machine.md](lock-state-machine.md) §3.5).

### I5 — Inception invariant

Before a ticker's `first_session`: position is 0, target is 0, and the action is unavailable (masked in the
logits and zeroed by the projection).

`tests/data/test_inception.py`. Includes the prohibition on synthesizing pre-inception history: assert every
feature and price is NaN or absent, never back-filled, before inception.

### I6 — Lookahead invariant

```python
features(full_data, D) == features(truncate(full_data, D), D)
```

for randomly chosen dates `D`, plus the stronger **future-mutation** test: randomly perturb data strictly
after `D`, recompute, and assert nothing at or before `D` changed by even a float.

`tests/features/test_lookahead.py`. Applies to per-ETF features, cross-sectional ranks, macro series (including
publication lag), and the risk envelope's stress estimators — the envelope is a lookahead surface too, and it
is the one most likely to be forgotten.

---

## 2. Additional tests this plan requires

Beyond the six, because the design has surfaces they do not cover:

| # | Test | Why | File |
|---|---|---|---|
| T1 | **Total-return reconstruction.** Buy-and-hold 100% of each ticker from inception; NAV must match the `close_adj` total-return series to float tolerance | Proves the dividend-reinvestment ledger. If TLT fails, everything downstream is wrong | `tests/portfolio/test_total_return.py` |
| T2 | Ledger + lock manager dict round-trip is lossless | D4 requirement, cheap now, expensive to retrofit | `tests/portfolio/test_serialization.py` |
| T3 | Projection idempotence: feasible in → identical out | Catches projections that perturb valid actions | `tests/constraints/test_projection.py` |
| T4 | Analytic vs CVXPY oracle: feasible, objective within tolerance, 10k random instances | D3's correctness guarantee | same |
| T5 | Risk **convexity** along `w(alpha)`, plus the pinned non-monotone counter-example and the proof that the projection never returns an infeasible action | The `alpha` bisection needs a convex sublevel set, not monotonicity — see [risk-envelope.md](risk-envelope.md) §5 | `tests/constraints/test_risk_envelope.py` |
| T6 | Fallback always returns a feasible point or flags market-forced; never raises | Terminating the episode is forbidden | same |
| T7 | No silent repair: diagnostics non-empty whenever `a_proj != a_raw` | Explicitly prohibited | `tests/constraints/test_projection.py` |
| T8 | Reset sampler produces only reachable states, and `D_t <= D_max` in normal mode | Reachable-state requirement | `tests/env/test_reset_sampler.py` |
| T9 | Determinism: same seed → byte-identical trajectory, across worker counts | [architecture.md](architecture.md) §5 | `tests/env/test_determinism.py` |
| T10 | Scaler fold isolation: a fold's scaler range never overlaps its evaluation window | Full-sample-scaling prohibition | `tests/features/test_scalers.py` |
| T11 | Walk-forward config: `train_end < val < test`, no overlap, no gaps | Walk-forward integrity | `tests/evaluation/test_walk_forward.py` |
| T12 | Model selection is lexicographic and never selects a risk-violating model at Level 1 | Selection rule | `tests/evaluation/test_selection.py` |
| T13 | Preventable-violation replay detector fires on a deliberately injected projection bug | A detector that has never fired is not known to work | `tests/evaluation/test_violation_taxonomy.py` |
| T14 | Observation matches the feature manifest; every mandatory Markov field present | Catches silent index shifts | `tests/env/test_observation.py` |
| T15 | Execution price within `[low_raw, high_raw]` on every fill | Data/timing bug detector | `tests/portfolio/test_execution.py` |
| T16 | Stage harness: `--dry-run` has no side effects; staleness detection fires; manifest written on failure | Everything depends on it | `tests/test_stage_harness.py` |

T13 deserves a note: an invariant checker that has never been observed to fail is untested infrastructure. The
test deliberately injects a bug into the projection, runs the detector, and asserts it catches it.

---

## 3. Which tests gate which stage (D8)

| Stage | Must pass |
|---|---|
| 0 | T16 |
| 1 | fetch pins (symbol quirks, row counts) |
| 2 | I5, T15 data-side, quality gate at zero violations |
| 3 | I6, T10, feature manifest completeness |
| **4** | **I1, I2, I3, I4, I5, I6, T1, T2, T3, T4, T5, T6, T7, T15** ← the gate |
| 5 | Baseline runs at zero lock/feasibility violations; `cash` has exactly zero drawdown and turnover |
| 6 | T8, T9, T14, plus zero invariant violations across 500 random episodes |
| 7 | Zero lock/feasibility violations across the full training run |
| 8 | T11, T12, T13 |
| 9–12 | `D_max` monotonicity assertion; acceptance table completeness |

Stage 4's list is long on purpose. It is the gate the whole project rests on.

---

## 4. Test data

- **Synthetic fixtures** for unit tests: small deterministic price paths where the correct answer is computable
  by hand. Fast, and they make failures diagnosable.
- **Real slices** for integration tests: a handful of tickers over the 2008 and 2020 windows, checked into
  `tests/fixtures/` as small parquet files so the suite runs offline and deterministically.
- **Full history** only in the stage acceptance runs, never in the unit suite.

The unit suite must run in under 60 seconds. A slow test suite stops being run, and this project's correctness
argument depends entirely on it being run.

---

## 5. Runtime assertions vs tests

Both, deliberately:

- **Tests** prove the invariants hold on the cases exercised.
- **Runtime assertions** in `Ledger`, `LockManager`, and the projection catch the cases nobody thought to test,
  at the step that caused them.

Runtime assertions stay on during training. If they measurably cost throughput (measure it in Stage 6 before
assuming), gate the expensive ones behind `config.debug.strict_assertions` — but the cheap ones (I1, I2, I3)
stay on unconditionally. A training run that silently violated the lock is a run that has to be thrown away,
which costs far more than the assertions ever will.
