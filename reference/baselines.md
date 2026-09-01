# Baselines

Built in [Stage 5](stages.md), **before any RL**, so the whole RL development period has something to beat.

**The non-negotiable rule:** every baseline runs through the *identical* simulator, lock manager, projection,
and risk envelope as the agent. A baseline is a weight sequence fed to
[Stage 4](stages.md); nothing else about it is special. Two reasons this matters:

1. The comparison is apples to apples — the agent is not being credited for constraints the baselines dodge.
2. A baseline that violates the lock or leverage is a **bug in the constraint layer**, and finding it here,
   against a strategy whose correct behavior is obvious, is far cheaper than finding it during training.

---

## 1. The set

Three primary benchmarks, specified by the user, plus three required for correctness and for the acceptance
criteria in [evaluation.md](evaluation.md) §5.

| # | Name | Role | Priority |
|---|---|---|---|
| B1 | `spy_buy_hold` | Buy and hold SPY | **Primary** |
| B2 | `momentum` | Simple cross-sectional momentum | **Primary** |
| B3 | `spy_tlt_60_40` | 60% SPY / 40% TLT | **Primary** |
| B4 | `cash` | 100% cash — the floor and a constraint-layer sanity check | Required |
| B5 | `equal_weight` | Equal weight across available tradables | Acceptance |
| B6 | `classical_optimizer` | Rolling mean-variance / risk parity | Acceptance |

**A note on B5 and B6.** The acceptance criteria ([evaluation.md](evaluation.md) §5) name cash, SPY, equal
weight, and a classical constrained baseline explicitly, so dropping B5/B6 would put the project out of
compliance with its own definition of an acceptable model. They are cheap — B5 is trivial, B6 reuses machinery the risk envelope already needs — and they stay.
B1-B3 are what the reports lead with.

Your three replace the vaguer placeholders from the first draft: `spy` becomes B1 (concretely
buy-and-hold, not rebalanced), and `static_risk_aware` becomes B3 (concretely 60/40, not a hand-wave).

---

## 2. B1 — `spy_buy_hold`

Buy SPY with 100% of NAV on the first session of the episode. Never trade again.

```yaml
spy_buy_hold:
  ticker: SPY
  rebalance: never
```

Notes:

- With a single asset there is no rebalancing to do: the weight stays 100% by construction as the price moves.
  Dividends reinvest into SPY via the ledger's total-return mechanism
  ([portfolio-ledger.md](portfolio-ledger.md) §3), which is what makes this a true total-return benchmark
  rather than a price-return one.
- **Lock interaction: none.** One buy, no sells, ever. It is legal at every `N` including 180.
- **This makes it a control.** B1's results must be *identical* across every value of `N`. If they are not,
  the lock manager is corrupting state it should not touch — a high-value invariant, and it is asserted as a
  test.
- **Risk envelope interaction:** the envelope can never force a sale (it only constrains *increases*, and
  capital-preservation mode caps positions at current levels). So B1 is also unaffected by `D_max`, and its
  realized drawdown is simply SPY's — which will exceed a tight `D_max` in 2008 and 2020. That is expected,
  correct, and a clean example of a **market-forced** violation ([evaluation.md](evaluation.md) §4) rather
  than a preventable one.

---

## 3. B2 — `momentum`

"Simple momentum" spans many designs, so this pins one down. Standard cross-sectional momentum, chosen
because it is the least arbitrary version and the one a reviewer will recognize.

```yaml
momentum:
  lookback_days: 252        # 12 months
  skip_days: 21             # skip the most recent month
  top_n: 5
  weighting: equal
  rebalance: monthly        # last session of each month
  absolute_filter: true     # only hold assets with positive lookback return
  cash_fallback: true       # unfilled slots go to cash
```

**Signal.** For each available tradable at rebalance date `t`:

```python
signal_i = close_adj[t - skip_days] / close_adj[t - skip_days - lookback_days] - 1
```

The 21-day skip is the standard "12-1" construction: it omits the most recent month to avoid the well-
documented short-term reversal effect, which otherwise contaminates the signal.

**Selection.** Rank available assets by `signal`, take the top 5, equal weight at 20% each. With
`absolute_filter`, an asset with a negative lookback return is excluded even if it ranks top-5 — so in a broad
bear market the strategy moves to cash rather than holding the least-bad loser. This makes B2 a genuinely
different *risk* profile from B1/B3, not just a different return stream, which is what makes it worth having.

**Constraint interactions — this is where B2 earns its place:**

- **The lock binds hard.** Monthly rebalancing means selling last month's losers, but every buy relocks that
  ETF for `N` calendar days. At `N = 30` roughly every position is still locked at the next rebalance; at
  `N = 90` the strategy is largely frozen and degenerates toward buy-and-hold of whatever it first bought.
  **B2 is therefore the most informative baseline for `N` sensitivity** ([robustness.md](robustness.md) §1.2),
  because its performance should degrade visibly and monotonically as `N` grows. If it does not, the lock is
  not actually binding, and that is a bug worth catching before it hides inside a trained policy.
- **Availability matters.** Only assets past inception are ranked ([features.md](features.md) §2). The
  selection universe grows from ~20 to ~25 names over the sample.
- **Warm-up:** needs `252 + 21 = 273` sessions of history. The 2003 buffer covers this for a 2004 start.

**Honest framing for the report:** this is a simple, standard momentum rule, not a tuned strategy. It exists as
a non-trivial active benchmark. It is not evidence about whether momentum works, and no lookback/top-N search
is performed — searching would make it a fitted strategy competing on unequal terms with a policy that had no
such search.

---

## 4. B3 — `spy_tlt_60_40`

```yaml
spy_tlt_60_40:
  weights: {SPY: 0.60, TLT: 0.40}
  rebalance: monthly
  drift_band: 0.05          # rebalance only if a weight drifts >5 percentage points
```

**Rebalancing.** Target weights drift as prices move; the strategy rebalances back to 60/40 monthly, but only
when a weight has drifted beyond the band. The band matters here more than in an unconstrained backtest:
**every rebalancing buy relocks the bought asset for `N` days**, so an unbanded monthly rebalance would keep
the whole book permanently locked for `N >= 30`. The band makes the strategy trade only when it meaningfully
needs to.

**Availability.** Both legs exist from the start of the study period — SPY from 1993, TLT from 2002-07 — so
B3 is well-defined across the entire 2004+ sample with no inception masking.

**Why this baseline is the sharpest test in the set.** 60/40 relies on the negative stock/bond correlation
that held for most of the sample and **broke in 2022**, when SPY and TLT fell together and TLT alone drew
down 31% ([data-pipeline.md](data-pipeline.md)). So:

- B3 should look excellent through 2021 and get badly hurt in 2022.
- If the trained agent has merely learned "bonds are the safe asset", it will fail in exactly the same year,
  in exactly the same way. Comparing the agent against B3 *specifically in the 2022 fold* is the cleanest
  available test of whether the multi-estimator risk envelope
  ([risk-envelope.md](risk-envelope.md) §4) bought anything real.

This is called out in [robustness.md](robustness.md) §1.1 as one of the two crisis windows that matter most
for this system.

---

## 5. B4-B6 — the required set

**B4 `cash`** — 100% cash, always. The floor. Also a constraint-layer sanity check with two exact
expectations: **zero drawdown and zero turnover**, always, at every `N` and `D_max`. Any deviation is a ledger
bug. It also proves the lock never *forces* risk-taking.

**B5 `equal_weight`** — equal weight across all available tradables, rebalanced monthly with the same drift
band as B3. Surprisingly hard to beat, and it moves with the expanding universe, which exercises the
availability mask continuously.

**B6 `classical_optimizer`** — rolling mean-variance or risk parity, **estimated on trailing visible data
only** and re-solved at each rebalance. This is the "classical constrained optimizer": the honest
representation of what a competent quant would do without RL. It reuses the covariance estimation the risk
envelope already computes ([risk-envelope.md](risk-envelope.md) §6), so it is cheap to add.

---

## 6. What Stage 5 produces

Per baseline, at each evaluated `(N, D_max)` cell:

```
artifacts/runs/<run_id>/baselines/<name>/trajectory.parquet   # standard schema
artifacts/runs/<run_id>/baseline_summary.json                 # standard metric set, all six
```

Every baseline's trajectory uses the identical schema as an RL run
([architecture.md](architecture.md) §4), so the metric code, plots, and the Stage 12 report have exactly one
input format.

---

## 7. Stage 5 acceptance

- [ ] All six produce complete trajectories over the full sample
- [ ] **Zero lock violations, zero feasibility violations**, all baselines, all `(N, D_max)` cells
- [ ] `cash`: exactly zero drawdown, exactly zero turnover
- [ ] `spy_buy_hold`: **identical results across all `N`** (the control — proves lock isolation)
- [ ] `momentum`: performance degrades monotonically as `N` increases (proves the lock actually binds)
- [ ] `spy_tlt_60_40`: 2022 drawdown clearly visible and severe (proves total-return bond pricing is right —
      if TLT's 2022 looks mild, the adjustment is wrong, cf. [portfolio-ledger.md](portfolio-ledger.md) §3)
- [ ] Standard metrics ([evaluation.md](evaluation.md) §3) computed for all six
- [ ] Initial-state reservoir populated from these runs ([env-mdp.md](env-mdp.md) §5)

The last three bullets are why baselines belong before RL: each one is a **test of the simulator** disguised as
a benchmark, and each has an obvious expected answer that a subtle accounting bug would break.
