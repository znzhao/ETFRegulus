# Portfolio Ledger, Execution and Valuation

The execution timeline and portfolio valuation. Implemented in
`src/portfolio/{ledger,execution,valuation}.py` and exercised by [Stage 4](stages.md).

Section 3 resolves the central design question here: *how do you keep a share/cash ledger at raw quoted
prices while producing a total-return NAV?*

---

## 1. Ledger state

```python
@dataclass
class Ledger:
    cash: float                     # USD, >= 0 always
    shares: dict[str, float]        # fractional, >= 0 always
    as_of: date                     # the session this state is valid at close of
```

Rules:

- **Fractional shares.** No rounding to whole shares anywhere. Rounding introduces a residual that silently
  breaks the weight invariant and is a well-known source of phantom alpha.
- **Long only.** `shares[t] >= 0` and `cash >= 0` at every point, enforced by assertion, not by convention.
- **No leverage, no margin.** `sum(market_value) <= NAV` by construction — every buy is funded from cash.
- Cash earns nothing at v1. (If a cash rate is added later it becomes a config field and an ablation; it is
  *not* silently on.)
- The ledger serializes to and from a plain dict losslessly. Required by D4 and tested.

---

## 2. The timeline

The single most common backtest bug is acting on information you would not have had. The timeline is
therefore fixed and mechanical:

```
Session t, CLOSE
    observe: prices through t's close, features through t, portfolio state, locks, NAV, peak, drawdown
    decide:  raw target weights a_raw
    project: a_proj = project(a_raw, state)      <- deterministic, uses only t-close information
    (no trading happens on session t)

Session t+1, OPEN
    execute: trade at open_raw[t+1] toward a_proj
    update:  shares, cash, lock state (see lock-state-machine.md)

Session t+1, CLOSE
    value:   NAV_{t+1} using close of t+1
    update:  peak, drawdown
    reward:  r_t = log(NAV_{t+1} / NAV_t)
```

Consequences that must hold in code:

- No feature computed from session `t+1` may influence the decision at `t`.
- The execution price is `open_raw[t+1]`, never `close[t]` and never `close[t+1]`.
- If `t+1` is not a session, there is no execution — the next session is used, and the lock clock keeps
  running in **calendar days** regardless (see [lock-state-machine.md](lock-state-machine.md)).
- Gap risk between `t` close and `t+1` open is real and is *not* smoothed away. The risk envelope is evaluated
  at `t` close and must account for it ([risk-envelope.md](risk-envelope.md)).

---

## 3. Total-return NAV from a raw-price ledger

**The problem.** Execution must happen at `close_raw`/`open_raw` (the actual quoted price you can transact
at). But `close_raw` excludes dividends, so an NAV computed purely from `close_raw` systematically
understates every income-paying ETF — TLT, LQD, HYG, XLU, TIP. Over 20 years this is not a rounding error;
[data-pipeline.md](data-pipeline.md) measured TLT over six months of 2024 at raw 98.31 → 94.18 versus
adjusted 88.05 → 84.62. An agent trained on raw-close NAV would rationally never hold a bond ETF, purely as
a data artifact.

Conversely, you cannot simply value the ledger at `close_adj`: `close_adj` is a *back-adjusted* series that is
rewritten every time a dividend is paid, so a share count multiplied by `close_adj` is not a dollar amount
that means anything, and it changes retroactively.

**The solution: explicit dividend reinvestment in the ledger.**

Derive a per-session, per-ticker dividend-per-share implied by the two series, then credit it as cash and
immediately reinvest it into the same ticker at the raw price. The ledger stays entirely in raw quoted prices
— which is what makes it real — and total return arrives through share accretion, which is what actually
happens with a reinvested distribution.

```python
# per ticker, per session t (derived once in Stage 2 or Stage 4 setup)
tr_t  = close_adj[t] / close_adj[t-1] - 1          # total return, the ground truth
pr_t  = close_raw[t] / close_raw[t-1] - 1          # price return
div_per_share[t] = (tr_t - pr_t) * close_raw[t-1]  # implied cash distribution per share
```

Then at each session close, for each held ticker:

```python
gross_div = shares[k] * div_per_share[t]
if gross_div > 0:
    shares[k] += gross_div / close_raw[t]          # reinvest at the raw close
```

Notes and guards:

- `div_per_share` is expected to be `~0` on the overwhelming majority of sessions and positive on ex-dividend
  dates. Small floating-point noise is clipped at a configured epsilon.
- A **negative** implied distribution is not physically meaningful for these ETFs and indicates an adjustment
  artifact or a split-handling problem. It fails the Stage 2 quality gate rather than being silently clipped.
- Splits are already reflected in both series consistently by yfinance; the derivation above is
  split-neutral because it is a ratio of consecutive values within each series.
- **Dividends do not touch the lock.** A reinvestment increases a share count but is not a discretionary buy.
  See [lock-state-machine.md](lock-state-machine.md) §5 — this is an explicit carve-out and is tested.

**The test that proves it** (in [testing.md](testing.md)): buy and hold 100% of one ticker with no rebalancing
from inception to today. The resulting NAV series must match the `close_adj` total-return series to within
floating-point tolerance, for *every* tradable ticker — including the high-yield ones. If TLT does not match,
the ledger is wrong.

---

## 4. Valuation, peak, drawdown

At each session close:

```python
nav      = cash + sum(shares[k] * close_raw[t, k] for k in shares)
peak     = max(peak, nav)                 # running maximum, monotone non-decreasing
drawdown = 1.0 - nav / peak               # in [0, 1)
weights  = {k: shares[k] * close_raw[t, k] / nav for k in shares}
w_cash   = cash / nav
```

- `peak` is part of the MDP state and must be carried through reset — a fresh episode starting from a
  reachable state inherits a plausible peak, not `nav`. See [env-mdp.md](env-mdp.md).
- `drawdown` uses the **total-return NAV** from section 3, so it is comparable to the drawdowns quoted for the ETFs
  themselves.
- `sum(weights) + w_cash == 1.0` to floating tolerance, always. This is the weight invariant.

---

## 5. Execution

```python
def execute(ledger, target_weights, open_raw_next, available_mask, cost_bps=0.0) -> Ledger:
```

Order of operations at the `t+1` open:

1. Compute `nav_at_open = cash + sum(shares * open_raw_next)`.
2. Target dollar position per ticker: `nav_at_open * target_weights[k]`.
3. Target shares: `target_dollars / open_raw_next[k]`. Unavailable tickers get 0.
4. **Sells first, then buys.** Buys are funded strictly from cash on hand after sells settle — this is what
   makes the no-margin constraint structural rather than a check after the fact.
5. Apply `cost_bps` to the traded notional on each side. **`cost_bps = 0.0` (D10)**, so this is a no-op — the
   code path exists and is unit-tested with a non-zero value, but the system runs frictionless.
6. Assert post-conditions: `cash >= -eps`, all `shares >= -eps`, realized weights match targets to tolerance
   (residual is attributable only to cost and to the availability mask).
7. Hand the executed trade vector to the lock manager, which decides what relocks.

**Sanity check, always on:** the execution price must satisfy `low_raw[t+1] <= open_raw[t+1] <= high_raw[t+1]`.
A fill outside the day's range is a data bug and raises.

---

## 6. What lives where

| Concern | Module |
|---|---|
| State container, invariant assertions, dict round-trip | `src/portfolio/ledger.py` |
| Next-open execution, sells-then-buys, cost application | `src/portfolio/execution.py` |
| NAV, dividend reinvestment, peak, drawdown, weights | `src/portfolio/valuation.py` |
| Lock bookkeeping | `src/portfolio/lock_manager.py` — see [lock-state-machine.md](lock-state-machine.md) |
