# The Resettable Holding Lock

Implemented in `src/portfolio/lock_manager.py`. This is a core module and is tested **independently** of the
environment.

---

## 0. Lock scope: per-ETF (D13)

**Buying ETF `i` resets only `unlock_date_i`.** Every other holding keeps its own independent clock.

This is worth stating explicitly because the alternative reading is natural and materially different: a
**portfolio-wide** lock, where any purchase relocks the entire book for `N` days. That variant is far more
restrictive — a single small buy would freeze every position — and it was considered and rejected.

```
Per-ETF (canonical)                    Portfolio-wide (rejected)
Day 1:  buy SPY, hold TLT              Day 1:  buy SPY, hold TLT
  unlock[SPY] = Day 1 + N                unlock[SPY] = Day 1 + N
  unlock[TLT] unchanged                  unlock[TLT] = Day 1 + N   <- also relocked
Day 10: add QQQ                        Day 10: add QQQ
  unlock[QQQ] = Day 10 + N               unlock[SPY] = Day 10 + N  <- relocked
  unlock[SPY], unlock[TLT] unchanged     unlock[TLT] = Day 10 + N  <- relocked
                                         unlock[QQQ] = Day 10 + N
```

A `lock.scope: {per_etf, portfolio}` config flag is retained because it is one branch in the reset function,
but `per_etf` is canonical and `portfolio` exists only as an ablation. Nothing else in this plan assumes the
portfolio-wide variant.

---

## 1. State

Per tradable ETF `i`:

```python
shares_i: float          # from the ledger
unlock_date_i: date|None # absolute date; None iff shares_i == 0
```

The environment stores the **absolute unlock date**, never a countdown — a countdown has to be decremented,
and anything that has to be decremented eventually gets decremented twice or not at all across a reset.

The *observation* exposes the derived countdown:

```python
remaining_days_i = max(0, (unlock_date_i - session_date).days)   # 0 if unlock_date_i is None
```

---

## 2. Calendar days vs sessions — the trap

`N` is in **calendar days**. The lock clock does not care about weekends, holidays, or trading halts.

```python
unlock_date_i = execution_date + timedelta(days=N)
```

But a *sale* can only occur on a session. So:

> **First legal sell date = the first XNYS session on or after `unlock_date_i`.**

```python
def is_sellable(unlock_date, session_date):
    return unlock_date is None or session_date >= unlock_date
```

Because `session_date` is by construction a session, comparing dates directly is correct — the "first session
on or after" resolution happens implicitly. Do **not** try to precompute a "first sellable session" by adding
`N` business days; that is a different and wrong rule.

Edge cases that must be tested:

- `N = 0` → `unlock_date = execution_date`, so the position is sellable the very next session (it cannot be
  sold on the execution date itself, because execution happens at that session's open and the next decision is
  at that session's close, for execution the session after).
- `unlock_date` falls on a weekend or holiday → the position becomes sellable on the next session, with no
  extra delay beyond the calendar.
- `unlock_date` falls on a half-day → a half-day is a normal session ([data-pipeline.md](data-pipeline.md)).

---

## 3. Transitions

`execution_date` is the session at whose **open** the trade executed.

### 3.1 Buy / add (`delta_shares_i > 0`)

```python
unlock_date_i = execution_date + timedelta(days=N)
```

The reset applies to the **entire** position in `i`, not just the newly bought portion. Shares are fungible;
there are no tax lots in this model.

```
Day 1:  buy SPY            -> unlock[SPY] = Day 1 + 30 = Day 31
Day 10: add to SPY         -> unlock[SPY] = Day 10 + 30 = Day 40
```

The shares bought on Day 1 are now locked until Day 40. This is the whole point of the constraint: adding to a
winner costs you the optionality to sell what you already hold in that name.

### 3.2 Reduce (`delta_shares_i < 0`, `shares_i` still > 0)

Permitted only if `is_sellable(unlock_date_i, execution_date)`.

`unlock_date_i` is **unchanged**. Selling does not extend, shorten, or reset the lock. A partial sale of an
unlocked position leaves the remainder unlocked.

### 3.3 Sell entirely (`shares_i` reaches 0)

```python
unlock_date_i = None
```

The slot is clean. A later purchase starts a fresh lock from that purchase's execution date.

### 3.4 Sell then rebuy on the same day

The genuinely subtle case, and the one fixed-fixture tests miss. The execution order is deterministic and
matches [portfolio-ledger.md](portfolio-ledger.md) §5:

1. Execute **all legal sells** first (illegal sells were already removed by the projection — see
   [feasibility-projection.md](feasibility-projection.md); if one reaches here it is a bug and raises).
2. Compute available cash.
3. Execute **buys**.
4. Set locks **after** the buys.

Therefore, if ETF `i` is fully sold and then rebought in the same execution:

```
unlock_date_i = execution_date + timedelta(days=N)
```

The intermediate `None` from step 3.3 is transient and never observed. Net-flat (sold and rebought to exactly
the same share count) still relocks, because a buy occurred. This is correct and intentional: it closes the
loophole of "sell to unlock, immediately rebuy, keep the lock clock frozen".

**Net position change is not what triggers the lock — an executed buy is.** The lock manager therefore
inspects the executed trade legs, not the start-to-end share delta.

### 3.5 Carve-out: dividend reinvestment does not relock

The share accretion from reinvesting a distribution ([portfolio-ledger.md](portfolio-ledger.md) §3) increases
`shares_i` but is **not** a discretionary buy and does **not** reset `unlock_date_i`. Any other reading would
let a corporate action silently freeze the portfolio.

This carve-out is implemented by routing reinvestment through a separate ledger method that the lock manager
does not observe, and it is explicitly tested.

---

## 4. Interface

```python
class LockManager:
    def is_locked(self, ticker: str, session: date) -> bool: ...
    def locked_mask(self, session: date) -> np.ndarray:            # bool per tradable
    def remaining_days(self, session: date) -> np.ndarray:         # for the observation
    def lower_bounds(self, ledger, session) -> np.ndarray:         # shares_i for locked i, else 0
    def apply_execution(self, legs: list[TradeLeg], session: date, n_days: int) -> None: ...
    def to_dict(self) -> dict: ...                                 # D4: lossless round-trip
    @classmethod
    def from_dict(cls, d: dict) -> "LockManager": ...
```

`lower_bounds` is what the projection consumes: a locked ETF's target share count may not fall below its
current share count ([feasibility-projection.md](feasibility-projection.md) §4).

---

## 5. Invariants

Asserted on every step, and property-tested with `hypothesis` over random buy/sell/hold sequences
(see [testing.md](testing.md)):

| # | Invariant |
|---|---|
| L1 | A sale never executes when `session_date < unlock_date_i` |
| L2 | `unlock_date_i is None` if and only if `shares_i == 0` |
| L3 | Any executed buy leg in `i` sets `unlock_date_i == execution_date + N` exactly |
| L4 | A sell-only or hold execution leaves `unlock_date_i` unchanged |
| L5 | Dividend reinvestment never changes any `unlock_date` |
| L6 | `unlock_date_i` is never in the past while `shares_i > 0` *and* the position is reported locked |
| L7 | `N = 0` behaves as a no-lock portfolio: the locked mask is all-false at every decision point |
| L8 | Round-trip: `from_dict(to_dict(lm))` reproduces every mask and bound bit-identically |

A violation of L1 anywhere in the system — including in a baseline run — is a hard stop, not a warning. Zero
lock violations is an acceptance criterion ([evaluation.md](evaluation.md) §5).
