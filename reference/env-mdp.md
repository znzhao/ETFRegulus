# The Environment: MDP Definition

State, Markov sufficiency, action, reward, parameter conditioning, reachable initial states, reset, and
episode length. Implemented in `src/env/`.

The environment is a **finite-horizon, parameter-conditioned, constrained portfolio MDP** — not an ordinary
unconstrained MDP, and the difference is load-bearing throughout.

---

## 1. State

```
s_t = ( m_t, p_t, l_t, R_t, D_t, N, D_max )
```

| Symbol | Content | Source |
|---|---|---|
| `m_t` | Market: per-ETF OHLCV features, cross-sectional, macro/regime | `data/features/` |
| `p_t` | Portfolio: cash, NAV, shares, market values, weights, availability mask | live ledger |
| `l_t` | Lock: absolute unlock dates, exposed as remaining calendar days | lock manager |
| `R_t` | Running peak NAV | live |
| `D_t` | Current drawdown `1 - V_t/R_t` | live |
| `N` | Holding-lock parameter, calendar days | episode parameter |
| `D_max` | Max drawdown parameter | episode parameter |

### The sufficiency requirement

> Nothing required for the Markov property may be missing from state.

The observation must be sufficient to reconstruct **transition feasibility** — i.e. the agent must be able to
tell, from the observation alone, which actions the projection will alter. That makes these mandatory, not
optional:

```
market_features
portfolio_weights, cash_weight
per_asset_position_indicator
per_asset_lock_remaining_days
per_asset_availability_mask
portfolio_nav_normalized, portfolio_peak_normalized
current_drawdown
drawdown_budget_remaining  B_t = D_max - D_t
N, D_max
time / regime features
```

If any of these is absent, the problem is not merely harder — it is non-Markov, and PPO's assumptions break.
A test asserts the observation vector contains every field the feature manifest declares mandatory.

### Observation layout

A single flat `float32` vector, assembled in a fixed documented order:

```
[ macro_block | cross_sectional_block | per_asset_block (K x F) | portfolio_block | param_block ]
```

Per-asset blocks are ordered by the canonical ticker order in `config/universe.yaml`, which never changes —
appending a ticker is a breaking change that invalidates trained policies, and the feature manifest hash in the
run manifest is what catches it.

Unavailable assets have their per-asset block **zeroed** and their availability mask bit set to 0. The mask is
what carries the information; the zeros are just filler.

---

## 2. Action and the projection

`a_t ∈ R^(K+1)` → softmax → weights over `[CASH, ETF_1..ETF_K]`. Then the deterministic pipeline in
[feasibility-projection.md](feasibility-projection.md) and [risk-envelope.md](risk-envelope.md) produces the
executable target.

**The projection lives inside `env.step`.** This is what makes D9 correct: SB3 stores the action the policy
sampled, and everything after is, by construction, part of the environment's dynamics.

---

## 3. Reward (D11)

```
r_t = log( V_{t+1}^close / V_t^close )
```

`V_t` is the close NAV at the decision state; the trade executes at the `t+1` open; `V_{t+1}` is the
mark-to-market at the `t+1` close after trading. Total-return NAV throughout
([portfolio-ledger.md](portfolio-ledger.md) §3).

**Not present, deliberately:** transaction costs (D10, `cost_bps = 0.0`), slippage, commission, turnover
penalty, volatility penalty, drawdown penalty. Risk is handled by the safety layer. Any alternative reward is
a logged ablation in [rl-training.md](rl-training.md), never the primary run.

---

## 4. Parameter conditioning

`N` and `D_max` are sampled per episode and enter the observation, so one policy serves the whole grid.

Canonical definition in [../config/constraints.yaml](../config/constraints.yaml) (D16):

```yaml
lock:
  hold_days:
    primary: 30                              # the deployment value
    values:  [15, 21, 30, 42, 60]            # the operating range: a sqrt(2) ladder,
    weights: [0.15, 0.20, 0.30, 0.20, 0.15]  #   geometrically centred on 30
    stress_values: [0, 7, 90, 180]           # Stage 9 only, reported as OOD
drawdown:
  max_drawdown:
    primary: 0.15
    values:  [0.05, 0.10, 0.15, 0.20, 0.25]
    weights: null                            # uniform
```

`N` is centred on 30 because that is where the constraint will actually sit; `D_max` is
uniform because there is no single expected value for it — the whole point of conditioning
on `D_max` is to serve a caller who picks their own ceiling.

Training on a single fixed pair would not be a parameter-conditioned policy at all. The curriculum
([rl-training.md](rl-training.md)) widens this distribution in stages rather than starting from the full grid.

At deployment the observation carries the actual production parameters, unchanged from training semantics —
which is only true if the normalization applied to `N` and `D_max` is fixed and not fitted per fold. It is
fixed: `N/365` and `D_max` as-is.

---

## 5. Reachable initial states

The requirement is to start from a **random legal** portfolio state. Naively randomizing weights and lock days
produces states that no legal trajectory could ever have reached — e.g. a position locked for 200 days under
`N = 30`, or a holding that predates its ETF's inception. Training on unreachable states wastes capacity and
distorts the value function.

Two generators, both implemented, with **Approach A as the v1 default**:

### Approach A — trajectory replay / state reservoir *(default)*

Run the baselines and a random policy through the real simulator, and snapshot `(ledger, lock state, peak,
NAV)` tuples into a reservoir keyed by `(session, N, D_max)`. Reset samples from it.

Reachable **by construction**, because every state in the reservoir was actually produced by a legal
trajectory. The reservoir is built in Stage 5 and refreshed periodically during training from the agent's own
trajectories, which also keeps the initial-state distribution close to the on-policy state distribution.

### Approach B — constructive legal generator

1. Sample a historical start date.
2. Sample from the assets available on that date.
3. Sample non-negative weights and a cash fraction.
4. Sample lock states, then **verify**: the unlock date is `<= date + N`, the holding's inception precedes the
   date, and the state is producible by some legal buy trajectory.
5. Reject and resample on any failure.

Used for deliberate coverage of states the reservoir under-samples (very high lock fractions, near-ceiling
drawdowns). Rejection sampling makes it slower, which is why it is not the default.

**Both generators must satisfy `D_t <= D_max` for a normal reset.** Deliberately-breached starting states are
a separate stress environment (`env.stress_reset: true`), never the default.

---

## 6. Reset

```
1. sample training date range
2. sample N          ~ P_N
3. sample D_max      ~ P_D
4. sample a reachable portfolio state (reservoir or constructive)
5. initialize peak NAV      <- from the sampled state, NOT set to current NAV
6. initialize current NAV
7. initialize lock state
8. initialize feature state (warm-up already satisfied by the 2003 buffer)
```

Step 5 deserves emphasis: setting `peak = nav` at reset would hand every episode a fresh zero drawdown and
teach the agent that drawdown resets for free. The sampled state carries its own peak, so an episode can and
should begin partway into a drawdown.

Guarantee: `D_t <= D_max` at reset in the normal mode.

---

## 7. Episode length

Sampled per episode from `{63, 126, 252, 504}` sessions rather than fixed at 252, so the policy does not
overfit to one horizon — which matters because `N` at the top of the operating range (60 calendar days,
~42 sessions) is a large fraction of a 63-session episode, and a policy trained only on 252-day episodes
would learn horizon-specific end effects.

**Evaluation is different:** walk-forward runs the full natural calendar year, unsegmented
([evaluation.md](evaluation.md)). Training-time episode chopping is a variance-reduction device, not a
property of the problem.

Truncation vs termination is handled the Gymnasium way: episode-length exhaustion is `truncated=True` (so
bootstrapping continues from the value function and the agent is not taught that time ending is a terminal
outcome), and there is **no** `terminated=True` condition at all — notably not on a drawdown breach, which
puts the environment into capital-preservation mode rather than ending the episode
([risk-envelope.md](risk-envelope.md) §2).

---

## 8. Gymnasium interface

```python
class ETFAllocationEnv(gymnasium.Env):
    observation_space: Box(low=-inf, high=inf, shape=(obs_dim,), dtype=float32)
    action_space:      Box(low=-inf, high=inf, shape=(K+1,), dtype=float32)  # pre-softmax logits

    def reset(self, seed=None, options=None) -> tuple[obs, info]: ...
    def step(self, action) -> tuple[obs, reward, terminated, truncated, info]: ...
```

`info` carries the full diagnostics every step — projection distance, safety intervention, capital
preservation, fallback, the executed weights, `N`, `D_max`. These are what build the trajectory artifact and
what the TensorBoard callbacks read. `terminated` is always `False`.

Vectorization: `SubprocVecEnv` over CPU workers, each with `seed + worker_index`
([architecture.md](architecture.md)). Worker count is chosen from the Stage 6 throughput benchmark, not
guessed.
