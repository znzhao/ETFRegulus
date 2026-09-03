# RL Training

The RL stack, the policy, the projection credit-assignment problem, and the training curriculum. Implemented in `src/agents/` and `src/training/`; run via [Stage 7](stages.md).

---

## 1. Stack (D2)

Gymnasium 1.3.0 + Stable-Baselines3 2.9.0 + torch 2.13.0+cu126, on CPU (D14 — the GPU is verified working and
benchmarked 2.7x slower end-to-end; see [gpu-setup.md](gpu-setup.md)). The novel content of this project is the constraint layer, not the RL
algorithm; a hand-rolled PPO would add a second independent source of bugs on top of a simulator that is
already the primary risk.

What is custom:

| Piece | Where |
|---|---|
| The environment, including the whole projection pipeline | `src/env/etf_env.py` |
| Observation normalization (fold-aware, no full-sample leakage) | `src/agents/normalization.py` |
| Feature extractor + policy head | `src/agents/policy.py` |
| Curriculum orchestration and warm-starting | `src/training/curriculum.py` |
| Diagnostic callbacks | `src/training/trainer.py` |

What is not: the PPO update itself.

---

## 2. Policy architecture

Start small and boring. The bottleneck is environment stepping, not network capacity (confirm this with the
Stage 6 throughput benchmark before touching architecture).

```
obs -> [macro | cross-sectional | per-asset (K x F) | portfolio | params]
     -> per-asset shared MLP encoder applied across the K asset blocks   (weight sharing)
     -> concat( pooled per-asset embedding, global blocks )
     -> MLP trunk (2 x 256, tanh)
     -> actor head  -> K+1 logits  -> softmax in the env
        critic head -> scalar value
```

Why the shared per-asset encoder: it makes the policy roughly permutation-equivariant across assets, so what
it learns about one sector ETF transfers to the others, and adding a ticker later does not require relearning
from scratch. It also cuts parameter count substantially versus a flat MLP over the full observation.

**Availability masking in the logits.** Unavailable assets get `-inf` before the softmax, so probability mass
is never spent on ETFs that do not exist yet. This measurably reduces projection distance in the early years
when a third of the universe has not launched.

Normalization: `VecNormalize` on observations, with statistics **frozen at evaluation** and **fitted only on
training-window data**. A `VecNormalize` whose running statistics keep updating during evaluation is a
full-sample-scaling leak wearing a disguise, and that is explicitly prohibited. Statistics are saved alongside
the policy and restored together — a policy loaded without its normalizer is meaningless.

---

## 3. Credit assignment with the projection (D9)

The problem: the policy proposes `a_raw`, the environment executes `a_proj`, and the reward reflects `a_proj`.
Which action does PPO learn from?

**Decision: store `a_raw`, execute `a_proj`.**

```python
a_raw  = policy.sample(obs)          # stored in the rollout buffer, log_prob'd
a_proj = project(a_raw, state)       # inside env.step
reward = f(a_proj)
info["proj_distance"] = l1(a_proj, a_raw)
```

Why: PPO's importance ratio must be evaluated under the distribution that generated the stored action. Storing
`a_proj` and computing `log pi(a_proj | s)` is wrong — projected actions frequently land on the boundary of the
simplex, where a continuous policy assigns them vanishing density, producing exploding or degenerate ratios.
Treating the projection as environment dynamics is both correct and standard.

**The acknowledged cost:** the policy gets no gradient explaining *why* its action was altered — only the
reward of the altered action. Mitigations, in strict order, and do not skip to 3:

1. **Measure first.** Log `proj_distance` every step. If it trends down during training, the policy is
   learning feasibility from the observation and nothing more is needed. This is the expected outcome and it
   costs nothing to check.
2. **Verify the masks reach the network.** The lock mask, availability mask, and drawdown headroom are all in
   the observation, so the information needed to propose feasible actions is present. If `proj_distance`
   is flat, first confirm those inputs are actually non-degenerate after normalization — a mask that
   `VecNormalize` has flattened to a constant is a silent failure and is the most likely cause.
3. **Auxiliary penalty, only if 1 and 2 fail.** Add `-lambda * ||a_proj - a_raw||_1` to the reward, with
   `lambda` ablated across at least `{0, 0.001, 0.01}`.

   > This does **not** violate D11. The prohibition is on implementing the *hard risk constraint* as a
   > reward penalty. This term penalizes the policy for proposing infeasible actions; the constraint itself
   > remains hard and enforced by the projection either way. Report the distinction explicitly whenever this
   > term is enabled, and treat any run using it as an ablation rather than the headline.

Never considered at v1: a differentiable projection inside the policy. It would conflict with D2 and adds
fragility for a benefit that mitigation 1 usually makes unnecessary.

---

## 4. Curriculum

Five stages, run in order, each warm-started from the previous checkpoint. One config file per stage.

| Stage | Config | `N` | `D_max` | Initial state | Episode len | Risk envelope |
|---|---|---|---|---|---|---|
| 1 | `ppo_stage1.yaml` | 0 | fixed 0.20 | flat (all cash) | 252 | off |
| 2 | `ppo_stage2.yaml` | {21, 30, 42} | fixed 0.20 | flat | 252 | off |
| 3 | `ppo_stage3.yaml` | operating range | full grid | flat | 252 | off |
| 4 | `ppo_stage4.yaml` | operating range | full grid | flat | 252 | **on** |
| 5 | `ppo_stage5.yaml` | operating range | full grid | **reservoir** | {63,126,252,504} | on |

Rationale for the ordering: Stage 1 isolates portfolio mechanics with nothing else to confound them, so if the
agent cannot beat cash there, the bug is in the simulator, not the RL. Each later stage adds exactly one source
of difficulty, so a regression localizes to the thing just added.

**Gate between stages:** a stage advances only if it beats the `cash` baseline on its training window *and*
records zero lock and zero feasibility violations. A violation means the constraint layer is broken — stop and
fix it rather than continuing into a stage that will mask it.

Warm-starting caveat: the observation dimension must be identical across curriculum stages, which it is — the
risk-envelope toggle changes the environment's dynamics, not its observation space.

---

## 5. Hyperparameters

Starting point in `config/training.yaml`, deliberately conservative:

```yaml
ppo:
  n_steps: 2048          # per env
  batch_size: 4096
  n_epochs: 10
  gamma: 0.999           # near-1: daily steps, long horizons
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.005        # some entropy: the softmax collapses to a corner easily
  vf_coef: 0.5
  max_grad_norm: 0.5
  learning_rate: 3.0e-4  # linear decay
  target_kl: 0.02
device: cpu              # D14 — measured, not assumed. See gpu-setup.md §3
```

Notes:

- `gamma = 0.999` is inherited from a superseded parameter range and is **open question Q7**. It was
  justified by "`N` up to 180 calendar days"; under D16 the lock is centred on 30 calendar days (~21
  sessions), for which a conventional 0.99 — effective horizon ~100 sessions — is already several times
  the constraint horizon. 0.999 gives ~1000 sessions, roughly four years, which is far longer than the
  longest episode (504 sessions). It is left unchanged here because it is a tuned value and D16 is a
  specification change; settle it on a validation year, which is the only place tuning is permitted.
- Entropy matters more than usual here: a softmax over 25+ assets collapses to a single-asset corner readily,
  and a collapsed policy looks stable while learning nothing.
- Reward scale: daily log returns are `~1e-3`. Use SB3's value normalization (`VecNormalize(norm_reward=True)`)
  during training, and remember the reported reward is then in normalized units — always report evaluation
  metrics from the trajectory artifact, never from training reward curves.
- Tuning happens on the **validation year** of a fold and nowhere else. Tuning against test performance
  is prohibited and is what the walk-forward test suite checks for.

---

## 6. What to watch during training

All on TensorBoard, all also written to `metrics.json`:

| Signal | Healthy | Trouble |
|---|---|---|
| `proj_distance` (D9) | trends down | flat or rising → mitigation 2 above |
| `safety_intervention_rate` | modest, rises as `D_max` tightens | ~100% → envelope over-calibrated ([risk-envelope.md](risk-envelope.md) §7) |
| `capital_preservation_rate` | rare outside crises | common → budget mis-calibrated |
| `cash_weight` distribution | varies with regime | pinned at 0 or 1 → collapsed policy |
| `entropy` | decays slowly | collapses early → raise `ent_coef` |
| `episode_return` vs baselines | above `cash`, then above `spy` | below `cash` → simulator or reward bug, not a policy problem |
| `lock_violations`, `feasibility_violations` | **exactly 0** | anything else → hard stop |

Baseline curves from Stage 5 are drawn as horizontal reference lines, so "is this good" is answerable at a
glance rather than after an analysis.

---

## 7. Baselines

Built in [Stage 5](stages.md), *before* any RL, so the RL development period always has a reference point.
Full specifications in
**[baselines.md](baselines.md)**; the primary three are `spy_buy_hold`, `momentum` (12-1 cross-sectional), and
`spy_tlt_60_40`, plus `cash`, `equal_weight`, and `classical_optimizer`.

All six run through the identical simulator and constraint layer, and all six are drawn as horizontal
reference lines on the training curves.

The bar to clear, stated plainly: a learned policy that does not beat `spy_tlt_60_40` on risk-adjusted terms
across folds has not demonstrated anything, however good the training curve looked. The interesting comparison
is not against `cash` — it is against a two-line static allocation that anyone could implement.

---

## 8. Ablations to record (not run by default)

Each is a config change plus a row in the Stage 12 report:

- Reward variants: log growth + drawdown penalty; differential Sharpe (D11 alternatives)
- Auxiliary projection penalty `lambda ∈ {0, 0.001, 0.01}` (§3 above)
- `lock.scope: portfolio` — the portfolio-wide lock variant ([lock-state-machine.md](lock-state-machine.md) §0)
- Risk aggregation `max` vs `mean`
- Flat MLP instead of the shared per-asset encoder
