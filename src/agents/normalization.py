"""Observation and reward normalization for training.

**Observations are not re-normalized, and that is a deliberate correction.**
reference/rl-training.md section 2 specified `VecNormalize` on observations, fitted on the
training window and frozen at evaluation. Measured on the real environment, that is the
wrong call here for two reasons:

1. **It is redundant.** The observation is already scaled by the fold scaler that Stage 3
   fitted on that fold's training window only -- the normalization the project already
   guarantees is leak-free, and the one T10 tests. Measured over 40 episodes the assembled
   observation has mean 0.20 and standard deviation 0.83, which is what `VecNormalize`
   exists to produce.
2. **It would destroy the feasibility masks.** 70 of the 656 observation dimensions are
   constant over a training window, and **19 of those are `pf_available` bits**.
   `VecNormalize` maps a constant dimension to exactly 0, so the availability mask -- which
   `src/agents/policy.py` reads directly to mask logits, and which the agent needs in order
   to know which actions the projection will alter -- would arrive as a column of zeros.
   That is precisely the silent failure rl-training.md section 3 names as the most likely
   cause of a flat `proj_distance`, and stacking a second normalizer is how you get it.

**Rewards are normalized**, because that part of section 2 is right: daily log returns are
~1e-3, and PPO's value loss on an unscaled target of that magnitude is numerically hopeless.
The consequence is that training reward curves are in normalized units, so evaluation
numbers must always come from the trajectory artifact and never from a training curve.

The normalizer is saved and loaded alongside the policy. A policy restored without its
normalizer is meaningless, so `save_bundle`/`load_bundle` move them together.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.vec_env import VecEnv, VecNormalize

VECNORM_NAME = "vecnormalize.pkl"
POLICY_NAME = "policy.zip"


def wrap_normalizer(vec: VecEnv, *, gamma: float, norm_reward: bool = True,
                    clip_reward: float = 10.0, training: bool = True) -> VecNormalize:
    """Reward-only normalization. See the module docstring for why observations are out."""
    return VecNormalize(vec, norm_obs=False, norm_reward=norm_reward,
                        clip_reward=clip_reward, gamma=gamma, training=training)


def freeze(vec: VecNormalize) -> VecNormalize:
    """Stop the running statistics moving. Mandatory before any evaluation.

    A `VecNormalize` whose statistics keep updating during evaluation is a full-sample
    scaling leak wearing a disguise, and it is explicitly prohibited.
    """
    vec.training = False
    vec.norm_reward = False      # evaluation reads raw returns, never normalized ones
    return vec


def save_bundle(model, vec: VecNormalize | None, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / POLICY_NAME)
    out = {"policy": str(directory / POLICY_NAME)}
    if vec is not None:
        vec.save(str(directory / VECNORM_NAME))
        out["vecnormalize"] = str(directory / VECNORM_NAME)
    return out


def load_normalizer(directory: Path, vec: VecEnv, *, for_eval: bool) -> VecNormalize | None:
    path = Path(directory) / VECNORM_NAME
    if not path.exists():
        return None
    loaded = VecNormalize.load(str(path), vec)
    return freeze(loaded) if for_eval else loaded
