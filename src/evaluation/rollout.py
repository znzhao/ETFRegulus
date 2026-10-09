"""Running a trained policy through the deterministic simulator.

The policy is wrapped as a **weight source** -- the same interface the six baselines use --
so it goes through `simulate()` and produces a `trajectory.parquet` in exactly the Stage 4
format. That is what makes an RL result and a baseline result the same artifact rather than
two formats somebody has to reconcile later, and it is why the Stage 12 report needs no
special case for the policy: it is one more column.

Evaluation is deterministic. `deterministic=True` takes the distribution's mean rather than
sampling it, because a walk-forward test year is a measurement, not an exploration -- and a
sampled rollout would make the same policy score differently on every read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.env.etf_env import softmax_weights
from src.env.feature_store import FeatureStore
from src.env.observation import ObservationSpec
from src.env.state_builder import build_observation
from src.sim.simulator import MarketData, StepContext


class PolicyWeightSource:
    """Adapts a trained SB3 policy to the `WeightFn` signature.

    Holds the observation buffer and the episode-start NAV, which is the normalizer for
    `pf_nav_norm` and `pf_peak_norm` -- it must be the NAV the *evaluation window* started
    at, not the NAV the policy was trained from, or the two blocks mean different things.
    """

    def __init__(self, model, spec: ObservationSpec, store: FeatureStore,
                 market: MarketData, *, hold_days: int, max_drawdown: float,
                 deterministic: bool = True):
        self.model = model
        self.spec = spec
        self.store = store
        self.market = market
        self.hold_days = int(hold_days)
        self.max_drawdown = float(max_drawdown)
        self.deterministic = bool(deterministic)
        self._buffer = np.zeros(spec.size, dtype=np.float32)
        self._nav_at_reset: float | None = None
        self.n_calls = 0

    def reset(self) -> None:
        self._nav_at_reset = None
        self.n_calls = 0

    def __call__(self, session: pd.Timestamp, ctx: StepContext) -> np.ndarray:
        if self._nav_at_reset is None:
            self._nav_at_reset = float(ctx.nav)
        row = self.store.index_of_session(session)
        obs = build_observation(
            self.spec, self.store, row, ctx,
            hold_days=self.hold_days, max_drawdown=self.max_drawdown,
            nav_at_reset=self._nav_at_reset, out=self._buffer)
        self.n_calls += 1
        return policy_weights(self.model, obs, deterministic=self.deterministic)


def policy_weights(model, obs, *, deterministic: bool = True) -> np.ndarray:
    """Portfolio weights from one policy, or the average from a list of them.

    A seed ensemble is the mean of the members' PORTFOLIOS, not of their raw actions:
    averaging actions (logits) would not give the average allocation.
    """
    if isinstance(model, (list, tuple)):
        return np.mean([softmax_weights(m.predict(obs, deterministic=deterministic)[0])
                        for m in model], axis=0)
    action, _ = model.predict(obs, deterministic=deterministic)
    return softmax_weights(action)


def load_policy(directory: Path, *, device: str = "cpu"):
    """Load a policy. Its normalizer is deliberately NOT applied here.

    Observations are not normalized by `VecNormalize` in this project (see
    `src/agents/normalization.py`), so a policy is complete on its own for inference. The
    saved `vecnormalize.pkl` only ever scaled the *reward*, which evaluation does not use --
    evaluation reads NAV from the trajectory.
    """
    from stable_baselines3 import PPO

    if isinstance(directory, (list, tuple)):
        return [load_policy(d, device=device) for d in directory]
    path = Path(directory)
    if path.is_dir():
        path = path / "policy.zip"
    if not path.exists():
        raise FileNotFoundError(f"no policy at {path}")
    return PPO.load(path, device=device)


def rollout(model, bundle, *, hold_days: int, max_drawdown: float,
            start_row: int, end_row: int, seed: int = 42,
            deterministic: bool = True):
    """One (policy, window, parameter cell) evaluation, as a Stage 4 trajectory."""
    from src.sim.runner import build_envelope, build_sim_config, make_projector_from
    from src.sim.simulator import simulate

    sim_cfg = build_sim_config(bundle.resolved, bundle.constraints,
                               hold_days=hold_days, max_drawdown=max_drawdown, seed=seed)
    source = PolicyWeightSource(model, bundle.spec, bundle.store, bundle.market,
                                hold_days=hold_days, max_drawdown=max_drawdown,
                                deterministic=deterministic)
    return simulate(
        bundle.market, source, sim_cfg,
        projector=make_projector_from(bundle.constraints),
        envelope=build_envelope(bundle.constraints) if sim_cfg.risk_enabled else None,
        start_row=start_row, end_row=end_row,
    )


def window_rows(market: MarketData, start, end) -> tuple[int, int]:
    """Row indices bounding a date window, clipped to the available calendar."""
    sessions = market.sessions
    rows = np.flatnonzero((sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end)))
    if rows.size < 2:
        raise ValueError(f"fewer than two sessions between {start} and {end}")
    return int(rows[0]), int(rows[-1])
