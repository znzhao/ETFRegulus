"""Building environments, singly and vectorized.

One place that knows how to go from a resolved config to a working environment, so the
Stage 6 smoke test, the Stage 7 trainer and the Stage 8 evaluator cannot construct three
subtly different ones.

**Seeding.** Each worker derives its seed as `base_seed + worker_index` and never shares
an RNG (reference/architecture.md section 5). The consequence that matters for T9: worker
`i` sees the same episode stream whether it is one of 4 workers or one of 16, so a
throughput comparison across worker counts is a comparison of speed and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.config.loader import load_typed, resolve_config
from src.config.schema import UniverseConfig
from src.constraints.risk_envelope import RiskEnvelope
from src.env.etf_env import EnvConfig, ETFAllocationEnv
from src.env.feature_store import FeatureStore, build_store
from src.env.observation import ObservationSpec
from src.env.reset_sampler import build_sampler
from src.sim.runner import (
    build_constraints,
    build_envelope,
    build_sim_config,
    load_market,
    make_projector_from,
)
from src.sim.simulator import MarketData


@dataclass
class EnvBundle:
    """The expensive, immutable half of an environment, built once and shared."""

    market: MarketData
    store: FeatureStore
    spec: ObservationSpec
    constraints: object
    env_cfg: EnvConfig
    resolved: dict
    reservoir_path: str | None = None

    @property
    def obs_dim(self) -> int:
        return self.spec.size


def env_config_from(resolved: dict, constraints) -> EnvConfig:
    """Read the (N, D_max) grid from `config/constraints.yaml`, never from a literal.

    D16 lives in one file. An environment that carried its own copy of the ladder would
    be one edit away from training on a grid the evaluation does not use.
    """
    env = resolved.get("environment", {}) or {}
    hd = constraints.lock.hold_days
    dd = constraints.drawdown.max_drawdown
    window = env.get("window")

    # A curriculum rung may narrow the grid (stage 2 trains on {21, 30, 42}). Anything it
    # does NOT name falls through to constraints.yaml, so D16's ladder is stated once.
    # A rung that narrows the values must supply its own weights or none: silently
    # inheriting a 5-entry weight vector for a 3-entry grid is the bug this avoids.
    hold_values = env.get("hold_days_values", hd.values)
    hold_weights = (env.get("hold_days_weights")
                    if "hold_days_values" in env
                    else env.get("hold_days_weights", hd.weights))
    dmax_values = env.get("max_drawdown_values", dd.values)
    dmax_weights = (env.get("max_drawdown_weights")
                    if "max_drawdown_values" in env
                    else env.get("max_drawdown_weights", dd.weights))

    if hold_weights is not None and len(hold_weights) != len(hold_values):
        raise ValueError(
            f"hold_days_weights has {len(hold_weights)} entries for "
            f"{len(hold_values)} values; a rung that narrows the grid must drop or "
            "restate the weights rather than inherit mismatched ones")

    return EnvConfig(
        hold_days_values=tuple(int(v) for v in hold_values),
        hold_days_weights=tuple(hold_weights) if hold_weights else None,
        max_drawdown_values=tuple(float(v) for v in dmax_values),
        max_drawdown_weights=tuple(dmax_weights) if dmax_weights else None,
        episode_lengths=tuple(int(v) for v in env.get("episode_lengths", (63, 126, 252, 504))),
        window=(window["start"], window["end"]) if window else None,
        stress_reset=bool(env.get("stress_reset", False)),
        risk_enabled=bool(env.get("risk_enabled", True)),
        flat_start=bool(env.get("flat_start", False)),
        strict=bool(env.get("strict", True)),
        fixed_hold_days=env.get("fixed_hold_days"),
        fixed_max_drawdown=env.get("fixed_max_drawdown"),
        decision_cadence=int(env.get("decision_cadence", 0)),
        action_mode=str(env.get("action_mode", "full")),
    )


def build_bundle(config_path: str | Path, *, fold_id: str | None = None,
                 start=None, end=None, resolved: dict | None = None) -> EnvBundle:
    resolved = resolved if resolved is not None else resolve_config(Path(config_path))
    market, ucfg = load_market(resolved, start=start, end=end)
    constraints = build_constraints(resolved)
    spec = ObservationSpec.from_config(resolved, ucfg.tradable_tickers)
    # The store is materialized on the PRICE calendar, so a feature row and a price row
    # are the same integer and cannot drift by a session.
    store = build_store(spec, market.sessions,
                        fold_id=fold_id or (resolved.get("environment", {}) or {}).get("fold_id"))
    return EnvBundle(market=market, store=store, spec=spec, constraints=constraints,
                     env_cfg=env_config_from(resolved, constraints), resolved=resolved,
                     reservoir_path=(resolved.get("environment", {}) or {}).get("reservoir"))


def make_env(bundle: EnvBundle, *, seed: int = 0,
             env_cfg: EnvConfig | None = None) -> ETFAllocationEnv:
    cfg = env_cfg or bundle.env_cfg
    sim_cfg = build_sim_config(bundle.resolved, bundle.constraints, seed=seed)
    envelope: RiskEnvelope | None = build_envelope(bundle.constraints) if cfg.risk_enabled else None
    sampler = build_sampler(bundle.market, reservoir_path=bundle.reservoir_path,
                            initial_cash=sim_cfg.initial_cash,
                            require_reservoir=False)
    env = ETFAllocationEnv(
        bundle.market, bundle.store, bundle.spec, sampler, cfg, sim_cfg,
        envelope=envelope, projector=make_projector_from(bundle.constraints), seed=seed,
    )
    return env


def env_factory(bundle: EnvBundle, index: int, *, base_seed: int = 0,
                env_cfg: EnvConfig | None = None) -> Callable[[], ETFAllocationEnv]:
    """A thunk for `DummyVecEnv` / `SubprocVecEnv`. Seed is `base_seed + index`."""

    def _make() -> ETFAllocationEnv:
        return make_env(bundle, seed=base_seed + index, env_cfg=env_cfg)

    return _make


def make_vec_env(bundle: EnvBundle, n_envs: int, *, base_seed: int = 0,
                 subproc: bool = False, env_cfg: EnvConfig | None = None):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    thunks = [env_factory(bundle, i, base_seed=base_seed, env_cfg=env_cfg)
              for i in range(n_envs)]
    if subproc:
        # `spawn` on Windows, and it is also the only start method that behaves the same
        # on every platform this may be rerun on.
        return SubprocVecEnv(thunks, start_method="spawn")
    return DummyVecEnv(thunks)
