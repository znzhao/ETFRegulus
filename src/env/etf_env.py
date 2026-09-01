"""The Gymnasium environment: a finite-horizon, parameter-conditioned, constrained MDP.

Specified in reference/env-mdp.md. Three things about it are load-bearing and easy to get
subtly wrong, so they are stated here next to the code that implements them:

**The projection lives inside `step`.** The policy proposes logits; the environment
softmaxes them, projects onto the feasible set, executes the projection, and rewards what
executed. That is what makes D9 correct -- SB3 stores the action the policy sampled, and
everything after it is by construction part of the environment's dynamics. Storing the
projected action instead would ask PPO for the density of a point that usually sits on the
boundary of the simplex, where a continuous policy assigns it almost none.

**There is no `terminated`.** Not on a drawdown breach, not ever. A breach puts the
environment into capital preservation; it does not end the episode. Running out of
episode is `truncated=True`, so the value function still bootstraps and the agent is not
taught that time ending is an outcome it caused.

**The peak is inherited from the sampled state.** Setting `peak = nav` at reset would hand
every episode a fresh zero drawdown and teach the agent that drawdown resets for free.

The per-step mechanics are not reimplemented here: `src/sim/engine.py` owns them and the
Stage 5 baselines ran through the same two functions.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.constraints.projector import FeasibilityProjector, make_projector
from src.constraints.risk_envelope import RiskEnvelope
from src.env.feature_store import FeatureStore
from src.env.observation import ObservationSpec
from src.env.reset_sampler import InitialState, ResetSampler
from src.env.state_builder import build_observation
from src.sim.engine import EngineState, advance, observe
from src.sim.simulator import MarketData, SimulationConfig

#: The action space is `Box(-1, 1)`, and the environment scales it by `LOGIT_SCALE` before
#: the softmax.
#:
#: reference/env-mdp.md section 8 specified an unbounded `Box(-inf, inf)` over raw logits.
#: That is wrong twice over, and the deviation is deliberate:
#:
#: * SB3 clips sampled actions to the action-space bounds, and clipping to +/-inf is a
#:   no-op -- an unbounded space removes the only guard against a diverging policy
#:   emitting a logit large enough to saturate the softmax into a one-hot, at which point
#:   the gradient through it vanishes and training silently stalls.
#: * Both `gymnasium` and SB3's `check_env` recommend a symmetric, normalized space, and
#:   the reason is not cosmetic. PPO's Gaussian head starts at mean 0 with std ~1, so on a
#:   +/-10 space the initial policy would only ever sample the middle 10% of the range --
#:   every action a near-uniform allocation -- and would have to learn to inflate its own
#:   logits before it could express a concentrated portfolio at all.
#:
#: Scaling inside the environment keeps both properties: the policy works in the units it
#: is good at, and the softmax still sees a range spanning a weight ratio of e^20, far
#: wider than any allocation the projection would leave intact.
LOGIT_SCALE = 10.0

#: Sampled per episode rather than fixed at 252 (env-mdp.md section 7): at the top of the
#: operating range N = 60 calendar days is ~42 sessions, a large fraction of a 63-session
#: episode, and a policy trained only on year-long episodes learns horizon-specific end
#: effects it cannot use.
EPISODE_LENGTHS: tuple[int, ...] = (63, 126, 252, 504)


@dataclass
class EnvConfig:
    """Everything about the environment that is not the market or the constraints."""

    hold_days_values: tuple[int, ...] = (15, 21, 30, 42, 60)
    hold_days_weights: tuple[float, ...] | None = (0.15, 0.20, 0.30, 0.20, 0.15)
    max_drawdown_values: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)
    max_drawdown_weights: tuple[float, ...] | None = None
    episode_lengths: tuple[int, ...] = EPISODE_LENGTHS
    window: tuple[str, str] | None = None
    stress_reset: bool = False
    risk_enabled: bool = True
    initial_cash: float = 1_000_000.0
    #: Assert every invariant on every step. On in Stage 6, off in Stage 7 -- it roughly
    #: doubles the step cost, and the smoke test exists so that training does not have to
    #: pay for it.
    strict: bool = True
    fixed_hold_days: int | None = None
    fixed_max_drawdown: float | None = None


class InvariantViolation(AssertionError):
    """An environment step produced a state the constraint layer forbids."""


@dataclass
class EpisodeSpec:
    """What one episode was drawn as. Recorded so a failure is reproducible."""

    hold_days: int
    max_drawdown: float
    length: int
    start_row: int
    start_session: pd.Timestamp
    source: str
    nav_at_reset: float
    peak_at_reset: float
    drawdown_at_reset: float

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["start_session"] = str(pd.Timestamp(self.start_session).date())
        return d


class ETFAllocationEnv(gym.Env):
    """One parameter-conditioned episode over the real price history."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        market: MarketData,
        store: FeatureStore,
        spec: ObservationSpec,
        sampler: ResetSampler,
        cfg: EnvConfig,
        sim_cfg: SimulationConfig,
        *,
        envelope: RiskEnvelope | None = None,
        projector: FeasibilityProjector | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.market = market
        self.store = store
        # `spec` is reserved by `gymnasium.Env` for its EnvSpec; the observation layout
        # must not shadow it, or the env checker reads the wrong object.
        self.obs_spec = spec
        self.sampler = sampler
        self.cfg = cfg
        self.sim_cfg = sim_cfg
        self.envelope = envelope
        self.projector = projector or make_projector("analytic")

        K = market.n_assets
        self.observation_space = spaces.Box(
            low=-np.float32(spec.clip), high=np.float32(spec.clip),
            shape=(spec.size,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(K + 1,), dtype=np.float32)

        self._window = self._resolve_window(cfg.window)
        self._obs_buffer = np.zeros(spec.size, dtype=np.float32)
        self.episode: EpisodeSpec | None = None
        self.state: EngineState | None = None
        self.violations: list[str] = []
        self._row = 0
        self._steps = 0
        self._nav_at_reset = 1.0
        self._feature_offset = 0
        if seed is not None:
            super().reset(seed=seed)

    # -------------------------------------------------------------------- helpers

    def _resolve_window(self, window) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if window is None:
            return None
        return (pd.Timestamp(window[0]), pd.Timestamp(window[1]))

    def _sample_param(self, values, weights) -> Any:
        if len(values) == 1:
            return values[0]
        p = None if weights is None else np.asarray(weights, dtype=float)
        if p is not None:
            p = p / p.sum()
        return values[int(self.np_random.choice(len(values), p=p))]

    # ---------------------------------------------------------------------- reset

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        options = options or {}
        cfg = self.cfg

        hold_days = int(options.get(
            "hold_days",
            cfg.fixed_hold_days if cfg.fixed_hold_days is not None
            else self._sample_param(cfg.hold_days_values, cfg.hold_days_weights)))
        max_drawdown = float(options.get(
            "max_drawdown",
            cfg.fixed_max_drawdown if cfg.fixed_max_drawdown is not None
            else self._sample_param(cfg.max_drawdown_values, cfg.max_drawdown_weights)))
        stress = bool(options.get("stress", cfg.stress_reset))

        init: InitialState = self.sampler.sample(
            self.np_random, hold_days=hold_days, max_drawdown=max_drawdown,
            window=options.get("window", self._window), stress=stress)

        row = int(self.market.sessions.get_indexer([pd.Timestamp(init.session)])[0])
        if row < 0:
            raise InvariantViolation(
                f"sampled state at {init.session}, which is not a trading session")

        length = int(options.get(
            "length", self._sample_param(cfg.episode_lengths, None)))
        # An episode needs one session beyond its last decision to execute into.
        length = max(1, min(length, len(self.market.sessions) - 1 - row))

        self.sim_cfg = SimulationConfig(
            hold_days=hold_days, max_drawdown=max_drawdown,
            cost_bps=self.sim_cfg.cost_bps, initial_cash=self.sim_cfg.initial_cash,
            lock_scope=self.sim_cfg.lock_scope, risk_enabled=cfg.risk_enabled,
            seed=self.sim_cfg.seed, risk_lookback=self.sim_cfg.risk_lookback,
        )
        # The peak is INHERITED from the sampled state, not reset to the NAV.
        self.state = EngineState(ledger=init.ledger.copy(),
                                 lock_manager=init.lock_manager.copy(),
                                 peak=float(init.peak))
        self._row = row
        self._steps = 0
        self.violations = []
        self._feature_offset = self.store.index_of_session(self.market.sessions[row]) - row

        dec = observe(self.market, self.state, self.sim_cfg, row, envelope=None)
        self._nav_at_reset = float(dec.nav)

        self.episode = EpisodeSpec(
            hold_days=hold_days, max_drawdown=max_drawdown, length=length,
            start_row=row, start_session=self.market.sessions[row], source=init.source,
            nav_at_reset=float(dec.nav), peak_at_reset=float(self.state.peak),
            drawdown_at_reset=float(dec.drawdown),
        )

        if self.cfg.strict and not stress and dec.drawdown > max_drawdown + 1e-9:
            raise InvariantViolation(
                f"reset produced D_t={dec.drawdown:.4f} > D_max={max_drawdown:.4f} "
                f"in normal mode ({init.source})")

        obs = self._observe(dec)
        return obs, {"episode": self.episode.to_dict(), **self._info_from(dec)}

    # ----------------------------------------------------------------------- step

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.state is None or self.episode is None:
            raise RuntimeError("step() before reset()")

        dec = observe(self.market, self.state, self.sim_cfg, self._row,
                      envelope=self.envelope if self.cfg.risk_enabled else None)
        before = self.state.ledger.share_vector(self.market.universe)

        a_raw = softmax_weights(action)
        out = advance(self.market, self.state, self.sim_cfg, self._row, dec,
                      a_raw, self.projector)

        self._row += 1
        self._steps += 1
        if self.cfg.strict:
            self._check(out, dec, before)

        truncated = self._steps >= self.episode.length or self._row >= len(
            self.market.sessions) - 1
        # There is no `terminated` condition at all -- notably not a drawdown breach.
        terminated = False

        next_dec = observe(self.market, self.state, self.sim_cfg, self._row, envelope=None)
        obs = self._observe(next_dec)
        info = {
            **self._info_from(next_dec),
            "proj_distance": out.row["proj_distance"],
            "safety_intervened": out.row["safety_intervened"],
            "capital_preservation": out.row["capital_preservation"],
            "infeasible_fallback": out.row["infeasible_fallback"],
            "de_risk_alpha": out.row["de_risk_alpha"],
            "turnover": out.row["turnover"],
            "executed_weights": out.projected.astype(np.float32),
            "raw_weights": a_raw.astype(np.float32),
            "lock_violations": out.lock_violations,
            "n_param": self.episode.hold_days,
            "dmax_param": self.episode.max_drawdown,
        }
        if truncated:
            info["episode_spec"] = self.episode.to_dict()
        return obs, float(out.reward), terminated, truncated, info

    # ------------------------------------------------------------------ internals

    def _observe(self, dec) -> np.ndarray:
        return build_observation(
            self.obs_spec, self.store, self._row + self._feature_offset, dec,
            hold_days=self.episode.hold_days if self.episode else self.sim_cfg.hold_days,
            max_drawdown=(self.episode.max_drawdown if self.episode
                          else self.sim_cfg.max_drawdown),
            nav_at_reset=self._nav_at_reset, out=self._obs_buffer,
        ).copy()

    def _info_from(self, dec) -> dict:
        return {
            "session": self.market.sessions[self._row],
            "nav": float(dec.nav), "peak": float(self.state.peak),
            "drawdown": float(dec.drawdown),
            "drawdown_budget": self.sim_cfg.max_drawdown - float(dec.drawdown),
            "cash": float(self.state.ledger.cash),
        }

    def _check(self, out, dec, shares_before: np.ndarray) -> None:
        """Every invariant, on every step. Stage 6's whole job."""
        st = self.state
        st.ledger.check()                                    # I1
        if out.lock_violations:
            self.violations.append(
                f"I3 lock: {out.lock_violations} locked position(s) shrank at "
                f"{out.row['session']}")
        exec_date = self.market.sessions[self._row].date()
        for j, ticker in enumerate(self.market.universe):
            if st.lock_manager.is_locked(ticker, exec_date):
                if st.ledger.get(ticker) < shares_before[j] - 1e-9:
                    self.violations.append(f"I3 lock: {ticker} shrank while locked")
        # I4: an executed buy sets unlock = exec + N exactly.
        for ticker, unlock in st.lock_manager.unlock_dates.items():
            if unlock > exec_date + dt.timedelta(days=self.sim_cfg.hold_days):
                self.violations.append(
                    f"I4 lock: {ticker} unlock {unlock} beyond exec+N")
        # I2: weights and cash sum to one.
        total = out.row["cash"] / out.row["nav"] + sum(
            out.row[f"w_{t}"] for t in self.market.universe)
        if abs(total - 1.0) > 1e-6:
            self.violations.append(f"I2 accounting: weights+cash = {total!r}")
        # I5: nothing may be held before its inception.
        avail = self.market.available[self._row]
        for j, ticker in enumerate(self.market.universe):
            if not avail[j] and st.ledger.get(ticker) > 1e-9:
                self.violations.append(f"I5 inception: {ticker} held pre-inception")
        if self.violations:
            raise InvariantViolation("; ".join(self.violations[:5]))


def softmax_weights(action, scale: float = LOGIT_SCALE) -> np.ndarray:
    """A normalized action -> a point on the simplex over [CASH, ETF_1..ETF_K].

    The action arrives in [-1, 1]; `scale` turns it into logits. Shifted by the max before
    exponentiating, which costs nothing and keeps the function correct if the scale is
    ever widened.
    """
    a = np.asarray(action, dtype=np.float64).ravel()
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
    a = np.clip(a, -1.0, 1.0) * scale
    e = np.exp(a - a.max())
    return e / e.sum()
