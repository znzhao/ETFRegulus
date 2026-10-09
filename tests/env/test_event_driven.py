"""Event-driven decisions and free-capital actions (redesign Phase 2, MODEL_REDESIGN_PLAN.md 5a-5c).

* the decision clock fires weekly, on an unlock, and on a capital-preservation flip;
* holding between decisions is a genuine no-op -- no trade, no relock;
* the environment returns one transition per DECISION, with the holding-period reward;
* free-capital actions respect the lock by construction;
* evaluation follows the same schedule as training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.env.etf_env import EnvConfig, onto_free_capital, target_weights
from src.env.factory import make_env
from src.sim.engine import DecisionClock

from .conftest import needs_data


@dataclass
class Ctx:
    step: int
    current_weights: np.ndarray
    lock_remaining_days: np.ndarray
    drawdown: float = 0.0
    max_drawdown: float = 0.05
    available: np.ndarray | None = None


W = np.array([0.4, 0.6, 0.0])           # [CASH, A, B]: holding A


def test_the_clock_fires_on_the_first_session_then_every_cadence():
    clock = DecisionClock(5)
    fired = [clock.due(Ctx(i, W, np.array([10.0, 0.0]))) for i in range(12)]
    assert fired == [True, False, False, False, False, True,
                     False, False, False, False, True, False]


def test_the_clock_fires_when_a_held_position_unlocks():
    clock = DecisionClock(5)
    assert clock.due(Ctx(0, W, np.array([2.0, 0.0])))
    assert not clock.due(Ctx(1, W, np.array([1.0, 0.0])))
    assert clock.due(Ctx(2, W, np.array([0.0, 0.0])))        # A unlocked: decide now
    assert not clock.due(Ctx(3, W, np.array([0.0, 0.0])))


def test_the_clock_fires_when_capital_preservation_flips():
    clock = DecisionClock(5)
    assert clock.due(Ctx(0, W, np.zeros(2), drawdown=0.03))
    assert not clock.due(Ctx(1, W, np.zeros(2), drawdown=0.04))
    assert clock.due(Ctx(2, W, np.zeros(2), drawdown=0.06))   # crossed D_max = 5%
    assert not clock.due(Ctx(3, W, np.zeros(2), drawdown=0.07))
    assert clock.due(Ctx(4, W, np.zeros(2), drawdown=0.04))   # back under


def test_cadence_zero_is_every_session():
    clock = DecisionClock(0)
    assert all(clock.due(Ctx(i, W, np.zeros(2))) for i in range(5))


def test_free_capital_keeps_every_locked_floor_and_spends_the_rest():
    ctx = Ctx(0, np.array([0.1, 0.5, 0.4]), np.array([7.0, 0.0]), available=np.array([True, True]))
    s = np.array([0.2, 0.3, 0.5])
    w = onto_free_capital(s, ctx)
    assert w.sum() == pytest.approx(1.0)
    assert w[1] >= 0.5 - 1e-12                         # A's floor is kept
    assert w[1] == pytest.approx(0.5 + 0.5 * 0.3)      # weight on A on top = adding
    assert w[2] == pytest.approx(0.5 * 0.5)            # B may be cut: it is unlocked
    assert np.allclose(target_weights(np.zeros(3), ctx, "full"), np.ones(3) / 3)


def test_free_capital_never_allocates_to_an_asset_that_does_not_exist_yet():
    ctx = Ctx(0, np.array([1.0, 0.0, 0.0]), np.zeros(2), available=np.array([True, False]))
    w = onto_free_capital(np.array([0.2, 0.3, 0.5]), ctx)
    assert w[2] == 0.0 and w.sum() == pytest.approx(1.0)
    assert w[0] / w[1] == pytest.approx(0.2 / 0.3)          # the rest keeps its proportions


# ----------------------------------------------------------- the real environment


def _event_env(bundle, **over):
    cfg = EnvConfig(**{**bundle.env_cfg.__dict__, "risk_enabled": False,
                       "episode_lengths": (60,), "decision_cadence": 5,
                       "action_mode": "free_capital", "strict": True, **over})
    return make_env(bundle, seed=3, env_cfg=cfg)


@needs_data
def test_one_transition_per_decision_with_the_holding_period_reward(bundle):
    env = _event_env(bundle)
    rng = np.random.default_rng(0)
    for ep in range(3):
        _, info = env.reset(seed=ep)
        nav0 = info["nav"]
        total, steps, sessions = 0.0, 0, 0
        done = False
        while not done:
            _, r, term, trunc, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
            assert 1 <= info["sessions_in_step"] <= 5
            total += r
            steps += 1
            sessions += info["sessions_in_step"]
            done = term or trunc
        assert sessions == info["episode_spec"]["length"]   # <= 60 near the window's end
        assert steps < 60                               # fewer decisions than sessions
        # The holding-period rewards add up to the episode's log return exactly.
        assert total == pytest.approx(np.log(info["nav"] / nav0), abs=1e-9)
        assert not env.violations


@needs_data
def test_holding_never_trades_or_relocks(bundle):
    """Between decisions the ledger only accretes dividends: no trade, no new lock."""
    from src.sim.engine import hold, observe

    env = _event_env(bundle)
    env.reset(seed=1)
    env.step(np.zeros(env.action_space.shape))          # buy something
    st, mk = env.state, env.market
    before_shares = st.ledger.share_vector(mk.universe).copy()
    before_locks = dict(st.lock_manager.unlock_dates)
    dec = observe(mk, st, env.sim_cfg, env._row, envelope=None)
    out = hold(mk, st, env.sim_cfg, env._row, dec)
    after = st.ledger.share_vector(mk.universe)
    assert out.row["turnover"] == 0.0 and out.row["decision"] is False
    assert dict(st.lock_manager.unlock_dates) == before_locks
    assert np.all(after >= before_shares - 1e-12)       # only accretion, never a sale


@needs_data
def test_free_capital_actions_never_need_the_lock_repaired(bundle):
    """Feasible under the lock by construction: the projection never lifts a lock floor."""
    env = _event_env(bundle)
    rng = np.random.default_rng(5)
    env.reset(seed=4)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
        raw, executed = info["raw_weights"], info["executed_weights"]
        # With risk off, the only thing the projector could change is a lock floor.
        assert np.allclose(raw, executed, atol=1e-6)
        done = term or trunc


@needs_data
def test_evaluation_decides_on_the_training_schedule(bundle):
    """A policy run through simulate() decides on the same clock and holds in between."""
    from dataclasses import replace

    from src.evaluation.rollout import rollout

    class Uniform:
        def predict(self, obs, deterministic=True):
            return np.zeros(len(bundle.market.universe) + 1, dtype=np.float32), None

    b = replace(bundle, env_cfg=EnvConfig(**{**bundle.env_cfg.__dict__,
                                            "decision_cadence": 5,
                                            "action_mode": "free_capital"}))
    out = rollout(Uniform(), b, hold_days=30, max_drawdown=0.15, start_row=0, end_row=80)
    traj = out.trajectory
    assert out.diagnostics["lock_violations"] == 0
    held = traj[~traj["decision"]]
    assert len(held) > 0 and (held["turnover"] == 0.0).all()
    assert traj["decision"].sum() <= 80 // 5 + 4        # weekly, plus a few events
