"""The environment's own contract: the MDP semantics, and the invariants on every step.

Stage 6 sweeps 500 random episodes; these are the same assertions in a form that runs in
seconds and names the specific property when it breaks. The invariants themselves (I1-I5)
are proven in `tests/portfolio/`; what is tested here is that the *environment* upholds
them, because it drives the engine through a different path than `simulate()` does.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.env.etf_env import EnvConfig, InvariantViolation
from src.env.factory import make_env
from src.sim.simulator import SimulationConfig


def _run(env, n=20, seed=0, **options):
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed, options=options or None)
    out = []
    for _ in range(n):
        obs, reward, term, trunc, info = env.step(
            rng.uniform(-1, 1, env.action_space.shape))
        out.append((obs, reward, term, trunc, info))
        if trunc:
            break
    return out


# ------------------------------------------------------------------ MDP semantics


def test_there_is_no_terminated_condition_at_all(risky_env):
    """Not on a drawdown breach, not ever. A breach puts the environment into capital
    preservation; it does not end the episode (env-mdp.md section 7)."""
    # `stress=True` is the designed way to START already breached. Waiting for a random
    # walk to breach instead does not work here, and the reason is the envelope doing its
    # job: over a 20-step episode it keeps the drawdown inside D_max, so the state under
    # test is exactly the one the safety layer exists to prevent.
    seen_preservation = False
    for seed in range(6):
        for obs, r, term, trunc, info in _run(risky_env, 20, seed, max_drawdown=0.05,
                                              hold_days=30, stress=True):
            assert term is False
            seen_preservation |= bool(info["capital_preservation"])
    assert seen_preservation, "no episode entered capital preservation"


def test_running_out_of_episode_is_truncation(env):
    steps = _run(env, 40, 1, length=6)
    assert steps[-1][3] is True and steps[-1][2] is False
    assert len(steps) == 6


def test_the_reward_is_the_log_nav_ratio(env):
    """r_t = log(V_{t+1}/V_t) on total-return NAV. Nothing else is in it: no transaction
    cost, no turnover penalty, no drawdown penalty (D11)."""
    rng = np.random.default_rng(0)
    env.reset(seed=2)
    prev = None
    for _ in range(10):
        obs, reward, _, trunc, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
        if prev is not None:
            assert reward == pytest.approx(np.log(info["nav"] / prev), abs=1e-9)
        prev = info["nav"]
        if trunc:
            break


def test_the_projection_happens_inside_step(env):
    """D9: the policy's action is stored, the projected action is executed. Both must be
    visible in `info`, and they must differ when the constraints bind."""
    rng = np.random.default_rng(0)
    env.reset(seed=3, options={"hold_days": 60})
    differed = 0
    for _ in range(15):
        obs, r, _, trunc, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
        assert info["raw_weights"].sum() == pytest.approx(1.0, abs=1e-6)
        assert info["executed_weights"].sum() == pytest.approx(1.0, abs=1e-6)
        differed += info["proj_distance"] > 1e-9
        if trunc:
            break
    assert differed > 0, "the projection never altered anything; it is not being applied"


def test_the_parameters_are_constant_within_an_episode(env):
    """`N` and `D_max` are sampled per EPISODE. If they moved mid-episode the conditioning
    would be meaningless and the lock semantics would change under the agent."""
    spec = env.obs_spec
    obs, _ = env.reset(seed=4, options={"hold_days": 21, "max_drawdown": 0.10})
    rng = np.random.default_rng(0)
    for _ in range(12):
        assert obs[spec.index_of("p:param_hold_days")] == pytest.approx(21 / 365, abs=1e-6)
        assert obs[spec.index_of("p:param_max_drawdown")] == pytest.approx(0.10, abs=1e-6)
        obs, _, _, trunc, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
        assert info["n_param"] == 21 and info["dmax_param"] == 0.10
        if trunc:
            break


# ---------------------------------------------------------------- the invariants


def test_no_invariant_is_violated_across_the_grid(bundle):
    """Strict mode raises on the first violation, so reaching the end is the assertion."""
    cfg = EnvConfig(**{**bundle.env_cfg.__dict__, "strict": True,
                       "risk_enabled": False, "episode_lengths": (15,)})
    env = make_env(bundle, seed=1, env_cfg=cfg)
    for i, (n, d) in enumerate([(n, d) for n in (15, 30, 60)
                                for d in (0.05, 0.15, 0.25)]):
        _run(env, 15, i, hold_days=n, max_drawdown=d)
        assert env.violations == []


def test_a_locked_position_never_shrinks(env):
    """I3, observed through the environment rather than the ledger."""
    rng = np.random.default_rng(0)
    env.reset(seed=5, options={"hold_days": 60})
    before = env.state.ledger.share_vector(env.market.universe)
    for _ in range(20):
        session = env.market.sessions[env._row].date()
        locked = [t for t in env.market.universe
                  if env.state.lock_manager.is_locked(t, session)]
        obs, r, _, trunc, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
        after = env.state.ledger.share_vector(env.market.universe)
        for t in locked:
            j = env.market.universe.index(t)
            assert after[j] >= before[j] - 1e-9, f"{t} shrank while locked"
        before = after
        if trunc:
            break


def test_the_strict_checker_actually_fires(bundle):
    """A checker that has never been observed to fail is untested infrastructure. This
    injects a corrupt ledger and asserts the environment notices."""
    cfg = EnvConfig(**{**bundle.env_cfg.__dict__, "strict": True,
                       "risk_enabled": False, "episode_lengths": (10,)})
    env = make_env(bundle, seed=1, env_cfg=cfg)
    env.reset(seed=1)
    # Negative cash is I1. The engine cannot produce it; the checker must catch it if
    # anything ever does.
    env.state.ledger.cash = -1.0
    with pytest.raises(Exception):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))


# ------------------------------------------- the two constraints, interacting


def test_the_lock_is_what_makes_the_risk_envelope_infeasible(bundle):
    """Measured in Stage 6, and the most important thing that stage found.

    Without a lock, `w_safe` is all cash: zero risk, always feasible, so the de-risking
    scan always succeeds and the fallback is never needed. With a lock, `w_safe` is
    floored at the locked holdings, which carry real risk -- and when the drawdown budget
    is small that floor alone can exceed it, leaving the feasible set EMPTY. The fallback
    is then correct behaviour and not a bug, but it means the agent's action is discarded
    entirely, which is D9's acknowledged cost showing up as a number.
    """
    cfg = EnvConfig(**{**bundle.env_cfg.__dict__, "strict": False,
                       "episode_lengths": (25,)})
    rates = {}
    for hold_days in (0, 30):
        env = make_env(bundle, seed=3, env_cfg=cfg)
        n = fallback = 0
        for seed in range(4):
            for _, _, _, _, info in _run(env, 25, seed, hold_days=hold_days,
                                         max_drawdown=0.05):
                n += 1
                fallback += bool(info["infeasible_fallback"])
        rates[hold_days] = fallback / max(n, 1)
    assert rates[0] < 0.05, (
        f"with no lock the feasible set should almost never be empty, got {rates[0]:.3f}")
    assert rates[30] > 4 * max(rates[0], 0.01), (
        f"the lock should drive the fallback rate up sharply: {rates}")


def test_capital_preservation_does_not_end_the_episode_but_does_bind(risky_env):
    """Layer one: the drawdown has already happened and the agent cannot undo it. The
    action set shrinks; the episode continues."""
    steps = []
    for seed in range(8):
        steps += _run(risky_env, 20, seed, max_drawdown=0.05, hold_days=30, stress=True)
    preserved = [s for s in steps if s[4]["capital_preservation"]]
    assert preserved, "capital preservation never engaged"
    assert all(not s[2] for s in preserved), "capital preservation terminated an episode"
