"""T5, T6 and the risk-envelope properties.

T5 is load-bearing: the `alpha` bisection in the projection is only valid if the feasible
set along `w(alpha)` is an interval containing `alpha = 0`. An estimator that breaks it
cannot be used with the analytic backend, and this test says so explicitly rather than
letting the projection fail mysteriously later.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from src.constraints.projector import (
    CASH,
    AnalyticProjector,
    safe_portfolio,
)
from src.constraints.risk_envelope import (
    RiskEnvelope,
    block_bootstrap_paths,
    crisis_window_paths,
    headroom,
    risk_budget,
    rolling_stress_paths,
)

K = 5
N = K + 1


def make_returns(rng, n=600, n_assets=N, vol=0.012):
    """Column 0 is cash and is identically zero, by construction."""
    r = rng.normal(0.0004, vol, size=(n, n_assets))
    r[:, CASH] = 0.0
    return r


def make_envelope(returns, *, measure="cvar", aggregation="max", as_of=None,
                  windows=None, seed=0):
    env = RiskEnvelope(quantile=0.01, horizon_days=5, block_length=10,
                       aggregation=aggregation, measure=measure,
                       n_bootstrap_paths=128, crisis_windows=windows or {})
    sessions = np.arange(np.datetime64("2010-01-01"),
                         np.datetime64("2010-01-01") + returns.shape[0])
    env.prepare(returns, sessions, as_of=as_of, seed=seed, n_assets=returns.shape[1])
    return env


# ------------------------------------------------------------------ the budget


def test_budget_shrinks_as_drawdown_deepens():
    d_max = 0.20
    at_peak = risk_budget(nav=100.0, peak=100.0, d_max=d_max)
    halfway = risk_budget(nav=90.0, peak=100.0, d_max=d_max)
    at_ceiling = risk_budget(nav=80.0, peak=100.0, d_max=d_max)

    assert at_peak == pytest.approx(0.20)
    assert 0.0 < halfway < at_peak
    assert at_ceiling == pytest.approx(0.0, abs=1e-12)


def test_budget_goes_negative_past_the_ceiling():
    """A negative budget means only `w_safe` is admissible -- capital preservation,
    consistent with the state-feasibility layer rather than a separate rule."""
    assert risk_budget(nav=70.0, peak=100.0, d_max=0.20) < 0.0


def test_headroom_is_dmax_minus_drawdown():
    assert headroom(nav=90.0, peak=100.0, d_max=0.20) == pytest.approx(0.10)
    assert headroom(nav=100.0, peak=100.0, d_max=0.20) == pytest.approx(0.20)


# -------------------------------------------------------------------- estimators


def test_cash_carries_no_stress():
    rng = np.random.default_rng(0)
    env = make_envelope(make_returns(rng))
    all_cash = np.zeros(N)
    all_cash[CASH] = 1.0
    assert env.stress_loss(all_cash) == pytest.approx(0.0, abs=1e-12)


def test_a_risky_portfolio_carries_positive_stress():
    rng = np.random.default_rng(0)
    env = make_envelope(make_returns(rng))
    w = np.zeros(N)
    w[1:] = 1.0 / K
    assert env.stress_loss(w) > 0.0


def test_a_more_volatile_book_is_scored_riskier():
    rng = np.random.default_rng(1)
    r = make_returns(rng)
    r[:, 1] *= 3.0                       # asset 1 is three times as volatile
    env = make_envelope(r)

    calm, wild = np.zeros(N), np.zeros(N)
    calm[2] = 1.0
    wild[1] = 1.0
    assert env.stress_loss(wild) > env.stress_loss(calm)


def test_rolling_paths_have_the_right_shape():
    rng = np.random.default_rng(2)
    r = make_returns(rng, n=100)
    paths = rolling_stress_paths(r, horizon=5)
    assert paths.shape == (95, N)
    assert np.allclose(paths[:, CASH], 0.0)


def test_the_bootstrap_is_not_iid():
    """Stationary bootstrap, never IID resampling: IID destroys volatility clustering and
    would produce a comfortable, wrong risk number.
    """
    rng = np.random.default_rng(3)
    n = 2000
    # A series with genuine volatility clustering.
    vol = np.ones(n) * 0.005
    vol[800:1000] = 0.05
    r = np.zeros((n, 2))
    r[:, 1] = rng.normal(0, 1, n) * vol

    long_blocks = block_bootstrap_paths(r, horizon=40, n_paths=4000, block_length=40,
                                        rng=np.random.default_rng(0))
    iid_like = block_bootstrap_paths(r, horizon=40, n_paths=4000, block_length=1,
                                     rng=np.random.default_rng(0))

    def kurtosis(x):
        return float((((x - x.mean()) / x.std()) ** 4).mean())

    # The discriminator is the TAIL, not the spread. A bootstrap preserves the marginal
    # variance by construction, so the standard deviations come out near-identical either
    # way (measured 0.094 vs 0.102) and comparing them would prove nothing. What IID
    # resampling destroys is the clustering, and that shows up as kurtosis.
    assert kurtosis(long_blocks[:, 1]) > 2 * kurtosis(iid_like[:, 1]), (
        "long blocks are not preserving volatility clustering; the bootstrap is "
        "behaving as IID and would produce a comfortable, wrong risk number"
    )
    assert np.quantile(long_blocks[:, 1], 0.001) < np.quantile(iid_like[:, 1], 0.001)


def test_the_bootstrap_is_deterministic_given_the_seed():
    rng = np.random.default_rng(4)
    r = make_returns(rng)
    a = block_bootstrap_paths(r, 5, 64, 10, np.random.default_rng(7))
    b = block_bootstrap_paths(r, 5, 64, 10, np.random.default_rng(7))
    assert np.array_equal(a, b)


# ---------------------------------------------------- the crisis-window lookahead


def test_only_crisis_windows_that_already_ended_are_visible_at_decision_time():
    """A 2005 decision cannot be stress-tested against 2008.

    This is a lookahead surface, and the one most likely to be forgotten.
    """
    rng = np.random.default_rng(5)
    n = 4000
    r = make_returns(rng, n=n)
    sessions = np.arange(np.datetime64("2003-01-01"), np.datetime64("2003-01-01") + n)
    windows = {"gfc": ["2007-10-01", "2009-03-31"], "covid": ["2010-02-19", "2010-03-23"]}

    early = crisis_window_paths(r, sessions, windows, horizon=5,
                                as_of=dt.date(2005, 1, 1))
    assert early.size == 0, "a future crisis was visible to a 2005 decision"

    later = crisis_window_paths(r, sessions, windows, horizon=5,
                                as_of=dt.date(2011, 1, 1))
    assert later.size > 0, "a past crisis was not available"

    # Stage 9 evaluation deliberately uses the full library: a separate code path.
    full = crisis_window_paths(r, sessions, windows, horizon=5, as_of=None)
    assert full.shape[0] >= later.shape[0]


def test_the_envelope_tightens_once_a_crisis_becomes_visible():
    rng = np.random.default_rng(6)
    n = 4000
    r = make_returns(rng, n=n)
    r[1700:2100, 1:] -= 0.03                       # a crash inside the 2007-2009 window
    sessions = np.arange(np.datetime64("2003-01-01"), np.datetime64("2003-01-01") + n)
    windows = {"gfc": ["2007-10-01", "2009-03-31"]}

    w = np.zeros(N)
    w[1:] = 1.0 / K

    env_before = RiskEnvelope(estimators=("crisis_windows",), crisis_windows=windows)
    env_before.prepare(r, sessions, as_of=dt.date(2005, 1, 1), seed=0, n_assets=N)
    env_after = RiskEnvelope(estimators=("crisis_windows",), crisis_windows=windows)
    env_after.prepare(r, sessions, as_of=dt.date(2011, 1, 1), seed=0, n_assets=N)

    assert env_before.stress_loss(w) == 0.0
    assert env_after.stress_loss(w) > 0.0


# ----------------------------------------------------------------------- T5


def test_the_safe_portfolio_is_not_always_the_least_risky_one():
    """The premise the original spec got wrong, demonstrated rather than argued.

    reference/risk-envelope.md section 5 required `stress_loss` to be monotone
    non-increasing as weight shifts toward `w_safe`. That is false whenever
    diversification is available: `w_safe` minimizes EXPOSURE, not RISK. A locked position
    plus an anti-correlated hedge is far safer than the same locked position plus cash, so
    moving toward `w_safe` can make things strictly worse.
    """
    n = 3000
    rng = np.random.default_rng(0)
    r = np.zeros((n, N))
    z = rng.normal(0, 0.02, n)
    r[:, 1] = z + rng.normal(0, 0.002, n)
    r[:, 2] = -z + rng.normal(0, 0.002, n)      # the hedge
    r[:, 3] = rng.normal(0, 0.02, n)
    env = make_envelope(r)

    lower = np.zeros(N)
    lower[1] = 0.5                              # asset 1 is locked at 50%
    hedged = np.zeros(N)
    hedged[1], hedged[2] = 0.5, 0.5
    w_safe = safe_portfolio(lower, np.ones(N, dtype=bool))

    assert env.stress_loss(w_safe) > env.stress_loss(hedged), (
        "this fixture no longer exhibits the non-monotone case, so it can no longer "
        "justify the convexity requirement that replaced monotonicity"
    )


@pytest.mark.parametrize("aggregation", ["max", "mean"])
def test_cvar_is_convex_along_the_derisking_segment(aggregation):
    """T5, restated: **convexity** is what makes the alpha bisection valid.

    The bisection needs the feasible set along `w(alpha)` to be an interval containing
    `alpha = 0`. Convexity gives that for every budget, and `w_safe` being feasible puts 0
    in the interval. Monotonicity is neither necessary nor true.
    """
    rng = np.random.default_rng(11)
    worst = 0.0
    for trial in range(40):
        r = make_returns(np.random.default_rng(trial))
        env = make_envelope(r, measure="cvar", aggregation=aggregation, seed=trial)

        w = rng.dirichlet(np.ones(N))
        lower = np.zeros(N)
        locked = rng.random(K) < 0.4
        if locked.any():
            lower[1:][locked] = rng.random(locked.sum()) * (0.7 / locked.sum())
        w_safe = safe_portfolio(lower, np.ones(N, dtype=bool))

        alphas = np.linspace(0.0, 1.0, 41)
        losses = np.array([env.stress_loss(a * w + (1 - a) * w_safe) for a in alphas])
        worst = max(worst, float(-np.diff(losses, 2).min()))

    assert worst < 1e-9, (
        f"CVaR is not convex along the segment (worst second difference {worst:.2e}); "
        f"the alpha bisection in the analytic projector is invalid"
    )


def test_var_is_not_convex_so_is_not_usable_with_the_analytic_backend():
    """Recorded explicitly rather than left to fail mysteriously later.

    A raw sample quantile is not convex, and the violation is the same order as the risk
    budget itself -- so `measure="var"` is a reporting statistic, not a constraint.
    """
    rng = np.random.default_rng(3)
    env = make_envelope(make_returns(rng, n=4000), measure="var")

    worst = 0.0
    for trial in range(60):
        g = np.random.default_rng(trial)
        a, b = g.dirichlet(np.ones(N)), g.dirichlet(np.ones(N))
        losses = np.array([env.stress_loss(t * a + (1 - t) * b)
                           for t in np.linspace(0, 1, 41)])
        worst = max(worst, float(-np.diff(losses, 2).min()))
    assert worst > 1e-6, (
        "VaR now looks convex on this fixture; if that is genuinely true, the "
        "cvar-only restriction can be revisited"
    )


def test_the_projection_always_returns_a_feasible_point_when_one_exists():
    """The property that actually matters downstream, asserted directly.

    Whatever the estimator does, the projector must never hand back an action that
    breaches the budget while a feasible one was reachable.
    """
    rng = np.random.default_rng(21)
    proj = AnalyticProjector()
    checked = 0
    for trial in range(60):
        env = make_envelope(make_returns(np.random.default_rng(trial), vol=0.02),
                            seed=trial)

        w = rng.dirichlet(np.ones(N))
        lower = np.zeros(N)
        locked = rng.random(K) < 0.4
        if locked.any():
            lower[1:][locked] = rng.random(locked.sum()) * (0.6 / locked.sum())
        available = np.ones(N, dtype=bool)
        w_safe = safe_portfolio(lower, available)

        # A budget that bites, but that `w_safe` can still meet.
        budget = max(env.stress_loss(w_safe), 1e-9) * 1.05
        env._budget = budget
        if env.stress_loss(w) <= budget:
            continue

        out = proj.project(w, lower_bounds=lower, available=available, risk=env)
        assert env.stress_loss(out.weights) <= budget + 1e-9, (
            f"the projection returned an infeasible action: "
            f"{env.stress_loss(out.weights):.6f} > {budget:.6f}"
        )
        checked += 1
    assert checked >= 10, "the fixture never forced a de-risk; the test proves nothing"


def test_the_bisection_finds_a_feasible_point_when_one_exists():
    rng = np.random.default_rng(12)
    r = make_returns(rng, vol=0.02)
    env = make_envelope(r)

    w = np.zeros(N)
    w[1:] = 1.0 / K
    unconstrained = env.stress_loss(w)
    env.set_budget(nav=100.0, peak=100.0, d_max=unconstrained / 2)   # forces a de-risk

    out = AnalyticProjector().project(w, lower_bounds=np.zeros(N),
                                      available=np.ones(N, dtype=bool), risk=env)
    assert out.diagnostics.risk_binding
    assert 0.0 <= out.diagnostics.de_risk_alpha < 1.0
    assert env.stress_loss(out.weights) <= env.budget() + 1e-9, "the result is infeasible"
    assert out.weights[CASH] > 0.0, "de-risking did not move anything to cash"


# ----------------------------------------------------------------------- T6


def test_the_fallback_returns_a_feasible_point_and_never_raises():
    """T6 / P6. Terminating the episode is forbidden -- it would teach the policy that
    entering a risky state ends the game, which is precisely the wrong lesson."""
    rng = np.random.default_rng(13)
    env = make_envelope(make_returns(rng, vol=0.03))
    env.set_budget(nav=100.0, peak=100.0, d_max=0.20)

    proj = AnalyticProjector()
    for _ in range(200):
        w = rng.dirichlet(np.ones(N))
        lower = np.zeros(N)
        locked = rng.random(K) < 0.5
        if locked.any():
            lower[1:][locked] = rng.random(locked.sum()) * (0.95 / locked.sum())
        out = proj.project(w, lower_bounds=lower, available=np.ones(N, dtype=bool),
                           risk=env)
        assert np.isfinite(out.weights).all()
        assert out.weights.sum() == pytest.approx(1.0, abs=1e-8)
        assert (out.weights >= -1e-9).all()


def test_a_market_forced_breach_is_flagged_rather_than_raising():
    """When even `w_safe` breaches, that is a market-forced violation, not an agent
    choice: record it and continue."""
    rng = np.random.default_rng(14)
    env = make_envelope(make_returns(rng, vol=0.05))
    lower = np.zeros(N)
    lower[1] = 0.95                       # a huge locked position that cannot be sold
    available = np.ones(N, dtype=bool)

    env.set_budget(nav=100.0, peak=100.0, d_max=0.0001)   # essentially no budget left
    out = AnalyticProjector().project(np.full(N, 1.0 / N), lower_bounds=lower,
                                      available=available, risk=env)

    assert out.diagnostics.infeasible_fallback, "the market-forced breach was not flagged"
    assert out.diagnostics.de_risk_alpha == 0.0
    assert out.weights[1] >= 0.95 - 1e-9, "the locked position was sold to escape"


def test_the_lock_outranks_capital_preservation():
    """The case where the two constraints genuinely conflict, and the lock wins.

    That means the drawdown can keep deepening while the agent is powerless, which is
    exactly the market-forced category.
    """
    lower = np.zeros(N)
    lower[1] = 0.60
    out = AnalyticProjector().project(
        np.full(N, 1.0 / N), lower_bounds=lower, available=np.ones(N, dtype=bool),
        capital_preservation=True, current_weights=lower.copy(),
    )
    assert out.weights[1] >= 0.60 - 1e-9, "capital preservation sold a locked position"
    assert out.diagnostics.capital_preservation
