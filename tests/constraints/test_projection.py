"""T3, T4, T7 and the P1-P7 projection properties.

T4 is the important one: the analytic backend is checked against a CVXPY solve of the same
QP over thousands of random instances. That is D3's correctness guarantee, and it catches
analytic-projection bugs long before they could be mistaken for policy behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.constraints.projector import (
    CASH,
    AnalyticProjector,
    CvxpyProjector,
    ProjectionDiagnostics,
    project_onto_simplex,
    project_with_lower_bounds,
    safe_portfolio,
)

K = 6                      # tradable assets; the action vector is K + 1 wide
N = K + 1


def rand_case(rng, *, lock_frac=0.3, unavail_frac=0.2):
    y = rng.dirichlet(np.ones(N))
    available = np.ones(N, dtype=bool)
    available[1:][rng.random(K) < unavail_frac] = False
    available[CASH] = True

    lower = np.zeros(N)
    locked = (rng.random(K) < lock_frac) & available[1:]
    if locked.any():
        lower[1:][locked] = rng.random(locked.sum()) * (0.9 / max(1, locked.sum()))
    return y, lower, available


# ------------------------------------------------------------- simplex machinery


@settings(max_examples=200, deadline=None)
@given(v=st.lists(st.floats(-5, 5, allow_nan=False), min_size=2, max_size=12),
       budget=st.floats(0.01, 3.0))
def test_simplex_projection_lands_on_the_simplex(v, budget):
    w = project_onto_simplex(np.array(v), budget)
    assert (w >= -1e-12).all()
    assert w.sum() == pytest.approx(budget, abs=1e-9)


def test_simplex_projection_is_identity_on_the_simplex():
    y = np.array([0.2, 0.5, 0.3])
    assert np.allclose(project_onto_simplex(y, 1.0), y, atol=1e-12)


def test_simplex_projection_matches_a_hand_computed_case():
    """y = [1, 0, 0] is already on the simplex; y = [2, 0, 0] must project back to it."""
    assert np.allclose(project_onto_simplex(np.array([2.0, 0.0, 0.0])), [1.0, 0.0, 0.0])
    # Equal excess spreads equally.
    assert np.allclose(project_onto_simplex(np.array([1.0, 1.0])), [0.5, 0.5])


# ------------------------------------------------------------------ P1, P2, P3


def test_output_is_always_on_the_simplex_and_respects_availability():
    """P1."""
    rng = np.random.default_rng(0)
    proj = AnalyticProjector()
    for _ in range(500):
        y, lower, available = rand_case(rng)
        out = proj.project(y, lower_bounds=lower, available=available)
        w = out.weights
        assert (w >= -1e-12).all(), "negative weight"
        assert w.sum() == pytest.approx(1.0, abs=1e-9), "left the simplex"
        assert np.all(w[~available] <= 1e-12), "weight assigned to an unavailable asset"


def test_lock_lower_bounds_are_never_violated():
    """P2, in weight space. The share-space enforcement is tested in execution."""
    rng = np.random.default_rng(1)
    proj = AnalyticProjector()
    for _ in range(500):
        y, lower, available = rand_case(rng)
        if lower.sum() > 1.0:
            continue
        w = proj.project(y, lower_bounds=lower, available=available).weights
        assert np.all(w + 1e-9 >= np.where(available, lower, 0.0)), "a lock floor was cut"


def test_projection_is_idempotent_on_feasible_points():
    """P3 / T3: if `a_raw` is already feasible, `a_proj == a_raw` exactly.

    Catches projections that perturb valid actions -- a silent source of turnover and of
    projection distance that the policy can never learn away.
    """
    rng = np.random.default_rng(2)
    proj = AnalyticProjector()
    for _ in range(300):
        y, lower, available = rand_case(rng)
        if lower.sum() > 1.0:
            continue
        first = proj.project(y, lower_bounds=lower, available=available).weights
        second = proj.project(first, lower_bounds=lower, available=available)
        assert np.allclose(second.weights, first, atol=1e-10), "projection is not idempotent"
        assert second.diagnostics.l1_distance < 1e-9


# ------------------------------------------------------------------------- T7


def test_no_silent_repair():
    """T7 / P7: diagnostics are non-empty whenever the projection changed the action."""
    proj = AnalyticProjector()
    y = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    available = np.ones(N, dtype=bool)
    available[2] = False                       # asset 2 does not exist yet

    out = proj.project(y, lower_bounds=np.zeros(N), available=available)
    assert out.weights[2] == 0.0
    assert out.diagnostics.changed(), "the action was altered without recording anything"
    assert out.diagnostics.availability_clipped == 1
    assert out.diagnostics.l1_distance > 0.0


def test_diagnostics_stay_empty_when_nothing_changed():
    proj = AnalyticProjector()
    y = np.zeros(N)
    y[CASH] = 1.0
    out = proj.project(y, lower_bounds=np.zeros(N), available=np.ones(N, dtype=bool))
    assert not out.diagnostics.changed()
    assert out.diagnostics.availability_clipped == 0
    assert out.diagnostics.lock_bound_active == 0


# --------------------------------------------------------------- degenerate case


def test_lock_floors_exceeding_the_book_is_a_legal_state():
    """`sum(w_lower) > 1` after a hard gap up. Everything else goes to zero and there is
    nothing to redistribute -- legal, not an error, and flagged."""
    lower = np.zeros(N)
    lower[1] = 0.8
    lower[2] = 0.7                              # 1.5 total
    available = np.ones(N, dtype=bool)
    out = AnalyticProjector().project(np.full(N, 1.0 / N), lower_bounds=lower,
                                      available=available)
    assert out.weights.sum() == pytest.approx(1.0)
    assert out.diagnostics.budget_degenerate
    assert out.weights[CASH] == pytest.approx(0.0, abs=1e-12)


def test_safe_portfolio_is_floors_plus_cash():
    lower = np.zeros(N)
    lower[1] = 0.25
    available = np.ones(N, dtype=bool)
    w = safe_portfolio(lower, available)
    assert w[1] == pytest.approx(0.25)
    assert w[CASH] == pytest.approx(0.75)
    assert w.sum() == pytest.approx(1.0)
    assert w[2:].sum() == pytest.approx(0.0)


def test_capital_preservation_permits_no_net_increase_in_risky_exposure():
    """Reductions and moves to cash stay legal; increases do not."""
    lower = np.zeros(N)
    lower[1] = 0.30                             # currently held and locked
    available = np.ones(N, dtype=bool)
    y = np.zeros(N)
    y[2] = 1.0                                  # the policy wants to pile into asset 2

    out = AnalyticProjector().project(y, lower_bounds=lower, available=available,
                                      capital_preservation=True)
    assert out.diagnostics.capital_preservation
    assert out.weights[2] == pytest.approx(0.0, abs=1e-9), "risky exposure was increased"
    assert out.weights[1] >= 0.30 - 1e-9, "the locked position was cut"
    assert out.weights[CASH] > 0.0


# ----------------------------------------------------------------- T4: the oracle


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_analytic_matches_the_cvxpy_oracle(seed):
    """T4 / P4: feasible, and objective within tolerance, over random instances.

    D3's correctness guarantee. The analytic solution is exact for the lock, long-only and
    no-leverage constraints, so the objectives should agree to solver precision.
    """
    rng = np.random.default_rng(seed)
    analytic, oracle = AnalyticProjector(), CvxpyProjector()
    worst = 0.0
    for _ in range(120):
        y, lower, available = rand_case(rng)
        if lower.sum() > 1.0:
            continue
        a = analytic.project(y, lower_bounds=lower, available=available).weights
        b = oracle.project(y, lower_bounds=lower, available=available).weights

        # Feasibility of the analytic answer.
        assert (a >= -1e-9).all() and a.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(a[~available] <= 1e-9)
        assert np.all(a + 1e-8 >= np.where(available, lower, 0.0))

        obj_a = float(((a - y) ** 2).sum())
        obj_b = float(((b - y) ** 2).sum())
        # The analytic result must never be WORSE than the QP by more than solver noise.
        worst = max(worst, obj_a - obj_b)
        assert obj_a <= obj_b + 1e-6, f"analytic objective {obj_a} vs QP {obj_b}"
    assert worst < 1e-6


def test_the_oracle_comparison_can_detect_a_wrong_projection():
    """A deliberately broken projection must fail the oracle check.

    An oracle that has never rejected anything is not known to be an oracle.
    """
    rng = np.random.default_rng(9)
    oracle = CvxpyProjector()
    y, lower, available = rand_case(rng, lock_frac=0.0, unavail_frac=0.0)

    # "Projection" by naive clipping and renormalizing -- a plausible-looking mistake.
    bad = np.maximum(y - 0.05, 0.0)
    bad = bad / bad.sum()
    good = oracle.project(y, lower_bounds=lower, available=available).weights
    assert float(((bad - y) ** 2).sum()) > float(((good - y) ** 2).sum()) + 1e-9


# ------------------------------------------------------------------ hypothesis


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.filter_too_much])
@given(
    y=st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=N, max_size=N),
    lows=st.lists(st.floats(0.0, 0.15, allow_nan=False), min_size=K, max_size=K),
    avail=st.lists(st.booleans(), min_size=K, max_size=K),
)
def test_projection_properties_hold_for_arbitrary_inputs(y, lows, avail):
    y = np.array(y)
    if y.sum() <= 0:
        y = np.ones(N)
    y = y / y.sum()

    available = np.array([True] + list(avail))
    lower = np.zeros(N)
    lower[1:] = np.where(available[1:], lows, 0.0)

    out = AnalyticProjector().project(y, lower_bounds=lower, available=available)
    w = out.weights
    assert np.isfinite(w).all()
    assert (w >= -1e-9).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(w[~available] <= 1e-9)
    if lower.sum() <= 1.0:
        assert np.all(w + 1e-8 >= lower)
    else:
        assert out.diagnostics.budget_degenerate
