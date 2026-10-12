"""O-2: past the drawdown ceiling, risk must come down step by step.

MODEL_REDESIGN_PLAN.md section 14 step 1 / section 15 R-4. Under the no-risk-increase
rule (D-F) with `derisk_step = k`, a decision in capital preservation must bring the
stressed loss to at most `max(budget, stress(w_safe), (1 - k) * stress(holdings))`:

* a proposal already under that bound executes as proposed (the agent may cut faster);
* otherwise it is blended toward `w_safe` (locked positions plus cash) until it meets it;
* locked positions are never cut, so the bound never goes below `stress(w_safe)`;
* within the ceiling, and with `k = 0`, the rule is exactly plain D-F.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.constraints.projector import AnalyticProjector, CvxpyProjector, make_projector, safe_portfolio

# [CASH, EQUITY, BOND]; the same hand-checkable scenario maximum as the D-F tests.
SCENARIOS = np.array([[0.0, -0.30, 0.10],
                      [0.0, -0.05, -0.08]])


class ScenarioRisk:
    def __init__(self, budget: float):
        self._budget = budget

    def stress_loss(self, w) -> float:
        return float(max(0.0, (-(SCENARIOS @ np.asarray(w, dtype=float))).max()))

    def budget(self) -> float:
        return self._budget


MASK = np.array([True, True, True])
NO_LOCKS = np.zeros(3)
HOLDINGS = np.array([0.10, 0.90, 0.00])          # stress 0.27
O2 = AnalyticProjector(over_budget_rule="no_risk_increase", derisk_step=0.2)


def project(proj, raw, *, current=HOLDINGS, budget=-0.1, lower=NO_LOCKS, preservation=True):
    risk = ScenarioRisk(budget)
    out = proj.project(np.asarray(raw, float), lower_bounds=np.asarray(lower, float),
                       available=MASK, risk=risk, capital_preservation=preservation,
                       current_weights=np.asarray(current, float))
    return out, risk


def test_holding_still_past_the_ceiling_is_cut_by_the_step():
    out, risk = project(O2, HOLDINGS)
    assert out.diagnostics.derisk_forced
    assert risk.stress_loss(out.weights) <= 0.8 * risk.stress_loss(HOLDINGS) + 1e-9
    # and not much more than needed: the bisection stops at the bound
    assert risk.stress_loss(out.weights) >= 0.8 * risk.stress_loss(HOLDINGS) - 0.01


def test_a_riskier_proposal_is_also_brought_under_the_bound():
    out, risk = project(O2, [0.0, 1.0, 0.0])
    assert out.diagnostics.derisk_forced
    assert risk.stress_loss(out.weights) <= 0.8 * risk.stress_loss(HOLDINGS) + 1e-9


def test_a_proposal_that_already_cuts_enough_executes_as_proposed():
    hedge = np.array([0.0, 0.5, 0.5])            # stress 0.10, well under 0.216
    out, _ = project(O2, hedge)
    assert not out.diagnostics.derisk_forced
    assert np.allclose(out.weights, hedge)


def test_locked_positions_bound_the_cut_from_below():
    locked = np.array([0.0, 0.85, 0.0])          # w_safe stress 0.255 > 0.8 * 0.27
    out, risk = project(O2, HOLDINGS, lower=locked)
    assert out.weights[1] >= 0.85 - 1e-9          # the lock is never cut
    w_safe = safe_portfolio(locked, MASK)
    assert risk.stress_loss(out.weights) <= max(0.8 * 0.27, risk.stress_loss(w_safe)) + 1e-9


def test_within_the_ceiling_the_rule_is_plain_d_f():
    plain = AnalyticProjector(over_budget_rule="no_risk_increase")
    a, _ = project(O2, HOLDINGS, preservation=False, budget=0.05)
    b, _ = project(plain, HOLDINGS, preservation=False, budget=0.05)
    assert np.allclose(a.weights, b.weights)
    assert not a.diagnostics.derisk_forced


def test_zero_step_is_plain_d_f_past_the_ceiling_too():
    plain = AnalyticProjector(over_budget_rule="no_risk_increase")
    zero = AnalyticProjector(over_budget_rule="no_risk_increase", derisk_step=0.0)
    for raw in (HOLDINGS, [0.0, 1.0, 0.0], [0.3, 0.3, 0.4]):
        a, _ = project(zero, raw)
        b, _ = project(plain, raw)
        assert np.allclose(a.weights, b.weights)


@pytest.mark.parametrize("raw", [[0.1, 0.9, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [1.0, 0.0, 0.0]])
def test_the_cvxpy_oracle_agrees(raw):
    pytest.importorskip("cvxpy")
    cv = CvxpyProjector(over_budget_rule="no_risk_increase", derisk_step=0.2)
    a, _ = project(O2, raw)
    b, _ = project(cv, raw)
    assert np.allclose(a.weights, b.weights, atol=1e-6)


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        make_projector(derisk_step=0.2)                       # needs no_risk_increase
    with pytest.raises(ValueError):
        make_projector(over_budget_rule="no_risk_increase", derisk_step=1.0)
    assert make_projector(over_budget_rule="no_risk_increase", derisk_step=0.2).derisk_step == 0.2
