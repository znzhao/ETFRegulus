"""Redesign decision D-F: over budget, the risk layer forbids adding risk -- nothing more.

MODEL_REDESIGN_PLAN.md section 5d. The rule replaces "freeze into the safe portfolio",
which also forbade hedging. Three things must hold, and each has a test here:

* a proposal that lowers risk relative to the CURRENT holdings is executed as proposed --
  the size of a hedge is the agent's choice;
* a riskier proposal is scaled back toward the current holdings until it is not riskier;
* no executed action is ever riskier than doing nothing (or the budget, if looser).

The risk model is a hand-built scenario maximum: convex in `w`, like the real CVaR
envelope, and small enough that every expected answer can be checked by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.constraints.projector import (
    AnalyticProjector,
    CvxpyProjector,
    make_projector,
    safe_portfolio,
)

# Assets: [CASH, EQUITY, BOND].
#   scenario 0, an equity crash:  equity -30%, bond +10%
#   scenario 1, a rates shock:    equity -5%,  bond -8%
SCENARIOS = np.array([[0.0, -0.30, 0.10],
                      [0.0, -0.05, -0.08]])


class ScenarioRisk:
    """Worst scenario loss. A maximum of linear functions, so convex in `w`."""

    def __init__(self, budget: float):
        self._budget = budget

    def stress_loss(self, w: np.ndarray) -> float:
        return float(max(0.0, (-(SCENARIOS @ np.asarray(w, dtype=float))).max()))

    def budget(self) -> float:
        return self._budget


MASK = np.array([True, True, True])
NO_LOCKS = np.zeros(3)


def project(proj, raw, *, current, budget, lower=NO_LOCKS, preservation=False):
    risk = ScenarioRisk(budget)
    out = proj.project(np.asarray(raw, float), lower_bounds=np.asarray(lower, float),
                       available=MASK, risk=risk, capital_preservation=preservation,
                       current_weights=np.asarray(current, float))
    return out, risk


D_F = AnalyticProjector(over_budget_rule="no_risk_increase")

# 60% equity, LOCKED: its stress loss alone (0.18) is far above a 5% budget.
CURRENT = np.array([0.40, 0.60, 0.00])
LOCKED = np.array([0.0, 0.60, 0.0])


def test_the_default_rule_still_freezes_into_the_safe_portfolio():
    """`freeze` is the original behaviour and must be untouched by D-F."""
    out, _ = project(AnalyticProjector(), [0.0, 0.6, 0.4], current=CURRENT,
                     budget=0.05, lower=LOCKED)
    assert out.diagnostics.infeasible_fallback
    assert not out.diagnostics.no_risk_increase
    assert np.allclose(out.weights, safe_portfolio(LOCKED, MASK))


def test_a_hedge_is_executed_exactly_as_proposed():
    """Locked equity plus 40% cash breaches; moving the cash into bonds hedges the crash.
    The rule must let the hedge through at the size the agent chose."""
    hedge = np.array([0.0, 0.60, 0.40])
    out, risk = project(D_F, hedge, current=CURRENT, budget=0.05, lower=LOCKED)
    assert out.diagnostics.no_risk_increase
    assert not out.diagnostics.infeasible_fallback
    assert np.allclose(out.weights, hedge)
    assert risk.stress_loss(out.weights) < risk.stress_loss(CURRENT)


def test_a_partial_hedge_is_also_executed_as_proposed():
    partial = np.array([0.25, 0.60, 0.15])
    out, _ = project(D_F, partial, current=CURRENT, budget=0.05, lower=LOCKED)
    assert np.allclose(out.weights, partial)


def test_adding_risk_is_scaled_back_to_no_riskier_than_doing_nothing():
    """Using the cash to buy MORE equity raises the crash loss: it is scaled back toward
    the current holdings until it is not riskier than them."""
    riskier = np.array([0.0, 1.0, 0.0])
    out, risk = project(D_F, riskier, current=CURRENT, budget=0.05, lower=LOCKED)
    assert out.diagnostics.no_risk_increase and out.diagnostics.risk_binding
    assert risk.stress_loss(out.weights) <= risk.stress_loss(CURRENT) + 1e-9
    assert out.diagnostics.de_risk_alpha < 1e-2           # essentially nothing added
    assert out.weights[1] >= LOCKED[1] - 1e-12             # the lock floor holds


def test_a_mixed_proposal_is_shrunk_as_a_whole_never_edited():
    """Hedge plus extra equity: the layer scales the WHOLE proposal toward the holdings
    rather than keeping the hedge and dropping the equity -- it never chooses for the agent."""
    current = np.array([0.40, 0.60, 0.00])
    mixed = np.array([0.0, 0.80, 0.20])
    out, risk = project(D_F, mixed, current=current, budget=0.05, lower=LOCKED)
    a = out.diagnostics.de_risk_alpha
    assert 0.0 <= a < 1.0
    assert np.allclose(out.weights, a * mixed + (1 - a) * current, atol=1e-9)
    assert risk.stress_loss(out.weights) <= risk.stress_loss(current) + 1e-9


def test_capital_preservation_allows_hedging_under_d_f():
    """In capital preservation the old rule capped every weight, which also forbade buying
    a hedge. Under D-F the hedge goes through; adding equity still does not."""
    current = np.array([0.40, 0.60, 0.00])
    out, _ = project(D_F, [0.0, 0.60, 0.40], current=current, budget=-0.02,
                     lower=LOCKED, preservation=True)
    assert out.diagnostics.capital_preservation and out.diagnostics.no_risk_increase
    assert out.weights[2] == pytest.approx(0.40)

    frozen, _ = project(AnalyticProjector(), [0.0, 0.60, 0.40], current=current,
                        budget=-0.02, lower=LOCKED, preservation=True)
    assert frozen.weights[2] == pytest.approx(0.0)        # the old rule: no hedge allowed


def test_within_budget_nothing_changes():
    """The rule only acts over budget. A proposal inside the budget is untouched."""
    calm = np.array([0.5, 0.1, 0.4])
    out, _ = project(D_F, calm, current=np.array([1.0, 0.0, 0.0]), budget=0.10)
    assert np.allclose(out.weights, calm)
    assert not out.diagnostics.no_risk_increase


@pytest.mark.parametrize("seed", range(40))
def test_no_executed_action_is_ever_riskier_than_doing_nothing(seed):
    """The guarantee, over random holdings, locks, budgets and proposals."""
    rng = np.random.default_rng(seed)
    current = rng.dirichlet(np.ones(3))
    lower = np.zeros(3)
    lower[1:] = current[1:] * rng.integers(0, 2, size=2)     # some positions locked
    raw = rng.dirichlet(np.ones(3) * 0.5)
    budget = float(rng.uniform(-0.05, 0.08))
    preservation = budget < 0
    out, risk = project(D_F, raw, current=current, budget=budget, lower=lower,
                        preservation=preservation)
    w = out.weights
    assert w.min() >= -1e-12 and w.sum() == pytest.approx(1.0)
    assert np.all(w >= lower - 1e-9), "a lock floor was breached"
    bound = max(budget, risk.stress_loss(current))
    if out.diagnostics.no_risk_increase:
        assert risk.stress_loss(w) <= bound + 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_the_cvxpy_oracle_agrees_under_d_f(seed):
    rng = np.random.default_rng(100 + seed)
    current = rng.dirichlet(np.ones(3))
    lower = np.zeros(3)
    lower[1] = current[1]
    raw = rng.dirichlet(np.ones(3))
    budget = float(rng.uniform(-0.02, 0.06))
    a, _ = project(D_F, raw, current=current, budget=budget, lower=lower,
                   preservation=budget < 0)
    c, _ = project(CvxpyProjector(over_budget_rule="no_risk_increase"), raw,
                   current=current, budget=budget, lower=lower, preservation=budget < 0)
    assert np.allclose(a.weights, c.weights, atol=1e-5)
    assert a.diagnostics.no_risk_increase == c.diagnostics.no_risk_increase


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="over_budget_rule"):
        make_projector("analytic", over_budget_rule="sunk")


# --------------------------------------------- the replay detector understands D-F


def _rows(grew_raised: bool):
    import pandas as pd

    universe = ["EQ", "BD"]
    idx = pd.to_datetime(["2020-03-02", "2020-03-03"])
    base = {"w_EQ": 0.6, "w_BD": 0.0, "shares_EQ": 100.0, "shares_BD": 0.0,
            "locked_EQ": True, "locked_BD": False, "capital_preservation": False,
            "no_risk_increase": False, "proj_weights": np.array([0.4, 0.6, 0.0])}
    second = dict(base, shares_BD=50.0, w_BD=0.3, no_risk_increase=True,
                  proj_weights=np.array([0.1, 0.6, 0.3]) if grew_raised
                  else np.array([0.4, 0.6, 0.0]))
    return pd.DataFrame([base, second], index=idx), universe


def test_the_detector_accepts_a_deliberate_hedge_under_d_f():
    from src.evaluation.violations import replay_feasibility

    traj, universe = _rows(grew_raised=True)
    findings = replay_feasibility(traj, universe, d_max=0.05)
    assert not [f for f in findings if f.kind == "preservation_cap_breached"]


def test_the_detector_flags_growth_the_decision_did_not_ask_for():
    """Drift-buying: shares grew although the projected weight did not rise."""
    from src.evaluation.violations import replay_feasibility

    traj, universe = _rows(grew_raised=False)
    findings = replay_feasibility(traj, universe, d_max=0.05)
    assert [f for f in findings if f.kind == "preservation_cap_breached"]
