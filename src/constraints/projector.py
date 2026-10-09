"""The feasibility projection.

Turns a policy's raw proposal into an executable target:

    a_raw -> availability mask -> long-only / no-leverage -> lock lower bounds
          -> risk envelope -> a_proj

Index 0 is CASH; indices 1..K are the tradable ETFs in canonical order. Cash is never
locked and never unavailable, which is what makes `w_safe` always reachable.

**Nothing is silently repaired.** Every layer records what it changed into
`ProjectionDiagnostics`, which is written to the trajectory and logged. A projection that
quietly fixes invalid actions hides both policy pathology and constraint-layer bugs.

Two backends behind one interface (D3): `analytic` is closed-form and fast enough to run
millions of times; `cvxpy` solves the same problem as a QP and is used as a **correctness
oracle in tests from day one**, long before it is used in training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol

import numpy as np

CASH = 0


@dataclass
class ProjectionDiagnostics:
    l1_distance: float = 0.0
    l2_distance: float = 0.0
    availability_clipped: int = 0
    lock_bound_active: int = 0
    risk_binding: bool = False
    de_risk_alpha: float = 1.0        # 1.0 = untouched, 0.0 = fully to the safe portfolio
    capital_preservation: bool = False
    infeasible_fallback: bool = False
    budget_degenerate: bool = False   # sum(w_lower) > 1 after a gap
    #: Over budget under the `no_risk_increase` rule (D-F): the action was held to "no
    #: riskier than the current holdings" instead of being replaced by `w_safe`.
    no_risk_increase: bool = False

    def changed(self) -> bool:
        return self.l1_distance > 1e-12

    def to_dict(self) -> dict:
        return {
            "l1_distance": self.l1_distance, "l2_distance": self.l2_distance,
            "availability_clipped": self.availability_clipped,
            "lock_bound_active": self.lock_bound_active,
            "risk_binding": self.risk_binding, "de_risk_alpha": self.de_risk_alpha,
            "capital_preservation": self.capital_preservation,
            "infeasible_fallback": self.infeasible_fallback,
            "budget_degenerate": self.budget_degenerate,
            "no_risk_increase": self.no_risk_increase,
        }


@dataclass
class ProjectedAction:
    weights: np.ndarray
    diagnostics: ProjectionDiagnostics = field(default_factory=ProjectionDiagnostics)


# --------------------------------------------------------------- simplex machinery


def project_onto_simplex(y: np.ndarray, budget: float = 1.0) -> np.ndarray:
    """Euclidean projection of `y` onto `{v >= 0, sum(v) = budget}`.

    The standard sort-and-threshold algorithm, exact in O(n log n).
    """
    if budget <= 0.0:
        return np.zeros_like(y)
    n = y.size
    u = np.sort(y)[::-1]
    css = np.cumsum(u) - budget
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    if not cond.any():
        # Every coordinate is pushed to the boundary; spread the budget uniformly.
        return np.full_like(y, budget / n)
    rho = int(np.nonzero(cond)[0][-1])
    theta = css[rho] / (rho + 1)
    return np.maximum(y - theta, 0.0)


def project_with_lower_bounds(
    y: np.ndarray, lower: np.ndarray, mask: np.ndarray, budget: float = 1.0,
    upper: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Projection onto the simplex with per-coordinate lower **and upper** bounds.

        minimize ||w - y||^2   s.t.  lower <= w <= upper,  sum(w) = budget,  w_i = 0 off mask

    With no upper bounds this is the scaled-simplex projection of the substitution
    `v_i = w_i - lower_i`. With them it is a box-constrained problem, which the sort-and-
    threshold shortcut does not solve -- the upper bounds matter for capital preservation,
    where every risky asset is capped at its current level and the residual must land in
    cash rather than being redistributed back into the assets that were just capped.

    Both cases are handled by the same dual bisection: `w_i(theta) = clip(y_i - theta,
    l_i, u_i)` is monotone non-increasing in `theta`, so there is a unique `theta` with
    `sum(w) = budget`. Exact to machine precision in ~60 iterations.

    Assets excluded by `mask` are **removed from the problem before solving**, not clipped
    after -- clipping after would leave the remaining weights un-renormalized.

    Returns `(weights, budget_degenerate)`.
    """
    w = np.zeros_like(y)
    if not mask.any():
        return w, False

    lo = np.where(mask, lower, 0.0)
    hi = np.where(mask, np.inf if upper is None else upper, 0.0)
    hi = np.maximum(hi, lo)                     # a box must not be empty
    total_lower = float(lo.sum())

    if total_lower > budget + 1e-12:
        # Locked positions gapped up hard enough that their floors exceed the whole book.
        # Everything else goes to zero and there is nothing to redistribute. A legal
        # state, not an error (reference/feasibility-projection.md section 3.4).
        w[mask] = lo[mask] * (budget / total_lower)
        return w, True

    idx = np.flatnonzero(mask)
    yi, li, ui = y[idx], lo[idx], hi[idx]

    if not np.isfinite(ui).all() or float(ui.sum()) >= budget - 1e-12:
        theta = _dual_theta(yi, li, ui, budget)
        w[idx] = np.clip(yi - theta, li, ui)
        # Residual from float error; push it into the coordinate with the most slack.
        gap = budget - float(w[idx].sum())
        if abs(gap) > 1e-12:
            slack = (ui - w[idx]) if gap > 0 else (w[idx] - li)
            j = int(np.argmax(slack))
            w[idx[j]] = float(np.clip(w[idx[j]] + gap, li[j], ui[j]))
        return w, False

    # Even every asset at its cap cannot absorb the budget. Degenerate, and flagged.
    w[idx] = ui
    return w, True


def _dual_theta(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, budget: float) -> float:
    """Find `theta` with `sum(clip(y - theta, lo, hi)) == budget`, by bisection."""
    span = float(np.max(np.abs(y)) + np.max(np.abs(lo)) +
                 (np.max(hi[np.isfinite(hi)]) if np.isfinite(hi).any() else 0.0) + 1.0)
    lo_t, hi_t = -span - budget, span + budget
    for _ in range(200):
        mid = 0.5 * (lo_t + hi_t)
        if float(np.clip(y - mid, lo, hi).sum()) > budget:
            lo_t = mid
        else:
            hi_t = mid
        if hi_t - lo_t < 1e-15 * max(1.0, span):
            break
    return 0.5 * (lo_t + hi_t)


# ------------------------------------------------------------------- the interface


class RiskModel(Protocol):
    """What the projector needs from the risk envelope, and nothing more."""

    def stress_loss(self, w: np.ndarray) -> float: ...
    def budget(self) -> float: ...


class FeasibilityProjector(Protocol):
    def project(
        self, raw_action: np.ndarray, *, lower_bounds: np.ndarray,
        available: np.ndarray, risk: RiskModel | None = None,
        capital_preservation: bool = False,
        current_weights: np.ndarray | None = None,
    ) -> ProjectedAction: ...


def capital_preservation_caps(
    lower: np.ndarray, current_weights: np.ndarray | None, mask: np.ndarray
) -> np.ndarray:
    """Upper bounds under capital preservation: no net increase in risky exposure.

    Every ETF is capped at its CURRENT weight, so the drawdown cannot be deepened by
    adding exposure. Reductions and moves to cash stay legal, and cash is uncapped. A
    locked position's cap can never fall below its floor -- the lock outranks capital
    preservation, and that is the case where the two constraints genuinely conflict
    (reference/risk-envelope.md section 2).
    """
    if current_weights is None:
        # Without the current book the only defensible cap is the lock floor itself:
        # hold what must be held, and put everything else in cash.
        caps = np.where(mask, lower, 0.0)
    else:
        caps = np.where(mask, np.maximum(np.asarray(current_weights, float), lower), 0.0)
    caps[CASH] = np.inf
    return caps


def safe_portfolio(lower_bounds: np.ndarray, available: np.ndarray) -> np.ndarray:
    """`w_safe`: every locked position at its floor, everything else in cash.

    The least-risk portfolio still reachable under the lock, and **always feasible by
    construction** -- which is what makes the fallback terminate.
    """
    w = np.zeros_like(lower_bounds)
    lo = np.where(available, lower_bounds, 0.0)
    lo[CASH] = 0.0
    total = float(lo.sum())
    if total > 1.0:
        return lo / total
    w[:] = lo
    w[CASH] = 1.0 - total
    return w


# ---------------------------------------------------------------------- analytic


@dataclass
class AnalyticProjector:
    """Closed-form projection plus a scalar de-risking scan (D3, v1)."""

    alpha_tolerance: float = 1e-3
    max_bisection_steps: int = 32
    #: See `ProjectionSpec.over_budget_rule`. `freeze` is the original behaviour.
    over_budget_rule: str = "freeze"

    def _uses_d_f(self, risk, current_weights) -> bool:
        """D-F needs a risk model to compare with, and the holdings to compare against."""
        return (self.over_budget_rule == "no_risk_increase" and risk is not None
                and current_weights is not None)

    def project(
        self, raw_action: np.ndarray, *, lower_bounds: np.ndarray,
        available: np.ndarray, risk: RiskModel | None = None,
        capital_preservation: bool = False,
        current_weights: np.ndarray | None = None,
    ) -> ProjectedAction:
        y = np.asarray(raw_action, dtype=float).copy()
        lower = np.asarray(lower_bounds, dtype=float).copy()
        mask = np.asarray(available, dtype=bool).copy()
        mask[CASH] = True            # cash always exists and is never locked
        lower[CASH] = 0.0
        lower = np.where(mask, lower, 0.0)

        diag = ProjectionDiagnostics()
        diag.availability_clipped = int(np.count_nonzero((~mask) & (y > 1e-12)))
        diag.lock_bound_active = int(np.count_nonzero(lower > 1e-12))

        # Under D-F, capital preservation is enforced by the risk rule (no action riskier
        # than the current holdings) rather than by capping every weight, which would also
        # forbid buying a hedge. The share-count guard against drift-buying stays, in
        # `engine.advance`.
        d_f = self._uses_d_f(risk, current_weights)
        upper = capital_preservation_caps(
            lower, current_weights, mask) if capital_preservation and not d_f else None
        if capital_preservation:
            diag.capital_preservation = True

        w, degenerate = project_with_lower_bounds(y, lower, mask, upper=upper)
        diag.budget_degenerate = degenerate

        if risk is not None:
            w = self._derisk(w, lower, mask, risk, diag,
                             current_weights=current_weights if d_f else None,
                             over_budget=capital_preservation)

        w = np.maximum(w, 0.0)
        total = w.sum()
        if total > 0:
            w = w / total

        diag.l1_distance = float(np.abs(w - raw_action).sum())
        diag.l2_distance = float(np.linalg.norm(w - raw_action))
        return ProjectedAction(weights=w, diagnostics=diag)

    def _derisk(self, w, lower, mask, risk: RiskModel, diag: ProjectionDiagnostics,
                current_weights: np.ndarray | None = None, over_budget: bool = False):
        """Blend toward `w_safe` until the stressed loss fits the budget.

        With `current_weights` given (the D-F rule), a state in which even `w_safe`
        breaches the budget -- or capital preservation -- is handled by
        `_no_risk_increase` instead of being frozen into `w_safe`.
        """
        budget = risk.budget()
        if current_weights is not None and over_budget:
            return self._no_risk_increase(w, mask, risk, diag, current_weights, budget)
        if risk.stress_loss(w) <= budget:
            return w

        diag.risk_binding = True
        w_safe = safe_portfolio(lower, mask)

        if current_weights is not None and risk.stress_loss(w_safe) > budget:
            return self._no_risk_increase(w, mask, risk, diag, current_weights, budget)

        if risk.stress_loss(w_safe) > budget:
            # Even the minimum-risk reachable portfolio breaches. That is a MARKET-FORCED
            # violation, not an agent choice: record it and continue. Terminating would
            # teach the policy that entering a risky state ends the game, which is
            # precisely the wrong lesson.
            diag.infeasible_fallback = True
            diag.de_risk_alpha = 0.0
            return w_safe

        # `stress_loss` is CONVEX along the segment (see RiskEnvelope.stress_loss), so
        # the feasible set is an interval, and `w_safe` being feasible puts 0 in it.
        # Bisection therefore finds its upper end. Note this needs convexity, NOT the
        # monotonicity the original spec assumed -- w_safe minimizes exposure, not risk.
        lo_a, hi_a = 0.0, 1.0
        for _ in range(self.max_bisection_steps):
            if hi_a - lo_a < self.alpha_tolerance:
                break
            mid = 0.5 * (lo_a + hi_a)
            if risk.stress_loss(mid * w + (1.0 - mid) * w_safe) <= budget:
                lo_a = mid
            else:
                hi_a = mid

        candidate = lo_a * w + (1.0 - lo_a) * w_safe
        if risk.stress_loss(candidate) > budget:
            # Cannot happen under a convex measure, and cheap to rule out. If a
            # non-convex estimator is ever wired in, this degrades to `w_safe` rather
            # than silently returning an infeasible action.
            diag.de_risk_alpha = 0.0
            return w_safe
        diag.de_risk_alpha = lo_a
        return candidate

    def _no_risk_increase(self, w, mask, risk: RiskModel, diag: ProjectionDiagnostics,
                          current_weights: np.ndarray, budget: float):
        """Redesign decision D-F: over budget, no executed action may add risk.

        The reference is the CURRENT holdings (doing nothing), which are reachable by
        construction: every locked position sits exactly at its floor in them. An action
        whose stressed loss is no larger than the reference's -- or within the budget, if
        that is looser -- is executed as proposed, so a hedge bought with cash goes
        through at whatever size the agent chose. A riskier action is scaled back along
        the segment toward the reference until it is not riskier. The layer never picks
        assets of its own: it is a floor, not a second strategy.

        The bisection is sound for the same reason as the `w_safe` one: `stress_loss` is
        convex along the segment, and the reference satisfies the bound.
        """
        ref = np.where(mask, np.maximum(np.asarray(current_weights, dtype=float), 0.0), 0.0)
        total = float(ref.sum())
        ref = ref / total if total > 0 else safe_portfolio(np.zeros_like(ref), mask)
        bound = max(budget, risk.stress_loss(ref))
        diag.no_risk_increase = True
        tol = 1e-12
        if risk.stress_loss(w) <= bound + tol:
            diag.de_risk_alpha = 1.0
            return w

        diag.risk_binding = True
        lo_a, hi_a = 0.0, 1.0
        for _ in range(self.max_bisection_steps):
            if hi_a - lo_a < self.alpha_tolerance:
                break
            mid = 0.5 * (lo_a + hi_a)
            if risk.stress_loss(mid * w + (1.0 - mid) * ref) <= bound + tol:
                lo_a = mid
            else:
                hi_a = mid
        candidate = lo_a * w + (1.0 - lo_a) * ref
        if risk.stress_loss(candidate) > bound + tol:
            diag.de_risk_alpha = 0.0
            return ref
        diag.de_risk_alpha = lo_a
        return candidate


# ------------------------------------------------------------------------- cvxpy


@dataclass
class CvxpyProjector:
    """The same problem as a QP. Slow; used as the analytic backend's correctness oracle.

    Only the hard portfolio constraints are expressed as a QP. The risk constraint is
    handled by the same `alpha` scan, because the stress estimators are sample quantiles
    rather than closed-form functions of `w`.
    """

    alpha_tolerance: float = 1e-3
    max_bisection_steps: int = 32
    solver: str | None = None
    over_budget_rule: str = "freeze"

    def project(
        self, raw_action: np.ndarray, *, lower_bounds: np.ndarray,
        available: np.ndarray, risk: RiskModel | None = None,
        capital_preservation: bool = False,
        current_weights: np.ndarray | None = None,
    ) -> ProjectedAction:
        import cvxpy as cp

        y = np.asarray(raw_action, dtype=float).copy()
        lower = np.asarray(lower_bounds, dtype=float).copy()
        mask = np.asarray(available, dtype=bool).copy()
        mask[CASH] = True
        lower[CASH] = 0.0
        lower = np.where(mask, lower, 0.0)

        diag = ProjectionDiagnostics()
        diag.availability_clipped = int(np.count_nonzero((~mask) & (y > 1e-12)))
        diag.lock_bound_active = int(np.count_nonzero(lower > 1e-12))

        analytic = AnalyticProjector(self.alpha_tolerance, self.max_bisection_steps,
                                     self.over_budget_rule)
        d_f = analytic._uses_d_f(risk, current_weights)
        upper = capital_preservation_caps(
            lower, current_weights, mask) if capital_preservation and not d_f else None
        if capital_preservation:
            diag.capital_preservation = True

        if float(lower.sum()) > 1.0 + 1e-12:
            w = lower / float(lower.sum())
            diag.budget_degenerate = True
        else:
            idx = np.flatnonzero(mask)
            v = cp.Variable(idx.size)
            objective = cp.Minimize(cp.sum_squares(v - y[idx]))
            constraints = [v >= lower[idx], cp.sum(v) == 1.0]
            if upper is not None:
                finite = np.isfinite(upper[idx])
                if finite.any():
                    constraints.append(v[finite] <= upper[idx][finite])
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=self.solver) if self.solver else problem.solve()
            if v.value is None:
                raise RuntimeError(f"cvxpy failed to solve: status={problem.status}")
            w = np.zeros_like(y)
            w[idx] = np.maximum(np.asarray(v.value).ravel(), 0.0)

        if risk is not None:
            w = analytic._derisk(w, lower, mask, risk, diag,
                                 current_weights=current_weights if d_f else None,
                                 over_budget=capital_preservation)

        w = np.maximum(w, 0.0)
        total = w.sum()
        if total > 0:
            w = w / total
        diag.l1_distance = float(np.abs(w - raw_action).sum())
        diag.l2_distance = float(np.linalg.norm(w - raw_action))
        return ProjectedAction(weights=w, diagnostics=diag)


def make_projector(backend: Literal["analytic", "cvxpy"] = "analytic",
                   alpha_tolerance: float = 1e-3,
                   over_budget_rule: str = "freeze") -> FeasibilityProjector:
    if over_budget_rule not in ("freeze", "no_risk_increase"):
        raise ValueError(f"unknown over_budget_rule: {over_budget_rule!r}")
    if backend == "analytic":
        return AnalyticProjector(alpha_tolerance=alpha_tolerance,
                                 over_budget_rule=over_budget_rule)
    if backend == "cvxpy":
        return CvxpyProjector(alpha_tolerance=alpha_tolerance,
                              over_budget_rule=over_budget_rule)
    raise ValueError(f"unknown projection backend: {backend!r}")
