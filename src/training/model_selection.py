"""Lexicographic model selection: risk criteria first, return second.

reference/evaluation.md section 2. The rule is deliberately *not* "maximize return":

    Level 1: keep only models satisfying the validation risk criteria
    Level 2: among those, maximize cumulative log return

    If NO model satisfies the risk criteria:
        select minimum D_max violation, then highest return
        mark the fold: "constraint validation failure"

Selecting a high-return model that violates the risk limit is prohibited outright, and the
prohibition has to live in code rather than in a reviewer's judgment -- it is the exact
trade every backtest is tempted to make, and the temptation is strongest when the
violating model is also the best-looking one.

A fold marked `constraint validation failure` stays marked all the way into the final
report. Stage 12 will not quietly drop it, and a report containing such folds must say so
in its headline.

Everything here is scored on the **validation** year. Tuning against test performance is
prohibited and is what the walk-forward test suite checks for, so no function in this
module accepts a test metric at all -- the rule is enforced by the signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    """One hyperparameter setting, with the metrics it earned on the validation year."""

    name: str
    params: dict = field(default_factory=dict)
    #: Validation metrics. Keys mirror `src/evaluation/metrics.py`.
    validation: dict = field(default_factory=dict)
    #: Filled in by `select`: which level eliminated it, and why.
    eliminated_at: int | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "params": self.params,
                "validation": self.validation,
                "eliminated_at": self.eliminated_at, "reason": self.reason}


@dataclass
class SelectionResult:
    chosen: Candidate | None
    candidates: list[Candidate]
    constraint_validation_failure: bool
    criterion: str
    fold_id: str = ""

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "chosen": self.chosen.name if self.chosen else None,
            "criterion": self.criterion,
            "constraint_validation_failure": self.constraint_validation_failure,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def risk_criteria_failures(metrics: dict, *, d_max: float,
                           allow_market_forced: bool = True) -> list[str]:
    """Level 1. Every reason this candidate is not admissible.

    The three criteria from evaluation.md section 2:

    * realized max drawdown <= D_max for each evaluated cell, **or** a breach classified
      as market-forced;
    * zero preventable violations;
    * zero lock violations, zero feasibility violations.
    """
    failures: list[str] = []

    if int(metrics.get("lock_violations", 0)):
        failures.append(f"{metrics['lock_violations']} lock violation(s)")
    if int(metrics.get("feasibility_violations", 0)):
        failures.append(f"{metrics['feasibility_violations']} feasibility violation(s)")
    if int(metrics.get("preventable_violations", 0)):
        failures.append(
            f"{metrics['preventable_violations']} PREVENTABLE D_max violation(s) -- an "
            "implementation failure, not a risk outcome")

    realized = float(metrics.get("max_drawdown", 0.0))
    if realized > d_max + 1e-9:
        # A breach the market forced is legitimate: the ceiling constrains the ACTION,
        # never the realized path. A breach with no such classification is not.
        forced = bool(metrics.get("all_breaches_market_forced", False))
        if not (allow_market_forced and forced):
            failures.append(
                f"realized drawdown {realized:.4f} exceeds D_max {d_max:.4f} and is not "
                "classified as market-forced")
    return failures


def select(candidates: list[Candidate], *, d_max: float, fold_id: str = "",
           allow_market_forced: bool = True) -> SelectionResult:
    """Apply the rule. Never returns a Level-1 failure unless *every* candidate fails."""
    if not candidates:
        raise ValueError("no candidates to select from")

    admissible: list[Candidate] = []
    for c in candidates:
        failures = risk_criteria_failures(c.validation, d_max=d_max,
                                          allow_market_forced=allow_market_forced)
        if failures:
            c.eliminated_at = 1
            c.reason = "; ".join(failures)
        else:
            admissible.append(c)

    if admissible:
        best = max(admissible, key=lambda c: float(
            c.validation.get("cumulative_log_return",
                             c.validation.get("cumulative_return", float("-inf")))))
        for c in admissible:
            if c is not best:
                c.eliminated_at = 2
                c.reason = "admissible, but not the highest validation return"
        return SelectionResult(
            chosen=best, candidates=candidates, constraint_validation_failure=False,
            fold_id=fold_id,
            criterion=("Level 1: risk criteria satisfied; Level 2: highest validation "
                       "cumulative log return"))

    # Nothing was admissible. Fall back, and MARK the fold -- this travels to the report.
    def breach_depth(c: Candidate) -> float:
        return max(0.0, float(c.validation.get("max_drawdown", 0.0)) - d_max)

    fallback = min(
        candidates,
        key=lambda c: (breach_depth(c),
                       -float(c.validation.get("cumulative_log_return",
                              c.validation.get("cumulative_return", 0.0)))))
    fallback.eliminated_at = None
    fallback.reason = ("selected under CONSTRAINT VALIDATION FAILURE: no candidate met "
                       "the risk criteria, so the smallest D_max violation was taken")
    return SelectionResult(
        chosen=fallback, candidates=candidates, constraint_validation_failure=True,
        fold_id=fold_id,
        criterion=("CONSTRAINT VALIDATION FAILURE: no candidate satisfied the Level 1 "
                   "risk criteria; selected minimum D_max violation, then highest return"))


def default_candidates(n: int, base_seed: int = 0) -> list[Candidate]:
    """The default hyperparameter sweep: `evaluation.md` section 6 sizes it at 4.

    Deliberately a small, boring spread over the two knobs most likely to matter here --
    entropy (a softmax over 25 assets collapses readily) and the learning rate. This is a
    sweep, not a search: the point of walk-forward is the protocol, and a large search per
    fold would multiply 13 folds of training by whatever it costs.
    """
    grid = [
        {"name": "base", "ent_coef": 0.005, "learning_rate": 3.0e-4},
        {"name": "high_entropy", "ent_coef": 0.02, "learning_rate": 3.0e-4},
        {"name": "slow_lr", "ent_coef": 0.005, "learning_rate": 1.0e-4},
        {"name": "high_entropy_slow", "ent_coef": 0.02, "learning_rate": 1.0e-4},
        {"name": "fast_lr", "ent_coef": 0.005, "learning_rate": 1.0e-3},
        {"name": "low_entropy", "ent_coef": 0.001, "learning_rate": 3.0e-4},
    ]
    out = []
    for i, spec in enumerate(grid[:max(1, n)]):
        params = {k: v for k, v in spec.items() if k != "name"}
        params["seed"] = base_seed + i
        out.append(Candidate(name=spec["name"], params=params))
    return out
