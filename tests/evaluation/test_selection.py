"""T12 -- selection is lexicographic and never picks a risk-violating model at Level 1.

This is the test guarding the exact trade every backtest is tempted to make. The
temptation is strongest precisely when the violating model is also the best-looking one,
so most of these tests hand the selector a candidate that wins on return and loses on risk,
and assert it is not chosen.
"""

from __future__ import annotations

import pytest

from src.training.model_selection import (
    Candidate,
    default_candidates,
    risk_criteria_failures,
    select,
)

D_MAX = 0.15


def cand(name: str, *, ret: float, dd: float = 0.05, lock: int = 0, feas: int = 0,
         preventable: int = 0, forced: bool = False) -> Candidate:
    return Candidate(name=name, params={"seed": 0}, validation={
        "cumulative_log_return": ret, "cumulative_return": ret,
        "max_drawdown": dd, "lock_violations": lock,
        "feasibility_violations": feas, "preventable_violations": preventable,
        "all_breaches_market_forced": forced})


# ------------------------------------------------------------- the Level 1 rule


def test_a_clean_candidate_has_no_failures():
    assert risk_criteria_failures(cand("ok", ret=0.1).validation, d_max=D_MAX) == []


def test_each_hard_criterion_is_detected():
    for kw, needle in (({"lock": 1}, "lock violation"),
                       ({"feas": 1}, "feasibility violation"),
                       ({"preventable": 1}, "PREVENTABLE"),
                       ({"dd": 0.30}, "exceeds D_max")):
        failures = risk_criteria_failures(cand("x", ret=0.1, **kw).validation, d_max=D_MAX)
        assert any(needle in f for f in failures), (kw, failures)


def test_a_market_forced_breach_is_admissible_but_an_unclassified_one_is_not():
    """`D_max` constrains the action, never the realized path -- so a breach the market
    forced is legitimate. A breach with no such classification is not."""
    forced = cand("forced", ret=0.1, dd=0.30, forced=True)
    assert risk_criteria_failures(forced.validation, d_max=D_MAX) == []
    unclassified = cand("unclassified", ret=0.1, dd=0.30, forced=False)
    assert risk_criteria_failures(unclassified.validation, d_max=D_MAX)


def test_a_preventable_violation_is_never_excused_by_market_forcing():
    """Preventable is an implementation failure. It is fixed, not classified away."""
    both = cand("both", ret=0.9, dd=0.30, preventable=1, forced=True)
    assert any("PREVENTABLE" in f
               for f in risk_criteria_failures(both.validation, d_max=D_MAX))


# --------------------------------------------------------------- the selection


def test_the_highest_return_model_is_rejected_when_it_violates_risk():
    """The whole point of T12."""
    result = select([cand("greedy", ret=0.90, dd=0.40),
                     cand("safe", ret=0.10, dd=0.05)], d_max=D_MAX)
    assert result.chosen.name == "safe"
    assert not result.constraint_validation_failure
    greedy = next(c for c in result.candidates if c.name == "greedy")
    assert greedy.eliminated_at == 1
    assert "exceeds D_max" in greedy.reason


def test_return_only_breaks_ties_among_the_admissible():
    result = select([cand("a", ret=0.10), cand("b", ret=0.30), cand("c", ret=0.20)],
                    d_max=D_MAX)
    assert result.chosen.name == "b"
    assert all(c.eliminated_at == 2 for c in result.candidates if c.name != "b")
    assert "Level 1" in result.criterion and "Level 2" in result.criterion


def test_a_lock_violation_disqualifies_however_good_the_return():
    result = select([cand("broken", ret=99.0, lock=1), cand("fine", ret=0.01)],
                    d_max=D_MAX)
    assert result.chosen.name == "fine"


def test_when_nothing_is_admissible_the_fold_is_marked():
    """A fold marked `constraint validation failure` stays marked all the way into the
    final report -- Stage 12 will not quietly drop it."""
    result = select([cand("bad_a", ret=0.5, dd=0.40),
                     cand("bad_b", ret=0.1, dd=0.20)], d_max=D_MAX)
    assert result.constraint_validation_failure
    assert result.chosen.name == "bad_b", "should take the SMALLEST D_max violation"
    assert "CONSTRAINT VALIDATION FAILURE" in result.criterion
    assert "CONSTRAINT VALIDATION FAILURE" in result.chosen.reason


def test_the_fallback_breaks_ties_on_return_after_breach_depth():
    result = select([cand("shallow_low", ret=0.10, dd=0.20),
                     cand("shallow_high", ret=0.50, dd=0.20),
                     cand("deep", ret=9.00, dd=0.60)], d_max=D_MAX)
    assert result.constraint_validation_failure
    assert result.chosen.name == "shallow_high"


def test_the_selection_record_names_every_candidate_and_its_fate():
    """The record is a required artifact, not a convenience: "no tuning on test" is the
    easiest prohibition to violate accidentally and the hardest to detect afterwards."""
    result = select([cand("a", ret=0.1), cand("b", ret=0.2, dd=0.9)],
                    d_max=D_MAX, fold_id="fold_2015")
    record = result.to_dict()
    assert record["fold_id"] == "fold_2015"
    assert record["chosen"] == "a"
    assert {c["name"] for c in record["candidates"]} == {"a", "b"}
    assert all("validation" in c for c in record["candidates"])
    assert record["candidates"][1]["eliminated_at"] == 1


def test_selection_never_sees_a_test_metric():
    """Enforced by the signature: there is no parameter to pass one in through."""
    import inspect

    params = set(inspect.signature(select).parameters)
    assert not any("test" in p for p in params), params
    assert set(inspect.signature(risk_criteria_failures).parameters) == {
        "metrics", "d_max", "allow_market_forced"}


def test_an_empty_candidate_list_is_an_error_not_a_none():
    with pytest.raises(ValueError):
        select([], d_max=D_MAX)


# -------------------------------------------------------------- the candidates


def test_the_default_sweep_is_distinct_and_sized_by_config():
    candidates = default_candidates(4)
    assert len(candidates) == 4
    assert len({c.name for c in candidates}) == 4
    assert len({c.params["seed"] for c in candidates}) == 4
    # It varies the two knobs most likely to matter: entropy and the learning rate.
    assert len({c.params["ent_coef"] for c in candidates}) > 1
    assert len({c.params["learning_rate"] for c in candidates}) > 1
