"""The curriculum ladder and the gate between rungs.

The gate is the part that matters. It exists to stop a run rather than to describe it, so
the tests here are mostly about it refusing to pass things: a rung that did not beat cash,
and — more importantly — a rung that recorded a constraint violation, which means the
constraint layer is broken and every later rung would be trained against a simulator that
is not enforcing its own rules.
"""

from __future__ import annotations

import pytest

from src.training.curriculum import CURRICULUM, STAGES_BY_INDEX, evaluate_gate
from src.training.trainer import DEFAULT_PPO, ppo_kwargs


def _diag(episode_log_return: float = 0.05, **kw) -> dict:
    base = {"episode_log_return": episode_log_return, "n_episodes": 50,
            "safety_intervened": 0.1, "cash_weight": 0.2}
    base.update(kw)
    return base


# ------------------------------------------------------------------ the ladder


def test_the_ladder_is_five_rungs_in_order():
    assert [s.index for s in CURRICULUM] == [1, 2, 3, 4, 5]
    assert set(STAGES_BY_INDEX) == {1, 2, 3, 4, 5}


def test_each_rung_adds_exactly_one_source_of_difficulty():
    """The ordering rationale: a regression must localize to the thing just added."""
    by_index = STAGES_BY_INDEX
    # 1: no lock at all.
    assert by_index[1].env["hold_days_values"] == (0,)
    assert by_index[1].env["risk_enabled"] is False
    # 2: the lock appears, the ceiling is still fixed and the envelope still off.
    assert by_index[2].env["hold_days_values"] == (21, 30, 42)
    assert by_index[2].env["max_drawdown_values"] == (0.20,)
    assert by_index[2].env["risk_enabled"] is False
    # 3: the grid opens up (inherited, so not restated) and the envelope is still off.
    assert "hold_days_values" not in by_index[3].env
    assert by_index[3].env["risk_enabled"] is False
    # 4: the envelope, and only the envelope.
    assert by_index[4].env["risk_enabled"] is True
    assert by_index[4].env["flat_start"] is True
    assert by_index[4].env["episode_lengths"] == (252,)
    # 5: reservoir starts and sampled episode lengths.
    assert by_index[5].env["flat_start"] is False
    assert by_index[5].env["episode_lengths"] == (63, 126, 252, 504)


def test_only_the_rung_that_narrows_the_grid_restates_it():
    """Everything else inherits D16 from constraints.yaml, so the ladder lives in one
    place and cannot drift from what evaluation uses."""
    restating = [s.index for s in CURRICULUM if "hold_days_values" in s.env]
    assert restating == [1, 2]


def test_every_rung_has_a_config_file():
    from pathlib import Path

    for s in CURRICULUM:
        assert (Path("config/experiments") / s.config_name).exists(), s.config_name


# -------------------------------------------------------------------- the gate


def test_a_healthy_rung_passes():
    gate = evaluate_gate(1, _diag(0.05))
    assert gate.passed
    assert "passed" in gate.explain()


def test_a_rung_that_does_not_beat_cash_fails():
    gate = evaluate_gate(1, _diag(-0.01))
    assert not gate.passed
    assert "did not beat cash" in gate.explain()


def test_failing_rung_one_points_at_the_simulator_not_the_policy():
    """Stage 1 has no lock, no envelope and a fixed ceiling. There is nothing else it
    could be, and the message must say so rather than leaving it to be rediscovered."""
    assert "simulator" in evaluate_gate(1, _diag(-0.01)).explain()


def test_any_lock_violation_fails_the_gate_however_good_the_return():
    """A violation is not a weaker result, it is an invalid one."""
    gate = evaluate_gate(3, _diag(0.5), lock_violations=1)
    assert not gate.passed
    assert "lock violation" in gate.explain()


def test_any_feasibility_violation_fails_the_gate():
    gate = evaluate_gate(4, _diag(0.5), feasibility_violations=2)
    assert not gate.passed
    assert "feasibility violation" in gate.explain()


def test_the_gate_notes_a_small_episode_sample():
    gate = evaluate_gate(1, _diag(0.05, n_episodes=3))
    assert gate.passed
    assert any("small sample" in n for n in gate.notes)


def test_the_gate_notes_an_over_calibrated_envelope():
    gate = evaluate_gate(4, _diag(0.05, safety_intervened=0.97))
    assert any("over-calibrated" in n for n in gate.notes)


def test_the_gate_notes_a_collapsed_policy():
    """Cash pinned at 0 or 1 is the quiet failure: it looks stable and learns nothing."""
    assert any("collapsed" in n for n in evaluate_gate(3, _diag(cash_weight=0.999)).notes)
    assert any("collapsed" in n for n in evaluate_gate(3, _diag(cash_weight=0.0)).notes)


def test_the_gate_result_serializes_for_the_run_record():
    d = evaluate_gate(2, _diag(0.05)).to_dict()
    assert set(d) >= {"stage", "passed", "beats_cash", "lock_violations",
                      "feasibility_violations", "episode_log_return"}


# ------------------------------------------------------------ hyperparameters


def test_ppo_hyperparameters_come_from_config_and_reject_unknown_keys():
    """A typo in a hyperparameter name must not silently cost a training run."""
    cfg = ppo_kwargs({"ppo": {"gamma": 0.99, "learnign_rate": 1.0}})
    assert cfg["gamma"] == 0.99
    assert "learnign_rate" not in cfg
    assert cfg["learning_rate"] == DEFAULT_PPO["learning_rate"]


def test_the_defaults_match_the_specification():
    """reference/rl-training.md section 5. Q7 is open on gamma; until it is settled the
    code and the document must at least agree on what the value currently is."""
    assert DEFAULT_PPO["gamma"] == 0.999
    assert DEFAULT_PPO["n_steps"] == 2048
    assert DEFAULT_PPO["ent_coef"] == 0.005
    assert DEFAULT_PPO["target_kl"] == 0.02
