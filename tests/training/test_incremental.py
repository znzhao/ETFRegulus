"""The incremental-training campaign (INCREMENTAL_TRAINING_PLAN.md).

The property everything else rests on is the slow test at the bottom: **stopping and
resuming produces exactly the weights an uninterrupted run would have**. Without it, a
learning curve built over many short sessions would measure the restarts, not the budget.
The fast tests pin the schedule, the crash recovery, and the session planner.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.training.incremental import (
    COMPLETE,
    CampaignError,
    check_config,
    checkpoint_schedule,
    chunk_seed,
    next_target,
    pending_checkpoint,
    plan_session,
    reconcile,
    recover_state,
    write_state,
)

ROLLOUT = 2048 * 8


# ------------------------------------------------------------------- schedule


def test_checkpoints_are_whole_rollouts_and_c0_is_the_existing_2pct_run():
    """The 2% walk-forward asked for 40,000 timesteps and SB3 ran 49,152 = 3 rollouts.
    C0 must be exactly that, or the existing models are not a valid starting point."""
    sched = checkpoint_schedule(2_000_000, ROLLOUT)
    assert [c["rollouts"] for c in sched] == [3, 5, 10, 20, 40, 79, 123]
    assert sched[0]["timesteps"] == 49_152
    assert [c["pct"] for c in sched] == [2, 4, 8, 16, 32, 64, 100]
    assert all(c["timesteps"] % ROLLOUT == 0 for c in sched)


def test_chunk_seeds_never_collide_across_candidates_or_rollouts():
    seeds = {chunk_seed(s, r) for s in (42, 43, 44, 45) for r in range(200)}
    assert len(seeds) == 4 * 200


def _campaign(rollouts: dict[str, int], reported=(), snapshots=None) -> dict:
    sched = checkpoint_schedule(2_000_000, ROLLOUT)
    cands = {}
    for key, r in rollouts.items():
        snaps = {c["label"]: {"n_updates": 0, "diag": {}} for c in sched
                 if c["rollouts"] <= r} if snapshots is None else snapshots.get(key, {})
        cands[key] = {"test_year": int(key.split("/")[0]), "name": key.split("/")[1],
                      "rollouts": r, "seed": 42, "snapshots": snaps}
    return {"schedule": sched, "candidates": cands, "rollout_size": ROLLOUT,
            "total_timesteps": 2_000_000,
            "checkpoints": {lab: {"status": "reported"} for lab in reported}}


def test_no_candidate_runs_ahead_of_the_next_checkpoint():
    """Plan P4: everyone reaches C1 before anyone trains toward C2."""
    camp = _campaign({"2012/a": 5, "2012/b": 3, "2013/a": 4})
    assert next_target(camp)["label"] == "C1"
    camp = _campaign({"2012/a": 5, "2012/b": 5})
    assert next_target(camp)["label"] == "C2"
    camp = _campaign({"2012/a": 123})
    assert next_target(camp) is None


def test_a_checkpoint_is_pending_until_it_is_reported():
    camp = _campaign({"2012/a": 5, "2012/b": 5}, reported=("C0",))
    assert pending_checkpoint(camp)["label"] == "C1"
    camp = _campaign({"2012/a": 5, "2012/b": 4}, reported=("C0",))
    assert pending_checkpoint(camp) is None
    camp = _campaign({"2012/a": 3})
    assert pending_checkpoint(camp)["label"] == "C0"


def test_a_config_change_is_refused():
    with pytest.raises(CampaignError, match="same training setup"):
        check_config({"config_hash": "aaaa"}, "bbbb")
    check_config({"config_hash": "aaaa"}, "aaaa")
    check_config({"config_hash": "aaaa"}, "bbbb", allow_change=True)


# ------------------------------------------------------------- crash recovery


class _FakeModel:
    def __init__(self, timesteps: int):
        self.num_timesteps = timesteps

    def save(self, path):
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("data", json.dumps({"num_timesteps": self.num_timesteps,
                                           "_n_updates": self.num_timesteps // ROLLOUT}))


class _FakeVec:
    def save(self, path):
        Path(path).write_bytes(b"normalizer")


def test_write_state_replaces_the_previous_state(tmp_path: Path):
    state = tmp_path / "current"
    write_state(_FakeModel(ROLLOUT * 3), _FakeVec(), state)
    write_state(_FakeModel(ROLLOUT * 4), _FakeVec(), state)
    assert recover_state(state) == ROLLOUT * 4
    assert not state.with_name("current.tmp").exists()
    assert not state.with_name("current.old").exists()


def test_a_crash_mid_save_keeps_the_previous_complete_state(tmp_path: Path):
    state = tmp_path / "current"
    write_state(_FakeModel(ROLLOUT * 3), _FakeVec(), state)
    half = state.with_name("current.tmp")
    half.mkdir()
    (half / "policy.zip").write_bytes(b"truncated")       # no COMPLETE marker
    assert recover_state(state) == ROLLOUT * 3
    assert not half.exists()


def test_a_crash_between_the_renames_recovers_the_newer_state(tmp_path: Path):
    """The current state was moved aside and the new one not yet moved in."""
    state = tmp_path / "current"
    write_state(_FakeModel(ROLLOUT * 3), _FakeVec(), state)
    state.rename(state.with_name("current.old"))
    newer = state.with_name("current.tmp")
    write_state(_FakeModel(ROLLOUT * 4), _FakeVec(), newer)   # a complete dir named .tmp
    assert recover_state(state) == ROLLOUT * 4
    assert (state / COMPLETE).exists()


def test_only_the_old_state_survives(tmp_path: Path):
    state = tmp_path / "current"
    write_state(_FakeModel(ROLLOUT * 3), _FakeVec(), state)
    state.rename(state.with_name("current.old"))
    assert recover_state(state) == ROLLOUT * 3


def test_no_usable_state_is_an_error_not_a_silent_restart(tmp_path: Path):
    with pytest.raises(CampaignError, match="no usable saved state"):
        recover_state(tmp_path / "current")


def test_reconcile_trusts_the_saved_state_over_the_campaign_file(tmp_path: Path):
    """A kill between saving a rollout and updating campaign.json: disk is ahead by one."""
    camp = _campaign({"2012/base": 3})
    mdir = tmp_path / "models" / "2012" / "base"
    write_state(_FakeModel(ROLLOUT * 4), _FakeVec(), mdir / "current")
    notes = reconcile(tmp_path, camp, ROLLOUT)
    assert camp["candidates"]["2012/base"]["rollouts"] == 4
    assert any("using the saved state" in n for n in notes)


def test_reconcile_recovers_a_checkpoint_snapshot_the_crash_skipped(tmp_path: Path):
    camp = _campaign({"2012/base": 3})
    mdir = tmp_path / "models" / "2012" / "base"
    write_state(_FakeModel(ROLLOUT * 5), _FakeVec(), mdir / "current")
    reconcile(tmp_path, camp, ROLLOUT)
    assert "C1" in camp["candidates"]["2012/base"]["snapshots"]
    assert (mdir / "C1" / COMPLETE).exists()


# ------------------------------------------------------ inserting a checkpoint


def _on_disk(tmp_path: Path, camp: dict) -> None:
    """Write current state + a snapshot dir per recorded snapshot, like a real campaign."""
    by_label = {c["label"]: c["rollouts"] for c in camp["schedule"]}
    for key, cand in camp["candidates"].items():
        mdir = tmp_path / "models" / key
        for lab in cand["snapshots"]:
            write_state(_FakeModel(ROLLOUT * by_label[lab]), _FakeVec(), mdir / lab)
        write_state(_FakeModel(ROLLOUT * cand["rollouts"]), _FakeVec(), mdir / "current")


def test_a_checkpoint_inserted_mid_campaign_rolls_back_whoever_passed_it(tmp_path: Path):
    from src.training.incremental import insert_checkpoint

    camp = _campaign({"2012/base": 20, "2013/base": 10})
    camp["checkpoints"] = {"C0": {"status": "reported"}, "C1": {"status": "reported"},
                           "C2": {"status": "reported"}}
    _on_disk(tmp_path, camp)
    insert_checkpoint(tmp_path, camp, 12)

    labels = [(c["label"], c["pct"], c["rollouts"]) for c in camp["schedule"]]
    assert labels[2:5] == [("C2", 8, 10), ("C2b", 12, 15), ("C3", 16, 20)]
    ahead = camp["candidates"]["2012/base"]
    assert ahead["rollouts"] == 10
    assert "C3" not in ahead["snapshots"]
    assert not (tmp_path / "models" / "2012" / "base" / "C3").exists()
    assert recover_state(tmp_path / "models" / "2012" / "base" / "current") == ROLLOUT * 10
    assert camp["candidates"]["2013/base"]["rollouts"] == 10
    assert next_target(camp)["label"] == "C2b"
    # The rolled-back state is consistent with what is on disk.
    assert reconcile(tmp_path, camp, ROLLOUT) == []


def test_a_checkpoint_cannot_be_inserted_before_one_already_evaluated():
    from src.training.incremental import insert_checkpoint

    camp = _campaign({"2012/base": 20})
    camp["checkpoints"] = {"C3": {"status": "evaluating"}}
    with pytest.raises(CampaignError, match="cannot be inserted"):
        insert_checkpoint(Path("."), camp, 12)


def test_a_duplicate_checkpoint_is_refused():
    from src.training.incremental import insert_checkpoint

    with pytest.raises(CampaignError, match="already a checkpoint"):
        insert_checkpoint(Path("."), _campaign({"2012/base": 10}), 8)


# ------------------------------------------------------------------- planning


def _full_campaign(rollouts: int, reported=("C0",)) -> dict:
    keys = {f"{y}/{n}": rollouts for y in range(2012, 2026)
            for n in ("base", "high_entropy", "slow_lr", "high_entropy_slow")}
    return _campaign(keys, reported=reported)


def test_a_short_session_trains_part_of_the_way():
    out = plan_session(_full_campaign(3), 0.5)
    assert len(out["steps"]) == 1
    assert "still needed for C1" in out["steps"][0]


def test_the_first_session_evaluates_c0_before_training():
    out = plan_session(_full_campaign(3, reported=()), 2)
    assert out["steps"][0].startswith("evaluate + report C0")
    assert any("train" in s for s in out["steps"][1:])


def test_a_long_session_crosses_a_checkpoint_and_evaluates_it():
    out = plan_session(_full_campaign(3), 3)
    assert out["steps"][0].startswith("train to C1")
    assert out["steps"][1].startswith("evaluate + report C1")


def test_an_evaluation_that_does_not_fit_is_not_started():
    camp = _full_campaign(5)                  # C1 reached, not reported
    out = plan_session(camp, 0.2)
    assert "does NOT fit" in out["steps"][0]


def test_an_unbounded_session_finishes_the_campaign():
    out = plan_session(_full_campaign(3), 10_000)
    assert out["steps"][-1] == "campaign complete"


# --------------------------------------------------------- the progress verdict


def test_the_paired_verdict_detects_improvement_regression_and_noise():
    from src.evaluation.learning_curve import paired_difference

    rng = np.random.default_rng(0)
    base = rng.normal(0.0002, 0.01, 2000)
    better = base + 0.0008               # same days, steadily higher
    worse = base - 0.0008
    assert paired_difference(base, better, replicates=200)["verdict"] == "improved"
    assert paired_difference(base, worse, replicates=200)["verdict"] == "regressed"
    assert paired_difference(base, base, replicates=200)["verdict"] == "no clear change"
    noisy = base + rng.normal(0.0, 0.01, 2000)
    assert paired_difference(base, noisy, replicates=200)["verdict"] == "no clear change"


def test_paired_series_must_cover_the_same_sessions():
    from src.evaluation.learning_curve import paired_difference

    with pytest.raises(ValueError, match="same sessions"):
        paired_difference(np.zeros(10), np.zeros(11))


def test_the_learning_curve_renders_every_checkpoint():
    from src.evaluation.learning_curve import render

    row = {"label": "C0", "pct": 2, "test_sharpe": 0.92,
           "test_sharpe_band": {"q05": 0.47, "q50": 0.95, "q95": 1.41},
           "test_max_drawdown": 0.165, "val_sharpe_selected": 1.1,
           "val_sharpe_all_candidates": 0.7,
           "baselines": {"spy_tlt_60_40": 0.90, "momentum_constrained": 0.8},
           "best_unconstrained": ["spy_tlt_60_40", 0.90],
           "best_constrained": ["momentum_constrained", 0.8],
           "gap_to_best_unconstrained": 0.02, "gap_to_best_constrained": 0.12,
           "selections": {"2012": "slow_lr"}, "hard_acceptance": {"passed": True},
           "training": {"epochs_per_rollout_fast": 1.0}, "vs_previous": None}
    second = {**row, "label": "C1", "pct": 4, "vs_previous": {
        "previous": "C0", "selection_changes": 1,
        "validation": {"observed": 0.1, "q05": -0.1, "q95": 0.3, "verdict": "no clear change"},
        "test": {"observed": 0.05, "q05": -0.2, "q95": 0.3, "verdict": "no clear change"}}}
    text = render([row, second], campaign="t")
    assert "| C0 | 2% |" in text and "| C1 | 4% |" in text
    assert "no clear change" in text
    assert "spy_tlt_60_40" in text


# --------------------------------------------------- exact resume (the big one)


@pytest.mark.slow
def test_stopping_and_resuming_gives_exactly_the_uninterrupted_weights(tmp_path: Path):
    """Two rollouts in one go vs one rollout, save, reload from disk, one more rollout.

    Bit-identical weights, optimizer state, reward normalizer and counters -- not merely
    "close". This is what makes the learning curve independent of where sessions stopped.
    """
    import torch

    from src.agents.normalization import wrap_normalizer
    from src.env.factory import build_bundle, make_vec_env
    from src.training.curriculum import STAGES_BY_INDEX
    from src.training.incremental import load_for_training, train_one_rollout
    from src.training.trainer import build_model, stage_env_config

    if not Path("data/features/feature_manifest.json").exists():
        pytest.skip("run Stages 1-3 first")

    bundle = build_bundle("config/training.yaml", start="2008-01-02", end="2009-12-31")
    resolved = dict(bundle.resolved)
    resolved["ppo"] = {**(resolved.get("ppo") or {}),
                       "n_steps": 32, "batch_size": 32, "n_epochs": 2}
    cfg = stage_env_config(bundle, STAGES_BY_INDEX[5],
                           {"episode_lengths": (20,), "risk_enabled": False,
                            "window": ("2008-01-02", "2009-12-31")})
    venv = make_vec_env(bundle, 2, base_seed=0, subproc=False, env_cfg=cfg)
    vec = wrap_normalizer(venv, gamma=0.999)
    start = build_model(bundle, vec, resolved, seed=5, tensorboard=None)
    write_state(start, vec, tmp_path / "start")

    # Uninterrupted: two rollouts on one in-memory model.
    a, a_vec = load_for_training(tmp_path / "start", venv)
    train_one_rollout(a, chunk_seed(5, 0), [])
    train_one_rollout(a, chunk_seed(5, 1), [])

    # Interrupted: one rollout, save, reload from disk, one more.
    b, b_vec = load_for_training(tmp_path / "start", venv)
    train_one_rollout(b, chunk_seed(5, 0), [])
    write_state(b, b_vec, tmp_path / "mid")
    b, b_vec = load_for_training(tmp_path / "mid", venv)
    train_one_rollout(b, chunk_seed(5, 1), [])

    assert a.num_timesteps == b.num_timesteps == 2 * 32 * 2
    assert a._n_updates == b._n_updates
    sa, sb = a.policy.state_dict(), b.policy.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"weights differ at {k}"
    oa, ob = a.policy.optimizer.state_dict(), b.policy.optimizer.state_dict()
    for k in oa["state"]:
        for name, value in oa["state"][k].items():
            assert torch.equal(torch.as_tensor(value), torch.as_tensor(ob["state"][k][name]))
    assert np.array_equal(a_vec.ret_rms.mean, b_vec.ret_rms.mean)
    assert np.array_equal(a_vec.ret_rms.var, b_vec.ret_rms.var)
    venv.close()


@pytest.mark.slow
def test_a_resumed_rollout_actually_trains(tmp_path: Path):
    """The equality above would also hold if neither path learned anything."""
    import torch

    from src.agents.normalization import wrap_normalizer
    from src.env.factory import build_bundle, make_vec_env
    from src.training.curriculum import STAGES_BY_INDEX
    from src.training.incremental import load_for_training, train_one_rollout
    from src.training.trainer import build_model, stage_env_config

    if not Path("data/features/feature_manifest.json").exists():
        pytest.skip("run Stages 1-3 first")

    bundle = build_bundle("config/training.yaml", start="2008-01-02", end="2009-12-31")
    resolved = dict(bundle.resolved)
    resolved["ppo"] = {**(resolved.get("ppo") or {}),
                       "n_steps": 32, "batch_size": 32, "n_epochs": 2, "target_kl": None}
    cfg = stage_env_config(bundle, STAGES_BY_INDEX[5],
                           {"episode_lengths": (20,), "risk_enabled": False,
                            "window": ("2008-01-02", "2009-12-31")})
    venv = make_vec_env(bundle, 2, base_seed=0, subproc=False, env_cfg=cfg)
    vec = wrap_normalizer(venv, gamma=0.999)
    write_state(build_model(bundle, vec, resolved, seed=5, tensorboard=None), vec,
                tmp_path / "start")
    model, _ = load_for_training(tmp_path / "start", venv)
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    epochs = train_one_rollout(model, chunk_seed(5, 0), [])
    assert epochs >= 1
    after = model.policy.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    venv.close()
