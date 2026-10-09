"""The incremental-training campaign: grow the budget in sessions, stop at any moment.

INCREMENTAL_TRAINING_PLAN.md is the specification. The parts that live here:

**The budget is counted in whole PPO rollouts.** One rollout is `n_steps x n_envs` =
16,384 timesteps, and SB3 always finishes a rollout it has started, so a timestep target
that is not a multiple silently overshoots -- the 2% walk-forward asked for 40,000 and ran
49,152. Checkpoints are rounded UP to whole rollouts for the same reason, which is why C0
is exactly the 3 rollouts that run already trained.

**A resumed run is the SAME run as an uninterrupted one.** Every rollout is its own
`learn()` call, preceded by an environment reset and a reseed derived from
`(candidate seed, rollouts done)`. The optimizer state, the reward normalizer and the
timestep counter travel in the saved files. So the weights after rollout k do not depend
on where the sessions happened to stop -- which is what lets a learning curve built over
many short sessions mean what it says. The cost is that an episode in progress at the end
of a rollout is cut short rather than continued; every collected step is still used.

**Nothing is lost when a run is stopped, beyond the rollout in progress.** State is
written to a temporary directory and swapped in by rename, and a COMPLETE marker records
how many timesteps it holds. The marker, not the campaign file, is the truth about how far
a candidate has trained: `reconcile` re-reads it at the start of every session.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path

from src.agents.normalization import POLICY_NAME, VECNORM_NAME, save_bundle

CAMPAIGNS = Path("artifacts/incremental")
COMPLETE = "COMPLETE"
STOP_FILE = "STOP"

#: Checkpoints, in percent of `training.total_timesteps`. The budget doubles between
#: them; a session can end anywhere in between.
CHECKPOINT_PCTS: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 100)

#: Starting estimates, replaced by measurements as soon as a session has run.
DEFAULT_THROUGHPUT = {
    "seconds_per_rollout": 20.0,     # 16,384 steps at ~940 steps/s, plus the save
    "seconds_per_env_build": 25.0,   # bundle + 8 spawned subprocess environments
    "seconds_per_fold_eval": 110.0,  # 4 candidates x 12 validation cells + 12 test cells
    "seconds_for_report": 180.0,     # Stage 12 report + bootstraps + learning curve
}


class CampaignError(RuntimeError):
    """The campaign is in a state it must not continue from."""


# --------------------------------------------------------------------- schedule


def checkpoint_schedule(total_timesteps: int, rollout_size: int,
                        pcts: tuple[int, ...] = CHECKPOINT_PCTS) -> list[dict]:
    out = []
    for i, pct in enumerate(pcts):
        rollouts = math.ceil(total_timesteps * pct / 100 / rollout_size)
        out.append({"label": f"C{i}", "pct": pct, "rollouts": rollouts,
                    "timesteps": rollouts * rollout_size})
    counts = [c["rollouts"] for c in out]
    if counts != sorted(set(counts)):
        raise CampaignError(f"checkpoint rollouts must strictly increase, got {counts}")
    return out


def chunk_seed(candidate_seed: int, rollouts_done: int) -> int:
    """The seed for a candidate's next rollout. A function of progress alone, so a run that
    stopped and resumed draws exactly what an uninterrupted run would have drawn."""
    return int(candidate_seed) * 100_003 + int(rollouts_done)


def next_target(campaign: dict) -> dict | None:
    """The checkpoint every candidate is currently being trained toward.

    All candidates reach a checkpoint before any passes it (plan P4): selection compares
    candidates, and one that is further along would win for the wrong reason.
    """
    lowest = min(c["rollouts"] for c in campaign["candidates"].values())
    for cp in campaign["schedule"]:
        if cp["rollouts"] > lowest:
            return cp
    return None


def checkpoint_reached(campaign: dict, cp: dict) -> bool:
    return all(cp["label"] in c.get("snapshots", {})
               for c in campaign["candidates"].values())


def pending_checkpoint(campaign: dict) -> dict | None:
    """The earliest checkpoint that every candidate has reached but is not yet reported."""
    for cp in campaign["schedule"]:
        state = campaign["checkpoints"].get(cp["label"], {})
        if checkpoint_reached(campaign, cp) and state.get("status") != "reported":
            return cp
    return None


# ----------------------------------------------------------- saved-model metadata


def saved_model_data(policy_zip: Path) -> dict:
    """The JSON block of an SB3 zip, read without importing torch."""
    with zipfile.ZipFile(policy_zip) as z:
        return json.loads(z.read("data"))


def saved_hparams(policy_zip: Path) -> dict:
    d = saved_model_data(policy_zip)
    keys = ("learning_rate", "ent_coef", "n_steps", "n_envs", "n_epochs", "batch_size",
            "gamma", "gae_lambda", "target_kl", "vf_coef", "max_grad_norm",
            "num_timesteps", "_n_updates", "seed")
    return {k: d.get(k) for k in keys}


# ------------------------------------------------------------- atomic state dirs


def _retry(fn, *args, attempts: int = 30, wait: float = 0.5, max_wait: float = 5.0):
    """Windows can briefly lock a freshly written file (indexer, antivirus).

    Backs off for up to ~2 minutes in total. The first session of `seeds_v1` died after 11 hours on a
    lock that outlasted the original 5-second window; the saved state survived intact
    (`recover_state` handles exactly that point), but the session did not.
    """
    for i in range(attempts):
        try:
            return fn(*args)
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(min(wait * (1.5 ** i), max_wait))


def _complete(directory: Path) -> int | None:
    marker = directory / COMPLETE
    if not (marker.exists() and (directory / POLICY_NAME).exists()
            and (directory / VECNORM_NAME).exists()):
        return None
    return int(marker.read_text(encoding="utf-8").strip())


def mark_complete(directory: Path, num_timesteps: int) -> None:
    (directory / COMPLETE).write_text(str(int(num_timesteps)), encoding="utf-8")


def write_state(model, vec, state_dir: Path) -> None:
    """Save policy + normalizer so that a kill at any instant leaves a usable state."""
    tmp = state_dir.with_name(state_dir.name + ".tmp")
    old = state_dir.with_name(state_dir.name + ".old")
    if tmp.exists():
        _retry(shutil.rmtree, tmp)
    save_bundle(model, vec, tmp)
    mark_complete(tmp, model.num_timesteps)
    if state_dir.exists():
        if old.exists():
            _retry(shutil.rmtree, old)
        _retry(os.replace, state_dir, old)
    _retry(os.replace, tmp, state_dir)
    if old.exists():
        _retry(shutil.rmtree, old)


def recover_state(state_dir: Path) -> int:
    """Resolve a swap a crash interrupted. Returns the timesteps the usable state holds.

    At most the rollout that was being saved is lost: a newer state is only ever preferred
    when the current one is gone, never by guessing.
    """
    tmp = state_dir.with_name(state_dir.name + ".tmp")
    old = state_dir.with_name(state_dir.name + ".old")
    current = _complete(state_dir) if state_dir.exists() else None
    if current is not None:
        for leftover in (tmp, old):
            if leftover.exists():
                _retry(shutil.rmtree, leftover)
        return current
    if state_dir.exists():
        _retry(shutil.rmtree, state_dir)          # half-written, unusable
    for candidate in (tmp, old):
        if candidate.exists() and _complete(candidate) is not None:
            _retry(os.replace, candidate, state_dir)
            for leftover in (tmp, old):
                if leftover.exists():
                    _retry(shutil.rmtree, leftover)
            return _complete(state_dir)
    raise CampaignError(f"no usable saved state at {state_dir} (nor .tmp / .old)")


def snapshot(state_dir: Path, dest: Path) -> None:
    """Freeze a copy of the state at a checkpoint, for evaluation and for later seeds."""
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        _retry(shutil.rmtree, tmp)
    shutil.copytree(state_dir, tmp)
    if dest.exists():
        _retry(shutil.rmtree, dest)
    _retry(os.replace, tmp, dest)


# --------------------------------------------------------------- campaign file


def campaign_dir(name: str) -> Path:
    return CAMPAIGNS / name


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_campaign(directory: Path) -> dict:
    path = directory / "campaign.json"
    if not path.exists():
        raise CampaignError(f"no campaign at {directory}; create one with --init")
    return json.loads(path.read_text(encoding="utf-8"))


def save_campaign(directory: Path, campaign: dict) -> None:
    path = directory / "campaign.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(campaign, indent=2, default=str), encoding="utf-8")
    _retry(os.replace, tmp, path)


def check_config(campaign: dict, config_hash: str, *, allow_change: bool = False) -> None:
    """Plan P6: a learning curve across two configs means nothing."""
    if campaign["config_hash"] != config_hash and not allow_change:
        raise CampaignError(
            f"the config hash is {config_hash} but this campaign was started under "
            f"{campaign['config_hash']}. Every point on the learning curve must come from "
            "the same training setup. Revert the config change, or start a new campaign. "
            "(--allow-config-change exists for a change you have verified does not touch "
            "training; using it is recorded in the campaign.)")


def model_dir(directory: Path, key: str) -> Path:
    year, name = key.split("/")
    return directory / "models" / year / name


def insert_checkpoint(directory: Path, campaign: dict, pct: float) -> list[str]:
    """Add a checkpoint to a running campaign's schedule. Returns notes.

    A checkpoint is an exact snapshot at its rollout count, so a candidate that has already
    trained past it is rolled back to its latest earlier snapshot and retrains from there.
    Training is a deterministic function of (state, chunk seed), so the retrained path is
    the one it already took; the cost is only the repeated rollouts. Refused if any later
    checkpoint has started evaluation, since its comparison to "the previous checkpoint"
    would change underneath it.
    """
    rollouts = math.ceil(campaign["total_timesteps"] * pct / 100 / campaign["rollout_size"])
    schedule = campaign["schedule"]
    if any(cp["rollouts"] == rollouts for cp in schedule):
        raise CampaignError(f"{pct}% is {rollouts} rollouts, which is already a checkpoint")
    later = [cp for cp in schedule if cp["rollouts"] > rollouts]
    for cp in later:
        if campaign["checkpoints"].get(cp["label"], {}).get("status") in ("evaluating",
                                                                          "reported"):
            raise CampaignError(f"{cp['label']} is already {campaign['checkpoints'][cp['label']]['status']}; "
                                "a checkpoint cannot be inserted before it")
    earlier = [cp for cp in schedule if cp["rollouts"] < rollouts]
    if not earlier:
        raise CampaignError("a checkpoint cannot be inserted before C0")
    base = earlier[-1]
    label = base["label"] + "b"
    while any(cp["label"] == label for cp in schedule):
        label += "b"
    entry = {"label": label, "pct": pct, "rollouts": rollouts,
             "timesteps": rollouts * campaign["rollout_size"]}

    notes = []
    later_labels = {cp["label"] for cp in later}
    for key, cand in campaign["candidates"].items():
        if cand["rollouts"] <= rollouts:
            continue
        mdir = model_dir(directory, key)
        snapshot(mdir / base["label"], mdir / "current")
        snap = cand["snapshots"][base["label"]]
        notes.append(f"{key}: rolled back {cand['rollouts']} -> {base['rollouts']} rollouts "
                     f"({base['label']}) to pass through {label}")
        cand["rollouts"] = base["rollouts"]
        cand["n_updates"] = snap["n_updates"]
        cand["last_diag"] = snap.get("diag", {})
        for lab in list(cand.get("snapshots", {})):
            if lab in later_labels:
                del cand["snapshots"][lab]
                stale = mdir / lab
                if stale.exists():
                    _retry(shutil.rmtree, stale)
    for cp in later:
        campaign["checkpoints"].pop(cp["label"], None)
    schedule.insert(schedule.index(later[0]) if later else len(schedule), entry)
    notes.append(f"checkpoint {label} added at {pct}% = {rollouts} rollouts "
                 f"({entry['timesteps']:,} timesteps)")
    return notes


def reconcile(directory: Path, campaign: dict, rollout_size: int) -> list[str]:
    """Make the campaign file agree with what is actually on disk. Returns notes."""
    notes = []
    labels = {cp["rollouts"]: cp["label"] for cp in campaign["schedule"]}
    for key, cand in campaign["candidates"].items():
        state = model_dir(directory, key) / "current"
        timesteps = recover_state(state)
        if timesteps % rollout_size:
            raise CampaignError(f"{key}: {timesteps} timesteps is not a whole number of "
                                f"{rollout_size}-step rollouts")
        rollouts = timesteps // rollout_size
        if rollouts != cand["rollouts"]:
            notes.append(f"{key}: campaign file said {cand['rollouts']} rollouts, saved "
                         f"state holds {rollouts}; using the saved state")
            cand["rollouts"] = rollouts
        label = labels.get(rollouts)
        if label and label not in cand.get("snapshots", {}):
            # The run stopped between saving the checkpoint state and copying it.
            snapshot(state, model_dir(directory, key) / label)
            cand.setdefault("snapshots", {})[label] = {
                "n_updates": saved_hparams(state / POLICY_NAME)["_n_updates"],
                "diag": cand.get("last_diag", {})}
            notes.append(f"{key}: recovered missing {label} snapshot")
        for cp in campaign["schedule"]:
            if rollouts > cp["rollouts"] and cp["label"] not in cand.get("snapshots", {}):
                raise CampaignError(f"{key} is past {cp['label']} without a snapshot of it")
    return notes


# ----------------------------------------------------------------- training


def load_for_training(state_dir: Path, venv, *, device: str = "cpu",
                      tensorboard: Path | None = None):
    """Policy, optimizer state, counters and reward normalizer, onto a live VecEnv."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    vec = VecNormalize.load(str(state_dir / VECNORM_NAME), venv)
    vec.training = True
    vec.norm_reward = True
    model = PPO.load(state_dir / POLICY_NAME, env=vec, device=device,
                     tensorboard_log=str(tensorboard) if tensorboard else None)
    return model, vec


def train_one_rollout(model, seed: int, callbacks: list) -> int:
    """Exactly one rollout and one update. Returns the epochs the update ran.

    `_last_obs = None` forces `learn()` to reset the environment, and the reseed makes that
    reset a function of progress: the two together are what make a resumed run identical
    to an uninterrupted one.
    """
    from stable_baselines3.common.callbacks import CallbackList

    rollout = model.n_steps * model.env.num_envs
    before_ts, before_updates = model.num_timesteps, model._n_updates
    model._last_obs = None
    model.set_random_seed(seed)
    model.learn(total_timesteps=rollout, callback=CallbackList(callbacks),
                reset_num_timesteps=False, tb_log_name="incremental", progress_bar=False)
    if model.num_timesteps - before_ts != rollout:
        raise CampaignError(f"expected one rollout of {rollout} timesteps, ran "
                            f"{model.num_timesteps - before_ts}")
    return model._n_updates - before_updates


# --------------------------------------------------------------- session planning


def ema(old: float | None, new: float, weight: float = 0.3) -> float:
    return new if old is None else (1 - weight) * old + weight * new


def plan_session(campaign: dict, hours: float, *, margin_minutes: float = 5.0) -> dict:
    """What a session of `hours` should get through, from the measured throughput.

    A model, not a promise: the session itself enforces the deadline chunk by chunk.
    """
    tp = {**DEFAULT_THROUGHPUT, **campaign.get("throughput", {})}
    budget = hours * 3600 - margin_minutes * 60
    n_folds = len({c["test_year"] for c in campaign["candidates"].values()})
    steps: list[str] = []
    rollouts = {k: c["rollouts"] for k, c in campaign["candidates"].items()}
    status = {k: dict(v) for k, v in campaign["checkpoints"].items()}
    schedule = campaign["schedule"]
    spent = 0.0

    def reached(cp):
        return all(r >= cp["rollouts"] for r in rollouts.values())

    while spent < budget:
        pending = next((cp for cp in schedule if reached(cp)
                        and status.get(cp["label"], {}).get("status") != "reported"), None)
        if pending:
            done = len(status.get(pending["label"], {}).get("folds_done", []))
            cost = ((n_folds - done) * tp["seconds_per_fold_eval"]
                    + tp["seconds_for_report"])
            if spent + cost > budget:
                steps.append(f"evaluate {pending['label']} -- does NOT fit "
                             f"(~{cost / 60:.0f} min needed); the session stops before it")
                break
            spent += cost
            steps.append(f"evaluate + report {pending['label']} ({pending['pct']}%) "
                         f"~{cost / 60:.0f} min")
            status.setdefault(pending["label"], {})["status"] = "reported"
            continue
        lowest = min(rollouts.values())
        target = next((cp for cp in schedule if cp["rollouts"] > lowest), None)
        if target is None:
            steps.append("campaign complete")
            break
        need = sum(max(0, target["rollouts"] - r) for r in rollouts.values())
        cost_all = need * tp["seconds_per_rollout"] + n_folds * tp["seconds_per_env_build"]
        if spent + cost_all <= budget:
            spent += cost_all
            for k in rollouts:
                rollouts[k] = max(rollouts[k], target["rollouts"])
            steps.append(f"train to {target['label']} ({target['pct']}%): {need} rollouts "
                         f"~{cost_all / 60:.0f} min")
            continue
        # Partial: walk the folds in the order the session does, paying one environment
        # build per fold touched.
        left = budget - spent
        fits = 0
        for year in sorted({k.split("/")[0] for k in rollouts}):
            fold_need = sum(max(0, target["rollouts"] - r)
                            for k, r in rollouts.items() if k.startswith(year + "/"))
            if not fold_need or left < tp["seconds_per_env_build"] + tp["seconds_per_rollout"]:
                continue
            left -= tp["seconds_per_env_build"]
            n = min(fold_need, int(left // tp["seconds_per_rollout"]))
            fits += n
            left -= n * tp["seconds_per_rollout"]
        now_total = sum(rollouts.values()) + fits
        pct = 100.0 * now_total * campaign["rollout_size"] / (
            len(rollouts) * campaign["total_timesteps"])
        steps.append(f"train ~{fits} of the {need} rollouts still needed for "
                     f"{target['label']} ({target['pct']}%); the session ends at ~{pct:.1f}% "
                     "average budget and the next one carries on from there")
        break

    return {"hours": hours, "usable_minutes": round(budget / 60, 1), "steps": steps,
            "throughput": tp}


def average_budget_pct(campaign: dict) -> float:
    total = sum(c["rollouts"] for c in campaign["candidates"].values())
    return 100.0 * total * campaign["rollout_size"] / (
        len(campaign["candidates"]) * campaign["total_timesteps"])


# ------------------------------------------------------------------ keep awake


class KeepAwake:
    """Ask Windows not to sleep while a session runs, and only while it runs."""

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __enter__(self):
        self._ok = False
        if os.name == "nt":
            import ctypes
            self._ok = bool(ctypes.windll.kernel32.SetThreadExecutionState(
                self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED))
        return self

    def __exit__(self, *exc):
        if self._ok:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
        return False
