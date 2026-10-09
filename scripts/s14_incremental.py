"""Incremental training -- grow the budget in idle-time sessions. See INCREMENTAL_TRAINING_PLAN.md.

    python -m scripts.s14_incremental --init s08_walk_forward_20260903T224114Z_a207776c
    python -m scripts.s14_incremental --status
    python -m scripts.s14_incremental --plan 2          # what 2 idle hours would buy; no compute
    python -m scripts.s14_incremental --hours 2         # a session: train / evaluate until the deadline
    python -m scripts.s14_incremental --stop            # ask a running session to stop after its rollout
    python -m scripts.s14_incremental --add-checkpoint 12   # insert a checkpoint at 12% of the budget
    python -m scripts.s14_incremental --campaign seeds_v1 --init-fresh --setting high_entropy --seeds 3

A session does, in a loop and until its deadline: evaluate any checkpoint every candidate
has reached (walk-forward validation + selection + test, then the Stage 12 report, then
the learning curve), otherwise train every candidate toward the next checkpoint, one PPO
rollout at a time, saving after each. It can be stopped at any moment -- `--stop`, Ctrl-C,
or killing the process -- and loses at most the rollout in progress. Evaluation is resumable
fold by fold.

The walk-forward protocol is Stage 8's, unchanged: same folds, same four candidates, same
lexicographic selection on the validation year, same test grid, same scoring code
(`src/evaluation/fold_eval.py`). What changes is only that each candidate's training is
continued rather than restarted.

Writes `artifacts/incremental/<campaign>/`:

    campaign.json                    the state: rollouts per candidate, checkpoints, sessions
    models/<year>/<candidate>/current/   the latest state (policy, optimizer, normalizer)
    models/<year>/<candidate>/C<k>/      frozen at each checkpoint
    checkpoints/C<k>/folds/<year>/   Stage 8-format results, readable by s10/s12
    checkpoints/C<k>/checkpoint.json the learning-curve row
    learning_curve.md                every checkpoint so far, against the baselines
    sessions/<stamp>.log
and the Stage 12 report per checkpoint under `artifacts/reports/incremental/<campaign>/C<k>/`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from src.config.loader import config_hash, resolve_config
from src.evaluation.walk_forward import (
    assert_folds_valid,
    load_folds,
    scaler_is_legal,
    select_folds,
)
from src.training.incremental import (
    COMPLETE,
    STOP_FILE,
    CampaignError,
    KeepAwake,
    average_budget_pct,
    campaign_dir,
    check_config,
    checkpoint_schedule,
    chunk_seed,
    ema,
    insert_checkpoint,
    load_campaign,
    load_for_training,
    mark_complete,
    model_dir,
    next_target,
    pending_checkpoint,
    plan_session,
    reconcile,
    save_campaign,
    saved_hparams,
    snapshot,
    train_one_rollout,
    utc_now,
    write_state,
)

DEFAULT_CAMPAIGN = "budget_v1"
EVAL_SEED = 42                 # Stage 8 evaluates with the run seed; the rollout is deterministic
REPORTS = Path("artifacts/reports/incremental")


class Log:
    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str = "") -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


# ------------------------------------------------------------------- helpers


def _settings(resolved: dict) -> dict:
    training = resolved.get("training", {}) or {}
    ppo = resolved.get("ppo", {}) or {}
    wf = resolved.get("walk_forward", {}) or {}
    return {
        "n_envs": int(training.get("n_envs", 8)),
        "subproc": bool(training.get("subproc", True)),
        "n_steps": int(ppo.get("n_steps", 2048)),
        "total_timesteps": int(training.get("total_timesteps", 2_000_000)),
        "first_year": int(wf.get("first_test_year", 2012)),
        "last_year": int(wf.get("last_test_year", 2025)),
        "candidates": int(wf.get("candidates", 4)),
    }


def _folds(campaign: dict):
    folds = load_folds()
    assert_folds_valid(folds)
    chosen = select_folds(folds, first_year=campaign["first_year"],
                          last_year=campaign["last_year"])
    years = sorted({c["test_year"] for c in campaign["candidates"].values()})
    if [f.test_year for f in chosen] != years:
        raise CampaignError(f"folds.json now yields {[f.test_year for f in chosen]}, the "
                            f"campaign was built on {years}")
    return chosen


def _fold_keys(campaign: dict, year: int) -> list[str]:
    return [k for k, c in campaign["candidates"].items() if c["test_year"] == year]


def _bundle(config_path: Path, resolved: dict, fold):
    from src.env.factory import build_bundle

    bundle = build_bundle(config_path, resolved=resolved, fold_id=fold.fold_id)
    problems = (scaler_is_legal(*bundle.store.fit_range, fold) if bundle.store.fit_range
                else ["no scaler is fitted; training on unscaled observations"])
    if problems:
        raise CampaignError(f"{fold.fold_id}: scaler is not legal for this fold: {problems}")
    return bundle


# ---------------------------------------------------------------------- init


def init(args, resolved: dict, chash: str, log: Log) -> None:
    from src.training.model_selection import default_candidates

    cdir = campaign_dir(args.campaign)
    if (cdir / "campaign.json").exists():
        raise CampaignError(f"campaign {args.campaign!r} already exists at {cdir}")
    run = Path(args.init)
    run = run if run.exists() else Path("artifacts/runs") / args.init
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "success":
        raise CampaignError(f"{run.name} did not finish successfully")
    if manifest["config_hash"] != chash:
        raise CampaignError(
            f"{run.name} was trained under config {manifest['config_hash']}, the config is "
            f"now {chash}. Its models are not a valid C0 for a campaign on this config.")

    s = _settings(resolved)
    rollout_size = s["n_steps"] * s["n_envs"]
    schedule = checkpoint_schedule(s["total_timesteps"], rollout_size)
    c0 = schedule[0]

    campaign = {
        "campaign": args.campaign, "created_at": utc_now(), "source_run": run.name,
        "config_path": str(args.config).replace("\\", "/"), "config_hash": chash,
        "rollout_size": rollout_size, "n_envs": s["n_envs"],
        "total_timesteps": s["total_timesteps"],
        "first_year": s["first_year"], "last_year": s["last_year"],
        "schedule": schedule, "checkpoints": {c0["label"]: {"status": "reached"}},
        "candidates": {}, "throughput": {}, "sessions": [],
    }
    folds = select_folds(load_folds(), first_year=s["first_year"], last_year=s["last_year"])
    seed = int(manifest.get("seed", 42))
    for fold in folds:
        for cand in default_candidates(s["candidates"], base_seed=seed):
            src = run / "folds" / str(fold.test_year) / "candidates" / cand.name / \
                "stage5_randomized"
            hp = saved_hparams(src / "policy.zip")
            expected = {"learning_rate": cand.params["learning_rate"],
                        "ent_coef": cand.params["ent_coef"], "seed": cand.params["seed"],
                        "num_timesteps": c0["timesteps"], "n_envs": s["n_envs"],
                        "n_steps": s["n_steps"]}
            wrong = {k: (hp[k], v) for k, v in expected.items() if hp[k] != v}
            if wrong:
                raise CampaignError(f"{src}: saved model disagrees with the campaign "
                                    f"(saved, expected): {wrong}")
            key = f"{fold.test_year}/{cand.name}"
            mdir = model_dir(cdir, key)
            current = mdir / "current"
            current.mkdir(parents=True, exist_ok=True)
            for name in ("policy.zip", "vecnormalize.pkl"):
                shutil.copy2(src / name, current / name)
            mark_complete(current, hp["num_timesteps"])
            snapshot(current, mdir / c0["label"])
            metrics = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
            diag = metrics.get("diagnostics", {})
            campaign["candidates"][key] = {
                "fold_id": fold.fold_id, "test_year": fold.test_year, "name": cand.name,
                "learning_rate": cand.params["learning_rate"],
                "ent_coef": cand.params["ent_coef"], "seed": cand.params["seed"],
                "rollouts": c0["rollouts"], "n_updates": hp["_n_updates"],
                "last_diag": diag,
                "snapshots": {c0["label"]: {"n_updates": hp["_n_updates"], "diag": diag}},
            }
            log(f"  {key:<24} {hp['num_timesteps']:,} timesteps, "
                f"{hp['_n_updates']} update epochs -> {c0['label']}")
    cdir.mkdir(parents=True, exist_ok=True)
    save_campaign(cdir, campaign)
    log(f"campaign {args.campaign!r} created from {run.name}: "
        f"{len(campaign['candidates'])} candidates at {c0['label']} ({c0['pct']}%). "
        f"Schedule (rollouts): {[c['rollouts'] for c in schedule]}")


def init_fresh(args, resolved: dict, chash: str, log: Log) -> None:
    """A seed-ensemble campaign: ONE hyperparameter setting, several seeds, from scratch.

    Redesign plan Phase 1 (decision D-C): per-year selection between four settings mostly
    selected noise, and pooled validation showed the settings themselves behave very
    differently with budget. So the setting is fixed once, and the compute goes into seeds,
    which measure the noise and are averaged into one portfolio.
    """
    from src.agents.normalization import wrap_normalizer
    from src.env.factory import make_vec_env
    from src.training.curriculum import STAGES_BY_INDEX
    from src.training.model_selection import default_candidates
    from src.training.trainer import build_model, ppo_kwargs, stage_env_config

    cdir = campaign_dir(args.campaign)
    if (cdir / "campaign.json").exists():
        raise CampaignError(f"campaign {args.campaign!r} already exists at {cdir}")
    settings = {c.name: c.params for c in default_candidates(6)}
    if args.setting not in settings:
        raise CampaignError(f"unknown setting {args.setting!r}; one of {sorted(settings)}")
    hp = {k: v for k, v in settings[args.setting].items() if k != "seed"}
    s = _settings(resolved)
    rollout_size = s["n_steps"] * s["n_envs"]
    seeds = [args.seed_base + i for i in range(args.seeds)]

    campaign = {
        "campaign": args.campaign, "created_at": utc_now(), "mode": "ensemble",
        "source_run": None, "setting": {"name": args.setting, **hp}, "seeds": seeds,
        "config_path": str(args.config).replace("\\", "/"), "config_hash": chash,
        "rollout_size": rollout_size, "n_envs": s["n_envs"],
        "total_timesteps": s["total_timesteps"],
        "first_year": s["first_year"], "last_year": s["last_year"],
        "schedule": checkpoint_schedule(s["total_timesteps"], rollout_size),
        "checkpoints": {}, "candidates": {}, "throughput": {}, "sessions": [],
    }
    stage_resolved = dict(resolved)
    stage_resolved["ppo"] = {**(resolved.get("ppo", {}) or {}), **hp}
    folds = select_folds(load_folds(), first_year=s["first_year"], last_year=s["last_year"])
    for fold in folds:
        bundle = _bundle(Path(args.config), resolved, fold)
        env_cfg = stage_env_config(bundle, STAGES_BY_INDEX[5],
                                   {"window": (fold.train_start, fold.train_end)})
        # In-process environments: they only supply the spaces to build the model; no
        # step is taken here, so the subprocess start-up cost would buy nothing.
        venv = make_vec_env(bundle, s["n_envs"], base_seed=0, subproc=False, env_cfg=env_cfg)
        try:
            for seed in seeds:
                vec = wrap_normalizer(venv, gamma=ppo_kwargs(stage_resolved)["gamma"])
                model = build_model(bundle, vec, stage_resolved, seed=seed, tensorboard=None)
                key = f"{fold.test_year}/seed{seed}"
                write_state(model, vec, model_dir(cdir, key) / "current")
                campaign["candidates"][key] = {
                    "fold_id": fold.fold_id, "test_year": fold.test_year,
                    "name": f"seed{seed}", "learning_rate": hp["learning_rate"],
                    "ent_coef": hp["ent_coef"], "seed": seed, "rollouts": 0,
                    "n_updates": 0, "last_diag": {}, "snapshots": {}}
        finally:
            venv.close()
        log(f"  fold {fold.test_year}: {len(seeds)} fresh models ({args.setting})")
    cdir.mkdir(parents=True, exist_ok=True)
    save_campaign(cdir, campaign)
    log(f"campaign {args.campaign!r} created: setting {args.setting} {hp}, seeds {seeds}, "
        f"{len(campaign['candidates'])} models at 0 rollouts. Schedule (rollouts): "
        f"{[c['rollouts'] for c in campaign['schedule']]}")


# ------------------------------------------------------------------- session


class Session:
    def __init__(self, cdir: Path, campaign: dict, hours: float, margin_minutes: float,
                 log: Log):
        self.cdir, self.campaign, self.log = cdir, campaign, log
        self.started = time.time()
        self.deadline = self.started + hours * 3600 - margin_minutes * 60
        self.reason = ""
        self.rollouts_trained = 0
        self.checkpoints_reported: list[str] = []
        self.record = {"started_at": utc_now(), "hours": hours}

    @property
    def tp(self) -> dict:
        from src.training.incremental import DEFAULT_THROUGHPUT
        return {**DEFAULT_THROUGHPUT, **self.campaign.get("throughput", {})}

    def measure(self, key: str, seconds: float) -> None:
        self.campaign.setdefault("throughput", {})[key] = round(
            ema(self.campaign["throughput"].get(key), seconds), 2)

    def fits(self, seconds: float) -> bool:
        return time.time() + seconds <= self.deadline

    def stop_requested(self) -> bool:
        if (self.cdir / STOP_FILE).exists():
            self.reason = self.reason or "stop requested (--stop)"
            return True
        return False

    def save(self) -> None:
        save_campaign(self.cdir, self.campaign)


def train_to_target(sess: Session, target: dict, folds, config_path, resolved) -> bool:
    """Bring every candidate up to `target`. False if the session must end first."""
    from src.env.factory import make_vec_env
    from src.training.callbacks import ConstraintMonitor, DiagnosticCallback, EntropyGuard
    from src.training.curriculum import STAGES_BY_INDEX
    from src.training.trainer import stage_env_config

    camp, log = sess.campaign, sess.log
    s = _settings(resolved)
    log(f"training toward {target['label']} ({target['pct']}%, "
        f"{target['rollouts']} rollouts per candidate)")
    for fold in folds:
        keys = [k for k in _fold_keys(camp, fold.test_year)
                if camp["candidates"][k]["rollouts"] < target["rollouts"]]
        if not keys:
            continue
        if sess.stop_requested():
            return False
        if not sess.fits(sess.tp["seconds_per_env_build"] + sess.tp["seconds_per_rollout"]):
            sess.reason = "deadline"
            return False

        t0 = time.time()
        bundle = _bundle(config_path, resolved, fold)
        env_cfg = stage_env_config(bundle, STAGES_BY_INDEX[5],
                                   {"window": (fold.train_start, fold.train_end)})
        venv = make_vec_env(bundle, s["n_envs"], base_seed=0, subproc=s["subproc"],
                            env_cfg=env_cfg)
        sess.measure("seconds_per_env_build", time.time() - t0)
        try:
            for key in keys:
                cand = camp["candidates"][key]
                mdir = model_dir(sess.cdir, key)
                model, vec = load_for_training(mdir / "current", venv,
                                               tensorboard=mdir / "tensorboard")
                if (abs(float(model.learning_rate) - cand["learning_rate"]) > 1e-12
                        or abs(float(model.ent_coef) - cand["ent_coef"]) > 1e-12):
                    raise CampaignError(f"{key}: loaded hyperparameters do not match")
                diag = DiagnosticCallback()
                callbacks = [ConstraintMonitor(), diag, EntropyGuard(verbose=1)]
                while cand["rollouts"] < target["rollouts"]:
                    if sess.stop_requested():
                        return False
                    if not sess.fits(sess.tp["seconds_per_rollout"]):
                        sess.reason = "deadline"
                        return False
                    t1 = time.time()
                    epochs = train_one_rollout(
                        model, chunk_seed(cand["seed"], cand["rollouts"]), callbacks)
                    write_state(model, vec, mdir / "current")
                    cand["rollouts"] += 1
                    cand["n_updates"] = int(model._n_updates)
                    cand["last_diag"] = diag.snapshot()
                    if cand["rollouts"] == target["rollouts"]:
                        snapshot(mdir / "current", mdir / target["label"])
                        cand.setdefault("snapshots", {})[target["label"]] = {
                            "n_updates": cand["n_updates"], "diag": cand["last_diag"]}
                    sess.rollouts_trained += 1
                    sess.measure("seconds_per_rollout", time.time() - t1)
                    sess.save()
                    log(f"  {key:<24} rollout {cand['rollouts']:>3}/{target['rollouts']}  "
                        f"{time.time() - t1:5.1f}s  epochs {epochs}  "
                        f"proj {cand['last_diag'].get('proj_distance', 0):.3f}  "
                        f"cash {cand['last_diag'].get('cash_weight', 0):.2f}")
        finally:
            venv.close()
    camp["checkpoints"].setdefault(target["label"], {})["status"] = "reached"
    sess.save()
    log(f"all candidates reached {target['label']}")
    return True


def _training_summary(camp: dict, label: str, previous: str | None, prev_rollouts: int,
                      rollouts: int) -> dict:
    fast, slow, diags = [], [], []
    for cand in camp["candidates"].values():
        snap = cand["snapshots"][label]
        before = cand["snapshots"][previous]["n_updates"] if previous else 0
        span = rollouts - (prev_rollouts if previous else 0)
        (fast if cand["learning_rate"] >= 2e-4 else slow).append(
            (snap["n_updates"] - before) / max(span, 1))
        diags.append(snap.get("diag", {}))

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(sum(xs) / len(xs)) if xs else None

    return {"epochs_per_rollout_fast": mean(fast), "epochs_per_rollout_slow": mean(slow),
            "proj_distance": mean([d.get("proj_distance") for d in diags]),
            "infeasible_fallback": mean([d.get("infeasible_fallback") for d in diags]),
            "cash_weight": mean([d.get("cash_weight") for d in diags])}


#: The three ceilings every seed is scored at (redesign plan, Phase 1 item 3).
SEED_CEILINGS = (0.05, 0.10, 0.15)


def _evaluate_fold_ensemble(sess: Session, fold, bundle, cells, label: str, fdir: Path,
                            hd: int, dm: float) -> None:
    """One fold of a seed-ensemble campaign: no selection.

    Every seed is scored on validation and test at the three ceilings, and kept on disk so
    the learning curve can report the spread between seeds. The REPORTED policy is the
    ensemble -- the average of the seeds' portfolios -- which gets the full Stage 8 grid,
    so the Stage 12 report and the hard-acceptance checks read it exactly as they would a
    selected model.
    """
    from src.evaluation.fold_eval import cell_key, evaluate_window
    from src.evaluation.rollout import load_policy
    from src.evaluation.violations import summarize as summarize_violations

    camp, log = sess.campaign, sess.log
    primary = cell_key(hd, dm)
    seed_cells = [(hd, d) for d in SEED_CEILINGS]
    keys = _fold_keys(camp, fold.test_year)
    dirs = [model_dir(sess.cdir, k) / label for k in keys]
    fdir.mkdir(parents=True, exist_ok=True)

    seeds, members = {}, []
    for key, pdir in zip(keys, dirs):
        c = camp["candidates"][key]
        model = load_policy(pdir)
        members.append(model)
        val, val_trajs = evaluate_window(model, bundle, (fold.val_start, fold.val_end),
                                         seed_cells, seed=EVAL_SEED,
                                         label=f"val {c['name']}", log=log)
        test, test_trajs = evaluate_window(model, bundle, (fold.test_start, fold.test_end),
                                           seed_cells, seed=EVAL_SEED,
                                           label=f"test {c['name']}", log=log)
        sdir = fdir / "seeds" / c["name"]
        sdir.mkdir(parents=True, exist_ok=True)
        for k, t in val_trajs.items():
            t.to_parquet(sdir / f"val_{k}.parquet")
        for k, t in test_trajs.items():
            t.to_parquet(sdir / f"test_{k}.parquet")
        seeds[c["name"]] = {"params": {"ent_coef": c["ent_coef"],
                                       "learning_rate": c["learning_rate"],
                                       "seed": c["seed"], "policy_dir": str(pdir)},
                            "validation": val, "test": test}

    ens_val, ens_val_trajs = evaluate_window(members, bundle, (fold.val_start, fold.val_end),
                                             seed_cells, seed=EVAL_SEED,
                                             label="val ENSEMBLE", log=log)
    ens_val_trajs[primary].to_parquet(fdir / "val_trajectory.parquet")
    (fdir / "val_cells").mkdir(exist_ok=True)
    for key, traj in ens_val_trajs.items():
        traj.to_parquet(fdir / "val_cells" / f"{key}.parquet")
    test_agg, trajs = evaluate_window(members, bundle, (fold.test_start, fold.test_end),
                                      cells, seed=EVAL_SEED,
                                      label=f"TEST {fold.test_year} ENSEMBLE", log=log)
    trajs[primary].to_parquet(fdir / "trajectory.parquet")
    (fdir / "cells").mkdir(exist_ok=True)
    for key, traj in trajs.items():
        traj.to_parquet(fdir / "cells" / f"{key}.parquet")
    taxonomy = summarize_violations(trajs[primary], bundle.market.universe,
                                    d_max=dm, market=bundle.market)
    (fdir / "violations.json").write_text(json.dumps(taxonomy, indent=2, default=str),
                                          encoding="utf-8")
    criterion = "seed ensemble: the average of every seed's portfolio, no selection"
    # selection.json keeps the Stage 8 shape so every reader of it still works: the
    # "candidates" are the seeds, the "chosen" one is the ensemble.
    (fdir / "selection.json").write_text(json.dumps({
        "fold_id": fold.fold_id, "chosen": "ensemble", "criterion": criterion,
        "constraint_validation_failure": False,
        "candidates": [{"name": n, "params": s["params"], "validation": s["validation"],
                        "eliminated_at": None, "reason": "ensemble member"}
                       for n, s in seeds.items()]}, indent=2, default=str), encoding="utf-8")
    record = {
        "fold_id": fold.fold_id, "test_year": fold.test_year, "fold": fold.to_dict(),
        "selected": "ensemble", "criterion": criterion,
        "constraint_validation_failure": False,
        "validation": ens_val, "test": test_agg,
        "violations": {k: v for k, v in taxonomy.items() if k != "replay_findings"},
        "seeds": {n: {"validation": s["validation"], "test": s["test"]} for n, s in seeds.items()},
    }
    (fdir / "record.json").write_text(json.dumps(record, indent=2, default=str),
                                      encoding="utf-8")


def evaluate_checkpoint(sess: Session, cp: dict, folds, config_path, resolved,
                        replicates: int) -> bool:
    """Stage 8's per-fold protocol on the frozen checkpoint models, then the reports."""
    from src.evaluation.fold_eval import cell_key, eval_grid, evaluate_window
    from src.evaluation.learning_curve import build_record, render
    from src.evaluation.rollout import load_policy
    from src.evaluation.violations import summarize as summarize_violations
    from src.training.model_selection import Candidate, select

    camp, log, label = sess.campaign, sess.log, cp["label"]
    state = camp["checkpoints"].setdefault(label, {})
    state["status"] = "evaluating"
    ckdir = sess.cdir / "checkpoints" / label
    log(f"evaluating {label} ({cp['pct']}%)")

    for fold in folds:
        fdir = ckdir / "folds" / str(fold.test_year)
        if (fdir / COMPLETE).exists():
            continue
        if sess.stop_requested():
            return False
        if not sess.fits(sess.tp["seconds_per_fold_eval"]):
            sess.reason = "deadline (evaluation resumes next session)"
            return False
        t0 = time.time()
        bundle = _bundle(config_path, resolved, fold)
        cells = eval_grid("default", bundle.constraints)
        hd = bundle.constraints.lock.hold_days.primary
        dm = bundle.constraints.drawdown.max_drawdown.primary
        primary = cell_key(hd, dm)
        log(f"  fold {fold.test_year}: val {fold.val_start[:4]}, test {fold.test_year}")

        if camp.get("mode") == "ensemble":
            _evaluate_fold_ensemble(sess, fold, bundle, cells, label, fdir, hd, dm)
            (fdir / COMPLETE).write_text(label, encoding="utf-8")
            state.setdefault("folds_done", []).append(fold.test_year)
            sess.measure("seconds_per_fold_eval", time.time() - t0)
            sess.save()
            continue

        candidates, val_paths = [], {}
        for key in _fold_keys(camp, fold.test_year):
            c = camp["candidates"][key]
            pdir = model_dir(sess.cdir, key) / label
            cand = Candidate(name=c["name"], params={
                "ent_coef": c["ent_coef"], "learning_rate": c["learning_rate"],
                "seed": c["seed"], "policy_dir": str(pdir)})
            cand.validation, trajs = evaluate_window(
                load_policy(pdir), bundle, (fold.val_start, fold.val_end), cells,
                seed=EVAL_SEED, label=f"val {c['name']}", log=log)
            val_paths[c["name"]] = trajs[primary]
            candidates.append(cand)

        result = select(candidates, d_max=dm, fold_id=fold.fold_id)
        log(f"    selected: {result.chosen.name}")
        fdir.mkdir(parents=True, exist_ok=True)
        (fdir / "selection.json").write_text(
            json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        val_paths[result.chosen.name].to_parquet(fdir / "val_trajectory.parquet")

        test_agg, trajs = evaluate_window(
            load_policy(result.chosen.params["policy_dir"]), bundle,
            (fold.test_start, fold.test_end), cells, seed=EVAL_SEED,
            label=f"TEST {fold.test_year}", log=log)
        trajs[primary].to_parquet(fdir / "trajectory.parquet")
        (fdir / "cells").mkdir(exist_ok=True)
        for key, traj in trajs.items():
            traj.to_parquet(fdir / "cells" / f"{key}.parquet")
        taxonomy = summarize_violations(trajs[primary], bundle.market.universe,
                                        d_max=dm, market=bundle.market)
        (fdir / "violations.json").write_text(
            json.dumps(taxonomy, indent=2, default=str), encoding="utf-8")
        record = {
            "fold_id": fold.fold_id, "test_year": fold.test_year, "fold": fold.to_dict(),
            "selected": result.chosen.name, "criterion": result.criterion,
            "constraint_validation_failure": result.constraint_validation_failure,
            "validation": result.chosen.validation, "test": test_agg,
            "violations": {k: v for k, v in taxonomy.items() if k != "replay_findings"},
        }
        (fdir / "record.json").write_text(json.dumps(record, indent=2, default=str),
                                          encoding="utf-8")
        (fdir / COMPLETE).write_text(label, encoding="utf-8")
        state.setdefault("folds_done", []).append(fold.test_year)
        sess.measure("seconds_per_fold_eval", time.time() - t0)
        sess.save()

    if sess.stop_requested():
        return False
    if not sess.fits(sess.tp["seconds_for_report"]):
        sess.reason = "deadline (the report runs first thing next session)"
        return False
    t0 = time.time()
    records = [json.loads((ckdir / "folds" / str(f.test_year) / "record.json")
                          .read_text(encoding="utf-8")) for f in folds]
    # In an ensemble campaign every seed's own runs count too: a lock or feasibility
    # violation anywhere is a simulator defect, whichever policy produced it.
    lock = sum(r["test"]["lock_violations"]
               + sum(s["test"]["lock_violations"] for s in r.get("seeds", {}).values())
               for r in records)
    feas = sum(r["test"]["feasibility_violations"]
               + sum(s["test"]["feasibility_violations"] for s in r.get("seeds", {}).values())
               for r in records)
    prev = sum(r["test"]["preventable_violations"] for r in records)
    hard = {"lock_violations": lock, "feasibility_violations": feas,
            "preventable_dmax_violations": prev,
            "passed": lock == 0 and feas == 0 and prev == 0,
            "constraint_validation_failures": [r["fold_id"] for r in records
                                               if r["constraint_validation_failure"]]}
    (ckdir / "walk_forward_summary.json").write_text(json.dumps({
        "campaign": camp["campaign"], "checkpoint": label, "pct": cp["pct"],
        "timesteps_per_candidate": cp["timesteps"], "n_folds": len(records),
        "hard_acceptance": hard, "folds": records}, indent=2, default=str), encoding="utf-8")

    # The Stage 12 report, unchanged: the checkpoint's folds are one more rl_policy column,
    # against the market benchmarks AND their constrained versions (plan 5b).
    from scripts.s12_report import main as report_main

    from src.sim.runner import build_constraints

    constraints = build_constraints(resolved)
    hd = constraints.lock.hold_days.primary
    dm = constraints.drawdown.max_drawdown.primary
    name = f"incremental/{camp['campaign']}/{label}"
    rc = report_main([
        "--config", camp["config_path"], "--policy-runs", str(ckdir),
        "--policy-cell", cell_key(hd, dm), "--hold-days", str(hd), "--max-drawdown", str(dm),
        "--first-year", str(camp["first_year"]), "--last-year", str(camp["last_year"]),
        "--no-acceptance", "--name", name,
        "--title", f"Incremental {camp['campaign']}: {label} ({cp['pct']}% budget)",
        "--force", "--seed", str(EVAL_SEED)])
    if rc != 0:
        raise CampaignError(f"the Stage 12 report for {label} failed (exit {rc})")

    labels = [c["label"] for c in camp["schedule"]]
    idx = labels.index(label)
    previous = labels[idx - 1] if idx > 0 else None
    prev_cp = camp["schedule"][idx - 1] if idx > 0 else None
    row = build_record(
        label=label, pct=cp["pct"], checkpoint_dir=ckdir,
        report_dir=Path("artifacts/reports") / name,
        previous_dir=(sess.cdir / "checkpoints" / previous) if previous else None,
        previous_label=previous,
        training=_training_summary(camp, label, previous,
                                   prev_cp["rollouts"] if prev_cp else 0, cp["rollouts"]),
        hard_acceptance=hard, primary_cell=cell_key(hd, dm), replicates=replicates)
    (ckdir / "checkpoint.json").write_text(json.dumps(row, indent=2, default=str),
                                           encoding="utf-8")

    rows = [json.loads((sess.cdir / "checkpoints" / lab / "checkpoint.json")
                       .read_text(encoding="utf-8"))
            for lab in labels[:idx + 1]
            if (sess.cdir / "checkpoints" / lab / "checkpoint.json").exists()]
    text = render(rows, campaign=camp["campaign"], primary_cell=cell_key(hd, dm))
    (sess.cdir / "learning_curve.md").write_text(text, encoding="utf-8")
    (REPORTS / camp["campaign"]).mkdir(parents=True, exist_ok=True)
    (REPORTS / camp["campaign"] / "learning_curve.md").write_text(text, encoding="utf-8")

    state["status"] = "reported"
    state["reported_at"] = utc_now()
    sess.measure("seconds_for_report", time.time() - t0)
    sess.checkpoints_reported.append(label)
    sess.save()

    vp = row.get("vs_previous") or {}
    log(f"{label} reported: test Sharpe {row['test_sharpe']:.2f} "
        f"(band {row['test_sharpe_band']['q05']:.2f}-{row['test_sharpe_band']['q95']:.2f}), "
        f"val Sharpe {row['val_sharpe_selected']:.2f}, hard acceptance "
        f"{'PASS' if hard['passed'] else 'FAIL'}"
        + (f"; vs {previous}: val {vp['validation']['verdict']}, test {vp['test']['verdict']}"
           if vp else ""))
    log(f"learning curve: {sess.cdir / 'learning_curve.md'}")
    return True


def run_session(args, resolved: dict, chash: str) -> int:
    cdir = campaign_dir(args.campaign)
    campaign = load_campaign(cdir)
    check_config(campaign, chash, allow_change=args.allow_config_change)
    if args.allow_config_change and campaign["config_hash"] != chash:
        campaign.setdefault("config_overrides", []).append(
            {"at": utc_now(), "from": campaign["config_hash"], "to": chash})

    stamp = time.strftime("%Y%m%dT%H%M%S")
    log = Log(cdir / "sessions" / f"{stamp}.log")
    stop = cdir / STOP_FILE
    if stop.exists():
        stop.unlink()
        log("removed a stale STOP file left from an earlier request")

    sess = Session(cdir, campaign, args.hours, args.margin_minutes, log)
    log(f"session: {args.hours} h (stops by {time.strftime('%H:%M', time.localtime(sess.deadline))}"
        f" incl. a {args.margin_minutes:.0f}-min margin); campaign {args.campaign}, "
        f"average budget {average_budget_pct(campaign):.2f}%")
    for note in reconcile(cdir, campaign, campaign["rollout_size"]):
        log(f"reconcile: {note}")
    sess.save()
    folds = _folds(campaign)
    config_path = Path(campaign["config_path"])

    status = 0
    with KeepAwake():
        try:
            while True:
                if sess.stop_requested():
                    break
                if not sess.fits(60):
                    sess.reason = sess.reason or "deadline"
                    break
                cp = pending_checkpoint(campaign)
                if cp is not None:
                    if not evaluate_checkpoint(sess, cp, folds, config_path, resolved,
                                               args.replicates):
                        break
                    continue
                target = next_target(campaign)
                if target is None:
                    sess.reason = "campaign complete: every checkpoint is reported"
                    break
                if not train_to_target(sess, target, folds, config_path, resolved):
                    break
        except KeyboardInterrupt:
            sess.reason = "interrupted (Ctrl-C); the rollout in progress was discarded"
            status = 130
        except Exception as exc:
            sess.reason = f"FAILED: {type(exc).__name__}: {exc}"
            status = 1
            import traceback
            log(traceback.format_exc())
        finally:
            if stop.exists():
                stop.unlink()
            sess.record.update({
                "ended_at": utc_now(), "minutes": round((time.time() - sess.started) / 60, 1),
                "rollouts_trained": sess.rollouts_trained,
                "checkpoints_reported": sess.checkpoints_reported,
                "stopped_because": sess.reason or "deadline"})
            campaign["sessions"].append(sess.record)
            sess.save()
            log("")
            log(f"session over after {sess.record['minutes']} min: {sess.record['stopped_because']}")
            log(f"trained {sess.rollouts_trained} rollouts; reported "
                f"{sess.checkpoints_reported or 'no checkpoint'}; average budget now "
                f"{average_budget_pct(campaign):.2f}%")
    return status


# -------------------------------------------------------------- status / plan


def status(args) -> None:
    cdir = campaign_dir(args.campaign)
    camp = load_campaign(cdir)
    rolls = [c["rollouts"] for c in camp["candidates"].values()]
    print(f"campaign {camp['campaign']}  (from {camp['source_run']}, config {camp['config_hash']})")
    print(f"average budget {average_budget_pct(camp):.2f}%  -- rollouts per candidate "
          f"min {min(rolls)} / max {max(rolls)} of {camp['schedule'][-1]['rollouts']}")
    print("checkpoints:")
    for cp in camp["schedule"]:
        st = camp["checkpoints"].get(cp["label"], {})
        extra = ""
        if st.get("status") == "evaluating":
            n_folds = len({c["test_year"] for c in camp["candidates"].values()})
            extra = f" ({len(st.get('folds_done', []))}/{n_folds} folds evaluated)"
        print(f"  {cp['label']:<3} {cp['pct']:>3}%  {cp['rollouts']:>3} rollouts  "
              f"{st.get('status', '-')}{extra}")
    tp = camp.get("throughput") or {}
    print(f"measured throughput: {tp or 'none yet (defaults in use)'}")
    print(f"sessions so far: {len(camp['sessions'])}")
    for s in camp["sessions"][-3:]:
        print(f"  {s['started_at']}  {s.get('minutes', '?')} min, "
              f"{s.get('rollouts_trained', 0)} rollouts, {s.get('stopped_because', '')}")
    curve = cdir / "learning_curve.md"
    print(f"learning curve: {curve if curve.exists() else 'none yet'}")
    if (cdir / STOP_FILE).exists():
        print("NOTE: a STOP file is present")


def plan(args) -> None:
    camp = load_campaign(campaign_dir(args.campaign))
    out = plan_session(camp, args.plan, margin_minutes=args.margin_minutes)
    print(f"{out['hours']} h idle -> ~{out['usable_minutes']} usable minutes "
          f"(average budget now {average_budget_pct(camp):.2f}%):")
    for step in out["steps"]:
        print(f"  - {step}")
    print("throughput assumed: " + ", ".join(f"{k} {v:.0f}s"
                                            for k, v in out["throughput"].items()))


# ---------------------------------------------------------------------- main


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m scripts.s14_incremental",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="config/evaluation.yaml")
    p.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--init", metavar="S08_RUN",
                      help="create the campaign from a Stage 8 run's models (its C0)")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--plan", type=float, metavar="HOURS",
                      help="estimate what a session of HOURS would do; runs nothing")
    mode.add_argument("--hours", type=float, help="run a session of this many hours")
    mode.add_argument("--stop", action="store_true",
                      help="ask a running session to stop after its current rollout")
    mode.add_argument("--init-fresh", action="store_true",
                      help="create a seed-ensemble campaign: one --setting, --seeds models per "
                           "fold, trained from scratch, no per-year selection")
    mode.add_argument("--add-checkpoint", type=float, metavar="PCT",
                      help="insert a checkpoint at PCT%% of the budget; candidates already "
                           "past it are rolled back to their previous snapshot")
    p.add_argument("--margin-minutes", type=float, default=5.0,
                   help="stop this long before the stated idle time runs out")
    p.add_argument("--replicates", type=int, default=1000,
                   help="bootstrap replicates for the learning-curve bands and verdicts")
    p.add_argument("--allow-config-change", action="store_true")
    p.add_argument("--setting", default="high_entropy",
                   help="--init-fresh: the hyperparameter setting (a model_selection name)")
    p.add_argument("--seeds", type=int, default=3, help="--init-fresh: seeds per fold")
    p.add_argument("--seed-base", type=int, default=1001,
                   help="--init-fresh: first seed; the rest follow consecutively")
    args = p.parse_args(argv)

    try:
        if args.status:
            status(args)
            return 0
        if args.plan is not None:
            plan(args)
            return 0
        if args.add_checkpoint is not None:
            cdir = campaign_dir(args.campaign)
            campaign = load_campaign(cdir)
            for note in reconcile(cdir, campaign, campaign["rollout_size"]):
                print(f"reconcile: {note}")
            for note in insert_checkpoint(cdir, campaign, args.add_checkpoint):
                print(note)
            save_campaign(cdir, campaign)
            return 0
        if args.stop:
            cdir = campaign_dir(args.campaign)
            load_campaign(cdir)
            (cdir / STOP_FILE).write_text(utc_now(), encoding="utf-8")
            print("stop requested: a running session stops after its current rollout "
                  "(or fold evaluation) and saves.")
            return 0
        resolved = resolve_config(Path(args.config))
        chash = config_hash(resolved)
        if args.init:
            init(args, resolved, chash, Log(None))
            return 0
        if args.init_fresh:
            init_fresh(args, resolved, chash, Log(None))
            return 0
        if args.hours <= 0:
            raise CampaignError("--hours must be positive")
        return run_session(args, resolved, chash)
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
