"""Stage 7 -- train PPO through the five-stage curriculum.

    python -m scripts.s07_train_ppo --config config/experiments/ppo_stage1.yaml
    python -m scripts.s07_train_ppo --config config/experiments/ppo_stage3.yaml --resume-from <run_id>
    python -m scripts.s07_train_ppo --config config/training.yaml --curriculum

One rung per invocation, each warm-started from the previous, or `--curriculum` to run all
five in order in a single run directory. Implements reference/rl-training.md: SB3 PPO (D2),
pre-projection action storage (D9), the shared per-asset policy, and the curriculum with
warm-starting.

**The gate between rungs is not advisory.** A stage advances only if it beats the `cash`
baseline on its own training window and records zero lock and zero feasibility violations.
Continuing past a violation trains the next rung against a constraint layer that is known
broken -- so `ConstraintMonitor` raises on the first one rather than logging it, and
`--curriculum` stops at the first failed gate unless told otherwise.

Writes `artifacts/runs/<run_id>/`:

    stage<k>_<name>/policy.zip          the trained policy
    stage<k>_<name>/vecnormalize.pkl    its reward normalizer -- meaningless apart
    stage<k>_<name>/metrics.json        diagnostics and the gate verdict
    stage<k>_<name>/checkpoints/        periodic snapshots
    tensorboard/                        the curves
    curriculum.json                     every rung's verdict in one place
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cli.stage import RUNS, StageContext, StageError, stage
from src.env.factory import build_bundle
from src.training.curriculum import CURRICULUM, STAGES_BY_INDEX
from src.training.trainer import train_stage


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stage", type=int, default=None,
                   help="curriculum rung 1-5; inferred from the config filename if omitted")
    p.add_argument("--curriculum", action="store_true",
                   help="run every rung in order, warm-starting each from the last")
    p.add_argument("--resume-from", default=None,
                   help="run id (or path) whose policy to warm-start from")
    p.add_argument("--timesteps", type=int, default=None,
                   help="override training.total_timesteps, per rung")
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--no-subproc", action="store_true",
                   help="DummyVecEnv instead of SubprocVecEnv; slower, easier to debug")
    p.add_argument("--continue-on-gate-failure", action="store_true",
                   help="do not stop the curriculum when a rung fails its gate")


def infer_stage(config_path: Path, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    name = config_path.stem
    for i in STAGES_BY_INDEX:
        if name.endswith(f"stage{i}"):
            return i
    raise StageError(
        f"cannot tell which curriculum rung {config_path} is. Name it "
        "`ppo_stage<k>.yaml`, pass --stage, or use --curriculum.")


def resolve_previous(resume_from: str | None) -> Path | None:
    """Accept a run id, a run directory, or a specific stage directory."""
    if not resume_from:
        return None
    candidate = Path(resume_from)
    if (candidate / "policy.zip").exists():
        return candidate
    if not candidate.exists():
        candidate = RUNS / resume_from
    if (candidate / "policy.zip").exists():
        return candidate
    stages = sorted(candidate.glob("stage*_*/policy.zip"))
    if stages:
        return stages[-1].parent
    raise StageError(
        f"no policy.zip under {resume_from!r}. Pass a Stage 7 run id, a run directory, "
        "or a specific stage directory.")


def baseline_reference() -> dict:
    """Stage 5's baselines as flat reference lines on the reward chart.

    Reported as mean DAILY log return, which is the unit the episode-return curve is in.
    The bar that matters is not `cash` -- it is `spy_tlt_60_40`, a two-line static
    allocation anyone could implement.
    """
    runs = sorted(Path("artifacts/runs").glob("s05_run_baselines_*/baseline_summary.json"))
    if not runs:
        return {"cash": 0.0}
    summary = json.loads(runs[-1].read_text(encoding="utf-8"))
    out = {"cash": 0.0}
    for name, m in summary.get("baselines", {}).items():
        sessions = m.get("n_sessions") or 0
        cumulative = m.get("cumulative_return")
        if sessions and cumulative is not None and cumulative > -1.0:
            import math

            out[name] = math.log1p(cumulative) / sessions
    return out


@stage(
    name="s07_train_ppo",
    config_default="config/training.yaml",
    inputs=["data/curated/prices.parquet", "data/features/feature_manifest.json"],
    outputs=["artifacts/runs/{run_id}/curriculum.json"],
    upstream="s06_smoke_env",
    add_args=_add_args,
)
def main(resolved: dict, ctx: StageContext) -> None:
    """Train one curriculum rung, or the whole ladder."""
    args = ctx.args
    training = resolved.get("training", {}) or {}
    n_envs = int(args.n_envs or training.get("n_envs", 8))
    timesteps = int(args.timesteps or training.get("total_timesteps", 2_000_000))
    subproc = bool(training.get("subproc", True)) and not args.no_subproc

    ctx.log("building the environment bundle")
    bundle = build_bundle(ctx.config_path, resolved=resolved)
    ctx.log(f"obs_dim={bundle.obs_dim}  assets={bundle.market.n_assets}  "
            f"fold={bundle.store.fold_id}  scaler fit {bundle.store.fit_range}")
    if bundle.store.fold_id is None:
        raise StageError(
            "environment.fold_id is null. Training on unscaled observations puts ~3.7% of "
            "them hard against the clip (measured in Stage 6); name a fold.")

    baselines = baseline_reference()
    ctx.log(f"baseline reference lines: "
            + ", ".join(f"{k}={v:+.2e}" for k, v in sorted(baselines.items())))

    rungs = list(CURRICULUM) if args.curriculum else [
        STAGES_BY_INDEX[infer_stage(ctx.config_path, args.stage)]]
    previous = resolve_previous(args.resume_from)

    outcomes = []
    for rung in rungs:
        ctx.log("")
        outcome = train_stage(
            bundle, rung, resolved, run_dir=ctx.run_dir, seed=ctx.seed,
            total_timesteps=timesteps, n_envs=n_envs, subproc=subproc,
            resume_from=previous, baselines=baselines,
            checkpoint_every=int(training.get("checkpoint_every", 0)),
            log=ctx.log,
        )
        outcomes.append(outcome)
        previous = outcome.run_dir
        if not outcome.passed and not args.continue_on_gate_failure:
            break

    record = {
        "run_id": ctx.run_id, "seed": ctx.seed, "n_envs": n_envs,
        "timesteps_per_stage": timesteps, "obs_dim": bundle.obs_dim,
        "fold_id": bundle.store.fold_id,
        "baselines": baselines,
        "stages": [{"stage": o.stage, "name": o.name, "passed": o.passed,
                    "timesteps": o.timesteps, "wall_seconds": o.wall_seconds,
                    "steps_per_second": round(o.timesteps / max(o.wall_seconds, 1e-9), 1),
                    "n_parameters": o.n_parameters,
                    "diagnostics": o.diagnostics, "gate": o.gate,
                    "dir": str(o.run_dir)} for o in outcomes],
    }
    ctx.out_path("curriculum.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8")

    ctx.record(stages_run=len(outcomes),
               stages_passed=sum(o.passed for o in outcomes),
               final_stage=outcomes[-1].stage if outcomes else None)

    ctx.log("")
    ctx.log(f"{'rung':<22} {'steps':>10} {'ep log ret':>11} {'proj dist':>10} "
            f"{'cash w':>7} {'gate':>6}")
    for o in outcomes:
        d = o.diagnostics
        ctx.log(f"{o.stage} {o.name:<20} {o.timesteps:>10,} "
                f"{d.get('episode_log_return', 0):>+11.4f} "
                f"{d.get('proj_distance', 0):>10.3f} "
                f"{d.get('cash_weight', 0):>7.3f} "
                f"{'PASS' if o.passed else 'FAIL':>6}")

    failed = [o for o in outcomes if not o.passed]
    if failed:
        raise StageError(
            "curriculum rung(s) failed the gate: "
            + ", ".join(f"{o.stage} ({o.name})" for o in failed)
            + f". See {ctx.run_dir / 'curriculum.json'}.\n"
            "A failed gate on rung 1 in particular points at the simulator or the reward "
            "rather than at the policy -- that rung has no lock, no envelope and a fixed "
            "ceiling, so there is nothing else for it to be."
        )
    ctx.log(f"\nall {len(outcomes)} rung(s) passed. "
            f"Policy: {outcomes[-1].run_dir / 'policy.zip'}")


if __name__ == "__main__":
    raise SystemExit(main())
