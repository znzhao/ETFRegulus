"""Stage 6 -- the environment smoke test.

    python -m scripts.s06_smoke_env --config config/training.yaml --episodes 500
    python -m scripts.s06_smoke_env --config config/training.yaml --skip-throughput

**Not a training run.** A random policy that has learned nothing must still be unable to
violate a constraint, and this is where that gets established -- cheaply, before Stage 7
spends hours discovering it. Six jobs:

1. `gymnasium.utils.env_checker.check_env` and SB3's `check_env`;
2. random-policy episodes across the full `(N, D_max)` grid and every reset mode, with
   every invariant asserted on *every* step;
3. observation bounds, dtype, and no NaN or inf anywhere, ever;
4. determinism -- same seed, byte-identical trajectory, and identical per worker
   regardless of how many workers there are (T9);
5. the reset sampler produces only reachable states, with `D_t <= D_max` in normal mode
   (T8);
6. the throughput benchmark. D5 depends on knowing steps/second before any hyperparameter
   is tuned, and D14's CPU-over-GPU verdict was measured on a smaller observation than
   the one this stage settles, so it is re-checked here rather than assumed.

Writes `artifacts/runs/<run_id>/smoke_report.json`. Fails the stage on any violation.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter

import numpy as np

from src.cli.stage import StageContext, StageError, stage
from src.env.etf_env import EnvConfig, InvariantViolation, softmax_weights
from src.env.factory import build_bundle, make_env, make_vec_env
from src.env.observation import MANDATORY_FIELDS
from src.env.reset_sampler import ConstructiveSampler, is_reachable


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--episodes", type=int, default=None,
                   help="random-policy episodes (default: smoke.episodes)")
    p.add_argument("--episode-length", type=int, default=None)
    p.add_argument("--skip-throughput", action="store_true")
    p.add_argument("--skip-checkers", action="store_true")
    p.add_argument("--fold", default=None, help="override environment.fold_id")


def _replace(cfg: EnvConfig, **kw) -> EnvConfig:
    return EnvConfig(**{**cfg.__dict__, **kw})


# --------------------------------------------------------------------- the checks


def run_env_checkers(bundle) -> dict:
    """Both checkers, with warnings captured rather than printed and forgotten."""
    import warnings

    from gymnasium.utils.env_checker import check_env as gym_check
    from stable_baselines3.common.env_checker import check_env as sb3_check

    out: dict = {}
    for name, fn in (("gymnasium", gym_check), ("stable_baselines3", sb3_check)):
        env = make_env(bundle, seed=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn(env, skip_render_check=True)
            out[name] = {"passed": True,
                         "warnings": [str(w.message) for w in caught]}
        env.close()
    return out


def sweep_episodes(bundle, *, n_episodes: int, length: int, base_seed: int,
                   log) -> dict:
    """Random-policy episodes across the whole grid, asserting everything, every step."""
    cfg = _replace(bundle.env_cfg, strict=True,
                   episode_lengths=(length,), stress_reset=False)
    env = make_env(bundle, seed=base_seed, env_cfg=cfg)
    rng = np.random.default_rng(base_seed)

    grid = [(n, d) for n in cfg.hold_days_values for d in cfg.max_drawdown_values]
    cells = Counter()
    sources = Counter()
    n_steps = 0
    obs_min, obs_max = np.inf, -np.inf
    stats = {k: 0 for k in ("safety_intervened", "capital_preservation",
                            "infeasible_fallback")}
    proj = []
    rewards = []
    terminated_ever = False
    violations: list[str] = []

    low = env.observation_space.low
    high = env.observation_space.high

    for ep in range(n_episodes):
        # Every episode is a deliberate grid cell rather than a draw, so 500 episodes
        # cover all 25 cells evenly instead of leaving the tails to chance.
        n, d = grid[ep % len(grid)]
        stress = (ep % 17 == 16)      # exercise the stress reset mode too
        obs, info = env.reset(seed=base_seed + ep,
                              options={"hold_days": n, "max_drawdown": d,
                                       "length": length, "stress": stress})
        cells[(n, d)] += 1
        sources[info["episode"]["source"]] += 1
        _check_obs(obs, low, high, env, violations, where=f"reset ep{ep}")

        # T8, restated at the point of use: the sampler's own guarantee, re-verified
        # against the environment that consumed it.
        if not stress and info["drawdown"] > d + 1e-9:
            violations.append(
                f"ep{ep}: reset D_t={info['drawdown']:.4f} > D_max={d:.4f}")

        done = False
        while not done:
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            obs, reward, term, trunc, info = env.step(action)
            n_steps += 1
            terminated_ever = terminated_ever or bool(term)
            _check_obs(obs, low, high, env, violations, where=f"step ep{ep}")
            obs_min = min(obs_min, float(obs.min()))
            obs_max = max(obs_max, float(obs.max()))
            for k in stats:
                stats[k] += int(bool(info[k]))
            proj.append(float(info["proj_distance"]))
            rewards.append(float(reward))
            if info["lock_violations"]:
                violations.append(f"ep{ep}: {info['lock_violations']} lock violation(s)")
            done = term or trunc
        violations.extend(env.violations)
        if ep and ep % 100 == 0:
            log(f"  {ep}/{n_episodes} episodes, {n_steps} steps, "
                f"{len(violations)} violations")

    env.close()
    return {
        "n_episodes": n_episodes, "n_steps": n_steps,
        "grid_cells_visited": len(cells), "grid_cells_total": len(grid),
        "min_visits_per_cell": min(cells.values()) if cells else 0,
        "reset_sources": dict(sources),
        "terminated_ever": terminated_ever,
        "obs_min": obs_min, "obs_max": obs_max,
        "mean_proj_distance": float(np.mean(proj)) if proj else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "rates": {k: v / max(n_steps, 1) for k, v in stats.items()},
        "violations": violations[:50],
        "n_violations": len(violations),
    }


def _check_obs(obs, low, high, env, violations: list[str], *, where: str) -> None:
    if obs.dtype != np.float32:
        violations.append(f"{where}: observation dtype {obs.dtype}, expected float32")
    if obs.shape != env.observation_space.shape:
        violations.append(f"{where}: shape {obs.shape} != {env.observation_space.shape}")
    if not np.isfinite(obs).all():
        violations.append(f"{where}: observation contains NaN or inf")
    elif ((obs < low - 1e-6) | (obs > high + 1e-6)).any():
        violations.append(f"{where}: observation outside its declared Box")


def check_reset_sampler(bundle, *, n_draws: int, base_seed: int) -> dict:
    """T8, directly on the sampler rather than through the environment.

    Both generators, every grid cell, and the constructive one also asked for the corners
    the reservoir under-samples -- a high locked fraction and a near-ceiling drawdown --
    because a reachability check that only ever sees easy states proves very little.
    """
    cfg = bundle.env_cfg
    env = make_env(bundle, seed=base_seed)
    sampler = env.sampler
    rng = np.random.default_rng(base_seed)
    window = env._window

    unreachable: list[str] = []
    breaches: list[str] = []
    by_source = Counter()
    per_cell = {}

    grid = [(n, d) for n in cfg.hold_days_values for d in cfg.max_drawdown_values]
    per_draw = max(1, n_draws // len(grid))
    for n, d in grid:
        ok = 0
        for _ in range(per_draw):
            state = sampler.sample(rng, hold_days=n, max_drawdown=d, window=window)
            by_source[state.source] += 1
            reasons = is_reachable(state, bundle.market, hold_days=n, max_drawdown=d)
            if reasons:
                unreachable.append(f"N={n} D={d}: {reasons[0]}")
            if state.drawdown > d + 1e-9:
                breaches.append(f"N={n} D={d}: reset drawdown {state.drawdown:.4f}")
            ok += not reasons
        per_cell[f"N{n}_D{d}"] = ok

    # Approach B, pushed into the corners on purpose.
    corner = ConstructiveSampler(bundle.market)
    n_corner_ok = 0
    for _ in range(100):
        state = corner.sample(rng, hold_days=30, max_drawdown=0.15, window=window,
                              min_locked_fraction=0.9, min_drawdown=0.14)
        if state is None:
            continue
        reasons = is_reachable(state, bundle.market, hold_days=30, max_drawdown=0.15)
        if reasons:
            unreachable.append(f"corner: {reasons[0]}")
        else:
            n_corner_ok += 1

    env.close()
    return {
        "n_draws": per_draw * len(grid), "sources": dict(by_source),
        "cells_all_reachable": all(v == per_draw for v in per_cell.values()),
        "n_unreachable": len(unreachable), "unreachable": unreachable[:20],
        "n_dmax_breaches_at_reset": len(breaches), "breaches": breaches[:20],
        "constructive_corner_ok": n_corner_ok,
        "constructive_rejections": dict(list(corner.rejections.items())[:10]),
    }


def check_determinism(bundle, *, base_seed: int, repeats: int, steps: int) -> dict:
    """T9. Same seed, byte-identical. Then the same, per worker, across worker counts."""
    cfg = _replace(bundle.env_cfg, strict=False, episode_lengths=(steps + 5,))

    def rollout(seed: int) -> tuple[np.ndarray, np.ndarray]:
        env = make_env(bundle, seed=seed, env_cfg=cfg)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        o, r = [obs.copy()], []
        for _ in range(steps):
            a = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
            obs, rew, _, trunc, _ = env.step(a)
            o.append(obs.copy())
            r.append(rew)
            if trunc:
                break
        env.close()
        return np.array(o), np.array(r)

    ref_o, ref_r = rollout(base_seed)
    identical = True
    for _ in range(repeats - 1):
        o, r = rollout(base_seed)
        identical &= bool(np.array_equal(o, ref_o) and np.array_equal(r, ref_r))

    # Across worker counts: worker i derives seed base+i, so worker 0's stream must not
    # depend on how many siblings it has.
    #
    # The actions must be generated PER WORKER, from that worker's own RNG. Drawing one
    # (n_envs, K+1) block from a shared generator advances it n_envs times per step, so
    # worker 0 would see different actions at 1 worker than at 4 -- which is a property of
    # the test harness, not of the environment, and reads as a determinism failure. This
    # mirrors the real rule: each worker owns its stream and never shares one.
    per_count = {}
    for n_envs in (1, 4):
        vec = make_vec_env(bundle, n_envs, base_seed=base_seed, env_cfg=cfg)
        obs = vec.reset()
        rngs = [np.random.default_rng(base_seed + i) for i in range(n_envs)]
        trace = [obs[0].copy()]
        for _ in range(10):
            a = np.stack([r.uniform(-1, 1, vec.action_space.shape) for r in rngs]
                         ).astype(np.float32)
            obs, rew, done, _ = vec.step(a)
            trace.append(obs[0].copy())
        vec.close()
        per_count[n_envs] = np.array(trace)

    worker0_stable = bool(np.array_equal(per_count[1], per_count[4]))
    return {
        "same_seed_identical": identical,
        "repeats": repeats,
        "worker0_identical_across_worker_counts": worker0_stable,
        "n_obs_compared": int(ref_o.size),
    }


def check_observation_contract(bundle) -> dict:
    """T14. Every mandatory Markov field present, and the layout addressable by name."""
    spec = bundle.spec
    report = spec.validate()
    names = spec.build_names()
    missing = [f for f in MANDATORY_FIELDS
               if not any(n.split(":")[-1] == f for n in names)]
    # The blocks must tile the vector exactly -- no gap, no overlap.
    edges = [spec.macro_slice, spec.per_asset_slice, spec.portfolio_slice,
             spec.param_slice]
    contiguous = (edges[0].start == 0 and edges[-1].stop == spec.size
                  and all(a.stop == b.start for a, b in zip(edges, edges[1:])))
    return {
        **report,
        "missing_mandatory_fields": missing,
        "blocks_tile_exactly": contiguous,
        "n_names": len(names),
        "names_unique": len(set(names)) == len(names),
        "layout": spec.describe(),
    }


# ---------------------------------------------------------------- the benchmark


def benchmark(bundle, *, worker_counts, steps_per_worker: int, devices, log) -> dict:
    """steps/second for DummyVecEnv vs SubprocVecEnv, and the device check for D14."""
    import torch

    # `strict=False` is the training configuration; benchmarking with the Stage 6 checks
    # on would measure this stage rather than Stage 7.
    cfg = _replace(bundle.env_cfg, strict=False)
    rows = []
    for kind, subproc in (("dummy", False), ("subproc", True)):
        for n in worker_counts:
            if kind == "dummy" and n > 8:
                continue           # DummyVecEnv is serial; past 8 it only measures itself
            try:
                vec = make_vec_env(bundle, n, base_seed=1, subproc=subproc, env_cfg=cfg)
                vec.reset()
                a = np.zeros((n, *vec.action_space.shape), dtype=np.float32)
                for _ in range(5):
                    vec.step(a)     # warm up: import cost is not step cost
                t0 = time.perf_counter()
                for _ in range(steps_per_worker):
                    vec.step(a)
                elapsed = time.perf_counter() - t0
                vec.close()
                rows.append({"vec": kind, "n_envs": n,
                             "env_steps": steps_per_worker * n,
                             "seconds": round(elapsed, 3),
                             "steps_per_second": round(steps_per_worker * n / elapsed, 1)})
                log(f"  {kind:>7} x{n:<3} {rows[-1]['steps_per_second']:>9.1f} steps/s")
            except Exception as exc:
                rows.append({"vec": kind, "n_envs": n, "error": f"{type(exc).__name__}: {exc}"})
                log(f"  {kind:>7} x{n:<3} FAILED: {exc}", )

    # D14 was decided on a ~300-dim observation; this one is larger, so the premise is
    # re-measured rather than inherited.
    device_rows = []
    obs_dim = bundle.obs_dim
    for dev in devices:
        if dev == "cuda" and not torch.cuda.is_available():
            device_rows.append({"device": dev, "available": False})
            continue
        net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256), torch.nn.Tanh(),
            torch.nn.Linear(256, 256), torch.nn.Tanh(),
            torch.nn.Linear(256, bundle.market.n_assets + 1)).to(dev)
        x = torch.randn(2048, obs_dim, device=dev)
        for _ in range(5):
            net(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            net(x).sum().backward()
        if dev == "cuda":
            torch.cuda.synchronize()
        device_rows.append({"device": dev, "available": True,
                            "seconds_50_fwd_bwd": round(time.perf_counter() - t0, 3)})

    best = max((r for r in rows if "steps_per_second" in r),
               key=lambda r: r["steps_per_second"], default=None)
    return {"table": rows, "devices": device_rows, "obs_dim": obs_dim,
            "best": best,
            "cpu_count": __import__("os").cpu_count(),
            "platform": platform.platform()}


# ---------------------------------------------------------------------- the stage


@stage(
    name="s06_smoke_env",
    config_default="config/training.yaml",
    inputs=["data/curated/prices.parquet", "data/features/feature_manifest.json"],
    outputs=["artifacts/runs/{run_id}/smoke_report.json"],
    upstream="s05_run_baselines",
    add_args=_add_args,
)
def main(resolved: dict, ctx: StageContext) -> None:
    """Random-policy sweep, invariant assertions, determinism, and the throughput table."""
    args = ctx.args
    smoke = resolved.get("smoke", {}) or {}
    n_episodes = int(args.episodes or smoke.get("episodes", 500))
    length = int(args.episode_length or smoke.get("episode_length", 63))

    ctx.log("building the environment bundle")
    bundle = build_bundle(ctx.config_path, resolved=resolved, fold_id=args.fold)
    ctx.log(f"obs_dim={bundle.obs_dim}  assets={bundle.market.n_assets}  "
            f"sessions={len(bundle.market.sessions)}  fold={bundle.store.fold_id}  "
            f"scaler fit {bundle.store.fit_range}")

    report: dict = {"obs_dim": bundle.obs_dim, "seed": ctx.seed,
                    "fold_id": bundle.store.fold_id,
                    "scaler_fit_range": list(bundle.store.fit_range or ())}

    ctx.log("T14: observation contract against the feature manifest")
    report["observation"] = check_observation_contract(bundle)

    if not args.skip_checkers and smoke.get("check_env", True):
        ctx.log("gymnasium and SB3 env checkers")
        report["env_checkers"] = run_env_checkers(bundle)

    ctx.log("T8: reset sampler reachability across the whole grid")
    report["reset_sampler"] = check_reset_sampler(bundle, n_draws=250,
                                                  base_seed=ctx.seed)

    ctx.log(f"random-policy sweep: {n_episodes} episodes of {length} steps")
    t0 = time.time()
    report["sweep"] = sweep_episodes(bundle, n_episodes=n_episodes, length=length,
                                     base_seed=ctx.seed, log=ctx.log)
    report["sweep"]["wall_seconds"] = round(time.time() - t0, 2)

    ctx.log("T9: determinism")
    report["determinism"] = check_determinism(
        bundle, base_seed=ctx.seed,
        repeats=int(smoke.get("determinism_repeats", 2)), steps=40)

    if not args.skip_throughput:
        tp = smoke.get("throughput", {}) or {}
        ctx.log("throughput benchmark (D5 / Q2 depend on this table)")
        report["throughput"] = benchmark(
            bundle,
            worker_counts=tp.get("worker_counts", [1, 4, 8, 16]),
            steps_per_worker=int(tp.get("steps_per_worker", 200)),
            devices=tp.get("devices", ["cpu", "cuda"]), log=ctx.log)

    # ------------------------------------------------------------- the gate
    checks = {
        "no_invariant_violations": {
            "passed": report["sweep"]["n_violations"] == 0,
            "detail": report["sweep"]["violations"][:5],
        },
        "every_grid_cell_visited": {
            "passed": (report["sweep"]["grid_cells_visited"]
                       == report["sweep"]["grid_cells_total"]),
            "detail": f"{report['sweep']['grid_cells_visited']}"
                      f"/{report['sweep']['grid_cells_total']}",
        },
        "never_terminated": {
            "passed": not report["sweep"]["terminated_ever"],
            "detail": "there is no terminated=True condition at all (env-mdp.md 7)",
        },
        "reset_states_reachable": {
            "passed": report["reset_sampler"]["n_unreachable"] == 0,
            "detail": report["reset_sampler"]["unreachable"][:5],
        },
        "reset_respects_dmax": {
            "passed": report["reset_sampler"]["n_dmax_breaches_at_reset"] == 0,
            "detail": report["reset_sampler"]["breaches"][:5],
        },
        "determinism": {
            "passed": (report["determinism"]["same_seed_identical"]
                       and report["determinism"]["worker0_identical_across_worker_counts"]),
            "detail": report["determinism"],
        },
        "observation_matches_manifest": {
            "passed": (not report["observation"]["missing_mandatory_fields"]
                       and report["observation"]["blocks_tile_exactly"]
                       and report["observation"]["names_unique"]),
            "detail": report["observation"]["missing_mandatory_fields"],
        },
    }
    report["checks"] = checks

    out = ctx.out_path("smoke_report.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    ctx.record(obs_dim=bundle.obs_dim,
               n_steps=report["sweep"]["n_steps"],
               n_violations=report["sweep"]["n_violations"],
               checks_passed=sum(c["passed"] for c in checks.values()),
               checks_total=len(checks))

    ctx.log("")
    for name, c in checks.items():
        ctx.log(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name}")
    failed = [n for n, c in checks.items() if not c["passed"]]
    if failed:
        raise StageError(
            f"Stage 6 gate failed: {failed}. Full report: {out}\n"
            "Stage 7 must not run until every check passes -- these are the bugs that "
            "otherwise surface six hours into a training run."
        )
    ctx.log(f"all {len(checks)} checks passed; report: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
