"""The shared stage harness.

Every entry point in `scripts/` is a thin file wrapping a function with `@stage`. The
decorator provides, uniformly: `--config`, `--dry-run`, `--force`, `--seed`, staleness
detection against upstream artifacts, a run id, a run manifest written on both success
and failure, and structured logging to stdout and to the run directory.

Everything downstream depends on this. See reference/architecture.md section 2.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

from src.config.loader import config_hash, resolve_config, strict_from_dict

ARTIFACTS = Path("artifacts")
RUNS = ARTIFACTS / "runs"

#: Which stage produces which artifact. Consulted only to print the exact command to
#: run when an input is missing or stale -- a wrong guess here costs a helpful message,
#: never correctness.
ARTIFACT_PRODUCERS: dict[str, str] = {
    "data/raw": "python -m scripts.s01_fetch_data --config config/universe.yaml",
    "data/curated": "python -m scripts.s02_curate_data --config config/universe.yaml",
    "data/features": "python -m scripts.s03_build_features --config config/features.yaml",
}

_PACKAGES_OF_INTEREST = (
    "numpy", "pandas", "pyarrow", "scipy", "torch", "stable-baselines3",
    "gymnasium", "cvxpy", "yfinance", "exchange-calendars",
)


class StageError(RuntimeError):
    """A stage refused to run, or failed. Carries a human-actionable message."""


# ------------------------------------------------------------------------ helpers


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _newest_mtime(path: Path) -> float:
    """Modification time of a file, or of the newest file under a directory."""
    if path.is_file():
        return path.stat().st_mtime
    times = [p.stat().st_mtime for p in path.rglob("*") if p.is_file()]
    if not times:
        raise StageError(f"{path} exists but contains no files")
    return max(times)


def _producer_hint(path: Path) -> str:
    key = str(path).replace("\\", "/")
    for prefix, cmd in sorted(ARTIFACT_PRODUCERS.items(), key=lambda kv: -len(kv[0])):
        if key.startswith(prefix):
            return cmd
    return "(no registered producer -- check reference/stages.md)"


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except Exception:
        return ""


def git_state() -> tuple[str, bool]:
    sha = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return sha, dirty


def package_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for name in _PACKAGES_OF_INTEREST:
        try:
            out[name] = md.version(name)
        except Exception:
            pass
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


# ------------------------------------------------------------------------ context


@dataclasses.dataclass
class StageContext:
    """What a stage body gets. Everything a stage needs to be reproducible."""

    stage: str
    run_id: str
    run_dir: Path
    seed: int
    config_path: Path
    config_hash: str
    resolved_config: dict
    inputs: list[Path]
    outputs: list[Path]
    dry_run: bool
    force: bool
    args: argparse.Namespace
    started_at: str = dataclasses.field(default_factory=utc_stamp)
    _log_fh: Any = None
    extra: dict = dataclasses.field(default_factory=dict)

    def log(self, msg: str, level: str = "INFO") -> None:
        now = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
        line = f"{now} {level:<5} [{self.stage}] {msg}"
        print(line, flush=True)
        if self._log_fh is not None:
            self._log_fh.write(line + "\n")
            self._log_fh.flush()

    def warn(self, msg: str) -> None:
        self.log(msg, level="WARN")

    def record(self, **kv: Any) -> None:
        """Attach stage-specific values to the manifest."""
        self.extra.update(kv)

    def out_path(self, name: str) -> Path:
        """A path inside this run's directory, with parents created."""
        p = self.run_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


# ---------------------------------------------------------------------- staleness


def check_inputs(inputs: Sequence[Path], force: bool) -> None:
    """Refuse to run if a declared input is missing, naming the stage to run first."""
    missing = [p for p in inputs if not p.exists()]
    if missing and not force:
        lines = ["missing required input artifact(s):"]
        for p in missing:
            lines.append(f"  {p}")
            lines.append(f"    run first:  {_producer_hint(p)}")
        raise StageError("\n".join(lines))


def outputs_are_fresh(inputs: Sequence[Path], outputs: Sequence[Path]) -> bool:
    """True when every output exists and is newer than every input."""
    existing = [p for p in outputs if p.exists()]
    if not outputs or len(existing) != len(outputs):
        return False
    newest_in = max((_newest_mtime(p) for p in inputs if p.exists()), default=0.0)
    oldest_out = min(_newest_mtime(p) for p in existing)
    return oldest_out >= newest_in


# ----------------------------------------------------------------------- manifest


def write_manifest(ctx: StageContext, status: str, error: str | None = None) -> Path:
    """Written on success *and* on failure -- a failed run is a run, and it is recorded."""
    sha, dirty = git_state()
    inputs = []
    for p in ctx.inputs:
        entry: dict[str, Any] = {"path": str(p)}
        if p.is_file():
            entry["sha256"] = sha256_file(p)
            entry["mtime"] = dt.datetime.fromtimestamp(
                p.stat().st_mtime, dt.timezone.utc
            ).isoformat()
        elif p.exists():
            entry["mtime"] = dt.datetime.fromtimestamp(
                _newest_mtime(p), dt.timezone.utc
            ).isoformat()
            entry["note"] = "directory; newest contained mtime"
        else:
            entry["note"] = "missing"
        inputs.append(entry)

    manifest = {
        "run_id": ctx.run_id,
        "stage": ctx.stage,
        "git_sha": sha,
        "git_dirty": dirty,
        "config_hash": ctx.config_hash,
        "config_path": str(ctx.config_path),
        "seed": ctx.seed,
        "inputs": inputs,
        "outputs": [str(p) for p in ctx.outputs],
        "started_at": ctx.started_at,
        "finished_at": utc_stamp(),
        "status": status,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "argv": sys.argv[1:],
    }
    if error:
        manifest["error"] = error
    manifest.update(ctx.extra)

    path = ctx.run_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------- decorator


def stage(
    *,
    name: str,
    config_default: str | None = None,
    inputs: Sequence[str] = (),
    outputs: Sequence[str] = (),
    upstream: str | None = None,
    config_cls: type | None = None,
    add_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> Callable:
    """Wrap a stage body into a runnable entry point.

    `inputs`/`outputs` are path templates; `{run_id}` is substituted. `config_cls`, when
    given, is the dataclass the resolved config is validated into -- an unknown key then
    fails before any work starts.
    """

    def decorate(fn: Callable[[Any, StageContext], Any]):
        def main(argv: Sequence[str] | None = None) -> int:
            parser = argparse.ArgumentParser(
                prog=f"python -m scripts.{name}", description=fn.__doc__
            )
            parser.add_argument(
                "--config", default=config_default, required=config_default is None,
                help="path to the stage config (resolved through `extends`)",
            )
            parser.add_argument(
                "--dry-run", action="store_true",
                help="print the resolved config and the I/O plan, then exit without side effects",
            )
            parser.add_argument(
                "--force", action="store_true",
                help="run even if outputs are newer than inputs",
            )
            parser.add_argument(
                "--seed", type=int, default=42,
                help="seeds random/numpy/torch; recorded in the manifest",
            )
            if add_args is not None:
                add_args(parser)
            args = parser.parse_args(list(argv) if argv is not None else None)

            config_path = Path(args.config)
            resolved = resolve_config(config_path)
            chash = config_hash(resolved)
            cfg = strict_from_dict(config_cls, resolved) if config_cls is not None else resolved

            run_id = f"{name}_{utc_stamp()}_{chash}"
            in_paths = [Path(p.format(run_id=run_id)) for p in inputs]
            out_paths = [Path(p.format(run_id=run_id)) for p in outputs]

            ctx = StageContext(
                stage=name, run_id=run_id, run_dir=RUNS / run_id, seed=args.seed,
                config_path=config_path, config_hash=chash, resolved_config=resolved,
                inputs=in_paths, outputs=out_paths, dry_run=args.dry_run,
                force=args.force, args=args,
            )

            # --dry-run must have zero side effects: no run dir, no log file, no manifest.
            if args.dry_run:
                print(f"# dry run: {name}")
                print(f"# run_id would be: {run_id}")
                print(f"# config: {config_path}  (hash {chash})")
                print(f"# seed:   {args.seed}")
                print("# --- resolved config ---")
                print(json.dumps(resolved, indent=2, default=str))
                print("# --- reads ---")
                for p in in_paths:
                    state = "OK" if p.exists() else "MISSING -> " + _producer_hint(p)
                    print(f"  {p}  {state}")
                print("# --- writes ---")
                for p in out_paths:
                    print(f"  {p}")
                if upstream:
                    print(f"# upstream stage: {upstream}")
                return 0

            try:
                check_inputs(in_paths, args.force)
            except StageError as exc:
                print(f"ERROR [{name}] {exc}", file=sys.stderr)
                if upstream:
                    print(f"upstream stage is `{upstream}`.", file=sys.stderr)
                return 2

            if not args.force and outputs_are_fresh(in_paths, out_paths):
                print(f"[{name}] outputs are up to date; nothing to do. Use --force to rerun.")
                return 0

            ctx.run_dir.mkdir(parents=True, exist_ok=True)
            seed_everything(args.seed)

            import yaml as _yaml

            snapshot = ctx.run_dir / "config.snapshot.yaml"
            snapshot.write_text(_yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")

            started = time.time()
            with (ctx.run_dir / "log.txt").open("w", encoding="utf-8") as fh:
                ctx._log_fh = fh
                ctx.log(f"run_id={run_id} config={config_path} hash={chash} seed={args.seed}")
                try:
                    fn(cfg, ctx)
                except Exception as exc:
                    ctx.log(f"FAILED: {exc!r}", level="ERROR")
                    fh.write(traceback.format_exc())
                    ctx.record(wall_seconds=round(time.time() - started, 3))
                    ctx._log_fh = None
                    write_manifest(ctx, status="failed", error=f"{type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    print(
                        f"\nERROR [{name}] failed. Manifest: {ctx.run_dir / 'manifest.json'}",
                        file=sys.stderr,
                    )
                    return 1
                ctx.record(wall_seconds=round(time.time() - started, 3))
                ctx.log(f"done in {time.time() - started:.2f}s")
                ctx._log_fh = None

            path = write_manifest(ctx, status="success")
            print(f"[{name}] manifest: {path}")
            return 0

        main.__name__ = f"{name}_main"
        main.__doc__ = fn.__doc__
        main.stage_name = name  # type: ignore[attr-defined]
        return main

    return decorate
