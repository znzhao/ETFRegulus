"""Stage 0 -- preflight.

Every check is pass/fail with a fix hint. The point is to convert silent degradation
(a CPU-only torch wheel, a torch installed without numpy, an expired API key) into a
hard failure *before* six hours of training discovers it.

    python -m scripts.s00_check_env
    python -m scripts.s00_check_env --skip-network
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from src.cli.stage import StageContext, StageError, stage
from src.config.loader import load_typed
from src.config.schema import UniverseConfig

# Package name -> import name, where they differ.
_IMPORT_NAME = {
    "stable-baselines3": "stable_baselines3",
    "exchange-calendars": "exchange_calendars",
    "PyYAML": "yaml",
    "python-dotenv": "dotenv",
    "pytest-benchmark": "pytest_benchmark",
    "opencv-python": "cv2",
}

EXPECTED_CAPABILITY = (8, 6)  # RTX 3060 Ti, sm_86 (reference/gpu-setup.md)
EXPECTED_ARCH = "sm_86"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    hint: str = ""
    fatal: bool = True


def _requirements(path: Path = Path("requirements.txt")) -> list[tuple[str, str]]:
    """Parse `name==version` lines, ignoring comments, flags, and local build tags."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*(.+)$", line)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


# ------------------------------------------------------------------------- checks


def check_python() -> Check:
    v = sys.version_info
    ok = (3, 11) <= (v.major, v.minor) < (3, 13)
    return Check(
        "python version", ok, f"{sys.version.split()[0]} at {sys.executable}",
        hint="This project is pinned to Python 3.11/3.12; 3.13 wheels are not verified.",
    )


def check_requirements() -> list[Check]:
    req = Path("requirements.txt")
    if not req.exists():
        return [Check("requirements.txt", False, "missing",
                      hint="Pin the verified versions (reference/gpu-setup.md section 5).")]
    checks = []
    for name, want in _requirements(req):
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            checks.append(Check(f"pkg {name}", False, "not installed",
                                hint=f"pip install {name}=={want}"))
            continue
        # A local build tag (`+cu126`) is part of the identity: torch==2.13.0 from PyPI
        # is a different, CPU-only wheel.
        ok = have == want or (want.split("+")[0] == have.split("+")[0] and "+" not in want)
        checks.append(Check(
            f"pkg {name}", ok, f"{have} (want {want})",
            hint=f"pip install {name}=={want}" if not ok else "",
        ))
    for name, _ in _requirements(req):
        mod = _IMPORT_NAME.get(name, name.replace("-", "_"))
        try:
            importlib.import_module(mod)
        except Exception as exc:
            checks.append(Check(f"import {mod}", False, repr(exc),
                                hint=f"{name} is installed but not importable."))
    return checks


def check_torch_cuda() -> list[Check]:
    """CUDA must be usable. A CPU-only wheel is a hard failure, not a fallback (D14)."""
    checks: list[Check] = []
    try:
        import torch
    except Exception as exc:
        return [Check("torch import", False, repr(exc), hint="See reference/gpu-setup.md.")]

    checks.append(Check("torch build", "+cu" in torch.__version__,
                        f"{torch.__version__} (cuda runtime {torch.version.cuda})",
                        hint="A build tag without `+cuXXX` is the CPU-only PyPI wheel. "
                             "Reinstall from https://download.pytorch.org/whl/cu126"))

    available = torch.cuda.is_available()
    checks.append(Check("cuda available", available, str(available),
                        hint="`is_available()` False means SB3 would silently fall back "
                             "to CPU with only a warning. Fix the install, do not proceed."))
    if not available:
        return checks

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch = torch.cuda.get_arch_list()
    free, total = torch.cuda.mem_get_info()
    checks.append(Check("cuda device", True, f"{name}, capability {cap}"))
    checks.append(Check(f"arch {EXPECTED_ARCH} in build", EXPECTED_ARCH in arch,
                        " ".join(arch),
                        hint="The wheel was not built for this card's architecture."))
    checks.append(Check("vram", free > 1 << 30,
                        f"{free / 2**30:.2f} / {total / 2**30:.2f} GB free",
                        fatal=False,
                        hint="Under 1 GB free -- close other GPU consumers."))

    # is_available() proves a driver was found, not that arithmetic is correct.
    try:
        a = torch.randn(2048, 2048, device="cuda")
        ok = torch.allclose((a @ a).cpu(), a.cpu() @ a.cpu(), atol=1e-3)
        del a
        torch.cuda.empty_cache()
        checks.append(Check("cuda matmul == cpu", ok, "allclose atol=1e-3",
                            hint="The device computes, but not correctly. Suspect the driver."))
    except Exception as exc:
        checks.append(Check("cuda matmul == cpu", False, repr(exc)))
    return checks


def check_torch_numpy() -> Check:
    """The numpy-less-torch check.

    `pip install torch` from the PyTorch index has installed without numpy on this
    machine. torch imports, CUDA works, every tensor op succeeds -- and any `.numpy()`
    conversion fails at runtime deep inside SB3. It surfaces only as a UserWarning.
    """
    try:
        import numpy as np
        import torch

        arr = np.asarray(torch.randn(4).cpu())
        back = torch.from_numpy(arr)
        ok = arr.shape == (4,) and back.shape == (4,)
        return Check("torch <-> numpy bridge", ok, f"numpy {np.__version__}, round-trip ok",
                     hint="torch was installed without numpy. `pip install numpy` and reinstall torch.")
    except Exception as exc:
        return Check("torch <-> numpy bridge", False, repr(exc),
                     hint="torch was installed without numpy -- a UserWarning at import, "
                          "a crash inside SB3 later. See reference/gpu-setup.md section 5.")


def check_sb3() -> Check:
    """A working matmul does not prove SB3 is wired up."""
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO

        model = PPO("MlpPolicy", gym.make("Pendulum-v1"), device="cpu", verbose=0)
        model.learn(256)
        return Check("sb3 end-to-end (Pendulum, 256 steps)", True, "PPO.learn returned")
    except Exception as exc:
        return Check("sb3 end-to-end (Pendulum, 256 steps)", False, repr(exc),
                     hint="Gymnasium/SB3 version mismatch. See requirements.txt.")


def check_calendar(cfg: UniverseConfig) -> Check:
    try:
        from src.data.calendar import sessions

        s = sessions(cfg.history_start, calendar=cfg.calendar)
        # The first session must actually land on history_start, not somewhere later:
        # a calendar built with default bounds starts ~20 years ago and would silently
        # truncate the warm-up buffer. See src/data/calendar.py CALENDAR_START.
        import pandas as pd

        covers = s[0] <= pd.Timestamp(cfg.history_start) + pd.Timedelta(days=7)
        ok = len(s) > 5000 and covers
        return Check(f"calendar {cfg.calendar}", ok,
                     f"{len(s)} sessions {s[0].date()} .. {s[-1].date()}",
                     hint=f"First session must be at/near history_start "
                          f"({cfg.history_start}); a default-bounded calendar starts "
                          f"~20 years ago and would drop the warm-up buffer.")
    except Exception as exc:
        return Check(f"calendar {cfg.calendar}", False, repr(exc),
                     hint="pip install exchange-calendars")


def check_fred(skip: bool) -> Check:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        return Check("FRED_API_KEY", False, "unset or empty",
                     hint="Copy .env.example to .env and add a free key from "
                          "https://fred.stlouisfed.org/docs/api/api_key.html")
    if skip:
        return Check("FRED_API_KEY", True, f"present ({len(key)} chars); fetch skipped",
                     fatal=False)
    try:
        from fredapi import Fred

        s = Fred(api_key=key).get_series("T10Y2Y", observation_start="2024-01-01",
                                         observation_end="2024-02-01")
        ok = len(s) > 10
        return Check("FRED live fetch (T10Y2Y)", ok, f"{len(s)} observations",
                     hint="Key present but the fetch returned nothing. Check the key is active.")
    except Exception as exc:
        return Check("FRED live fetch (T10Y2Y)", False, repr(exc),
                     hint="Network or key problem.")


def check_yfinance(cfg: UniverseConfig, skip: bool) -> list[Check]:
    if skip:
        return [Check("yfinance live fetch", True, "skipped (--skip-network)", fatal=False)]
    checks = []
    # One tradable and one caret-prefixed symbol: the caret symbols are the ones that
    # break (reference/data-pipeline.md, known yfinance hazards).
    for symbol in (cfg.tradable_tickers[0], "^VIX"):
        try:
            import yfinance as yf

            df = yf.download(symbol, start="2024-01-01", end="2024-02-01",
                             auto_adjust=False, actions=False, progress=False,
                             threads=False)
            ok = df is not None and len(df) > 10
            checks.append(Check(f"yfinance {symbol}", ok,
                                f"{0 if df is None else len(df)} rows",
                                hint="Empty frame -- yfinance returns these silently on "
                                     "transient failure. Retry before concluding."))
        except Exception as exc:
            checks.append(Check(f"yfinance {symbol}", False, repr(exc)))
    return checks


def check_paths() -> list[Check]:
    checks = []
    for d in ("data", "artifacts"):
        p = Path(d)
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            checks.append(Check(f"{d}/ writable", True, str(p.resolve())))
        except Exception as exc:
            checks.append(Check(f"{d}/ writable", False, repr(exc)))

    # D6: data/ and artifacts/ are git-ignored. A committed parquet tree is a mess to undo.
    for d in ("data/probe.parquet", "artifacts/probe.json"):
        try:
            res = subprocess.run(["git", "check-ignore", "-q", d],
                                 capture_output=True, timeout=10)
            ignored = res.returncode == 0
        except Exception as exc:
            checks.append(Check(f"{d} git-ignored", False, repr(exc)))
            continue
        checks.append(Check(f"{d.split('/')[0]}/ git-ignored", ignored,
                            "ignored" if ignored else "TRACKED",
                            hint="Add it to .gitignore (D6)."))
    return checks


# -------------------------------------------------------------------------- stage


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--skip-network", action="store_true",
                   help="skip the yfinance and FRED live fetches (offline preflight)")
    p.add_argument("--allow-no-cuda", action="store_true",
                   help="downgrade the CUDA checks to warnings; for a CI box with no GPU")


@stage(
    name="s00_check_env",
    config_default="config/universe.yaml",
    config_cls=UniverseConfig,
    outputs=["artifacts/runs/{run_id}/preflight.json"],
    add_args=_add_args,
)
def main(cfg: UniverseConfig, ctx: StageContext) -> None:
    """Dependency, API-key, calendar and device preflight."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        ctx.warn("python-dotenv not installed; relying on the ambient environment")

    skip_net = ctx.args.skip_network

    checks: list[Check] = [check_python()]
    checks += check_requirements()
    cuda_checks = check_torch_cuda()
    if ctx.args.allow_no_cuda:
        for c in cuda_checks:
            c.fatal = False
    checks += cuda_checks
    checks.append(check_torch_numpy())
    checks.append(check_sb3())
    checks.append(check_calendar(cfg))
    checks.append(check_fred(skip_net))
    checks += check_yfinance(cfg, skip_net)
    checks += check_paths()

    width = max(len(c.name) for c in checks)
    ctx.log("")
    ctx.log(f"{'CHECK'.ljust(width)}  RESULT  DETAIL")
    for c in checks:
        mark = "PASS" if c.passed else ("FAIL" if c.fatal else "WARN")
        ctx.log(f"{c.name.ljust(width)}  {mark:<6}  {c.detail}")
        if not c.passed and c.hint:
            ctx.log(f"{' ' * width}          hint: {c.hint}")

    failed = [c for c in checks if not c.passed and c.fatal]
    warned = [c for c in checks if not c.passed and not c.fatal]

    report = {
        "checks": [asdict(c) for c in checks],
        "n_passed": sum(c.passed for c in checks),
        "n_failed": len(failed),
        "n_warned": len(warned),
        "skip_network": skip_net,
        "universe": {
            "tradable": cfg.tradable_tickers,
            "n_tradable": len(cfg.tradable_tickers),
            "feature_only": cfg.feature_symbols,
            "fred": list(cfg.fred),
        },
    }
    path = ctx.out_path("preflight.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ctx.record(n_failed=len(failed), n_warned=len(warned))
    ctx.log("")
    ctx.log(f"{len(checks) - len(failed) - len(warned)} passed, "
            f"{len(warned)} warned, {len(failed)} failed -> {path}")

    if failed:
        names = ", ".join(c.name for c in failed)
        raise StageError(f"preflight FAILED: {names}")


if __name__ == "__main__":
    raise SystemExit(main())
