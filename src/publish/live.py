"""The live record (WEBAPP_PLAN.md Step 4): an append-only log of what the app recommended.

One JSON line per published session, in `<log_dir>/YYYY-MM.jsonl`:

    {"date": "2026-10-09", "portfolios": {"momentum_D5": {"d1": ..., "weights": {...}, "changes": [...]}}}

`d1` is the strategy's return over that session; `weights` and `changes` are the plan made at
its close for the next open. Lines are only ever appended, never edited: the backtest is
recomputed daily and may shift with data revisions or code changes, but the log keeps what
the app actually said. The first logged session is the launch date.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.publish.build import SCHEMA_VERSION


def read_log(log_dir: str | Path) -> list[dict]:
    """Every logged line, oldest first."""
    root = Path(log_dir)
    lines = []
    for f in sorted(root.glob("*.jsonl")) if root.is_dir() else []:
        for raw in f.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                lines.append(json.loads(raw))
    return sorted(lines, key=lambda x: x["date"])


def line_from_today(today: dict) -> dict:
    return {"date": today["data_date"],
            "portfolios": {pid: {"d1": p["returns"]["d1"] if p["returns"]["d1"] is not None else 0.0,
                                 "weights": p["weights"], "changes": p["next_trades"]}
                           for pid, p in today["portfolios"].items()}}


def is_new(lines: list[dict], line: dict) -> bool:
    """Only a session after the last logged one is appended (a re-run or an older data date is not)."""
    return not lines or line["date"] > lines[-1]["date"]


def append(log_dir: str | Path, line: dict) -> Path:
    path = Path(log_dir) / f"{line['date'][:7]}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def live_doc(lines: list[dict], portfolios: list[str]) -> dict:
    out: dict[str, list] = {pid: [] for pid in portfolios}
    for line in lines:
        for pid, e in line["portfolios"].items():
            if pid in out:
                out[pid].append({"date": line["date"], "d1": e["d1"], "weights": e["weights"],
                                 "changes": e["changes"]})
    return {"schema_version": SCHEMA_VERSION, "launch_date": lines[0]["date"] if lines else None,
            "portfolios": out if lines else {}}
