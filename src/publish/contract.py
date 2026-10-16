"""The web app's data contract: JSON Schemas plus the checks a schema cannot express.

    python -m src.publish.contract <data_dir>

Exits non-zero and lists every problem if the directory is not a valid site data set. The
daily pipeline runs this before deploying; nothing that fails it is published.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

from src.publish.build import portfolio_id

SCHEMA_DIR = Path(__file__).with_name("schema")
KINDS = ("manifest", "universe", "today", "backtest", "risk", "live")
WEIGHT_TOL = 1e-3


@lru_cache(maxsize=None)
def validator(kind: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / f"{kind}.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def schema_errors(kind: str, doc) -> list[str]:
    return [f"{kind}: {'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}"
            for e in sorted(validator(kind).iter_errors(doc), key=lambda e: list(e.absolute_path))]


def _weights_ok(where: str, weights: dict, known: set[str]) -> list[str]:
    errs = []
    total = sum(weights.values())
    if abs(total - 1.0) > WEIGHT_TOL:
        errs.append(f"{where}: weights sum to {total:.4f}, not 1")
    unknown = set(weights) - known
    if unknown:
        errs.append(f"{where}: unknown assets {sorted(unknown)}")
    return errs


def validate_site(data_dir: str | Path) -> list[str]:
    """Every problem with the data set in `data_dir` (empty list = valid)."""
    root = Path(data_dir)
    errs: list[str] = []

    def load(rel: str, kind: str):
        path = root / rel
        if not path.is_file():
            errs.append(f"{kind}: missing file {rel}")
            return None
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errs.append(f"{kind}: {rel} is not valid JSON ({e})")
            return None
        found = schema_errors(kind, doc)
        errs.extend(f"{rel}: {m}" for m in found)
        return None if found else doc

    man = load("manifest.json", "manifest")
    if man is None:
        return errs
    date = man["data_date"]
    files = man["files"]
    uni, today, risk, live = (load(files[k], k) for k in ("universe", "today", "risk", "live"))

    expected = {portfolio_id(s["id"], c, k) for s in man["strategies"] for c in man["ceilings"]
                for k in man.get("holdings", [0])}
    if set(man["portfolios"]) != expected:
        errs.append(f"manifest: portfolios do not match strategies x ceilings x holdings "
                    f"(missing {sorted(expected - set(man['portfolios']))}, "
                    f"extra {sorted(set(man['portfolios']) - expected)})")
    if set(files["backtest"]) != set(man["portfolios"]):
        errs.append("manifest: files.backtest keys do not match portfolios")
    if man["backtest_start"] > date:
        errs.append("manifest: backtest_start is after data_date")

    tickers: set[str] = set()
    if uni:
        tickers = {a["ticker"] for a in uni["assets"]}
        if len(tickers) != len(uni["assets"]):
            errs.append("universe: duplicate tickers")
        if uni["data_date"] != date:
            errs.append(f"universe: data_date {uni['data_date']} != manifest {date}")
        cal = uni["calendar"]["sessions"]
        if cal != sorted(set(cal)) or cal[0] != date:
            errs.append("universe: calendar sessions must be ascending, unique, and start at data_date")
    known = tickers | {"CASH"}

    if today:
        if today["data_date"] != date:
            errs.append(f"today: data_date {today['data_date']} != manifest {date}")
        if set(today["portfolios"]) != set(man["portfolios"]):
            errs.append("today: portfolios do not match the manifest")
        for pid, p in today["portfolios"].items():
            if pid != portfolio_id(p["strategy"], p["ceiling"], p.get("holdings", 0)):
                errs.append(f"today/{pid}: id does not match strategy, ceiling and holdings")
            if tickers:
                errs.extend(_weights_ok(f"today/{pid}", p["weights"], known))
                errs.extend(_weights_ok(f"today/{pid}/current", p["current_weights"], known))
            for c in p["next_trades"]:
                cur, tgt = p["current_weights"].get(c["ticker"], 0.0), p["weights"].get(c["ticker"], 0.0)
                if abs(c["from"] - cur) > WEIGHT_TOL or abs(c["to"] - tgt) > WEIGHT_TOL:
                    errs.append(f"today/{pid}: next_trades {c['ticker']} disagrees with the weights")
            for t, until in p["locks"].items():
                if t not in p["current_weights"]:
                    errs.append(f"today/{pid}: lock on {t}, which is not held")
                if until <= date:
                    errs.append(f"today/{pid}: lock on {t} already expired ({until})")
            rb = p["last_rebalance"]
            if rb and (rb["date"] > date or p["traded_today"] != (rb["date"] == date)):
                errs.append(f"today/{pid}: last_rebalance and traded_today disagree")

    for pid in sorted(set(files["backtest"]) & set(man["portfolios"])):
        bt = load(files["backtest"][pid], "backtest")
        if bt is None:
            continue
        where = f"backtest/{pid}"
        if bt["portfolio"] != pid:
            errs.append(f"{where}: portfolio field is {bt['portfolio']}")
        d, eq = bt["dates"], bt["equity"]
        if len(d) != len(eq):
            errs.append(f"{where}: {len(d)} dates but {len(eq)} equity values")
        if d != sorted(set(d)):
            errs.append(f"{where}: dates are not ascending and unique")
        if d and (d[0] != bt["start"] or d[-1] != bt["end"] or bt["end"] != date or bt["start"] != man["backtest_start"]):
            errs.append(f"{where}: start/end do not match its dates, the manifest backtest_start, or data_date")
        w = bt["weights"]
        if len(w["dates"]) != len(w["values"]) or any(len(r) != len(w["assets"]) for r in w["values"]):
            errs.append(f"{where}: weights table shape is inconsistent")
        if w["dates"] and w["dates"][-1] != date:
            errs.append(f"{where}: weights sample must end on data_date")
        if tickers and set(w["assets"]) - known:
            errs.append(f"{where}: unknown assets in weights")
        for row in w["values"]:
            if abs(sum(row) - 1.0) > WEIGHT_TOL:
                errs.append(f"{where}: a weights row sums to {sum(row):.4f}")
                break
        if any(r["date"] < bt["start"] or r["date"] > date for r in bt["rebalances"]):
            errs.append(f"{where}: rebalance outside the backtest window")

    if risk:
        if risk["data_date"] != date:
            errs.append(f"risk: data_date {risk['data_date']} != manifest {date}")
        n = len(risk["assets"])
        if tickers and set(risk["assets"]) - known:
            errs.append("risk: unknown assets")
        for s in risk["samples"]:
            if any(len(row) != n for row in s["paths"]):
                errs.append(f"risk/{s['name']}: a path does not have {n} columns")
            elif any(row[0] != 0 for row in s["paths"]):
                errs.append(f"risk/{s['name']}: CASH column must be 0")

    if live:
        if set(live["portfolios"]) - set(man["portfolios"]):
            errs.append("live: portfolios not in the manifest")
        for pid, entries in live["portfolios"].items():
            ds = [e["date"] for e in entries]
            if ds != sorted(set(ds)):
                errs.append(f"live/{pid}: dates are not ascending and unique")
            if ds and (live["launch_date"] is None or ds[0] < live["launch_date"] or ds[-1] > date):
                errs.append(f"live/{pid}: entries outside launch_date..data_date")
            for e in entries:
                if tickers:
                    errs.extend(_weights_ok(f"live/{pid}/{e['date']}", e["weights"], known))
    return errs


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__)
        return 2
    errs = validate_site(args[0])
    for e in errs:
        print(e)
    print(f"{len(errs)} problem(s)" if errs else "valid")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
