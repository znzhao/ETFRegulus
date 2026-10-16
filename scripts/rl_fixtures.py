"""Test cases for the browser's RL plan (`app/rl.js` in the site repository).

    python -m scripts.rl_fixtures --out ../etfregulus-site/tests/fixtures/rl_cases.json

Builds the published RL document and risk samples for the latest curated session, then
asks `src.publish.rl_live.user_plan` -- the reference -- for the plan of a set of made-up
users: different books, buy dates, cash, 12-month highs (some past the ceiling), ceilings
and holdings limits. The site's tests run the same users through the JavaScript and
compare. To keep the fixture small it carries one seed and the first paths of each risk
sample; the reference is computed on exactly that reduced input.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from src.config.loader import resolve_config
from src.publish import pipeline, rl_live
from src.sim.runner import build_constraints, build_envelope, load_market

N_PATHS = 120


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", type=int, default=24)
    a = ap.parse_args()

    resolved = resolve_config(pipeline.Settings.config)
    market, _ = load_market(resolved, start="2022-01-03")
    data_date = market.sessions[-1].date()
    doc = pipeline.rl_doc(market, resolved, data_date)
    doc["seeds"] = doc["seeds"][:1]
    cons = build_constraints(resolved)
    env = build_envelope(cons)
    i = len(market.sessions) - 1
    env.prepare(market.returns[max(0, i - 756):i + 1], market.sessions[max(0, i - 756):i + 1].to_numpy(),
                as_of=data_date, seed=42 + i, n_assets=len(market.universe) + 1)
    risk = pipeline.risk_doc(env, resolved, list(market.universe), data_date)
    for s in risk["samples"]:
        s["paths"] = s["paths"][:N_PATHS]

    closes = market.prices_at(i, "close_raw")
    prices = {t: round(float(closes[t]), 4) for t in market.universe}
    rng = np.random.default_rng(20261016)
    tickers = list(market.universe)
    cases = []
    for c in range(a.cases):
        n_hold = int(rng.integers(0, 9))
        picks = rng.choice(len(tickers), size=n_hold, replace=False)
        holdings = []
        for j in picks:
            t = tickers[j]
            sh = int(rng.integers(1, 60))
            d = (data_date - dt.timedelta(days=int(rng.integers(0, 50)))).isoformat()
            holdings.append({"t": t, "sh": sh, "d": d})
        cash = float(rng.choice([0, 500, 5000, 20000]))
        value = cash + sum(h["sh"] * prices[h["t"]] for h in holdings)
        if value <= 0:
            cash, value = 10000.0, 10000.0
        dd = float(rng.choice([0.0, 0.02, 0.04, 0.08, 0.12, 0.2]))
        user = {"holdings": holdings, "cash": cash, "peak": round(value / (1 - dd), 2) if dd else None,
                "prices": prices, "hold_days": 30,
                "ceiling": int(rng.choice([5, 10, 15])), "max_holdings": int(rng.choice([0, 3, 5, 8]))}
        plan = rl_live.user_plan(doc, risk, user)
        cases.append({"user": user, "expect": {
            "raw": [float(x) for x in plan["raw"]], "target": [float(x) for x in plan["target"]],
            "shares": [float(x) for x in plan["shares"]], "cash_after": plan["cash_after"],
            "drawdown": plan["drawdown"], "budget": plan["budget"], "risk_state": plan["risk_state"],
            "derisk_forced": plan["derisk_forced"]}})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"doc": doc, "risk": risk, "cases": cases}, separators=(",", ":")), encoding="utf-8")
    states = [c["expect"]["risk_state"] for c in cases]
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.2f} MB): {len(cases)} cases, "
          + ", ".join(f"{s} {states.count(s)}" for s in sorted(set(states))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
