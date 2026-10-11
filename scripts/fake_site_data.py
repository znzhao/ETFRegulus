"""Fake web-app data set, in the exact shape the daily pipeline publishes (WEBAPP_PLAN.md Step 2).

    python -m scripts.fake_site_data --out webapp/fake_data
    python -m scripts.fake_site_data --out webapp/fake_data_failed --failed

Prices are a seeded random walk with three crisis episodes; the strategies are simplified
stand-ins with a 30-day lock and a crude drawdown brake. The numbers mean nothing. The
point is a complete, valid data set -- real tickers, the real XNYS calendar, every file of
the contract -- for building and testing the front end before the real pipeline exists.
The output is checked against the contract before the script reports success.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from src.data.calendar import get_calendar
from src.publish import build, contract, names

ROOT = Path(__file__).resolve().parents[1]
START, END = dt.date(2004, 1, 2), dt.date(2026, 10, 9)
CEILINGS = (5, 10, 15)
LOCK_DAYS = 30
SAFE = {"SHY"}

# beta to the market factor, idiosyncratic daily vol, daily drift, inception
PARAMS = {
    "SPY": (1.0, .002, 0, None), "QQQ": (1.15, .006, .0001, None), "IWM": (1.1, .007, 0, None),
    "XLK": (1.2, .008, .0001, None), "XLC": (1.0, .009, 0, "2018-06-19"), "XLY": (1.1, .008, 0, None),
    "XLP": (.6, .007, 0, None), "XLE": (1.0, .013, -.0001, None), "XLF": (1.2, .009, -.0001, None),
    "XLI": (1.05, .007, 0, None), "XLB": (1.05, .009, 0, None), "XLV": (.75, .007, .00005, None),
    "XLU": (.55, .008, 0, None), "XLRE": (.9, .009, -.00005, "2015-10-08"),
    "SHY": (-.02, .0006, .0001, None), "IEF": (-.12, .003, .00012, None), "TLT": (-.3, .008, .00015, None),
    "TIP": (-.05, .003, .00012, None), "LQD": (.1, .004, .00015, None), "HYG": (.4, .003, .0001, "2007-04-11"),
    "GLD": (.05, .011, .00025, "2004-11-18"), "DBC": (.4, .012, -.0001, "2006-02-06"),
    "VEA": (.95, .006, -.0001, "2007-07-26"), "VWO": (1.05, .009, -.00005, "2005-03-10"),
}
DURATION = {"SHY": .1, "IEF": .5, "TLT": 1.2, "TIP": .5, "LQD": .8, "HYG": .4}
REGIMES = [  # start, end, market drift, market vol, bond-beta multiplier, bond drift per duration
    ("2008-09-01", "2009-03-09", -.0025, .028, 1, 0), ("2009-03-10", "2009-12-31", .002, .014, 1, 0),
    ("2020-02-20", "2020-03-23", -.012, .045, 1, 0), ("2020-03-24", "2020-08-31", .003, .016, 1, 0),
    ("2022-01-03", "2022-10-12", -.0009, .015, -2, -.0006),
]
CRISES = [("2008-09-01", "2009-03-09"), ("2020-02-20", "2020-03-23"), ("2022-01-03", "2022-10-12")]


def universe() -> tuple[list[str], dict[str, str]]:
    cfg = yaml.safe_load((ROOT / "config" / "universe.yaml").read_text(encoding="utf-8"))
    tickers, cls = [], {}
    for group, ts in cfg["tradable"].items():
        for t in ts:
            tickers.append(t)
            cls[t] = group
    return tickers, cls


def market(tickers, sessions, rng):
    T, A = len(sessions), len(tickers)
    regime = np.zeros((T, 4))
    regime[:] = (.0004, .0105, 1, 0)
    for a, b, *v in REGIMES:
        m = (sessions >= np.datetime64(a)) & (sessions <= np.datetime64(b))
        regime[m] = v
    mkt = regime[:, 0] + regime[:, 1] * rng.standard_normal(T)
    R = np.zeros((T, A))
    for j, t in enumerate(tickers):
        beta, idio, mu, _ = PARAMS[t]
        bond = t in DURATION
        b = beta * regime[:, 2] if bond and beta < .2 else beta
        R[:, j] = mu + b * mkt + idio * rng.standard_normal(T) + (regime[:, 3] * DURATION[t] if bond else 0)
    avail = np.ones((T, A), bool)
    for j, t in enumerate(tickers):
        if PARAMS[t][3]:
            avail[:, j] = sessions >= np.datetime64(PARAMS[t][3])
    R[~avail] = 0.0
    start_px = rng.uniform(40, 200, A)
    P = np.where(avail, start_px * np.cumprod(1 + np.where(avail, R, 0), axis=0), np.nan)
    return R, P, avail


def target(strategy, t, P, avail, tickers):
    A = len(tickers)
    w = np.zeros(A + 1)
    ok = np.flatnonzero(avail[t])
    col = {tk: j for j, tk in enumerate(tickers)}
    if strategy == "cash":
        w[0] = 1
    elif strategy == "spy_buy_hold":
        w[1 + col["SPY"]] = 1
    elif strategy == "spy_tlt_60_40":
        w[1 + col["SPY"]], w[1 + col["TLT"]] = .6, .4
    elif strategy == "equal_weight":
        w[1 + ok] = 1 / len(ok)
    elif strategy == "momentum":
        cand = [j for j in ok if t >= 252 and avail[t - 252, j]]
        score = sorted(((P[t, j] / P[t - 252, j] - 1, j) for j in cand), reverse=True)[:3]
        top = [j for s, j in score if s > 0]
        w[[1 + j for j in top]] = 1 / 3
        w[0] = 1 - len(top) / 3
    else:  # classical_optimizer stand-in: inverse 63-day vol, capped at 25%
        cand = [j for j in ok if t >= 63 and avail[t - 63, j]]
        if not cand:
            w[0] = 1
            return w
        r = P[t - 62:t + 1, cand] / P[t - 63:t, cand] - 1
        iv = 1 / np.sqrt((r ** 2).mean(axis=0) + 1e-8)
        w[[1 + j for j in cand]] = np.minimum(.25, iv / iv.sum())
        w[0] = max(0.0, 1 - w.sum())
    return w


def first_session_on_or_after(d: dt.date, cal_dates: list[dt.date]) -> dt.date:
    for s in cal_dates:
        if s >= d:
            return s
    return cal_dates[-1]


def simulate(strategy, ceiling, dates, R, P, avail, tickers, cal_dates) -> build.PortfolioPath:
    T, A = R.shape
    D = ceiling / 100
    w = np.zeros(A + 1)
    w[0] = 1
    W = np.zeros((T, A + 1))
    eq = np.ones(T)
    last_buy: dict[int, dt.date] = {}
    rebal: list[build.Rebalance] = []
    e, mode, hist = 1.0, -1, []
    for t in range(T):
        if t:
            r = float(w[1:] @ R[t])
            e *= 1 + r
            w = np.concatenate([[w[0]], w[1:] * (1 + R[t])]) / (1 + r)
        eq[t] = e
        hist.append(e)
        hist = hist[-252:]
        dd = 1 - e / max(hist)
        m = 2 if dd > D else 1 if dd > .6 * D else 0
        month_start = t == 0 or dates[t].month != dates[t - 1].month
        if month_start or m != mode:
            tg = target(strategy, t, P, avail, tickers)
            sc = (1, .6, .2)[m]
            for j, tk in enumerate(tickers):
                if tk not in SAFE:
                    tg[0] += tg[1 + j] * (1 - sc)
                    tg[1 + j] *= sc
            locked = {j for j in range(A) if w[1 + j] > tg[1 + j]
                      and j in last_buy and (dates[t] - last_buy[j]).days < LOCK_DAYS}
            extra = sum(w[1 + j] - tg[1 + j] for j in locked)
            for j in locked:
                tg[1 + j] = w[1 + j]
            take = min(extra, tg[0])
            tg[0] -= take
            extra -= take
            free = sum(tg[1 + j] for j in range(A) if j not in locked)
            if extra > 1e-12 and free > 0:
                for j in range(A):
                    if j not in locked:
                        tg[1 + j] *= 1 - extra / free
            changes = []
            for j, tk in enumerate(tickers):
                if abs(tg[1 + j] - w[1 + j]) > .002:
                    changes.append(build.Change(tk, w[1 + j], tg[1 + j]))
                    if tg[1 + j] > w[1 + j]:
                        last_buy[j] = dates[t]
            if changes:
                rebal.append(build.Rebalance(dates[t], changes))
            w, mode = tg, m
        W[t] = w
    held = {j for j in range(A) if W[-1, 1 + j] > 1e-6}
    locks = {tickers[j]: first_session_on_or_after(last_buy[j] + dt.timedelta(days=LOCK_DAYS), cal_dates)
             for j in held if j in last_buy}
    return build.PortfolioPath(
        strategy=strategy, ceiling=ceiling, dates=dates, equity=eq, weights=W, assets=tickers,
        rebalances=rebal, locks=locks, drawdown=float(dd),
        risk_state=build.RISK_STATES[mode])


def risk_doc(dates, R, avail, tickers, rng) -> dict:
    h = 5
    Rc = np.where(avail, R, 0.0)
    log = np.log1p(Rc)
    cum = lambda block: np.expm1(block.sum(axis=0))
    trail = log[-252:]
    rolling = [cum(trail[i:i + h]) for i in range(len(trail) - h + 1)]
    boot = []
    for _ in range(256):
        starts = rng.integers(0, len(trail) - 10, size=1)
        boot.append(cum(trail[starts[0]:starts[0] + h]))
    sess = np.array(dates, dtype="datetime64[D]")
    crisis = []
    for a, b in CRISES:
        idx = np.flatnonzero((sess >= np.datetime64(a)) & (sess <= np.datetime64(b)))
        crisis += [cum(log[i:i + h]) for i in idx[:-h]]

    def rows(paths):
        return [[0.0] + [round(float(x), 4) for x in p] for p in paths]

    return {"schema_version": 1, "data_date": dates[-1].isoformat(), "measure": "cvar", "quantile": 0.01,
            "horizon_days": h, "aggregation": "max", "over_budget_rule": "no_risk_increase",
            "assets": ["CASH", *tickers],
            # Empty samples are dropped, as RiskEnvelope.prepare does.
            "samples": [{"name": n, "paths": rows(ps)} for n, ps in
                        (("rolling", rolling), ("block_bootstrap", boot), ("crisis_windows", crisis)) if ps]}


def live_doc(paths: list[build.PortfolioPath], days: int) -> dict:
    if days <= 0:
        return {"schema_version": 1, "launch_date": None, "portfolios": {}}
    out = {}
    for p in paths:
        rb = {r.date: r for r in p.rebalances}
        entries = []
        for i in range(len(p.dates) - days, len(p.dates)):
            d = p.dates[i]
            entries.append({"date": d.isoformat(), "d1": round(float(p.equity[i] / p.equity[i - 1] - 1), 6),
                            "weights": build.weight_dict(p.weights[i], p.assets),
                            "changes": [c.doc() for c in rb[d].changes] if d in rb else []})
        out[p.id] = entries
    return {"schema_version": 1, "launch_date": paths[0].dates[-days].isoformat(), "portfolios": out}


def dump(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def generate(out: Path, failed: bool = False, seed: int = 20261009, live_days: int = 10,
             start: dt.date = START) -> list[str]:
    rng = np.random.default_rng(seed)
    tickers, cls = universe()
    cal = get_calendar("XNYS")
    all_sessions = cal.sessions_in_range(start.isoformat(), (END + dt.timedelta(days=200)).isoformat())
    all_dates = [s.date() for s in all_sessions]
    dates = [d for d in all_dates if d <= END]
    if failed:  # the last good run was one session earlier
        dates = dates[:-1]
    data_date = dates[-1]
    future = [d for d in all_dates if d >= data_date]
    sessions = np.array(dates, dtype="datetime64[D]")
    R, P, avail = market(tickers, sessions, rng)

    paths = [simulate(s, c, dates, R, P, avail, tickers, future)
             for s in names.STRATEGIES for c in CEILINGS]

    if out.exists():
        shutil.rmtree(out)
    files = {"universe": "universe.json", "today": "today.json", "risk": "risk.json",
             "live": "live/index.json", "backtest": {p.id: f"backtest/{p.id}.json" for p in paths}}
    stamp = "2026-10-09T22:41:07Z"
    dump(out / "manifest.json", {
        "schema_version": 1, "data_date": data_date.isoformat(),
        "generated_at": "2026-10-08T22:40:51Z" if failed else stamp,
        "status": "failed" if failed else "ok",
        "last_attempt": {"at": "2026-10-09T22:38:12Z" if failed else stamp, "status": "failed" if failed else "ok",
                         "message": "yfinance returned no rows for 3 tickers" if failed else None},
        "next_run": "2026-10-12T22:30:00Z",
        "code": {"repo": "znzhao/ETFRegulus", "tag": None, "commit": "aea3e2e"},
        "config": {"name": "redesign_df", "hash": "f8adbcc9"},
        "backtest_start": dates[0].isoformat(), "lock_days": LOCK_DAYS, "ceilings": list(CEILINGS),
        "strategies": names.strategy_entries(), "portfolios": [p.id for p in paths], "files": files,
    })
    dump(out / files["universe"], {
        "schema_version": 1, "data_date": data_date.isoformat(),
        "assets": [{"ticker": t, "name": names.text(names.TICKERS[t]), "asset_class": cls[t],
                    "inception": PARAMS[t][3] or "2003-01-02",
                    "close": round(float(P[-1, j]), 2), "prev_close": round(float(P[-2, j]), 2)}
                   for j, t in enumerate(tickers)],
        "calendar": {"exchange": "XNYS", "sessions": [d.isoformat() for d in future[:90]]},
    })
    dump(out / files["today"], build.today_doc(data_date, paths))
    dump(out / files["risk"], risk_doc(dates, R, avail, tickers, rng))
    dump(out / files["live"], live_doc(paths, live_days))
    for p in paths:
        dump(out / files["backtest"][p.id], build.backtest_doc(p))
    return contract.validate_site(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "webapp" / "fake_data")
    ap.add_argument("--failed", action="store_true", help="simulate a failed update (data one session old)")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--live-days", type=int, default=10, help="sessions of sample live record (0 = before launch)")
    a = ap.parse_args()
    errs = generate(a.out, a.failed, a.seed, a.live_days)
    for e in errs:
        print(e)
    size = sum(f.stat().st_size for f in a.out.rglob("*.json"))
    print(f"wrote {a.out} ({size / 1e6:.1f} MB): " + (f"{len(errs)} contract problem(s)" if errs else "valid"))
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
