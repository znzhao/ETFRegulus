"""Build the web app's data set from the curated prices (WEBAPP_PLAN.md Step 3).

Every strategy x ceiling runs once, from the backtest start to the last curated session,
through the same simulator, lock manager, projection and risk envelope as the research
(`src/sim`). The decision taken at the final close -- which executes at the next open --
is projected without being executed and published as "today's plan".
"""

from __future__ import annotations

import datetime as dt
import itertools
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.baselines.strategies import BASELINES
from src.config.loader import config_hash, resolve_config
from src.data.calendar import get_calendar
from src.publish import build, names
from src.sim.engine import plan_next
from src.sim.runner import build_constraints, build_envelope, build_sim_config, load_market, make_projector_from
from src.sim.simulator import MarketData, simulate

CEILINGS = (5, 10, 15)
HOLDINGS = (0, 3, 5, 8)      # most ETFs held at once; 0 = no limit
CALENDAR_AHEAD = 90          # sessions of future calendar shipped for lock expiry dates
RISK_DECIMALS = 4


@dataclass
class Settings:
    config: str = "config/experiments/webapp.yaml"
    start: str = "2004-01-01"
    end: str | None = None
    ceilings: tuple[int, ...] = CEILINGS
    holdings: tuple[int, ...] = HOLDINGS
    strategies: tuple[str, ...] = tuple(names.STRATEGIES)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def code_info() -> dict:
    return {"repo": "znzhao/ETFRegulus",
            "tag": _git("describe", "--tags", "--exact-match"),
            "commit": _git("rev-parse", "--short=7", "HEAD") or "0000000"}


# --------------------------------------------------------------------------- the RL model

RL_STRATEGY = "rl"
RL_MODEL_DIR = Path("models/webapp/seeds_v2_C5bb")
RL_FOLD = "fold_2025"


class _YearlyPolicySource:
    """The trained ensemble as a weight source over a multi-year path.

    The policy learned on episodes of at most two years, and two of its inputs are the NAV
    and the peak relative to the NAV the episode started at. Over a 20-year path that
    ratio leaves anything seen in training, so the reference NAV restarts every calendar
    year, as each walk-forward test year does.
    """

    def __init__(self, source):
        self.source, self.year = source, None

    def __call__(self, session, ctx):
        if session.year != self.year:
            self.year = session.year
            self.source._nav_at_reset = float(ctx.nav)
        return self.source(session, ctx)


def rl_weight_source(market: MarketData, resolved: dict, *, hold_days: int, max_drawdown: float):
    from src.env.factory import env_config_from
    from src.env.feature_store import build_store
    from src.env.observation import ObservationSpec
    from src.evaluation.rollout import PolicyWeightSource, load_policy

    spec = ObservationSpec.from_config(resolved, list(market.universe))
    store = build_store(spec, market.sessions, fold_id=RL_FOLD, scalers_dir=RL_MODEL_DIR)
    env_cfg = env_config_from(resolved, build_constraints(resolved))
    models = load_policy(sorted(p for p in RL_MODEL_DIR.iterdir() if p.is_dir()))
    source = PolicyWeightSource(models, spec, store, market, hold_days=hold_days,
                                max_drawdown=max_drawdown, decision_cadence=env_cfg.decision_cadence,
                                preservation_daily=env_cfg.preservation_daily,
                                action_mode=env_cfg.action_mode)
    return _YearlyPolicySource(source)


# --------------------------------------------------------------------------- one portfolio

def _path_from_trajectory(market: MarketData, traj: pd.DataFrame, strategy: str, ceiling: int,
                          initial_cash: float) -> build.PortfolioPath:
    """Session dates, equity, weights and executed trades, with the opening all-cash row."""
    universe = list(market.universe)
    dates = [market.sessions[0].date()] + [d.date() for d in traj.index]
    nav = np.concatenate([[initial_cash], traj["nav"].to_numpy(float)])
    w = np.zeros((len(dates), len(universe) + 1))
    w[0, 0] = 1.0
    w[1:, 1:] = traj[[f"w_{t}" for t in universe]].to_numpy(float)
    w[1:, 0] = traj["cash"].to_numpy(float) / traj["nav"].to_numpy(float)

    # A purchase relocks (the unlock date moves); a sale lowers the share count. Dividend
    # reinvestment does neither, so it is not mistaken for a trade. Trades that move a
    # weight by less than `build.MIN_CHANGE` (top-ups the risk rule lets through almost
    # daily) are left out of the published list; the lock dates still reflect them.
    shares = np.vstack([np.zeros(len(universe)), traj[[f"shares_{t}" for t in universe]].to_numpy(float)])
    unlock = pd.concat([pd.DataFrame([[pd.NaT] * len(universe)], columns=universe),
                        traj[[f"unlock_date_{t}" for t in universe]].set_axis(universe, axis=1)],
                       ignore_index=True)
    rebalances = []
    for k in range(1, len(dates)):
        changes = []
        for j, t in enumerate(universe):
            bought = pd.notna(unlock.iat[k, j]) and unlock.iat[k, j] != unlock.iat[k - 1, j]
            sold = shares[k, j] < shares[k - 1, j] - 1e-9
            if (bought or sold) and abs(w[k, j + 1] - w[k - 1, j + 1]) >= build.MIN_CHANGE:
                changes.append(build.Change(t, w[k - 1, j + 1], w[k, j + 1]))
        if changes:
            rebalances.append(build.Rebalance(dates[k], changes))

    last = traj.iloc[-1]
    locks = {t: last[f"unlock_date_{t}"].date() for t in universe
             if last[f"shares_{t}"] > 0 and pd.notna(last[f"unlock_date_{t}"])}
    return build.PortfolioPath(strategy=strategy, ceiling=ceiling, dates=dates, equity=nav,
                               weights=w, assets=universe, rebalances=rebalances, locks=locks)


def run_portfolio(market: MarketData, resolved: dict, strategy: str, ceiling: int, holdings: int = 0):
    """Simulate one strategy x ceiling x holdings limit and plan its next decision.
    Returns (path, envelope)."""
    constraints = build_constraints(resolved)
    projector = make_projector_from(constraints)
    cfg = build_sim_config(resolved, constraints, max_drawdown=ceiling / 100, seed=42,
                           max_holdings=holdings)
    if strategy == RL_STRATEGY:
        fn = rl_weight_source(market, resolved, hold_days=cfg.hold_days, max_drawdown=ceiling / 100)
    else:
        params = resolved.get("baselines", {}).get("params", {}).get(strategy, {})
        fn = BASELINES[strategy].build(market, params)
    envelope = build_envelope(constraints)
    res = simulate(market, fn, cfg, projector=projector, envelope=envelope)
    if res.diagnostics["lock_violations"] or res.diagnostics["feasibility_violations"]:
        raise RuntimeError(f"{strategy} D{ceiling} K{holdings}: constraint violations in the simulation")

    path = _path_from_trajectory(market, res.trajectory, strategy, ceiling, cfg.initial_cash)
    path.holdings = holdings
    dec, projected = plan_next(market, res.final_state, cfg, len(market.sessions) - 1, fn, projector,
                               envelope=envelope)
    diag = projected.diagnostics
    path.target = np.asarray(projected.weights, dtype=float)
    path.drawdown = float(dec.drawdown)
    path.risk_state = ("capital_preservation" if dec.capital_preservation
                       else "over_budget" if diag.no_risk_increase else "normal")
    return path, envelope


# --------------------------------------------------------------------------- other documents

def universe_doc(market: MarketData, ucfg) -> dict:
    groups = {t: g for g, ts in ucfg.tradable.items() for t in ts}
    last = len(market.sessions) - 1
    close, prev = market.prices_at(last, "close_raw"), market.prices_at(last - 1, "close_raw")
    data_date = market.sessions[last].date()
    first = [market.sessions[int(np.argmax(market.available[:, j]))].date() for j in range(len(market.universe))]
    cal = get_calendar("XNYS")
    ahead = cal.sessions_in_range(data_date.isoformat(), (data_date + dt.timedelta(days=200)).isoformat())
    return {
        "schema_version": build.SCHEMA_VERSION, "data_date": data_date.isoformat(),
        "assets": [{"ticker": t, "name": names.text(names.TICKERS[t]), "asset_class": groups[t],
                    "inception": first[j].isoformat(), "close": round(float(close[t]), 4),
                    "prev_close": round(float(prev[t]), 4) if prev.get(t) else None}
                   for j, t in enumerate(market.universe)],
        "calendar": {"exchange": "XNYS", "sessions": [s.date().isoformat() for s in ahead[:CALENDAR_AHEAD]]},
    }


def risk_doc(envelope, resolved: dict, assets: list[str], data_date: dt.date) -> dict:
    r = resolved["risk"]
    return {
        "schema_version": build.SCHEMA_VERSION, "data_date": data_date.isoformat(),
        "measure": r.get("measure", "cvar"), "quantile": float(r["quantile"]),
        "horizon_days": int(r["horizon_days"]), "aggregation": r["aggregation"],
        "over_budget_rule": resolved.get("projection", {}).get("over_budget_rule", "freeze"),
        "assets": ["CASH", *assets],
        "samples": [{"name": s.name, "paths": np.round(s.paths, RISK_DECIMALS).tolist()}
                    for s in envelope.samples],
    }


def next_run(data_date: dt.date) -> str:
    cal = get_calendar("XNYS")
    nxt = cal.sessions_in_range((data_date + dt.timedelta(days=1)).isoformat(),
                                (data_date + dt.timedelta(days=10)).isoformat())[0]
    return f"{nxt.date().isoformat()}T22:30:00Z"


# --------------------------------------------------------------------------- the whole site

def build_site(settings: Settings, log: Callable[[str], None] = print) -> dict[str, dict]:
    """All documents of the data set, keyed by their path relative to the data directory."""
    resolved = resolve_config(settings.config)
    market, ucfg = load_market(resolved, start=settings.start, end=settings.end)
    data_date = market.sessions[-1].date()
    log(f"{len(market.universe)} ETFs, {len(market.sessions)} sessions "
        f"{market.sessions[0].date()} .. {data_date}; config {settings.config} ({config_hash(resolved)})")

    paths, envelope = [], None
    for strategy in settings.strategies:
        for c, k in itertools.product(settings.ceilings, settings.holdings):
            t0 = time.time()
            p, envelope = run_portfolio(market, resolved, strategy, c, k)
            paths.append(p)
            m = build.metrics(p.equity / p.equity[0])
            log(f"  {p.id:<26} {time.time() - t0:5.1f}s  Sharpe {m['sharpe']:5.2f}  maxDD {m['max_drawdown']:.3f}  "
                f"{p.risk_state:<20} trades next open: {len(build.trades(p.weights[-1], p.target, p.assets))}")

    files = {"universe": "universe.json", "today": "today.json", "risk": "risk.json",
             "live": "live/index.json", "backtest": {p.id: f"backtest/{p.id}.json" for p in paths}}
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    docs = {
        "manifest.json": {
            "schema_version": build.SCHEMA_VERSION, "data_date": data_date.isoformat(),
            "generated_at": stamp, "status": "ok",
            "last_attempt": {"at": stamp, "status": "ok", "message": None},
            "next_run": next_run(data_date), "code": code_info(),
            "config": {"name": Path(settings.config).stem, "hash": config_hash(resolved)},
            "backtest_start": paths[0].dates[0].isoformat(),
            "lock_days": int(build_sim_config(resolved, build_constraints(resolved)).hold_days),
            "ceilings": list(settings.ceilings),
            "holdings": list(settings.holdings),
            "strategies": [s for s in names.strategy_entries() if s["id"] in settings.strategies],
            "portfolios": [p.id for p in paths], "files": files,
        },
        files["universe"]: universe_doc(market, ucfg),
        files["today"]: build.today_doc(data_date, paths),
        files["risk"]: risk_doc(envelope, resolved, list(market.universe), data_date),
        # Step 4 fills the live record from the append-only log.
        files["live"]: {"schema_version": build.SCHEMA_VERSION, "launch_date": None, "portfolios": {}},
    }
    for p in paths:
        docs[files["backtest"][p.id]] = build.backtest_doc(p)
    return docs
