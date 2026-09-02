"""The standard metric set.

Computed from `trajectory.parquet`, so every backtest -- baseline, RL, stress cell -- goes
through the identical metric code and there is exactly one input format
(reference/evaluation.md section 3).

The constraint row is not diagnostic colour. `lock_violations == 0` and
`feasibility_violations == 0` are hard acceptance criteria: a fold failing either is not a
weaker result, it is an invalid one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _ann_return(nav: pd.Series) -> float:
    if len(nav) < 2:
        return 0.0
    years = len(nav) / TRADING_DAYS
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0


def drawdown_episodes(drawdown: pd.Series, threshold: float = 1e-9) -> list[dict]:
    """Contiguous runs where the portfolio is under water."""
    under = drawdown > threshold
    episodes, start = [], None
    for session, flag in under.items():
        if flag and start is None:
            start = session
        elif not flag and start is not None:
            block = drawdown.loc[start:session]
            episodes.append({"start": start, "end": session,
                             "depth": float(block.max()), "duration": len(block)})
            start = None
    if start is not None:
        block = drawdown.loc[start:]
        episodes.append({"start": start, "end": block.index[-1],
                         "depth": float(block.max()), "duration": len(block)})
    return episodes


def compute_metrics(traj: pd.DataFrame, *, d_max: float | None = None) -> dict:
    """The standard set for one trajectory."""
    nav = traj["nav"].astype(float)
    ret = nav.pct_change().dropna()
    logret = np.log(nav / nav.shift(1)).dropna()
    dd = traj["drawdown"].astype(float)

    vol = float(ret.std() * np.sqrt(TRADING_DAYS)) if len(ret) > 1 else 0.0
    ann = _ann_return(nav)
    # Downside deviation, over EVERY observation rather than only the losing ones.
    # Averaging the squared shortfalls over just the losers flatters a strategy that
    # loses rarely, and this must agree with `src/evaluation/report.py` -- two different
    # Sortinos in one codebase is the kind of silent inconsistency that makes a number
    # unciteable.
    shortfall = np.minimum(ret.to_numpy(), 0.0)
    downside_vol = float(np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(TRADING_DAYS)) \
        if len(ret) else 0.0
    max_dd = float(dd.max())

    episodes = drawdown_episodes(dd)
    cash_w = (traj["cash"] / nav) if "cash" in traj else pd.Series(dtype=float)

    out = {
        # -- return
        "cumulative_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "annualized_return": ann,
        "final_nav": float(nav.iloc[-1]),
        # -- risk
        "volatility": vol,
        "max_drawdown": max_dd,
        "drawdown_duration_max": max((e["duration"] for e in episodes), default=0),
        "time_under_water": float((dd > 1e-9).mean()),
        "sharpe": float(ann / vol) if vol > 1e-12 else 0.0,
        "sortino": float(ann / downside_vol) if downside_vol > 1e-12 else 0.0,
        "calmar": float(ann / max_dd) if max_dd > 1e-12 else 0.0,
        # -- tails
        "worst_1d": float(ret.min()) if len(ret) else 0.0,
        "worst_5d": float(logret.rolling(5).sum().min()) if len(logret) >= 5 else 0.0,
        "worst_21d": float(logret.rolling(21).sum().min()) if len(logret) >= 21 else 0.0,
        # -- behaviour
        "mean_cash_weight": float(cash_w.mean()) if len(cash_w) else 0.0,
        "median_cash_weight": float(cash_w.median()) if len(cash_w) else 0.0,
        "turnover_mean": float(traj["turnover"].mean()) if "turnover" in traj else 0.0,
        "turnover_total": float(traj["turnover"].sum()) if "turnover" in traj else 0.0,
        "n_sessions": int(len(traj)),
    }

    # -- constraints: hard acceptance criteria, not colour
    if "safety_intervened" in traj:
        out["safety_intervention_rate"] = float(traj["safety_intervened"].mean())
    if "capital_preservation" in traj:
        out["capital_preservation_rate"] = float(traj["capital_preservation"].mean())
    if "infeasible_fallback" in traj:
        out["infeasible_fallback_rate"] = float(traj["infeasible_fallback"].mean())
    if "proj_distance" in traj:
        out["mean_proj_distance"] = float(traj["proj_distance"].mean())

    if d_max is not None:
        breaches = [e for e in episodes if e["depth"] > d_max]
        out["d_max"] = float(d_max)
        out["n_dmax_breaches"] = len(breaches)
        out["worst_breach_depth"] = max((e["depth"] for e in breaches), default=0.0)
        out["worst_breach_duration"] = max((e["duration"] for e in breaches), default=0)
    return out


def summarize(traj: pd.DataFrame, diagnostics: dict, d_max: float | None = None) -> dict:
    """Metrics plus the constraint counters the simulator recorded."""
    out = compute_metrics(traj, d_max=d_max)
    out["lock_violations"] = int(diagnostics.get("lock_violations", 0))
    out["feasibility_violations"] = int(diagnostics.get("feasibility_violations", 0))
    out["n_param"] = diagnostics.get("hold_days")
    out["dmax_param"] = diagnostics.get("max_drawdown")
    return out


def metrics_frame(rows: dict[str, dict]) -> pd.DataFrame:
    """A comparison table, one row per named run."""
    return pd.DataFrame(rows).T
