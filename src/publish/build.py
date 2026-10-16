"""Turn a simulated portfolio path into the web app's JSON documents (WEBAPP_PLAN.md §3).

The inputs are what any simulation produces: session dates, an equity curve, a weight
matrix with CASH in column 0, and the rebalances. The real pipeline (Step 3) and the
fake-data generator both call these functions, so the derived numbers -- metrics, monthly
returns, period returns, weekly samples -- are computed one way only.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

SCHEMA_VERSION = 1
CASH = "CASH"
TRADING_DAYS = 252
W_DECIMALS = 4          # weights
MIN_CHANGE = 0.001      # smallest weight change published as a trade (0.1 percentage point)
EQ_DIGITS = 6           # significant digits of the equity curve
RISK_STATES = ("normal", "over_budget", "capital_preservation")


def portfolio_id(strategy: str, ceiling: int, holdings: int = 0) -> str:
    """`<strategy>_D<ceiling>`, plus `_K<n>` when at most n ETFs may be held (0 = no limit)."""
    return f"{strategy}_D{ceiling}" + (f"_K{holdings}" if holdings else "")


def iso(d: dt.date) -> str:
    return d.isoformat()


def _r(x: float, nd: int = 6) -> float:
    return float(round(float(x), nd))


def _sig(x: float, digits: int = EQ_DIGITS) -> float:
    return float(f"{x:.{digits}g}")


@dataclass
class Change:
    ticker: str
    before: float
    after: float

    def doc(self) -> dict:
        return {"ticker": self.ticker, "from": _r(self.before, W_DECIMALS), "to": _r(self.after, W_DECIMALS)}


@dataclass
class Rebalance:
    date: dt.date
    changes: list[Change]

    def doc(self) -> dict:
        return {"date": iso(self.date), "changes": [c.doc() for c in self.changes]}


@dataclass
class PortfolioPath:
    """One strategy x ceiling, as simulated through `data_date`."""

    strategy: str
    ceiling: int                    # percent
    dates: Sequence[dt.date]
    equity: np.ndarray              # (T,), value of 1 invested at dates[0]
    weights: np.ndarray             # (T, 1 + n_assets), column 0 = CASH, end-of-session
    assets: Sequence[str]           # tickers for columns 1..
    rebalances: list[Rebalance] = field(default_factory=list)
    locks: dict[str, dt.date] = field(default_factory=dict)   # held ticker -> first legal sell session
    drawdown: float = 0.0           # as the risk layer measures it at data_date
    risk_state: str = "normal"      # of the planned decision
    #: The decision taken at the final close, for the next open (1 + n_assets, CASH first).
    #: None means "hold the current book".
    target: np.ndarray | None = None
    holdings: int = 0               # most ETFs held at once; 0 = no limit

    @property
    def id(self) -> str:
        return portfolio_id(self.strategy, self.ceiling, self.holdings)


# --------------------------------------------------------------------------- numbers

def daily_returns(equity: np.ndarray) -> np.ndarray:
    return equity[1:] / equity[:-1] - 1.0


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def metrics(equity: np.ndarray) -> dict:
    r = daily_returns(equity)
    years = len(r) / TRADING_DAYS
    cagr = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    sd = float(np.std(r))
    mdd = max_drawdown(equity)
    return {
        "cagr": _r(cagr), "volatility": _r(sd * np.sqrt(TRADING_DAYS)),
        "sharpe": _r(float(np.mean(r)) / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0, 4),
        "max_drawdown": _r(mdd), "calmar": _r(cagr / mdd if mdd > 0 else 0.0, 4),
        "best_day": _r(float(r.max())), "worst_day": _r(float(r.min())),
        "up_days": _r(float(np.mean(r > 0)), 4),
    }


def calendar_returns(dates: Sequence[dt.date], equity: np.ndarray) -> tuple[dict, dict]:
    """Monthly and annual returns. The first period starts at dates[0]'s close."""
    monthly: dict[str, list] = {}
    annual: dict[str, float] = {}
    m_start = y_start = equity[0]
    for i, d in enumerate(dates):
        last = i == len(dates) - 1
        nxt = None if last else dates[i + 1]
        if last or (nxt.year, nxt.month) != (d.year, d.month):
            monthly.setdefault(str(d.year), [None] * 12)[d.month - 1] = _r(equity[i] / m_start - 1.0)
            m_start = equity[i]
        if last or nxt.year != d.year:
            annual[str(d.year)] = _r(equity[i] / y_start - 1.0)
            y_start = equity[i]
    return monthly, annual


def period_returns(dates: Sequence[dt.date], equity: np.ndarray) -> dict:
    n = len(equity) - 1

    def back(k: int):
        return _r(equity[-1] / equity[n - k] - 1.0) if n - k >= 0 else None

    year = dates[-1].year
    prior = [i for i, d in enumerate(dates) if d.year < year]
    ytd = _r(equity[-1] / equity[prior[-1]] - 1.0) if prior else None
    return {"d1": back(1), "m1": back(21), "ytd": ytd, "y1": back(TRADING_DAYS)}


def weight_dict(row: np.ndarray, assets: Sequence[str]) -> dict[str, float]:
    """Nonzero weights, rounded, with the rounding residue folded into the largest entry."""
    names = [CASH, *assets]
    out = {names[j]: round(float(w), W_DECIMALS) for j, w in enumerate(row) if round(float(w), W_DECIMALS) > 0}
    if out:
        top = max(out, key=out.get)
        out[top] = round(out[top] + 1.0 - sum(out.values()), W_DECIMALS)
    return out


def trades(before: np.ndarray, after: np.ndarray, assets: Sequence[str],
           min_change: float = MIN_CHANGE) -> list[Change]:
    """Per-ETF weight changes of at least `min_change` (cash is the residual, not a trade)."""
    return [Change(t, float(before[j + 1]), float(after[j + 1])) for j, t in enumerate(assets)
            if abs(after[j + 1] - before[j + 1]) >= min_change]


def round_row(row: Sequence[float]) -> list[float]:
    """Round weights, folding the rounding residue into the largest so the row sums to 1."""
    out = [round(max(float(x), 0.0), W_DECIMALS) for x in row]
    if out:
        k = max(range(len(out)), key=out.__getitem__)
        out[k] = round(out[k] + 1.0 - sum(out), W_DECIMALS)
    return out


def spark(dates: Sequence[dt.date], equity: np.ndarray, sessions: int = TRADING_DAYS) -> list[float]:
    """Weekly points over the last `sessions`, scaled to 1 at the first."""
    lo = max(0, len(dates) - sessions - 1)
    idx = [i for i in weekly_sample(dates) if i >= lo]
    if idx[0] != lo:
        idx.insert(0, lo)
    idx = idx[-60:]
    return [_sig(equity[i] / equity[idx[0]], 5) for i in idx]


def weekly_sample(dates: Sequence[dt.date]) -> list[int]:
    """Index of the last session of each ISO week, plus the final session."""
    idx = [i for i in range(len(dates) - 1) if dates[i].isocalendar()[:2] != dates[i + 1].isocalendar()[:2]]
    idx.append(len(dates) - 1)
    return idx


# --------------------------------------------------------------------------- documents

def backtest_doc(p: PortfolioPath) -> dict:
    eq = np.asarray(p.equity, dtype=float) / float(p.equity[0])
    monthly, annual = calendar_returns(p.dates, eq)
    held = [j for j in range(p.weights.shape[1]) if np.any(p.weights[:, j] > 1e-6)]
    names = [CASH, *p.assets]
    sample = weekly_sample(p.dates)
    return {
        "schema_version": SCHEMA_VERSION,
        "portfolio": p.id, "strategy": p.strategy, "ceiling": p.ceiling, "holdings": p.holdings,
        "start": iso(p.dates[0]), "end": iso(p.dates[-1]),
        "dates": [iso(d) for d in p.dates],
        "equity": [_sig(x) for x in eq],
        "weights": {
            "assets": [names[j] for j in held],
            "dates": [iso(p.dates[i]) for i in sample],
            "values": [round_row([p.weights[i, j] for j in held]) for i in sample],
        },
        "rebalances": [rb.doc() for rb in p.rebalances],
        "metrics": metrics(eq),
        "monthly_returns": monthly,
        "annual_returns": annual,
    }


def today_entry(p: PortfolioPath) -> dict:
    eq = np.asarray(p.equity, dtype=float)
    last = p.rebalances[-1] if p.rebalances else None
    if p.risk_state not in RISK_STATES:
        raise ValueError(f"unknown risk_state {p.risk_state!r}")
    current = np.asarray(p.weights[-1], dtype=float)
    plan = current if p.target is None else np.asarray(p.target, dtype=float)
    held = weight_dict(current, p.assets)
    return {
        "strategy": p.strategy, "ceiling": p.ceiling, "holdings": p.holdings,
        "weights": weight_dict(plan, p.assets),
        "current_weights": held,
        "next_trades": [c.doc() for c in trades(current, plan, p.assets)],
        "drawdown": _r(min(max(p.drawdown, 0.0), 1.0)),
        "risk_state": p.risk_state,
        # A position too small to survive rounding is not shown, so neither is its lock.
        "locks": {t: iso(d) for t, d in sorted(p.locks.items()) if d > p.dates[-1] and t in held},
        "returns": period_returns(p.dates, eq),
        "summary": {k: v for k, v in metrics(eq).items() if k in ("cagr", "sharpe", "max_drawdown")},
        "spark": spark(p.dates, eq),
        "last_rebalance": last.doc() if last else None,
        "traded_today": bool(last and last.date == p.dates[-1]),
    }


def today_doc(data_date: dt.date, paths: Sequence[PortfolioPath]) -> dict:
    return {"schema_version": SCHEMA_VERSION, "data_date": iso(data_date),
            "portfolios": {p.id: today_entry(p) for p in paths}}
