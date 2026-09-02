"""Stage 10 -- block bootstrap confidence bands.

reference/robustness.md section 2. **Stationary (Politis-Romano) or moving-block, never
IID.** Independent resampling destroys volatility clustering and serial dependence, which
are precisely the properties the entire risk layer is built to handle; an IID bootstrap
produces comfortable, meaningless confidence intervals.

**What is resampled, and why that preserves the cross-section.** Blocks are drawn over
*sessions*, and a session carries the whole portfolio at once. Because the same block
indices apply to every asset simultaneously, cross-asset correlation is preserved by
construction -- resampling each asset independently would manufacture diversification that
does not exist and would flatter every drawdown number.

**What this does and does not measure.** It quantifies sampling uncertainty in the
realized strategy path *within the observed regime distribution*. It is not a statement
about regimes that have not occurred, and it does not re-run the policy on counterfactual
prices: doing that would require synthesizing price and feature histories, which leaves
the support of real history -- the same objection robustness.md section 3 raises against
synthetic shocks. Every band this module produces carries that caveat into the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252

#: robustness.md section 2.
DEFAULT_REPLICATES = 1000
DEFAULT_MEAN_BLOCK = 10
BLOCK_LENGTH_SENSITIVITY: tuple[int, ...] = (5, 10, 21, 63)


def stationary_block_indices(n: int, mean_block: float,
                             rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano: geometric block lengths, wrapping at the end of the sample.

    The geometric length is what makes it *stationary* -- a fixed block length imposes a
    period on the resampled series, and that period shows up in any autocorrelation-
    sensitive statistic.
    """
    if n < 2:
        return np.arange(n)
    p = 1.0 / max(mean_block, 1.0)
    out = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        length = min(int(rng.geometric(p)), n - i)
        out[i:i + length] = (start + np.arange(length)) % n
        i += length
    return out


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed-length blocks, wrapping. The simpler alternative, offered for comparison."""
    if n < 2:
        return np.arange(n)
    block = max(1, min(int(block), n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(s + np.arange(block)) % n for s in starts])
    return idx[:n]


def resample_indices(n: int, *, method: str, mean_block: float,
                     rng: np.random.Generator) -> np.ndarray:
    if method == "stationary":
        return stationary_block_indices(n, mean_block, rng)
    if method == "moving_block":
        return moving_block_indices(n, int(mean_block), rng)
    raise ValueError(
        f"unknown bootstrap method {method!r}. IID resampling is deliberately not "
        "offered: it destroys the volatility clustering the risk layer exists to handle.")


def metrics_from_returns(returns: np.ndarray) -> dict:
    """The standard set, recomputed from a resampled daily return path."""
    if returns.size < 2:
        return {k: 0.0 for k in
                ("total_return", "annualized_return", "volatility", "sharpe",
                 "sortino", "max_drawdown", "worst_1d", "worst_5d", "calmar")}
    nav = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(nav)
    max_dd = float((1.0 - nav / peak).max())
    years = returns.size / TRADING_DAYS
    total = float(nav[-1] - 1.0)
    ann = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    shortfall = np.minimum(returns, 0.0)
    downside = float(np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(TRADING_DAYS))
    rolling5 = pd.Series(np.log1p(returns)).rolling(5).sum().min()
    return {
        "total_return": total,
        "annualized_return": ann,
        "volatility": vol,
        # rf = 0, matching the report: CASH here returns exactly zero.
        "sharpe": float(ann / vol) if vol > 1e-12 else 0.0,
        "sortino": float(ann / downside) if downside > 1e-12 else 0.0,
        "max_drawdown": max_dd,
        "calmar": float(ann / max_dd) if max_dd > 1e-12 else 0.0,
        "worst_1d": float(returns.min()),
        "worst_5d": float(rolling5) if np.isfinite(rolling5) else 0.0,
    }


@dataclass
class BootstrapResult:
    method: str
    mean_block_length: float
    replicates: int
    n_sessions: int
    observed: dict = field(default_factory=dict)
    bands: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"method": self.method, "mean_block_length": self.mean_block_length,
                "replicates": self.replicates, "n_sessions": self.n_sessions,
                "observed": self.observed, "bands": self.bands}


def bootstrap(returns: np.ndarray, *, replicates: int = DEFAULT_REPLICATES,
              method: str = "stationary", mean_block: float = DEFAULT_MEAN_BLOCK,
              seed: int = 42, quantiles=(0.05, 0.25, 0.50, 0.75, 0.95)
              ) -> BootstrapResult:
    """Confidence bands for every standard metric."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    rng = np.random.default_rng(seed)
    n = returns.size

    draws: dict[str, list[float]] = {}
    for _ in range(replicates):
        idx = resample_indices(n, method=method, mean_block=mean_block, rng=rng)
        for key, value in metrics_from_returns(returns[idx]).items():
            draws.setdefault(key, []).append(value)

    bands = {}
    for key, values in draws.items():
        arr = np.asarray(values, dtype=float)
        bands[key] = {
            "mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
            **{f"q{int(q * 100):02d}": float(np.quantile(arr, q)) for q in quantiles},
        }
    return BootstrapResult(
        method=method, mean_block_length=float(mean_block), replicates=replicates,
        n_sessions=n, observed=metrics_from_returns(returns), bands=bands)


def block_length_sensitivity(returns: np.ndarray, *, lengths=BLOCK_LENGTH_SENSITIVITY,
                             replicates: int = 250, method: str = "stationary",
                             seed: int = 42) -> dict:
    """Do the conclusions move with the block length?

    Reported whether or not it is flattering. Picking the block length that produces the
    tightest band is a way of choosing an answer, and the whole point of a sensitivity
    table is to make that visible.
    """
    out = {}
    for length in lengths:
        result = bootstrap(returns, replicates=replicates, method=method,
                           mean_block=length, seed=seed)
        out[str(length)] = {
            key: {"q05": result.bands[key]["q05"], "q50": result.bands[key]["q50"],
                  "q95": result.bands[key]["q95"]}
            for key in ("annualized_return", "volatility", "sharpe", "sortino",
                        "max_drawdown")
        }

    # A crude but honest summary: how much the 90% band width for the headline metrics
    # moves across block lengths.
    spread = {}
    for key in ("annualized_return", "max_drawdown", "sortino"):
        widths = [out[str(b)][key]["q95"] - out[str(b)][key]["q05"] for b in lengths]
        spread[key] = {"min_width": float(min(widths)), "max_width": float(max(widths)),
                       "ratio": float(max(widths) / max(min(widths), 1e-12))}
    return {"by_block_length": out, "band_width_spread": spread,
            "note": ("If a conclusion changes materially with block length, the report "
                     "says so rather than quoting the most favourable choice.")}


def portfolio_returns(trajectories) -> np.ndarray:
    """Daily returns from one or more trajectories, concatenated in date order.

    Concatenating *returns* rather than NAV levels is what keeps a fold boundary from
    manufacturing a spurious jump: each fold restarts at its own capital, so the level
    series is discontinuous by design while the return series is not.
    """
    frames = trajectories if isinstance(trajectories, (list, tuple)) else [trajectories]
    pieces = []
    for traj in sorted(frames, key=lambda f: f.index[0]):
        pieces.append(traj["nav"].astype(float).pct_change().dropna().to_numpy())
    return np.concatenate(pieces) if pieces else np.zeros(0)
