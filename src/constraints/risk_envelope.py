"""The drawdown engine and the action-level risk safety envelope.

The honest statement of what is enforced, and the one this implementation commits to:

> The maximum-drawdown constraint is an **action-level safety constraint**, not a
> guarantee about the realized path.

At every decision the agent may not take an action whose *stressed* outcome would breach
`D_max`, under a stress estimate built from data visible at that moment. Claiming a hard
guarantee on realized drawdown would be false, and every report must say so.

Never a reward term (D11). The budget is a constraint consumed by the projection.

**Speed.** This is called once per step across millions of steps, so everything that does
not depend on `w` is computed once per session: the trailing return matrix, the bootstrap
paths, the date-filtered crisis paths. Given a cached matrix of stress return paths `R`
(paths x assets), the loss for any `w` is a matrix-vector product and a quantile.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

Measure = Literal["var", "cvar"]
Aggregation = Literal["max", "mean", "quantile"]


def _loss_from_paths(paths: np.ndarray, w: np.ndarray, q: float,
                     measure: Measure) -> float:
    """Stressed loss fraction for weights `w`, positive for losses.

    `paths` are horizon-cumulative returns per asset, one row per scenario. Index 0 is
    cash, whose return is identically zero.

    `cvar` is the default and the only measure usable with the analytic backend, because
    it is **convex in `w`** and the de-risking bisection depends on that. `var` (a raw
    sample quantile) is not convex -- measured violation ~1e-3, the same order as the
    risk budget itself -- so it is available for reporting only. See
    `RiskEnvelope.stress_loss`.
    """
    if paths.size == 0:
        return 0.0
    port = paths @ w
    if measure == "var":
        return float(-np.quantile(port, q))

    # Empirical CVaR with a FRACTIONAL tail weight (Rockafellar-Uryasev). Averaging
    # "every value at or below the q-quantile" instead would make the tail size jump by
    # whole observations as `w` moves, putting small kinks in an otherwise convex
    # function. The fractional form is exactly convex.
    losses = -np.asarray(port, dtype=float)
    n = losses.size
    k = q * n
    if k <= 0:
        return float(losses.max())
    ordered = np.sort(losses)[::-1]
    whole = int(np.floor(k))
    total = ordered[:whole].sum()
    if whole < n:
        total += (k - whole) * ordered[whole]
    return float(total / k)


# ------------------------------------------------------------------- estimators


@dataclass
class StressSample:
    """Cached horizon-cumulative return paths for one estimator, at one session."""

    name: str
    paths: np.ndarray            # (n_paths, n_assets), index 0 = cash (all zeros)


def rolling_stress_paths(returns: np.ndarray, horizon: int) -> np.ndarray:
    """Overlapping horizon-day cumulative returns from the visible trailing window.

    Cheap, and the default component. Its weakness is that it only sees what the trailing
    window contains, so it under-reacts entering a regime change -- which is precisely why
    it is not used alone.
    """
    n = returns.shape[0]
    if n < horizon + 1:
        return np.empty((0, returns.shape[1]))
    # Cumulative log-free approximation: sum of simple returns over the horizon is close
    # enough at daily scale, and is what keeps this a single matrix operation.
    csum = np.cumsum(returns, axis=0)
    return csum[horizon:] - csum[:-horizon]


def block_bootstrap_paths(
    returns: np.ndarray, horizon: int, n_paths: int, block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary bootstrap (Politis-Romano) over historical return blocks.

    **Never IID resampling.** It destroys volatility clustering and serial dependence, and
    would produce a comfortable, wrong risk number.
    """
    n = returns.shape[0]
    if n < max(horizon, block_length) + 1:
        return np.empty((0, returns.shape[1]))

    p = 1.0 / max(1, block_length)
    out = np.empty((n_paths, returns.shape[1]))
    starts = rng.integers(0, n, size=(n_paths, horizon))
    continues = rng.random((n_paths, horizon)) > p
    for i in range(n_paths):
        idx = np.empty(horizon, dtype=int)
        cur = starts[i, 0]
        for t in range(horizon):
            if t > 0:
                cur = (cur + 1) % n if continues[i, t] else starts[i, t]
            idx[t] = cur
        out[i] = returns[idx].sum(axis=0)
    return out


def crisis_window_paths(
    returns: np.ndarray, sessions: np.ndarray, windows: dict[str, Sequence[str]],
    horizon: int, as_of: dt.date | None,
) -> np.ndarray:
    """Fixed historical scenario paths, **date-filtered**.

    At decision time `t`, only crisis windows that ENDED before `t` may be used. A 2005
    decision cannot be stress-tested against 2008. Stage 9 evaluation deliberately passes
    `as_of=None` to use the full library -- that is a stress test, not a decision input,
    and the two code paths are kept distinct by this argument.
    """
    rows = []
    for _name, (start, end) in windows.items():
        w_start = np.datetime64(str(start))
        w_end = np.datetime64(str(end))
        if as_of is not None and w_end >= np.datetime64(as_of):
            continue
        sel = (sessions >= w_start) & (sessions <= w_end)
        block = returns[sel]
        if block.shape[0] < horizon:
            continue
        csum = np.cumsum(block, axis=0)
        rows.append(csum[horizon:] - csum[:-horizon])
    if not rows:
        return np.empty((0, returns.shape[1]))
    return np.vstack(rows)


# --------------------------------------------------------------------- envelope


@dataclass
class RiskEnvelope:
    """Aggregates several stress estimators into one action-level constraint."""

    quantile: float = 0.01
    horizon_days: int = 5
    block_length: int = 10
    aggregation: Aggregation = "max"
    measure: Measure = "cvar"   # the only measure valid with the analytic backend
    n_bootstrap_paths: int = 256
    estimators: tuple[str, ...] = ("rolling", "block_bootstrap", "crisis_windows")
    crisis_windows: dict[str, Sequence[str]] = field(default_factory=dict)

    #: Set per session by `prepare`.
    _samples: list[StressSample] = field(default_factory=list, repr=False)
    _budget: float = field(default=1.0, repr=False)

    # ----------------------------------------------------------------- per session

    def prepare(
        self, returns: np.ndarray, sessions: np.ndarray, *, as_of: dt.date | None,
        seed: int, n_assets: int,
    ) -> None:
        """Compute everything that does not depend on `w`, once for this session.

        `returns` is the VISIBLE trailing history only, shape (T, n_assets), with column 0
        for cash (identically zero). The bootstrap draws from a session-derived seed, so
        the estimator is deterministic given the seed.
        """
        self._samples = []
        if returns.size == 0:
            return
        rng = np.random.default_rng(seed)

        if "rolling" in self.estimators:
            self._samples.append(StressSample(
                "rolling", rolling_stress_paths(returns, self.horizon_days)))
        if "block_bootstrap" in self.estimators:
            self._samples.append(StressSample(
                "block_bootstrap",
                block_bootstrap_paths(returns, self.horizon_days,
                                      self.n_bootstrap_paths, self.block_length, rng)))
        if "crisis_windows" in self.estimators and self.crisis_windows:
            self._samples.append(StressSample(
                "crisis_windows",
                crisis_window_paths(returns, sessions, self.crisis_windows,
                                    self.horizon_days, as_of)))
        self._samples = [s for s in self._samples if s.paths.size]

    # --------------------------------------------------------------------- query

    def stress_loss(self, w: np.ndarray) -> float:
        """Aggregate stressed loss fraction over the decision horizon, positive for losses.

        `max` is the default and the right one for a hard constraint: it is the most
        conservative combination, so no single estimator's blind spot can wave an action
        through.

        **Convexity, not monotonicity, is the property the projection needs.**

        reference/risk-envelope.md section 5 originally required `stress_loss` to be
        monotone non-increasing as weight shifts toward `w_safe`. That requirement is
        false whenever diversification is available: `w_safe` minimizes *exposure*, not
        *risk*. A book holding a locked position plus an anti-correlated hedge can be many
        times safer than the same locked position plus cash -- measured at 0.0088 versus
        0.0605 on a constructed pair, a factor of seven the wrong way.

        What the alpha bisection actually needs is that the feasible set along the segment
        `w(alpha)` be an interval containing `alpha = 0`. With `measure="cvar"` each
        component is convex in `w`, and `max` and `mean` of convex functions are convex,
        so every sublevel set along the segment is an interval; `w_safe` being feasible
        puts 0 in it. That is exactly the guarantee, and it holds with no monotonicity
        assumption. `measure="var"` is not convex and does not get it.
        """
        if not self._samples:
            return 0.0
        losses = [_loss_from_paths(s.paths, w, self.quantile, self.measure)
                  for s in self._samples]
        if self.aggregation == "max":
            return max(losses)
        if self.aggregation == "mean":
            return float(np.mean(losses))
        return float(np.quantile(losses, 0.9))

    def set_budget(self, nav: float, peak: float, d_max: float) -> float:
        self._budget = risk_budget(nav, peak, d_max)
        return self._budget

    def budget(self) -> float:
        return self._budget

    def is_feasible(self, w: np.ndarray) -> bool:
        return self.stress_loss(w) <= self._budget

    def component_losses(self, w: np.ndarray) -> dict[str, float]:
        """Per-estimator breakdown. Used by the Stage 5 calibration table."""
        return {s.name: _loss_from_paths(s.paths, w, self.quantile, self.measure)
                for s in self._samples}


def risk_budget(nav: float, peak: float, d_max: float) -> float:
    """`RiskBudget_t = 1 - (1 - D_max) * P_t / V_t`.

    From requiring that a stressed loss keeps the projected drawdown legal:

        D_projected(w) = 1 - V_t (1 - L_stress(w)) / P_t  <=  D_max

    Shrinks as drawdown deepens and goes to zero, or negative, at the ceiling. A negative
    budget means only `w_safe` is admissible, which is capital preservation -- consistent
    with the state-feasibility layer rather than a separate rule.
    """
    if nav <= 0.0:
        return -np.inf
    return 1.0 - (1.0 - d_max) * (peak / nav)


def headroom(nav: float, peak: float, d_max: float) -> float:
    """`H_t = D_max - D_t`. Mandatory in the observation."""
    drawdown = 1.0 - nav / peak if peak > 0 else 0.0
    return d_max - drawdown
