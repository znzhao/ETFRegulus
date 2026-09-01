"""The drawdown violation taxonomy: preventable versus market-forced.

reference/evaluation.md section 4. When realized drawdown exceeds `D_max` the report
*must* distinguish two categories, because conflating them is how a broken safety layer
gets excused as bad luck:

**A. Preventable** -- the agent's action should have been blocked and was not. A bug in the
projection, or the safety engine failing to execute as defined. **Any preventable violation
is a project-blocking defect.** It is not tuned away; it is fixed.

**B. Market-forced** -- a locked asset that could not be sold, an overnight gap, or prices
that simply exceeded the limit despite legal behaviour. Legitimate: the honest consequence
of the constraint being action-level rather than a path guarantee.

Detection is a **replay**, not an inference. For every decision the trajectory recorded the
action the projection actually produced (`proj_weights`), so this module can rebuild the
feasible set from the state as it stood at that decision and ask whether that action was in
it. Inferring legality from the position that resulted would be much weaker -- prices move
between the decision and the close, so a resulting position tells you little about whether
the action that caused it was legal.

The detector is exercised by T13 against a deliberately injected projection bug, because an
invariant checker that has never been observed to fail is untested infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Slack on every bound. The projection works to ~1e-9; anything above this is a real
#: excursion rather than float noise, and the threshold is stated rather than tuned.
TOL = 1e-6


@dataclass
class ReplayFinding:
    session: str
    kind: str
    detail: str
    magnitude: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class BreachRecord:
    """One drawdown breach, with the evidence that classifies it.

    `actions_blocked_by_lock` is the field that carries the argument: it shows what the
    agent tried to do and could not. Without it, "market-forced" is an assertion.
    """

    breach_start: str
    breach_end: str
    breach_depth: float
    breach_duration: int
    locked_exposure_at_breach: float
    available_cash_at_breach: float
    actions_blocked_by_lock: int
    classification: str = "market_forced"
    preventable_findings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------- the replay


def _executed_shares(shares, div, close):
    """Strip reinvested distributions back out of a recorded share count.

    A trajectory records shares at the close, *after* dividends were reinvested as share
    accretion. Accretion is deliberately exempt from both the lock and the
    capital-preservation cap -- it is a corporate action, not a trade, and
    `Ledger.accrue_shares` is a separate entry point precisely so the lock manager never
    observes it. Comparing recorded closes directly therefore reports every distribution
    as a cap breach.

    Since `shares_close = shares_executed * (1 + div/close)`, the inversion is exact and
    needs no tolerance. That matters: XLE paid $0.2637 on 2020-03-23 into a collapsed
    $11.79 price, a 2.24% one-day accretion, which no plausible fixed threshold would
    have separated from real dip-buying.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where((close > 0) & np.isfinite(close) & np.isfinite(div),
                        div / close, 0.0)
    return shares / (1.0 + np.maximum(rate, 0.0))


def replay_feasibility(traj: pd.DataFrame, universe: list[str], *,
                       d_max: float, tol: float = TOL,
                       market=None) -> list[ReplayFinding]:
    """Re-derive each decision's feasible set and check the action that was taken.

    Four bounds, all reconstructible from what the trajectory recorded:

    1. the action is a point on the simplex (weights non-negative, summing to one);
    2. availability -- no weight on an asset that did not exist yet;
    3. the lock floor -- a locked position's weight was not projected below what the lock
       required it to keep;
    4. capital preservation -- while breached, share counts did not rise.

    Any excursion is a preventable violation *by definition*: the projection is the thing
    that was supposed to prevent it.

    `market` is optional but strongly recommended: without it, reinvested distributions
    cannot be separated from purchases, and every dividend paid during a
    capital-preservation window reads as a cap breach.
    """
    findings: list[ReplayFinding] = []
    if "proj_weights" not in traj.columns:
        raise KeyError(
            "trajectory has no `proj_weights` column, so the decision cannot be replayed. "
            "It was written by an older version of src/sim/engine.py -- rerun the "
            "simulation rather than weakening the check.")

    share_cols = [f"shares_{t}" for t in universe]
    locked_cols = [f"locked_{t}" for t in universe]
    prev_shares = None

    # Per-session dividend and close, so accretion can be removed exactly.
    div_of: dict = {}
    if market is not None:
        rows = market.sessions.get_indexer(pd.DatetimeIndex(traj.index))
        for k, r in enumerate(rows):
            if r >= 0:
                div_of[traj.index[k]] = (market.div_per_share[r], market.close_raw[r])

    for session, row in traj.iterrows():
        stamp = str(pd.Timestamp(session).date())
        w = np.asarray(row["proj_weights"], dtype=float)

        # 1. still on the simplex
        if w.min() < -tol:
            findings.append(ReplayFinding(
                stamp, "negative_weight",
                f"projected weight {w.min():.3e} < 0", abs(float(w.min()))))
        total = float(w.sum())
        if abs(total - 1.0) > 1e-6:
            findings.append(ReplayFinding(
                stamp, "not_on_simplex",
                f"projected weights sum to {total:.9f}", abs(total - 1.0)))

        shares = np.array([float(row[c]) for c in share_cols])
        locked = np.array([bool(row[c]) for c in locked_cols])
        if session in div_of:
            div, close = div_of[session]
            executed = _executed_shares(shares, np.asarray(div, dtype=float),
                                        np.asarray(close, dtype=float))
        else:
            executed = shares

        # 2. availability: a position in an asset with no price is pre-inception.
        for j, ticker in enumerate(universe):
            if shares[j] > 1e-9 and not np.isfinite(row.get(f"w_{ticker}", np.nan)):
                findings.append(ReplayFinding(
                    stamp, "pre_inception", f"{ticker} held with no price"))

        # 3. the lock floor, checked against the previous session's holdings.
        if prev_shares is not None:
            shrank = locked & (executed < prev_shares - 1e-9)
            for j in np.flatnonzero(shrank):
                findings.append(ReplayFinding(
                    stamp, "lock_floor_breached",
                    f"{universe[j]} fell from {prev_shares[j]:.4f} to "
                    f"{executed[j]:.4f} while locked",
                    float(prev_shares[j] - executed[j])))

            # 4. capital preservation caps SHARE COUNTS, not weights.
            if bool(row.get("capital_preservation", False)):
                # Net of accretion: the cap constrains buying, not corporate actions.
                grew = executed > prev_shares * (1.0 + 1e-9) + 1e-6
                for j in np.flatnonzero(grew):
                    findings.append(ReplayFinding(
                        stamp, "preservation_cap_breached",
                        f"{universe[j]} grew from {prev_shares[j]:.4f} to "
                        f"{executed[j]:.4f} during capital preservation "
                        "(net of reinvested distributions)",
                        float(executed[j] - prev_shares[j])))
        prev_shares = shares
    return findings


# ------------------------------------------------------------ classification


def breach_episodes(traj: pd.DataFrame, d_max: float) -> list[tuple]:
    """Contiguous runs where realized drawdown exceeded the ceiling."""
    over = traj["drawdown"].astype(float) > d_max + 1e-12
    episodes, start = [], None
    index = list(traj.index)
    for i, (session, flag) in enumerate(zip(index, over)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, len(index) - 1))
    return episodes


def classify(traj: pd.DataFrame, universe: list[str], *, d_max: float,
             findings: list[ReplayFinding] | None = None,
             market=None) -> list[BreachRecord]:
    """Build one record per breach, classified by whether a replay finding sits inside it.

    A breach is **preventable** if and only if the replay found an illegal action within
    it. Otherwise it is market-forced -- and the record carries the evidence for that:
    what was locked, what cash was on hand, and how many actions the lock blocked.
    """
    findings = findings if findings is not None else replay_feasibility(
        traj, universe, d_max=d_max, market=market)
    by_session: dict[str, list[ReplayFinding]] = {}
    for f in findings:
        by_session.setdefault(f.session, []).append(f)

    records: list[BreachRecord] = []
    index = list(traj.index)
    locked_cols = [f"locked_{t}" for t in universe]
    weight_cols = [f"w_{t}" for t in universe]

    for lo, hi in breach_episodes(traj, d_max):
        window = traj.iloc[lo:hi + 1]
        first = traj.iloc[lo]
        sessions = {str(pd.Timestamp(s).date()) for s in index[lo:hi + 1]}
        inside = [f for s in sessions for f in by_session.get(s, [])]

        locked_mask = np.array([bool(first[c]) for c in locked_cols])
        weights = np.array([float(first[c]) for c in weight_cols])
        nav = float(first["nav"])

        records.append(BreachRecord(
            breach_start=str(pd.Timestamp(index[lo]).date()),
            breach_end=str(pd.Timestamp(index[hi]).date()),
            breach_depth=float(window["drawdown"].max()),
            breach_duration=int(hi - lo + 1),
            locked_exposure_at_breach=float(weights[locked_mask].sum())
            if locked_mask.any() else 0.0,
            available_cash_at_breach=float(first["cash"]) / nav if nav else 0.0,
            # What the agent tried to do and could not.
            actions_blocked_by_lock=int(window.get(
                "share_floor_binding", pd.Series([0])).sum()),
            classification="preventable" if inside else "market_forced",
            preventable_findings=[f.to_dict() for f in inside],
        ))
    return records


def summarize(traj: pd.DataFrame, universe: list[str], *, d_max: float,
              market=None) -> dict:
    """Everything Stage 12's risk-reporting requirements need, in one place.

    Preventable and market-forced are reported separately and are **never** summed into a
    single "violations" count.
    """
    findings = replay_feasibility(traj, universe, d_max=d_max, market=market)
    breaches = classify(traj, universe, d_max=d_max, findings=findings, market=market)
    preventable = [b for b in breaches if b.classification == "preventable"]
    forced = [b for b in breaches if b.classification == "market_forced"]

    return {
        "max_drawdown": float(traj["drawdown"].max()),
        "d_max": d_max,
        "n_breaches": len(breaches),
        "n_preventable": len(preventable),
        "n_market_forced": len(forced),
        # Findings outside any breach still matter: an illegal action that happened not to
        # cause a breach is the same defect, caught earlier.
        "n_replay_findings": len(findings),
        "replay_findings": [f.to_dict() for f in findings[:50]],
        "all_breaches_market_forced": len(preventable) == 0,
        "worst_breach_depth": max((b.breach_depth for b in breaches), default=0.0),
        "longest_breach_duration": max((b.breach_duration for b in breaches), default=0),
        "breaches": [b.to_dict() for b in breaches],
        "safety_intervention_rate": float(traj["safety_intervened"].mean())
        if "safety_intervened" in traj else 0.0,
    }
