"""Assemble the observation vector from market features plus the live portfolio.

Named in reference/features.md section 4, which lists the portfolio fields; the layout and
the Markov floor are in `src/env/observation.py`. This module is the one that actually
writes the numbers into the vector, in the declared order.

The one non-obvious point is `unlock_proximity`. `lock_remaining` alone is ambiguous
across the parameter grid: 20 days remaining means "just bought" under `N = 21` and
"two thirds of the way out" under `N = 60`. Both forms are present, and both are divided
by `N`, so the policy can read either the absolute clock or the fraction of the lock
already served without having to divide by an observed parameter itself.
"""

from __future__ import annotations

import numpy as np

from src.env.feature_store import FeatureStore
from src.env.observation import MANDATORY_GLOBAL, ObservationSpec, finite_or_raise
from src.sim.engine import Decision

#: `N = 0` is a legal stress value (the no-lock control), so every division by N guards
#: against it rather than assuming the operating range.
_MIN_N = 1.0


def build_observation(
    spec: ObservationSpec,
    store: FeatureStore,
    feature_row: int,
    dec: Decision,
    *,
    hold_days: int,
    max_drawdown: float,
    nav_at_reset: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Write one observation. `out` may be reused across steps to avoid reallocating."""
    K = spec.n_assets
    F = spec.n_per_asset
    n_market = len(spec.per_asset_market)
    obs = np.zeros(spec.size, dtype=np.float32) if out is None else out
    obs.fill(0.0)

    per_asset, glob = store.at(feature_row)

    # ---- global market block --------------------------------------------------
    obs[spec.macro_slice] = glob

    # ---- per-asset block ------------------------------------------------------
    ctx = dec.ctx
    available = np.asarray(ctx.available, dtype=bool)
    weights = np.asarray(ctx.current_weights, dtype=np.float64)[1:]
    shares = ctx.ledger.share_vector(list(spec.tickers))
    remaining = np.asarray(ctx.lock_remaining_days, dtype=np.float64)
    locked = remaining > 0
    n = max(float(hold_days), _MIN_N)

    block = np.zeros((K, F), dtype=np.float32)
    # Unavailable assets are ZEROED and carry mask bit 0. The mask is what carries the
    # information; the zeros are filler (env-mdp.md section 1).
    block[:, :n_market] = np.where(available[:, None], per_asset, 0.0)
    block[:, n_market + 0] = weights
    block[:, n_market + 1] = shares > 0
    block[:, n_market + 2] = locked
    block[:, n_market + 3] = remaining / n
    # 0 when unlocked, rising to 1 as the lock is served out.
    block[:, n_market + 4] = np.where(locked, 1.0 - np.minimum(remaining, n) / n, 0.0)
    block[:, n_market + 5] = available
    obs[spec.per_asset_slice] = block.reshape(-1)

    # ---- portfolio block ------------------------------------------------------
    nav = float(ctx.nav)
    locked_nav_frac = float(weights[locked].sum()) if locked.any() else 0.0
    wavg_lock = (float((weights[locked] * remaining[locked]).sum() / weights[locked].sum())
                 if locked.any() and weights[locked].sum() > 1e-12 else 0.0)
    values = {
        "pf_cash_weight": float(ctx.current_weights[0]),
        # Normalized against the EPISODE START, not against a constant, so the scale is
        # comparable across episodes that begin at very different NAVs.
        "pf_nav_norm": nav / max(nav_at_reset, 1e-9),
        "pf_peak_norm": float(ctx.peak) / max(nav_at_reset, 1e-9),
        "pf_drawdown": float(ctx.drawdown),
        # B_t = D_max - D_t. Mandatory. May go negative in capital preservation, and the
        # sign is exactly the information the agent needs, so it is not clipped at 0.
        "pf_drawdown_budget": float(max_drawdown) - float(ctx.drawdown),
        "pf_locked_count": float(locked.sum()) / max(K, 1),
        "pf_locked_nav_frac": locked_nav_frac,
        "pf_wavg_lock_days": wavg_lock / n,
    }
    obs[spec.portfolio_slice] = np.array(
        [values[k] for k in MANDATORY_GLOBAL], dtype=np.float32)

    # ---- parameter block ------------------------------------------------------
    # Fixed normalization, never fitted -- this is what makes deployment semantics
    # identical to training semantics (env-mdp.md section 4).
    obs[spec.param_slice] = np.array(
        [hold_days / spec.hold_days_divisor, max_drawdown], dtype=np.float32)

    if spec.clip:
        np.clip(obs, -spec.clip, spec.clip, out=obs)
    finite_or_raise(obs, spec, where=f"session {ctx.session.date()}")
    return obs
