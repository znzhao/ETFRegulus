"""The browser's RL plan (reference `src.publish.rl_live`) equals the simulator's plan.

The user's book is set to the simulator's final book (last buy = unlock date - N); the
reference must then build the same observation and propose the same projected target
as `engine.plan_next` with the policy wrapped as in the web app.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from src.config.loader import resolve_config
from src.publish import pipeline, rl_live
from src.sim.engine import plan_next
from src.sim.runner import build_constraints, build_envelope, build_sim_config, load_market, make_projector_from
from src.sim.simulator import simulate

pytestmark = pytest.mark.skipif(
    not (Path("data/curated/prices.parquet").exists() and pipeline.RL_MODEL_DIR.exists()),
    reason="needs curated data and the published model")


@pytest.mark.parametrize("ceiling,k", [(5, 0), (10, 3)])
def test_reference_plan_equals_the_simulator(ceiling, k):
    r = resolve_config(pipeline.Settings.config)
    market, _ = load_market(r, start="2024-06-03", end="2025-06-30")
    cons = build_constraints(r)
    cfg = build_sim_config(r, cons, max_drawdown=ceiling / 100, seed=42, max_holdings=k)
    env, proj = build_envelope(cons), make_projector_from(cons)
    fn = pipeline.rl_weight_source(market, r, hold_days=cfg.hold_days, max_drawdown=ceiling / 100, max_holdings=k)
    st = simulate(market, fn, cfg, projector=proj, envelope=env).final_state
    i = len(market.sessions) - 1
    src = pipeline.rl_weight_source(market, r, hold_days=cfg.hold_days, max_drawdown=ceiling / 100,
                                    max_holdings=k).source
    _, projected = plan_next(market, st, cfg, i, src, proj, envelope=env)

    doc = pipeline.rl_doc(market, r, market.sessions[-1].date())
    closes = market.prices_at(i, "close_raw")
    unlock = st.lock_manager.unlock_dates
    user = {"holdings": [{"t": t, "sh": st.ledger.get(t),
                          "d": (unlock[t] - dt.timedelta(days=cfg.hold_days)).isoformat() if t in unlock else ""}
                         for t in market.universe if st.ledger.get(t) > 0],
            "cash": st.ledger.cash, "peak": st.peak, "hold_days": cfg.hold_days,
            "prices": {t: float(closes[t]) for t in market.universe}, "ceiling": ceiling, "max_holdings": k}
    plan = rl_live.user_plan(doc, pipeline.risk_doc(env, r, list(market.universe), market.sessions[-1].date()), user)
    np.testing.assert_allclose(plan["target"], projected.weights, atol=1e-4)


def test_the_published_document_meets_its_schema():
    from src.publish.contract import schema_errors
    r = resolve_config(pipeline.Settings.config)
    market, _ = load_market(r, start="2025-01-02", end="2025-06-30")
    doc = pipeline.rl_doc(market, r, market.sessions[-1].date())
    assert schema_errors("rl", doc) == []
    assert len(doc["inputs"]["per_asset"]) == len(doc["tickers"]) == len(doc["inputs"]["available"])
