"""The web app's publish pipeline (WEBAPP_PLAN.md Step 3), on a short real-data window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import s15_publish
from src.config.loader import resolve_config
from src.publish import build, contract, pipeline
from src.sim.runner import load_market

CURATED = Path("data/curated/prices.parquet")
pytestmark = pytest.mark.skipif(not CURATED.exists(), reason="run Stage 2 first")
START, END = "2019-01-02", "2021-12-31"


@pytest.fixture(scope="module")
def setup():
    resolved = resolve_config(pipeline.Settings.config)
    market, _ = load_market(resolved, start=START, end=END)
    return resolved, market


@pytest.fixture(scope="module")
def momentum(setup):
    resolved, market = setup
    return pipeline.run_portfolio(market, resolved, "momentum", 5)[0]


def test_webapp_config_is_the_research_main_line():
    r = resolve_config(pipeline.Settings.config)
    assert r["drawdown"]["peak_window_sessions"] == 252
    assert r["projection"]["over_budget_rule"] == "no_risk_increase"
    assert r["projection"]["derisk_step"] > 0
    assert r["execution"]["no_trade_band"] > 0
    main_line = resolve_config("config/experiments/redesign_v2.yaml")
    for k in ("drawdown", "projection", "execution", "lock", "risk"):
        assert r[k] == main_line[k]


def test_a_monthly_strategy_trades_about_monthly(momentum):
    """Between rebalances a hold must not trade at the next open (the daily top-up bug,
    fixed by the no-trade band, R-1)."""
    days = {rb.date for rb in momentum.rebalances}
    months = len({(d.year, d.month) for d in momentum.dates})
    assert len(days) <= 2 * months, f"{len(days)} trade days in {months} months"


def test_path_is_consistent_with_the_simulation(momentum):
    p = momentum
    assert len(p.dates) == len(p.equity) == len(p.weights)
    np.testing.assert_allclose(p.weights.sum(axis=1), 1.0, atol=1e-9)
    assert p.weights[0, 0] == 1.0 and p.equity[0] == 1_000_000.0
    assert p.target is not None and abs(p.target.sum() - 1.0) < 1e-9
    assert p.risk_state in build.RISK_STATES
    for rb in p.rebalances:
        assert all(abs(c.after - c.before) >= build.MIN_CHANGE for c in rb.changes)
    held = {t for j, t in enumerate(p.assets) if p.weights[-1, j + 1] > 0}
    assert set(p.locks) <= held


def test_cli_writes_a_valid_site_and_keeps_the_old_one_on_failure(tmp_path, monkeypatch):
    out = tmp_path / "site"
    args = ["--out", str(out), "--start", START, "--end", END, "--only", "spy_tlt_60_40", "--ceilings", "5"]
    assert s15_publish.main(args) == 0
    assert contract.validate_site(out) == []
    before = (out / "manifest.json").read_text(encoding="utf-8")

    monkeypatch.setattr(s15_publish.contract, "validate_site", lambda d: ["forced failure"])
    assert s15_publish.main(args) == 1
    assert (out / "manifest.json").read_text(encoding="utf-8") == before


def test_live_log_appends_each_session_once(tmp_path):
    out, log = tmp_path / "site", tmp_path / "log"
    args = ["--out", str(out), "--start", START, "--end", END, "--only", "spy_tlt_60_40",
            "--ceilings", "5", "--log-dir", str(log)]
    assert s15_publish.main(args) == 0
    assert s15_publish.main(args) == 0          # same data date: nothing new to append
    files = list(log.glob("*.jsonl"))
    assert [f.name for f in files] == ["2021-12.jsonl"]
    assert len(files[0].read_text(encoding="utf-8").splitlines()) == 1
    import json
    doc = json.loads((out / "live" / "index.json").read_text(encoding="utf-8"))
    assert doc["launch_date"] == "2021-12-31"
    assert [e["date"] for e in doc["portfolios"]["spy_tlt_60_40_D5"]] == ["2021-12-31"]
    assert contract.validate_site(out) == []


def test_live_log_never_appends_an_older_session():
    from src.publish import live
    lines = [{"date": "2026-10-09", "portfolios": {}}]
    assert not live.is_new(lines, {"date": "2026-10-08", "portfolios": {}})
    assert not live.is_new(lines, {"date": "2026-10-09", "portfolios": {}})
    assert live.is_new(lines, {"date": "2026-10-12", "portfolios": {}})
