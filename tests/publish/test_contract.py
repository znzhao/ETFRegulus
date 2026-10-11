"""The web app's data contract (WEBAPP_PLAN.md Step 2): the fake data set is valid, and the
validator rejects the mistakes a pipeline could plausibly make."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts import fake_site_data
from src.publish import build, contract

# Two years keep the tests fast; the full 2004 start is exercised by the script itself.
SHORT = dt.date(2024, 10, 1)


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("site") / "data"
    errs = fake_site_data.generate(out, start=SHORT)
    assert errs == []
    return out


@pytest.fixture
def copy(site, tmp_path) -> Path:
    dst = tmp_path / "data"
    shutil.copytree(site, dst)
    return dst


def edit(root: Path, rel: str, fn) -> None:
    p = root / rel
    doc = json.loads(p.read_text(encoding="utf-8"))
    fn(doc)
    p.write_text(json.dumps(doc), encoding="utf-8")


def test_schemas_are_valid_json_schema():
    for kind in contract.KINDS:
        contract.validator(kind)


def test_fake_site_is_valid(site):
    assert contract.validate_site(site) == []
    man = json.loads((site / "manifest.json").read_text(encoding="utf-8"))
    assert len(man["portfolios"]) == 18


def test_failed_variant_is_valid_and_one_session_older(tmp_path):
    assert fake_site_data.generate(tmp_path / "f", failed=True, live_days=0, start=SHORT) == []
    man = json.loads((tmp_path / "f" / "manifest.json").read_text(encoding="utf-8"))
    assert man["status"] == "failed" and man["data_date"] == "2026-10-08"
    live = json.loads((tmp_path / "f" / "live" / "index.json").read_text(encoding="utf-8"))
    assert live["launch_date"] is None


@pytest.mark.parametrize("rel, mutate, expect", [
    ("manifest.json", lambda d: d.update(schema_version=2), "schema_version"),
    ("manifest.json", lambda d: d["portfolios"].pop(), "portfolios"),
    ("today.json", lambda d: d["portfolios"]["momentum_D5"]["weights"].update(CASH=0.9, SPY=0.5), "sum to"),
    ("today.json", lambda d: d["portfolios"]["momentum_D5"]["weights"].update(ZZZ=0.0), "unknown assets"),
    ("today.json", lambda d: d.update(data_date="2026-10-08"), "data_date"),
    ("today.json", lambda d: d["portfolios"]["spy_buy_hold_D5"]["locks"].update(QQQ="2027-01-04"), "not held"),
    ("today.json", lambda d: d["portfolios"]["spy_buy_hold_D5"].update(risk_state="panic"), "risk_state"),
    ("backtest/cash_D5.json", lambda d: d["equity"].pop(), "equity values"),
    ("backtest/cash_D5.json", lambda d: d.update(end="2026-10-08"), "start/end"),
    ("universe.json", lambda d: d["assets"].append(dict(d["assets"][0])), "duplicate"),
    ("risk.json", lambda d: d["samples"][0]["paths"][0].append(0.0), "columns"),
    ("live/index.json", lambda d: d.update(launch_date=None), "launch_date"),
])
def test_validator_rejects(copy, rel, mutate, expect):
    edit(copy, rel, mutate)
    errs = contract.validate_site(copy)
    assert errs and any(expect in e for e in errs), errs


def test_missing_backtest_file_is_reported(copy):
    (copy / "backtest" / "momentum_D10.json").unlink()
    assert any("missing file" in e for e in contract.validate_site(copy))


# --------------------------------------------------------------------------- build helpers

def _path(equity, dates):
    T = len(equity)
    w = np.zeros((T, 2))
    w[:, 1] = 1.0
    return build.PortfolioPath("spy_buy_hold", 5, dates, np.asarray(equity, float), w, ["SPY"])


def test_metrics_and_calendar_returns():
    dates = [dt.date(2025, 12, 30), dt.date(2025, 12, 31), dt.date(2026, 1, 2), dt.date(2026, 1, 5)]
    eq = np.array([1.0, 1.1, 0.99, 1.089])
    monthly, annual = build.calendar_returns(dates, eq)
    assert monthly["2025"][11] == pytest.approx(0.1)
    assert monthly["2026"][0] == pytest.approx(-0.01)
    assert annual == {"2025": pytest.approx(0.1), "2026": pytest.approx(-0.01)}
    assert build.max_drawdown(eq) == pytest.approx(0.1)
    r = build.period_returns(dates, eq)
    assert r["d1"] == pytest.approx(0.1) and r["ytd"] == pytest.approx(-0.01) and r["y1"] is None


def test_weight_dict_rounds_to_exactly_one():
    w = build.weight_dict(np.array([1 / 3, 1 / 3, 1 / 3]), ["SPY", "TLT"])
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)
    assert build.weight_dict(np.array([0.0, 1.0]), ["SPY"]) == {"SPY": 1.0}


def test_weekly_sample_ends_on_last_session():
    dates = [dt.date(2026, 10, d) for d in (5, 6, 7, 8, 9, 12, 13)]
    assert build.weekly_sample(dates) == [4, 6]


def test_today_entry_drops_expired_locks_and_flags_trading():
    dates = [dt.date(2026, 10, 8), dt.date(2026, 10, 9)]
    p = _path([1.0, 1.01], dates)
    p.locks = {"SPY": dt.date(2026, 10, 9)}
    p.rebalances = [build.Rebalance(dates[-1], [build.Change("SPY", 0.5, 1.0)])]
    e = build.today_entry(p)
    assert e["locks"] == {} and e["traded_today"] is True
    p.risk_state = "panic"
    with pytest.raises(ValueError):
        build.today_entry(p)
