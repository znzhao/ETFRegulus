"""Phase 3: the stress gate, the bootstrap, and the adversarial construction.

Three properties carry the weight here, and each is easy to get subtly wrong in a way that
produces comfortable numbers:

* **`D_max` monotonicity** is the single most important robustness check in the project. A
  tighter ceiling producing a deeper realized drawdown means the safety layer is not doing
  what it claims. The check must therefore *fire*, so it is handed a violation.
* **The bootstrap must not be IID.** An independent resample destroys the volatility
  clustering the whole risk layer exists to handle, and produces comfortable, meaningless
  intervals. The tests assert the block structure actually survives.
* **The adversarial splice must not invent returns.** A price series stitched from disjoint
  windows manufactures an enormous fake return at every seam. The test reconstructs the
  spliced returns and checks every one against the real history it came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.bootstrap import (
    bootstrap,
    metrics_from_returns,
    moving_block_indices,
    resample_indices,
    stationary_block_indices,
)
from src.evaluation.stress import (
    N_OUT_OF_DISTRIBUTION,
    N_SWEEP,
    Cell,
    check_monotonicity,
    is_ood,
    lock_is_binding,
)


# ------------------------------------------------------- the monotonicity gate


def cell(n: int, d: float, drawdown: float, **extra) -> Cell:
    metrics = {"max_drawdown": drawdown, "turnover": 10.0,
               "locked_nav_fraction": 0.3, **extra}
    return Cell(hold_days=n, max_drawdown=d, window="grid", metrics=metrics,
                out_of_distribution=is_ood(n))


def test_a_monotone_grid_passes():
    """A tighter ceiling produces a shallower realized drawdown, as it must."""
    cells = [cell(30, d, dd) for d, dd in
             ((0.05, 0.04), (0.10, 0.09), (0.15, 0.14), (0.20, 0.19), (0.25, 0.24))]
    result = check_monotonicity(cells)
    assert result.passed
    assert "monotone non-decreasing" in result.explain()


def test_the_gate_fires_when_a_tighter_ceiling_is_worse():
    """The check that has to work. A violation here voids every risk number."""
    cells = [cell(30, 0.05, 0.25), cell(30, 0.25, 0.10)]
    result = check_monotonicity(cells, tolerance=0.02)
    assert not result.passed
    assert "MONOTONICITY VIOLATED" in result.explain()
    assert "every risk number in the report is void" in result.explain()
    v = result.violations[0]
    assert v["tighter_d_max"] == 0.05 and v["looser_d_max"] == 0.25
    assert v["excess"] == pytest.approx(0.15)


def test_a_violation_inside_tolerance_is_allowed():
    """A single seed is stochastic; the tolerance is stated rather than tuned."""
    cells = [cell(30, 0.05, 0.111), cell(30, 0.10, 0.10)]
    assert check_monotonicity(cells, tolerance=0.02).passed
    assert not check_monotonicity(cells, tolerance=0.005).passed


def test_monotonicity_is_checked_within_each_N_never_across_it():
    """Comparing N=15 to N=60 would confound the ceiling with the lock -- and the lock
    genuinely can deepen a drawdown by preventing a sale."""
    cells = [cell(15, 0.05, 0.04), cell(15, 0.25, 0.20),
             cell(60, 0.05, 0.10), cell(60, 0.25, 0.30)]
    result = check_monotonicity(cells)
    assert result.passed, "a cross-N comparison leaked into the check"
    assert result.n_hold_days == 2


def test_the_out_of_distribution_labels_match_the_spec():
    assert N_OUT_OF_DISTRIBUTION == {0, 7, 90, 180}
    assert set(N_SWEEP) >= {15, 21, 30, 42, 60}
    assert all(is_ood(n) for n in (0, 7, 90, 180))
    assert not any(is_ood(n) for n in (15, 21, 30, 42, 60))


def test_an_inert_lock_is_detectable():
    """robustness.md 1.2: if performance is flat in N the lock may not be binding, and a
    silently-inert constraint looks exactly like this."""
    flat = [cell(n, 0.15, 0.10, turnover=100.0, locked_nav_fraction=0.0)
            for n in (15, 30, 60)]
    assert lock_is_binding(flat)["binding"] is False

    binding = [cell(15, 0.15, 0.10, turnover=100.0, locked_nav_fraction=0.10),
               cell(30, 0.15, 0.10, turnover=40.0, locked_nav_fraction=0.30),
               cell(60, 0.15, 0.10, turnover=5.0, locked_nav_fraction=0.55)]
    out = lock_is_binding(binding)
    assert out["binding"] is True
    assert out["turnover_ratio_low_to_high_N"] == pytest.approx(20.0)


# ------------------------------------------------------------- the bootstrap


def test_iid_resampling_is_refused_by_name():
    """Not an oversight: an IID bootstrap destroys volatility clustering and produces
    comfortable, meaningless intervals."""
    with pytest.raises(ValueError, match="volatility clustering"):
        resample_indices(100, method="iid", mean_block=10,
                         rng=np.random.default_rng(0))


def test_the_stationary_bootstrap_returns_a_full_length_index():
    rng = np.random.default_rng(0)
    idx = stationary_block_indices(500, 10, rng)
    assert idx.size == 500
    assert idx.min() >= 0 and idx.max() < 500


def test_blocks_actually_preserve_serial_structure():
    """The property the whole method exists for. A block resample of a strongly
    autocorrelated series must retain far more of that autocorrelation than an IID one."""
    rng = np.random.default_rng(0)
    n = 4000
    series = np.zeros(n)
    for i in range(1, n):                       # AR(1), rho = 0.9
        series[i] = 0.9 * series[i - 1] + rng.normal(scale=0.01)

    def rho(x):
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    block = series[stationary_block_indices(n, 50, np.random.default_rng(1))]
    iid = series[np.random.default_rng(1).integers(0, n, size=n)]
    assert rho(block) > 0.5, "the block bootstrap lost the serial dependence"
    assert abs(rho(iid)) < 0.1, "the IID control should destroy it"


def test_the_moving_block_bootstrap_uses_contiguous_runs():
    idx = moving_block_indices(200, 20, np.random.default_rng(0))
    steps = np.diff(idx)
    # Within a block the index advances by exactly one; only seams differ.
    assert (steps == 1).mean() > 0.85


def test_bands_are_produced_for_every_standard_metric():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0004, 0.01, size=1500)
    result = bootstrap(returns, replicates=60, mean_block=10, seed=1)
    for key in ("annualized_return", "volatility", "sharpe", "max_drawdown",
                "worst_1d", "worst_5d", "calmar"):
        band = result.bands[key]
        assert band["q05"] <= band["q50"] <= band["q95"], key
    assert result.n_sessions == 1500


def test_the_bootstrap_is_deterministic_given_a_seed():
    returns = np.random.default_rng(3).normal(0.0004, 0.01, size=800)
    a = bootstrap(returns, replicates=40, seed=7).bands["max_drawdown"]
    b = bootstrap(returns, replicates=40, seed=7).bands["max_drawdown"]
    assert a == b


def test_metrics_on_a_hand_computable_path():
    returns = np.array([0.10, -0.20, 0.25])
    m = metrics_from_returns(returns)
    assert m["total_return"] == pytest.approx(1.10 * 0.80 * 1.25 - 1.0)
    # Peak 1.10, trough 0.88 -> 20%.
    assert m["max_drawdown"] == pytest.approx(0.20)
    assert m["worst_1d"] == pytest.approx(-0.20)


# --------------------------------------------------------- the adversarial splice


@pytest.fixture(scope="module")
def market():
    from pathlib import Path

    from src.config.loader import resolve_config
    from src.sim.runner import load_market

    if not Path("data/curated/prices.parquet").exists():
        pytest.skip("run Stage 2 first")
    mkt, _ = load_market(resolve_config(Path("config/evaluation.yaml")))
    return mkt


def test_the_splice_invents_no_returns(market):
    """The property that makes an adversarial path admissible evidence: every return in it
    is one that actually happened. A price series stitched from disjoint windows would
    manufacture a huge fake return at each seam."""
    from src.evaluation.adversarial import splice

    rows = np.concatenate([np.arange(500, 521), np.arange(3000, 3021),
                           np.arange(1200, 1221)])
    spliced = splice(market, rows)

    j = market.universe.index("SPY")
    got = spliced.close_raw[1:, j] / spliced.close_raw[:-1, j]
    want = []
    for r in rows[1:]:
        want.append(market.close_raw[r, j] / market.close_raw[r - 1, j])
    assert np.allclose(got, want, rtol=1e-9), (
        "a spliced return does not match the real return from its source session")


def test_the_splice_preserves_ohlc_relationships(market):
    from src.evaluation.adversarial import splice

    rows = np.concatenate([np.arange(800, 821), np.arange(2500, 2521)])
    s = splice(market, rows)
    ok = np.isfinite(s.low_raw) & np.isfinite(s.high_raw) & np.isfinite(s.close_raw)
    assert (s.low_raw[ok] <= s.high_raw[ok] + 1e-9).all()
    assert (s.close_raw[ok] >= s.low_raw[ok] - 1e-9).all()
    assert (s.close_raw[ok] <= s.high_raw[ok] + 1e-9).all()
    assert (s.close_raw[ok] > 0).all()


def test_the_spliced_calendar_is_strictly_increasing(market):
    """The lock counts CALENDAR days, so a non-monotone calendar would corrupt it."""
    from src.evaluation.adversarial import splice

    rows = np.concatenate([np.arange(400, 421), np.arange(2000, 2021)])
    s = splice(market, rows)
    assert s.sessions.is_monotonic_increasing
    assert s.sessions.is_unique
    assert len(s.sessions) == rows.size


def test_the_scenarios_select_real_disjoint_blocks(market):
    from src.evaluation.adversarial import build_scenarios

    scenarios = build_scenarios(market, block=21, n_blocks=4)
    assert {s.name for s in scenarios} == {
        "equity_shock_credit_widening", "duration_loss",
        "correlation_spike", "diversification_breakdown"}
    for scenario in scenarios:
        assert scenario.blocks, scenario.name
        for b in scenario.blocks:
            assert b.length == 21
            assert b.reason
        rows = scenario.rows()
        assert rows.size == len(set(rows.tolist())), (
            f"{scenario.name} reused a session; the blocks should be disjoint")


def test_the_diversification_scenario_finds_positive_stock_bond_correlation(market):
    """The 2022 failure mode: the diversifier stops diversifying."""
    from src.evaluation.adversarial import stock_bond_flip_blocks

    blocks = stock_bond_flip_blocks(market, block=21, n=3, reason="test")
    assert blocks
    spy = market.universe.index("SPY")
    tlt = market.universe.index("TLT")
    for b in blocks:
        a = market.returns[b.start_row:b.end_row + 1, spy + 1]
        c = market.returns[b.start_row:b.end_row + 1, tlt + 1]
        assert float(np.corrcoef(a, c)[0, 1]) > 0.0, (
            "a block selected for a stock/bond correlation flip is not positively "
            "correlated")


def test_the_disclaimer_is_not_optional():
    """robustness.md section 3: the disclaimer is part of the deliverable."""
    from src.evaluation.adversarial import DISCLAIMER

    assert "not a forward-looking worst-case guarantee" in DISCLAIMER.lower()
    assert "actually happened" in DISCLAIMER
    assert "not a probability statement about the future" in DISCLAIMER.lower()
