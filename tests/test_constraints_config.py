"""The (N, D_max) parameter space (D16).

`config/constraints.yaml` is canonical; reference/decisions.md D16 is the rationale. These
tests pin the properties that make the grid mean what D16 says it means, so a later edit
that widens or re-centres it has to be deliberate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config.loader import ConfigError, load_typed, strict_from_dict
from src.config.schema import ConstraintsConfig, HoldDaysSpec


@pytest.fixture(scope="module")
def constraints() -> ConstraintsConfig:
    cfg, _ = load_typed("config/constraints.yaml", ConstraintsConfig)
    return cfg


def test_the_lock_is_centred_on_thirty_calendar_days(constraints):
    hd = constraints.lock.hold_days
    assert hd.primary == 30
    assert hd.primary in hd.values

    # Centred GEOMETRICALLY, because a holding period is a multiplicative quantity:
    # 30 -> 60 is the same size of change as 30 -> 15.
    weights = np.array(hd.weights)
    geometric_mean = float(np.exp(np.dot(np.log(hd.values), weights)))
    assert geometric_mean == pytest.approx(30.0, rel=0.02), (
        f"the weighted geometric mean is {geometric_mean:.2f}, not ~30 -- the grid is no "
        f"longer centred on the deployment value"
    )

    # And the mode is the primary value: most episodes run at the deployment lock.
    assert hd.values[int(np.argmax(weights))] == hd.primary


def test_the_operating_range_varies_but_stays_near_thirty(constraints):
    hd = constraints.lock.hold_days
    assert len(hd.values) >= 3, "a single value is not a parameter-conditioned policy"
    assert min(hd.values) >= 14, "below a fortnight is out of the intended operating range"
    assert max(hd.values) <= 63, "beyond ~two months is out of the intended operating range"
    assert hd.values == sorted(hd.values)
    assert len(set(hd.values)) == len(hd.values)


def test_the_ladder_steps_by_a_constant_factor(constraints):
    """A geometric ladder, so each step away from 30 is the same RELATIVE change.

    The factor itself is not pinned -- it was sqrt(2) at five rungs and is 2^(1/4) at
    nine -- because refining the grid is a legitimate change. What must hold is that the
    steps stay even: an evenly *spaced* grid over the same endpoints would put more
    resolution above 30 than below it while looking symmetric.
    """
    v = constraints.lock.hold_days.values
    ratios = [v[i + 1] / v[i] for i in range(len(v) - 1)]
    assert max(ratios) - min(ratios) < 0.1, f"uneven ladder: {ratios}"
    assert all(1.1 < r < 1.55 for r in ratios), f"not a geometric ladder: {ratios}"
    # Rounding to whole days is what makes the ratios inexact; the span must still be
    # the one the endpoints imply.
    implied = (v[-1] / v[0]) ** (1.0 / (len(v) - 1))
    assert min(ratios) <= implied <= max(ratios), (implied, ratios)


def test_stress_points_are_outside_the_operating_range(constraints):
    """A stress point inside the training distribution measures nothing."""
    hd = constraints.lock.hold_days
    assert not (set(hd.values) & set(hd.stress_values))
    assert 0 in hd.stress_values, "the no-lock control must remain available"
    assert 180 in hd.stress_values, "the long-lock extreme must remain available"


def test_weights_are_a_distribution(constraints):
    hd = constraints.lock.hold_days
    assert sum(hd.weights) == pytest.approx(1.0)
    assert all(w > 0 for w in hd.weights), "a zero-weight value is never sampled"


def test_max_drawdown_is_uniform_by_design(constraints):
    """Unlike N there is no expected deployment value: the caller picks their ceiling."""
    dd = constraints.drawdown.max_drawdown
    assert dd.weights is None
    assert dd.primary in dd.values
    assert dd.values == sorted(dd.values)


# ------------------------------------------------------------------ validation


def test_a_primary_outside_the_grid_is_rejected():
    with pytest.raises(ValueError, match="must appear in values"):
        HoldDaysSpec(primary=35, values=[15, 21, 30, 42, 60],
                     weights=[0.2] * 5, stress_values=[0])


def test_mismatched_weights_are_rejected():
    with pytest.raises(ValueError, match="weights has"):
        HoldDaysSpec(primary=30, values=[15, 30, 60], weights=[0.5, 0.5], stress_values=[0])


def test_weights_that_do_not_sum_to_one_are_rejected():
    with pytest.raises(ValueError, match="sum to"):
        HoldDaysSpec(primary=30, values=[15, 30, 60], weights=[0.5, 0.5, 0.5],
                     stress_values=[0])


def test_a_stress_point_inside_the_operating_range_is_rejected():
    with pytest.raises(ValueError, match="out-of-distribution"):
        HoldDaysSpec(primary=30, values=[15, 30, 60], weights=[0.3, 0.4, 0.3],
                     stress_values=[30])


def test_unknown_constraint_keys_are_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        strict_from_dict(ConstraintsConfig, {"lock": {"scope": "per_etf", "hold_dayz": {}}})


# ------------------------------------------------------------- doc consistency


def test_the_documents_quote_the_shipped_grid(constraints):
    """D16 and env-mdp.md must not drift from config/constraints.yaml."""
    values = str(constraints.lock.hold_days.values)          # "[15, 21, 30, 42, 60]"
    stress = str(constraints.lock.hold_days.stress_values)   # "[0, 7, 90, 180]"
    for doc in ("reference/decisions.md", "reference/env-mdp.md",
                "IMPLEMENTATION_PLAN.md"):
        text = Path(doc).read_text(encoding="utf-8")
        assert values in text, f"{doc} does not quote the operating range {values}"
    assert stress in Path("reference/decisions.md").read_text(encoding="utf-8")


def test_no_document_still_quotes_the_superseded_grid_as_current():
    """The pre-D16 grid may appear only where it is explicitly named as superseded.

    A stale grid left in a spec is worse than no grid: it reads as authoritative, and
    whoever implements against it will not know it was replaced.
    """
    superseded = "[0, 7, 14, 30, 60, 90, 180]"
    docs = list(Path("reference").glob("*.md")) + [Path("IMPLEMENTATION_PLAN.md")]
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.split("\n"), 1):
            if superseded not in line:
                continue
            context = "\n".join(text.split("\n")[max(0, line_no - 3):line_no + 2])
            assert "supersede" in context.lower(), (
                f"{doc}:{line_no} quotes the pre-D16 grid without marking it superseded"
            )


def test_crisis_windows_are_well_formed(constraints):
    import pandas as pd

    windows = constraints.risk.crisis_windows
    assert {"gfc", "covid", "rate_shock_22"} <= set(windows)
    for name, (start, end) in windows.items():
        assert pd.Timestamp(start) < pd.Timestamp(end), f"{name}: window runs backwards"


def test_v1_is_frictionless(constraints):
    """D10, settled and not revisited."""
    assert constraints.execution.cost_bps == 0.0
    assert constraints.lock.scope == "per_etf"     # D13
