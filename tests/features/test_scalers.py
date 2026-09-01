"""T10 -- scaler fold isolation, and the fold algebra it rests on.

Full-sample scaling is explicitly prohibited: it is lookahead that contaminates every
walk-forward fold at once, invisibly. The defence is that a scaler records the window it
was fitted over, and that window is asserted never to reach the fold's evaluation period.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.folds import Fold, build_folds
from src.features.scalers import SCALERS_DIR, Scaler, is_bounded

STUDY_START = "2004-01-02"


# ------------------------------------------------------------------ fold algebra


def test_folds_are_expanding_and_never_overlap():
    folds = build_folds(2012, 2020, today=pd.Timestamp("2026-01-01").date())
    assert [f.test_year for f in folds] == list(range(2012, 2021))

    for f in folds:
        train_start, train_end = f.train_range(STUDY_START)
        val_start, val_end = f.val_range
        test_start, _ = f.test_range
        assert pd.Timestamp(train_end) < pd.Timestamp(val_start), "train reaches into validation"
        assert pd.Timestamp(val_end) < pd.Timestamp(test_start), "validation reaches into test"
        assert pd.Timestamp(train_start) == pd.Timestamp(STUDY_START), "the window must expand, not roll"

    # Expanding: each fold's training window strictly contains the previous one's.
    ends = [pd.Timestamp(f.train_range(STUDY_START)[1]) for f in folds]
    assert ends == sorted(ends) and len(set(ends)) == len(ends)


def test_an_incomplete_test_year_is_excluded():
    """A fold for a year still in progress would be evaluated on a partial year and
    silently compared against full ones."""
    folds = build_folds(2012, 2030, today=pd.Timestamp("2026-09-01").date())
    assert max(f.test_year for f in folds) == 2025


# ---------------------------------------------------------------------- fit/apply


@pytest.fixture
def frame():
    rng = np.random.default_rng(3)
    n = 500
    return pd.DataFrame({
        "ret_21": rng.normal(0, 0.05, n),
        "vol_21": rng.gamma(2.0, 0.05, n),
        "rank_ret_21": rng.uniform(0, 1, n),      # bounded: must be left alone
        "constant": np.ones(n),                    # no scale: must not become inf
    })


def test_robust_scaling_centers_and_scales(frame, features_cfg):
    sc = Scaler.fit(frame, fold_id="fold_test", fit_start="a", fit_end="b",
                    spec=features_cfg.scaling)
    out = sc.transform(frame)
    assert abs(np.median(out["ret_21"])) < 1e-9, "the median was not centered"
    assert np.isfinite(out.to_numpy()).all(), "scaling produced a non-finite value"


def test_bounded_features_are_left_alone(frame, features_cfg):
    sc = Scaler.fit(frame, fold_id="fold_test", fit_start="a", fit_end="b",
                    spec=features_cfg.scaling)
    assert "rank_ret_21" in sc.bounded
    out = sc.transform(frame)
    pd.testing.assert_series_equal(out["rank_ret_21"], frame["rank_ret_21"])
    assert is_bounded("rank_ret_21") and not is_bounded("ret_21")


def test_a_constant_feature_does_not_become_infinite(frame, features_cfg):
    sc = Scaler.fit(frame, fold_id="fold_test", fit_start="a", fit_end="b",
                    spec=features_cfg.scaling)
    out = sc.transform(frame)
    assert np.isfinite(out["constant"]).all()


def test_clipping_is_counted_not_silent(frame, features_cfg):
    """A single crisis observation must not dominate a batch -- and the fact that it was
    clipped must be reported rather than disappearing."""
    sc = Scaler.fit(frame, fold_id="fold_test", fit_start="a", fit_end="b",
                    spec=features_cfg.scaling)
    extreme = frame.copy()
    extreme.loc[0, "ret_21"] = 1e6
    out = sc.transform(extreme)
    assert out["ret_21"].abs().max() <= sc.clip + 1e-12
    assert sc.clip_report(extreme).get("ret_21", 0) >= 1


def test_scaler_round_trips_through_disk(frame, features_cfg, tmp_path):
    sc = Scaler.fit(frame, fold_id="fold_2015", fit_start="2004-01-02",
                    fit_end="2013-12-31", spec=features_cfg.scaling)
    sc.save(tmp_path)
    back = Scaler.load("fold_2015", tmp_path)
    pd.testing.assert_frame_equal(sc.transform(frame), back.transform(frame))
    assert back.fit_end == "2013-12-31"


# ------------------------------------------------------------------------- T10


def test_a_fold_scaler_is_fitted_only_on_that_folds_training_window(frame, features_cfg):
    """The scaler must not have seen the evaluation data. Demonstrated by construction:
    statistics fitted on the training window differ from full-sample ones."""
    train = frame.iloc[:300]
    train_only = Scaler.fit(train, fold_id="f", fit_start="a", fit_end="b",
                            spec=features_cfg.scaling)
    full_sample = Scaler.fit(frame, fold_id="f", fit_start="a", fit_end="b",
                             spec=features_cfg.scaling)
    assert train_only.center["ret_21"] != full_sample.center["ret_21"], (
        "the training-window scaler is indistinguishable from a full-sample one; this "
        "test cannot detect the prohibited case"
    )
    assert train_only.n_rows_fitted == 300


@pytest.mark.skipif(not SCALERS_DIR.exists(), reason="run Stage 3 first")
def test_shipped_scalers_never_overlap_their_evaluation_window():
    """T10 against the real artifacts: a fitted range must not touch validation or test."""
    folds = {f["fold_id"]: f for f in
             json.loads(Path("data/features/folds.json").read_text(encoding="utf-8"))}
    assert folds, "Stage 3 wrote no folds"

    for fold_id, spec in folds.items():
        sc = Scaler.load(fold_id)
        assert sc.fit_start == spec["train_start"]
        assert sc.fit_end == spec["train_end"]
        assert pd.Timestamp(sc.fit_end) < pd.Timestamp(spec["val_start"]), (
            f"{fold_id}: the scaler's fitted range reaches into its validation year"
        )
        assert pd.Timestamp(sc.fit_end) < pd.Timestamp(spec["test_start"]), (
            f"{fold_id}: the scaler's fitted range reaches into its test year"
        )
        assert sc.n_rows_fitted > 0


@pytest.mark.skipif(not SCALERS_DIR.exists(), reason="run Stage 3 first")
def test_later_folds_are_fitted_on_strictly_more_data():
    """The expanding window, verified on the artifacts rather than the config."""
    folds = json.loads(Path("data/features/folds.json").read_text(encoding="utf-8"))
    counts = [Scaler.load(f["fold_id"]).n_rows_fitted for f in folds]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)
