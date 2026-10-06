"""T11 -- walk-forward integrity: `train_end < val < test`, no overlap, no gaps.

Ordinary k-fold is prohibited here because it would train on the future. What replaces it
is an expanding-window annual split, and the value of that split depends entirely on the
three windows staying in order and staying disjoint. A fold that silently overlaps its
validation and test years does not error -- it just produces the best result in the table.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.evaluation.walk_forward import (
    Fold,
    WalkForwardError,
    assert_folds_valid,
    load_folds,
    scaler_is_legal,
    select_folds,
    validate_folds,
)


def make_fold(year: int = 2015, **kw) -> Fold:
    base = dict(
        fold_id=f"fold_{year}", test_year=year,
        train_start="2004-01-02", train_end=f"{year - 2}-12-31",
        val_start=f"{year - 1}-01-01", val_end=f"{year - 1}-12-31",
        test_start=f"{year}-01-01", test_end=f"{year}-12-31")
    base.update(kw)
    return Fold(**base)


def chain(years) -> list[Fold]:
    return [make_fold(y) for y in years]


# ---------------------------------------------------------------- the real folds


def test_the_projects_own_folds_are_valid():
    """The folds Stage 3 wrote. This is the assertion that actually protects the run."""
    folds = load_folds()
    assert validate_folds(folds) == []
    assert len(folds) >= 13


def test_the_real_folds_expand_rather_than_roll():
    """~20 years of data, and the crisis regimes in it are exactly what the risk layer
    needs to have seen -- so the training window must never drop its early years."""
    folds = load_folds()
    assert len({f.train_start for f in folds}) == 1
    ends = [f.train_end for f in folds]
    assert ends == sorted(ends)


# ------------------------------------------------------------------- the rules


def test_a_valid_chain_passes():
    assert validate_folds(chain([2012, 2013, 2014])) == []


def test_validation_overlapping_the_test_year_is_rejected():
    bad = make_fold(2015, val_end="2015-06-30")
    assert any("val_end" in p for p in validate_folds([bad]))


def test_training_running_into_the_validation_year_is_rejected():
    """The one that matters most: training on Y-1 while selecting on Y-1."""
    bad = make_fold(2015, train_end="2014-06-30")
    problems = validate_folds([bad])
    assert any("train_end" in p or "training ends" in p for p in problems)


def test_a_gap_between_windows_is_rejected():
    bad = make_fold(2015, val_start="2014-06-01", val_end="2014-12-31",
                    train_end="2013-12-31")
    # train_end 2013 -> val 2014 is contiguous by year, but the gap check is on days.
    bad = dataclasses.replace(bad, train_end="2013-01-01")
    assert any("gap" in p for p in validate_folds([bad]))


def test_a_rolling_window_is_rejected():
    folds = [make_fold(2012), dataclasses.replace(make_fold(2013),
                                                  train_start="2006-01-02")]
    assert any("expands rather than rolls" in p for p in validate_folds(folds))


def test_a_training_window_that_does_not_grow_is_rejected():
    folds = [make_fold(2012), dataclasses.replace(make_fold(2013),
                                                  train_end="2010-12-31")]
    assert any("did not expand" in p for p in validate_folds(folds))


def test_duplicate_and_out_of_order_folds_are_rejected():
    assert any("duplicate" in p for p in validate_folds(chain([2012, 2012])))
    assert any("chronological" in p for p in validate_folds(chain([2014, 2013])))


def test_assert_folds_valid_raises_with_every_problem_named():
    with pytest.raises(WalkForwardError) as exc:
        assert_folds_valid([make_fold(2015, val_end="2015-06-30")])
    assert "val_end" in str(exc.value)


# ------------------------------------------------------------- the scaler rule


def test_a_scaler_fitted_past_the_training_window_is_a_leak():
    """Invisible in results: a contaminated scaler makes every fold look slightly better
    and nothing anywhere reports an error."""
    fold = make_fold(2015)                      # trains to 2013-12-31
    assert scaler_is_legal("2004-01-02", "2013-12-31", fold) == []
    problems = scaler_is_legal("2004-01-02", "2014-06-30", fold)
    assert problems and "past the training window" in problems[0]


def test_a_scaler_fitted_before_the_training_window_is_rejected():
    fold = make_fold(2015)
    assert scaler_is_legal("2001-01-02", "2013-12-31", fold)


def test_the_real_scalers_are_legal_for_their_own_folds():
    from src.features.scalers import Scaler

    for fold in load_folds()[:4]:
        try:
            scaler = Scaler.load(fold.fold_id)
        except FileNotFoundError:
            pytest.skip("run Stage 3 first")
        assert scaler_is_legal(scaler.fit_start, scaler.fit_end, fold) == [], (
            f"{fold.fold_id}: scaler fitted {scaler.fit_start}..{scaler.fit_end}")


# ---------------------------------------------------------------- fold picking


def test_selecting_folds_by_year():
    folds = chain([2012, 2013, 2014, 2015])
    assert [f.test_year for f in select_folds(folds, first_year=2013)] == [2013, 2014, 2015]
    assert [f.test_year for f in select_folds(folds, last_year=2013)] == [2012, 2013]
    assert [f.test_year for f in select_folds(folds, only=[2012, 2015])] == [2012, 2015]
