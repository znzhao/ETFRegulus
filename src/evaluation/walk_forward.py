"""Expanding-window annual folds, and the integrity rules they must satisfy.

reference/evaluation.md section 1. The requirement is retraining every year, so the split
is by year and ordinary k-fold is prohibited outright -- it would train on the future.

    Train: [2004 .. Y-2]    Validate: Y-1    Test: Y

The window *expands* rather than rolls, because there is only ~20 years of data and the
crisis regimes in it (2008, 2020, 2022) are exactly what the risk layer needs to have seen.

`validate_folds` is T11. Three prohibitions are enforced here rather than by discipline:
no mid-year retraining on test-year data, no scaler fitted outside the training window, and
no gap or overlap between the three windows. The third sounds pedantic until a fold that
silently overlaps its validation and test years produces the best result in the table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FOLDS_PATH = Path("data/features/folds.json")


class WalkForwardError(ValueError):
    """A fold definition violates walk-forward integrity."""


@dataclass(frozen=True)
class Fold:
    fold_id: str
    test_year: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

    @property
    def train(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp(self.train_start), pd.Timestamp(self.train_end)

    @property
    def validation(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp(self.val_start), pd.Timestamp(self.val_end)

    @property
    def test(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp(self.test_start), pd.Timestamp(self.test_end)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def load_folds(path: Path = FOLDS_PATH) -> list[Fold]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run "
            "`python -m scripts.s03_build_features --config config/features.yaml`")
    return [Fold(**f) for f in json.loads(path.read_text(encoding="utf-8"))]


def validate_folds(folds: list[Fold]) -> list[str]:
    """T11. Return every integrity problem; an empty list means the folds are legal."""
    problems: list[str] = []
    for f in folds:
        train_start, train_end = f.train
        val_start, val_end = f.validation
        test_start, test_end = f.test

        # The ordering rule, stated as the document states it.
        if not train_end < val_start:
            problems.append(f"{f.fold_id}: train_end {f.train_end} is not before "
                            f"val_start {f.val_start}")
        if not val_end < test_start:
            problems.append(f"{f.fold_id}: val_end {f.val_end} is not before "
                            f"test_start {f.test_start}")
        if not train_start < train_end:
            problems.append(f"{f.fold_id}: empty training window")

        # No gap: a missing year between windows is data thrown away silently.
        if (val_start - train_end).days > 2:
            problems.append(f"{f.fold_id}: {(val_start - train_end).days}-day gap between "
                            f"train and validation")
        if (test_start - val_end).days > 2:
            problems.append(f"{f.fold_id}: {(test_start - val_end).days}-day gap between "
                            f"validation and test")

        # The windows must be the years they claim to be.
        if val_start.year != f.test_year - 1 or val_end.year != f.test_year - 1:
            problems.append(f"{f.fold_id}: validation window is not the year "
                            f"{f.test_year - 1}")
        if test_start.year != f.test_year or test_end.year != f.test_year:
            problems.append(f"{f.fold_id}: test window is not {f.test_year}")
        if train_end.year != f.test_year - 2:
            problems.append(f"{f.fold_id}: training ends {train_end.year}, expected "
                            f"{f.test_year - 2} (train [start..Y-2])")

    years = [f.test_year for f in folds]
    if len(set(years)) != len(years):
        problems.append(f"duplicate test years: {sorted(years)}")
    if years != sorted(years):
        problems.append("folds are not in chronological order")

    # The window must expand, never roll: the crisis regimes have to stay in it.
    for a, b in zip(folds, folds[1:]):
        if b.train_start != a.train_start:
            problems.append(
                f"{b.fold_id}: train_start moved to {b.train_start} from {a.train_start}. "
                "The window expands rather than rolls -- 2008, 2020 and 2022 are exactly "
                "what the risk layer needs to have seen.")
        if pd.Timestamp(b.train_end) <= pd.Timestamp(a.train_end):
            problems.append(f"{b.fold_id}: training window did not expand")
    return problems


def assert_folds_valid(folds: list[Fold]) -> None:
    problems = validate_folds(folds)
    if problems:
        raise WalkForwardError("walk-forward integrity violated:\n  "
                               + "\n  ".join(problems))


def scaler_is_legal(fit_start: str, fit_end: str, fold: Fold) -> list[str]:
    """A scaler fitted anywhere that overlaps validation or test is a leak.

    This is the second of the three prohibitions, and the one that is invisible in
    results: a contaminated scaler makes every fold look slightly better and nothing
    anywhere reports an error.
    """
    start, end = pd.Timestamp(fit_start), pd.Timestamp(fit_end)
    problems = []
    if end > fold.train[1]:
        problems.append(
            f"scaler fitted to {fit_end}, past the training window's end "
            f"{fold.train_end} -- it has seen the validation or test year")
    if start < fold.train[0]:
        problems.append(f"scaler fitted from {fit_start}, before the training window "
                        f"starts at {fold.train_start}")
    return problems


def select_folds(folds: list[Fold], *, first_year: int | None = None,
                 last_year: int | None = None,
                 only: list[int] | None = None) -> list[Fold]:
    out = folds
    if only:
        out = [f for f in out if f.test_year in set(only)]
    if first_year is not None:
        out = [f for f in out if f.test_year >= first_year]
    if last_year is not None:
        out = [f for f in out if f.test_year <= last_year]
    return out
