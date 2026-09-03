"""Walk-forward fold definitions.

Expanding annual windows (reference/evaluation.md section 1):

    Train [study_start .. Y-2]   Validate Y-1   Test Y

The window expands rather than rolls because there is only ~20 years of data, and the
crisis regimes in it -- 2008, 2020, 2022 -- are exactly what the risk layer needs to have
seen. Ordinary k-fold is prohibited: it would train on the future.

A fold's identity lives here and nowhere else, so the scaler artifact, the training run,
and the evaluation report cannot disagree about what "fold 2015" means.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    test_year: int

    @property
    def fold_id(self) -> str:
        return f"fold_{self.test_year}"

    def train_range(self, study_start: str) -> tuple[str, str]:
        return (study_start, f"{self.test_year - 2}-12-31")

    @property
    def val_range(self) -> tuple[str, str]:
        return (f"{self.test_year - 1}-01-01", f"{self.test_year - 1}-12-31")

    @property
    def test_range(self) -> tuple[str, str]:
        return (f"{self.test_year}-01-01", f"{self.test_year}-12-31")

    def as_dict(self, study_start: str) -> dict:
        tr = self.train_range(study_start)
        return {
            "fold_id": self.fold_id,
            "test_year": self.test_year,
            "train_start": tr[0], "train_end": tr[1],
            "val_start": self.val_range[0], "val_end": self.val_range[1],
            "test_start": self.test_range[0], "test_end": self.test_range[1],
        }


def build_folds(first_test_year: int, last_test_year: int,
                today: dt.date | None = None) -> list[Fold]:
    """Every fold whose test year has actually happened.

    A fold for a year that is still in progress would be evaluated on a partial year and
    silently compared against full ones.
    """
    today = today or dt.date.today()
    last_complete = today.year - 1
    end = min(last_test_year, last_complete)
    return [Fold(y) for y in range(first_test_year, end + 1)]
