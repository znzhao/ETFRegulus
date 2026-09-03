"""Per-fold feature scalers.

**No full-sample scaling.** It is explicitly prohibited: it is lookahead that contaminates
every walk-forward fold at once, invisibly. A scaler is fitted on a fold's *training
window only*, saved with the range it was fitted over, and T10 asserts that range never
overlaps the fold's evaluation period.

Robust (median / IQR) by default: daily financial features have fat tails that mean/std
handles badly, and one crisis observation should not set the scale for a whole decade.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.schema import ScalingSpec

SCALERS_DIR = Path("data/features/scalers")

#: Features already living in [0, 1] by construction: percentile ranks, %B, masks.
#: Scaling them would destroy the property that makes them comparable across time.
BOUNDED_PREFIXES = ("rank_", "close_location", "bollinger_pct_b", "is_available")


def is_bounded(name: str) -> bool:
    return name.startswith(BOUNDED_PREFIXES)


@dataclass
class Scaler:
    """Fitted scaling statistics for one fold."""

    fold_id: str
    method: str
    clip: float
    fit_start: str
    fit_end: str
    center: dict[str, float] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    bounded: list[str] = field(default_factory=list)
    n_rows_fitted: int = 0

    # ------------------------------------------------------------------ fit/apply

    @classmethod
    def fit(cls, df: pd.DataFrame, *, fold_id: str, fit_start: str, fit_end: str,
            spec: ScalingSpec) -> "Scaler":
        """Fit on `df`, which MUST already be restricted to the training window."""
        sc = cls(fold_id=fold_id, method=spec.method, clip=spec.clip,
                 fit_start=fit_start, fit_end=fit_end, n_rows_fitted=int(len(df)))
        for col in df.columns:
            if spec.leave_bounded_unscaled and is_bounded(col):
                sc.bounded.append(col)
                continue
            s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                sc.center[col], sc.scale[col] = 0.0, 1.0
                continue
            if spec.method == "robust":
                center = float(s.median())
                iqr = float(s.quantile(0.75) - s.quantile(0.25))
                scale = iqr if iqr > 1e-12 else float(s.std() or 1.0)
            elif spec.method == "standard":
                center, scale = float(s.mean()), float(s.std() or 1.0)
            else:
                center, scale = 0.0, 1.0
            # A constant feature in the training window has no scale. Leaving it at 1.0
            # keeps it constant rather than producing inf.
            sc.center[col] = center
            sc.scale[col] = scale if abs(scale) > 1e-12 else 1.0
        return sc

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if col in self.bounded or col not in self.center:
                continue
            out[col] = (out[col] - self.center[col]) / self.scale[col]
        if self.clip:
            cols = [c for c in out.columns if c not in self.bounded]
            out[cols] = out[cols].clip(-self.clip, self.clip)
        return out

    def clip_report(self, df: pd.DataFrame) -> dict[str, int]:
        """How many values each feature had clipped. Counted, never silent."""
        scaled = self.transform(df)
        report = {}
        for col in scaled.columns:
            if col in self.bounded or col not in self.center:
                continue
            raw = (df[col] - self.center[col]) / self.scale[col]
            n = int((raw.abs() > self.clip).sum())
            if n:
                report[col] = n
        return report

    # ------------------------------------------------------------------ storage

    def save(self, directory: Path = SCALERS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.fold_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, fold_id: str, directory: Path = SCALERS_DIR) -> "Scaler":
        path = directory / f"{fold_id}.json"
        return cls(**json.loads(path.read_text(encoding="utf-8")))
