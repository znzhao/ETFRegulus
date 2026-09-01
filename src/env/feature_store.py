"""Stage 3's feature frames, selected, scaled and pre-pivoted into dense arrays.

The environment steps once per session. Doing a pandas lookup per step would make the
feature read dominate the step cost and would put the throughput benchmark in Stage 6 at
the mercy of pandas indexing rather than of the actual simulator. So everything is
materialized once, at construction, into two arrays:

    per_asset  (T, K, F)   float32, canonical ticker order
    globals    (T, G)      float32

and a step is then two slices.

**Scaling is fold-scoped.** A `FeatureStore` is built for one fold and carries that fold's
scaler, fitted on that fold's training window only. There is no code path that fits a
scaler on the data being observed -- `Scaler.load` reads statistics that Stage 3 already
committed to disk, and `fit_range` is carried through to the run manifest so T10 can check
it against the evaluation window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.env.observation import ObservationSpec
from src.features.scalers import SCALERS_DIR, Scaler

FEATURES_DIR = Path("data/features")


@dataclass
class FeatureStore:
    sessions: pd.DatetimeIndex
    tickers: tuple[str, ...]
    per_asset: np.ndarray      # (T, K, F)
    globals: np.ndarray        # (T, G)
    fold_id: str | None
    fit_range: tuple[str, str] | None
    n_filled: int = 0

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    def at(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.per_asset[i], self.globals[i]

    def index_of_session(self, session) -> int:
        """Position of a session, or an error naming the range -- an off-by-one between
        the price calendar and the feature calendar must not become a silent shift."""
        loc = self.sessions.get_indexer([pd.Timestamp(session)])[0]
        if loc < 0:
            raise KeyError(
                f"session {session} is not in the feature store "
                f"({self.sessions[0].date()}..{self.sessions[-1].date()})"
            )
        return int(loc)


def load_fold(fold_id: str, folds_path: Path = FEATURES_DIR / "folds.json") -> dict:
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    for f in folds:
        if f["fold_id"] == fold_id:
            return f
    raise KeyError(f"unknown fold {fold_id!r}; have {[f['fold_id'] for f in folds]}")


def build_store(
    spec: ObservationSpec,
    sessions: Sequence[pd.Timestamp],
    *,
    fold_id: str | None = None,
    features_dir: Path = FEATURES_DIR,
    scalers_dir: Path = SCALERS_DIR,
) -> FeatureStore:
    """Materialize the selected features over `sessions`, in canonical ticker order.

    `sessions` is the *price* calendar the simulator runs on; the feature frames are
    reindexed onto it, so the two can never drift apart by a session.
    """
    index = pd.DatetimeIndex(sessions)
    tickers = list(spec.tickers)

    etf = pd.read_parquet(features_dir / "etf.parquet",
                          columns=list(spec.per_asset_etf) or None)
    cs_cols = sorted(set(spec.per_asset_cross_sectional) | set(spec.global_cross_sectional))
    cross = pd.read_parquet(features_dir / "cross_sectional.parquet",
                            columns=cs_cols or None)
    macro = pd.read_parquet(features_dir / "macro.parquet",
                            columns=list(spec.macro) or None)

    scaler = Scaler.load(fold_id, scalers_dir) if fold_id else None
    fit_range = (scaler.fit_start, scaler.fit_end) if scaler else None
    if scaler is not None:
        etf = scaler.transform(etf)
        cross = scaler.transform(cross)
        macro = scaler.transform(macro)

    # ---- per-asset block ------------------------------------------------------
    n_filled = 0
    blocks: list[np.ndarray] = []
    for source, cols in ((etf, spec.per_asset_etf),
                         (cross, spec.per_asset_cross_sectional)):
        for col in cols:
            wide = (source[col].unstack("ticker")
                    .reindex(columns=tickers).reindex(index))
            arr = wide.to_numpy(dtype=np.float32)
            n_filled += int(np.isnan(arr).sum())
            blocks.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
    per_asset = (np.stack(blocks, axis=-1) if blocks
                 else np.zeros((len(index), len(tickers), 0), dtype=np.float32))

    # ---- global block ---------------------------------------------------------
    # `universe_size` and friends are stored per (session, ticker) but are per-session
    # scalars; take the cross-sectional first value rather than repeating them K times.
    g_cols: list[np.ndarray] = []
    for col in spec.macro:
        s = macro[col].reindex(index)
        g_cols.append(s.to_numpy(dtype=np.float32))
    for col in spec.global_cross_sectional:
        s = cross[col].groupby(level="session").first().reindex(index)
        g_cols.append(s.to_numpy(dtype=np.float32))
    if g_cols:
        stacked = np.stack(g_cols, axis=-1)
        n_filled += int(np.isnan(stacked).sum())
        globals_ = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        globals_ = np.zeros((len(index), 0), dtype=np.float32)

    return FeatureStore(
        sessions=index, tickers=tuple(tickers),
        per_asset=np.ascontiguousarray(per_asset, dtype=np.float32),
        globals=np.ascontiguousarray(globals_, dtype=np.float32),
        fold_id=fold_id, fit_range=fit_range, n_filled=n_filled,
    )
