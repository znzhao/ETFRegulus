"""I6 -- the lookahead invariant.

Two forms, and the second is the one that finds real bugs:

1. **Truncation.** `features(full, D) == features(truncate(full, D), D)`.
2. **Future mutation.** Randomly perturb the data strictly after `D`, recompute, and
   assert nothing at or before `D` changed by even a float.

Truncation catches a feature that reads ahead. Mutation also catches one that reads ahead
*and* happens to be insensitive to the truncation boundary -- a centered rolling window
with a short tail, say. Both run over several random dates.

Coverage is deliberately wide: per-ETF features, cross-sectional ranks, and the macro
block including publication lag. The risk envelope's stress estimators are a lookahead
surface too and are the one most likely to be forgotten; they are tested in Stage 4,
where they exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.cross_sectional import build_cross_sectional
from src.features.etf import build_etf_features

CUT_DATES = [0.35, 0.5, 0.65, 0.8, 0.92]


def _cut(index: pd.DatetimeIndex, frac: float) -> pd.Timestamp:
    return index[int(len(index) * frac)]


def _assert_identical_up_to(a: pd.DataFrame, b: pd.DataFrame, cut: pd.Timestamp) -> None:
    left = a.loc[a.index <= cut]
    right = b.loc[b.index <= cut]
    assert list(left.columns) == list(right.columns)
    for col in left.columns:
        x, y = left[col].to_numpy(float), right[col].to_numpy(float)
        both_nan = np.isnan(x) & np.isnan(y)
        assert np.array_equal(np.isnan(x), np.isnan(y)), f"{col}: NaN pattern moved"
        assert np.allclose(x[~both_nan], y[~both_nan], rtol=0, atol=0), (
            f"{col}: a value at or before {cut.date()} changed"
        )


# ------------------------------------------------------------------ per-ETF block


@pytest.mark.parametrize("frac", CUT_DATES)
def test_etf_features_survive_truncation(synthetic_prices, features_cfg, frac):
    cut = _cut(synthetic_prices.index, frac)
    full = build_etf_features(synthetic_prices, features_cfg)
    partial = build_etf_features(synthetic_prices.loc[:cut], features_cfg)
    _assert_identical_up_to(full, partial, cut)


@pytest.mark.parametrize("frac", CUT_DATES)
def test_etf_features_survive_future_mutation(synthetic_prices, features_cfg, frac):
    """The stronger form: scramble the future, and the past must not move by a float."""
    cut = _cut(synthetic_prices.index, frac)
    rng = np.random.default_rng(int(frac * 1000))

    mutated = synthetic_prices.copy()
    after = mutated.index > cut
    for col in ("close_adj", "close_raw", "open_raw", "high_raw", "low_raw", "volume"):
        noise = rng.uniform(0.5, 2.0, int(after.sum()))
        mutated.loc[after, col] = mutated.loc[after, col].to_numpy() * noise
    # Keep the mutated bars internally consistent so the failure is about lookahead,
    # not about a nonsensical OHLC ordering.
    mutated.loc[after, "high_raw"] = mutated.loc[after, ["high_raw", "close_raw", "open_raw"]].max(axis=1)
    mutated.loc[after, "low_raw"] = mutated.loc[after, ["low_raw", "close_raw", "open_raw"]].min(axis=1)

    full = build_etf_features(synthetic_prices, features_cfg)
    perturbed = build_etf_features(mutated, features_cfg)
    _assert_identical_up_to(full, perturbed, cut)


def test_the_lookahead_detector_actually_fires(synthetic_prices, features_cfg):
    """A leak is injected deliberately, because a detector that has never fired is not
    known to work."""
    cut = _cut(synthetic_prices.index, 0.5)
    full = build_etf_features(synthetic_prices, features_cfg)

    leaky = full.copy()
    leaky["ret_1"] = synthetic_prices["close_adj"].pct_change().shift(-1)  # tomorrow's return

    mutated = synthetic_prices.copy()
    after = mutated.index > cut
    mutated.loc[after, "close_adj"] *= 1.5
    leaky_mutated = build_etf_features(mutated, features_cfg)
    leaky_mutated["ret_1"] = mutated["close_adj"].pct_change().shift(-1)

    with pytest.raises(AssertionError):
        _assert_identical_up_to(leaky, leaky_mutated, cut)


# ------------------------------------------------------------ cross-sectional block


@pytest.mark.parametrize("frac", [0.4, 0.7])
def test_cross_sectional_ranks_survive_future_mutation(price_slice, features_cfg, frac):
    """Ranks are computed per session across assets, so a leak here would be a
    cross-asset one -- a different failure mode from the per-ETF block."""
    tickers = ["SPY", "TLT", "HYG", "GLD"]
    sess = price_slice.index.get_level_values("session").unique().sort_values()
    cut = _cut(sess, frac)
    rng = np.random.default_rng(7)

    def build(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for t in tickers:
            g = frame.xs(t, level="ticker").sort_index()
            f = build_etf_features(g, features_cfg)
            f["ticker"] = t
            rows.append(f.reset_index().set_index(["session", "ticker"]))
        etf = pd.concat(rows).sort_index()
        return build_cross_sectional(etf, {t: "g" for t in tickers})

    mutated = price_slice.copy()
    after = mutated.index.get_level_values("session") > cut
    mutated.loc[after, "close_adj"] = (
        mutated.loc[after, "close_adj"].to_numpy() * rng.uniform(0.5, 2.0, int(after.sum()))
    )

    base = build(price_slice).groupby(level="session").mean()
    perturbed = build(mutated).groupby(level="session").mean()
    _assert_identical_up_to(base, perturbed, cut)


# ---------------------------------------------------------------------- macro lag


def test_a_monthly_series_is_not_visible_before_its_publication_lag(universe_cfg,
                                                                    macro_slice):
    """The lookahead that flatters every regime-detection claim in the project.

    FRED stamps an observation with the period it describes, not its release date. The
    curated series must carry the configured lag, so a value cannot appear on a session
    earlier than `observation date + lag_days`.
    """
    import datetime as dt

    from src.data.calendar import sessions
    from src.data.curate import curate_macro

    raw = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(["2019-01-01", "2019-02-01", "2019-03-01"], name="date"),
    )
    idx = sessions("2019-01-01", "2019-12-31")
    macro, violations = curate_macro(universe_cfg, {"UNRATE": raw}, idx)

    lag = universe_cfg.fred["UNRATE"].lag_days
    for obs_date, value in zip(raw.index, raw["value"]):
        visible_from = obs_date + dt.timedelta(days=lag)
        early = macro.loc[macro.index < visible_from, "UNRATE"]
        assert not (early == value).any(), (
            f"the {obs_date.date()} observation was visible before "
            f"{visible_from.date()} (lag {lag}d)"
        )
    del violations


def test_the_real_curated_macro_respects_every_configured_lag(universe_cfg, macro_slice):
    """Same property, against the shipped artifact rather than a constructed one."""
    import datetime as dt
    from pathlib import Path

    raw_dir = Path("data/raw/fred")
    checked = 0
    for sid, spec in universe_cfg.fred.items():
        path = raw_dir / f"{sid}.parquet"
        if not path.exists() or sid not in macro_slice.columns:
            continue
        raw = pd.read_parquet(path)["value"].dropna()
        aligned = macro_slice[sid].dropna()
        if aligned.empty:
            continue
        # The first session carrying any value must be at least `lag` days after the
        # observation it came from.
        first_session = aligned.index[0]
        candidates = raw.index[raw.index <= first_session - dt.timedelta(days=spec.lag_days)]
        assert len(candidates), (
            f"{sid}: a value appears on {first_session.date()} with no observation old "
            f"enough to have been published by then (lag {spec.lag_days}d)"
        )
        checked += 1
    assert checked, "no FRED series were available to check"
