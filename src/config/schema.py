"""Typed config schemas.

One dataclass per config file. `strict_from_dict` rejects unknown keys, so a typo in a
key name fails at load time rather than silently doing nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# --------------------------------------------------------------- config/universe.yaml


@dataclass(frozen=True)
class SymbolSpec:
    kind: Literal["level", "price"]
    role: str
    scale: float = 1.0
    #: Hard sanity band on the level. None -> quality.default_level_range.
    range: list[float] | None = None


@dataclass(frozen=True)
class FredSpec:
    freq: Literal["daily", "monthly"]
    lag_days: int
    kind: Literal["level", "price"]
    role: str


@dataclass(frozen=True)
class FetchSpec:
    batch_size: int = 20
    batch_pause_seconds: float = 2.0
    retries: int = 3
    backoff_seconds: float = 5.0
    incremental_days: int = 90


@dataclass(frozen=True)
class QualitySpec:
    max_forward_fill_sessions: int = 1
    max_macro_forward_fill_sessions: int = 10
    max_abs_daily_return: float = 0.25
    max_abs_daily_return_volatile: float = 0.40
    volatile_tickers: list[str] = field(default_factory=list)
    split_jump_threshold: float = 0.40
    default_level_range: list[float] = field(default_factory=lambda: [-1.0, 20.0])
    spy_vti_min_correlation: float = 0.95
    spy_vti_correlation_window: int = 252
    require_positive_volume: bool = True
    macro_staleness_buffer_days: int = 10


@dataclass(frozen=True)
class KnownException:
    ticker: str
    check: str
    reason: str
    sessions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UniverseConfig:
    calendar: str
    history_start: str
    study_start: str
    tradable: dict[str, list[str]]
    synthetic_asset: str
    feature_only: dict[str, SymbolSpec]
    fred: dict[str, FredSpec]
    inception_pins: dict[str, str]
    fetch: FetchSpec = field(default_factory=FetchSpec)
    quality: QualitySpec = field(default_factory=QualitySpec)
    inception_tolerance_sessions: int = 3
    known_exceptions: list[KnownException] = field(default_factory=list)

    # -- derived views; the canonical ticker order lives here and never changes ------

    @property
    def tradable_tickers(self) -> list[str]:
        """Canonical order: group order as declared, tickers in declared order.

        Appending a ticker is a breaking change that invalidates trained policies
        (reference/env-mdp.md section 1).
        """
        out: list[str] = []
        for group in self.tradable.values():
            out.extend(group)
        return out

    @property
    def group_of(self) -> dict[str, str]:
        return {t: g for g, tickers in self.tradable.items() for t in tickers}

    @property
    def feature_symbols(self) -> list[str]:
        return list(self.feature_only)

    @property
    def all_yfinance_symbols(self) -> list[str]:
        return self.tradable_tickers + self.feature_symbols

    @property
    def price_symbols(self) -> set[str]:
        """Feature-only symbols that ARE prices, so return-based checks apply to them."""
        return {s for s, spec in self.feature_only.items() if spec.kind == "price"}

    @property
    def level_symbols(self) -> set[str]:
        """Symbols whose values are levels. Computing a return on one of these is a bug."""
        return {s for s, spec in self.feature_only.items() if spec.kind == "level"}


# --------------------------------------------------------------- config/features.yaml


@dataclass(frozen=True)
class ScalingSpec:
    method: Literal["robust", "standard", "none"] = "robust"
    clip: float = 10.0
    # Already in [0,1] by construction: percentile ranks, %B, masks.
    leave_bounded_unscaled: bool = True


@dataclass(frozen=True)
class WindowSpec:
    returns: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 21, 63, 126, 252])
    volatility: list[int] = field(default_factory=lambda: [5, 10, 21, 63, 126, 252])
    drawdown: list[int] = field(default_factory=lambda: [21, 63, 252])
    volume: list[int] = field(default_factory=lambda: [21, 63])
    trend: list[int] = field(default_factory=lambda: [20, 50, 100, 200])


@dataclass(frozen=True)
class TechnicalSpec:
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0


@dataclass(frozen=True)
class FoldSpec:
    """Expanding annual walk-forward folds (reference/evaluation.md section 1)."""

    first_test_year: int = 2012
    last_test_year: int = 2025


@dataclass(frozen=True)
class FeaturesConfig:
    universe_config: str
    windows: WindowSpec = field(default_factory=WindowSpec)
    technical: TechnicalSpec = field(default_factory=TechnicalSpec)
    scaling: ScalingSpec = field(default_factory=ScalingSpec)
    folds: FoldSpec = field(default_factory=FoldSpec)
    epsilon: float = 1e-12
    correlation_report_threshold: float = 0.95
