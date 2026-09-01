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


# ------------------------------------------------------------ config/constraints.yaml


@dataclass(frozen=True)
class HoldDaysSpec:
    """The holding-lock parameter space, in CALENDAR days (D16).

    Centred on `primary`; `values` is the operating range the policy is trained and
    evaluated over; `stress_values` sit deliberately outside it and are used only by the
    Stage 9 sensitivity sweep.
    """

    primary: int = 30
    values: list[int] = field(default_factory=lambda: [14, 21, 30, 45, 60])
    weights: list[float] | None = field(
        default_factory=lambda: [0.15, 0.20, 0.30, 0.20, 0.15])
    stress_values: list[int] = field(default_factory=lambda: [0, 7, 90, 180])

    def __post_init__(self) -> None:
        if self.primary not in self.values:
            raise ValueError(
                f"hold_days.primary ({self.primary}) must appear in values {self.values}"
            )
        if self.weights is not None and len(self.weights) != len(self.values):
            raise ValueError(
                f"hold_days.weights has {len(self.weights)} entries for "
                f"{len(self.values)} values"
            )
        if self.weights is not None and abs(sum(self.weights) - 1.0) > 1e-9:
            raise ValueError(f"hold_days.weights sum to {sum(self.weights)}, not 1.0")
        overlap = set(self.values) & set(self.stress_values)
        if overlap:
            raise ValueError(
                f"stress_values {sorted(overlap)} are inside the operating range; a "
                f"stress point must be out-of-distribution to mean anything"
            )


@dataclass(frozen=True)
class MaxDrawdownSpec:
    primary: float = 0.15
    values: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.25])
    weights: list[float] | None = None

    def __post_init__(self) -> None:
        if self.primary not in self.values:
            raise ValueError(
                f"max_drawdown.primary ({self.primary}) must appear in values {self.values}"
            )


@dataclass(frozen=True)
class LockSpec:
    scope: Literal["per_etf", "portfolio"] = "per_etf"
    hold_days: HoldDaysSpec = field(default_factory=HoldDaysSpec)


@dataclass(frozen=True)
class DrawdownSpec:
    max_drawdown: MaxDrawdownSpec = field(default_factory=MaxDrawdownSpec)


@dataclass(frozen=True)
class ProjectionSpec:
    backend: Literal["analytic", "cvxpy"] = "analytic"
    alpha_tolerance: float = 1e-3


@dataclass(frozen=True)
class RiskSpec:
    quantile: float = 0.01
    horizon_days: int = 5
    block_length: int = 10
    aggregation: Literal["max", "mean", "quantile"] = "max"
    #: Only `cvar` is convex in `w`, and the analytic projector's bisection needs that.
    measure: Literal["var", "cvar"] = "cvar"
    intervention_rate_ceiling: float = 0.50
    estimators: list[str] = field(
        default_factory=lambda: ["rolling", "block_bootstrap", "crisis_windows"])
    crisis_windows: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionSpec:
    cost_bps: float = 0.0     # D10 -- frictionless at v1


@dataclass(frozen=True)
class ConstraintsConfig:
    lock: LockSpec = field(default_factory=LockSpec)
    drawdown: DrawdownSpec = field(default_factory=DrawdownSpec)
    projection: ProjectionSpec = field(default_factory=ProjectionSpec)
    risk: RiskSpec = field(default_factory=RiskSpec)
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)

    @property
    def hold_days_grid(self) -> list[int]:
        """The operating range. Stress points are NOT included -- ask for them by name."""
        return list(self.lock.hold_days.values)

    @property
    def full_hold_days_grid(self) -> list[int]:
        """Operating range plus the out-of-distribution stress points, sorted."""
        hd = self.lock.hold_days
        return sorted(set(hd.values) | set(hd.stress_values))
