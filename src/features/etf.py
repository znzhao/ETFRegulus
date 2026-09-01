"""Per-ETF OHLCV features.

Every feature at session `t` uses data through `t`'s close and nothing after. In pandas
terms that means trailing `rolling`/`ewm` windows only -- never `center=True`, never a
negative `shift`. The lookahead test (I6) perturbs the future and asserts nothing at or
before `t` moves, so a violation here fails loudly rather than flattering the backtest.

Two standing rules from reference/features.md:

* Returns come from `close_adj`; execution and intraday ratios come from the raw OHLC.
  Never mix.
* Every division guards its denominator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.schema import FeaturesConfig

ANNUALIZATION = np.sqrt(252.0)


def _safe_div(num: pd.Series, den: pd.Series, eps: float) -> pd.Series:
    """Division with a clipped denominator. Most acute for the OHLC ratios."""
    return num / den.where(den.abs() > eps, np.nan)


def build_etf_features(g: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Features for one ticker, from its session-indexed curated block.

    `g` is indexed by session and carries close_adj / close_raw / OHLC / volume.
    """
    eps = cfg.epsilon
    w = cfg.windows
    adj = g["close_adj"]
    out: dict[str, pd.Series] = {}

    # -- returns ----------------------------------------------------------------
    log_adj = np.log(adj.where(adj > eps))
    r1 = adj.pct_change()
    lr1 = log_adj.diff()
    for n in w.returns:
        # For n > 1 these are the cumulative n-session returns; both scalings are kept
        # because both appear downstream (ranks want arithmetic, vol wants log). The
        # redundancy is intentional and is surfaced by the correlation diagnostic.
        out[f"ret_{n}"] = adj.pct_change(n)
        out[f"logret_{n}"] = log_adj.diff(n)

    # -- volatility -------------------------------------------------------------
    for n in w.volatility:
        out[f"vol_{n}"] = lr1.rolling(n, min_periods=n).std() * ANNUALIZATION

    # -- drawdown ---------------------------------------------------------------
    for n in w.drawdown:
        roll_max = adj.rolling(n, min_periods=n).max()
        out[f"drawdown_{n}"] = 1.0 - _safe_div(adj, roll_max, eps)
        # Sessions since the rolling maximum -- a different signal from its depth.
        out[f"days_since_max_{n}"] = (
            adj.rolling(n, min_periods=n).apply(lambda x: len(x) - 1 - int(np.argmax(x)),
                                                raw=True)
        )

    # -- intraday shape (raw OHLC; adjustment cancels within a session) ----------
    o, h, l, c = g["open_raw"], g["high_raw"], g["low_raw"], g["close_raw"]
    out["hl_over_c"] = _safe_div(h - l, c, eps)
    out["c_over_o"] = _safe_div(c - o, o, eps)
    # A zero-range bar (high == low) leaves the close simultaneously at both, so 0.5 is
    # the exact value rather than a guess. Without this it is an interior NaN, and an
    # interior NaN is a bug rather than something to fill.
    rng = h - l
    out["close_location"] = _safe_div(c - l, rng, eps).where(rng.abs() > eps, 0.5)

    # -- volume -----------------------------------------------------------------
    # A zero-volume print on a listed ETF means the bar is bad, not that nobody traded
    # it. Stage 2's gate has ALREADY adjudicated every such run -- failing the job on any
    # run longer than the forward-fill policy allows, unless it carries an explicit
    # waiver in `known_exceptions`. So carrying the last good volume forward here invents
    # nothing the gate has not already accepted, and it keeps the volume features free of
    # interior NaN.
    vol = g["volume"].where(g["volume"] > 0).ffill()
    for n in w.volume:
        mean = vol.rolling(n, min_periods=n).mean()
        std = vol.rolling(n, min_periods=n).std()
        out[f"volume_ratio_{n}"] = _safe_div(vol, mean, eps)
        out[f"volume_z_{n}"] = _safe_div(vol - mean, std, eps)
    out["volume_change"] = vol.pct_change()

    # -- trend ------------------------------------------------------------------
    for n in w.trend:
        sma = adj.rolling(n, min_periods=n).mean()
        out[f"close_over_sma_{n}"] = _safe_div(adj, sma, eps)
        out[f"sma_slope_{n}"] = _safe_div(sma.diff(n), sma.shift(n), eps)

    # -- technicals -------------------------------------------------------------
    t = cfg.technical
    out[f"rsi_{t.rsi_period}"] = _rsi(adj, t.rsi_period, eps)

    ema_fast = adj.ewm(span=t.macd_fast, adjust=False, min_periods=t.macd_slow).mean()
    ema_slow = adj.ewm(span=t.macd_slow, adjust=False, min_periods=t.macd_slow).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=t.macd_signal, adjust=False, min_periods=t.macd_signal).mean()
    # Normalized by price so the scale is comparable across a $40 and a $600 ETF.
    out["macd"] = _safe_div(macd, adj, eps)
    out["macd_signal"] = _safe_div(signal, adj, eps)
    out["macd_hist"] = _safe_div(macd - signal, adj, eps)

    out["atr_norm"] = _safe_div(_atr(g, t.atr_period), c, eps)

    mid = adj.rolling(t.bollinger_period, min_periods=t.bollinger_period).mean()
    sd = adj.rolling(t.bollinger_period, min_periods=t.bollinger_period).std()
    upper, lower = mid + t.bollinger_std * sd, mid - t.bollinger_std * sd
    out["bollinger_pct_b"] = _safe_div(adj - lower, upper - lower, eps)

    df = pd.DataFrame(out, index=g.index)
    df.index.name = "session"
    return df


def _rsi(close: pd.Series, period: int, eps: float) -> pd.Series:
    """Wilder's RSI. `ewm(alpha=1/period)` is Wilder's smoothing, and it is causal."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = _safe_div(avg_gain, avg_loss, eps)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 means an unbroken run of gains: RSI is 100 by definition, not NaN.
    return rsi.where(avg_loss > eps, 100.0).where(avg_gain.notna())


def _atr(g: pd.DataFrame, period: int) -> pd.Series:
    h, l, prev_c = g["high_raw"], g["low_raw"], g["close_raw"].shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    # True range is undefined on the first bar: it needs a previous close. `max` would
    # otherwise silently fall back to the high-low range and shorten the warm-up by one.
    tr = tr.where(prev_c.notna())
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def expected_warmup(name: str, cfg: FeaturesConfig) -> int:
    """Sessions of NaN a feature must have at the start of a ticker's history.

    Recorded in the feature manifest so Stage 3 can assert the NaN pattern *exactly*.
    Any other NaN is a bug, not something to fill.
    """
    t = cfg.technical
    if "_" in name and name.rsplit("_", 1)[1].isdigit():
        stem, n = name.rsplit("_", 1)
        n = int(n)
        if stem in ("ret", "logret", "volume_change"):
            return n
        if stem in ("vol", "drawdown", "days_since_max", "volume_ratio", "volume_z",
                    "close_over_sma"):
            return n - 1 + (1 if stem == "vol" else 0)
        if stem == "sma_slope":
            return 2 * n - 1
        if stem == "rsi":
            return n
    return {
        "hl_over_c": 0, "c_over_o": 0, "close_location": 0, "volume_change": 1,
        "macd": t.macd_slow - 1, "macd_signal": t.macd_slow + t.macd_signal - 2,
        "macd_hist": t.macd_slow + t.macd_signal - 2,
        "atr_norm": t.atr_period,
        "bollinger_pct_b": t.bollinger_period - 1,
    }.get(name, 0)
