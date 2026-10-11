"""Display names, descriptions and colours for the web app (WEBAPP_PLAN.md).

Shared by the real publish pipeline and the fake-data generator, so both write the same
labels. Text is bilingual: the app switches between Chinese and English.
"""

from __future__ import annotations

TICKERS: dict[str, tuple[str, str]] = {
    "SPY": ("标普500", "S&P 500"),
    "QQQ": ("纳斯达克100", "Nasdaq-100"),
    "IWM": ("罗素2000", "Russell 2000"),
    "XLK": ("科技", "Technology"),
    "XLC": ("通信", "Communication services"),
    "XLY": ("可选消费", "Consumer discretionary"),
    "XLP": ("必需消费", "Consumer staples"),
    "XLE": ("能源", "Energy"),
    "XLF": ("金融", "Financials"),
    "XLI": ("工业", "Industrials"),
    "XLB": ("材料", "Materials"),
    "XLV": ("医疗", "Health care"),
    "XLU": ("公用事业", "Utilities"),
    "XLRE": ("房地产", "Real estate"),
    "SHY": ("1-3年美债", "1-3Y Treasury"),
    "IEF": ("7-10年美债", "7-10Y Treasury"),
    "TLT": ("20年以上美债", "20Y+ Treasury"),
    "TIP": ("通胀保值美债", "TIPS"),
    "LQD": ("投资级公司债", "Investment-grade corporate"),
    "HYG": ("高收益债", "High yield"),
    "GLD": ("黄金", "Gold"),
    "DBC": ("大宗商品", "Commodities"),
    "VEA": ("发达市场", "Developed markets ex-US"),
    "VWO": ("新兴市场", "Emerging markets"),
}

# Order is the display order in the app.
STRATEGIES: dict[str, dict] = {
    "spy_tlt_60_40": {
        "name": ("60/40 股债", "60/40 SPY-TLT"),
        "description": ("60% 标普 500（SPY）+ 40% 长期美债（TLT）。每月末检查，偏离目标超过 5 个百分点才再平衡。",
                        "60% S&P 500 (SPY) and 40% long Treasuries (TLT). Checked at each month-end; rebalanced only when a weight drifts more than 5 points from target."),
        "color": "#7a5bb5",
    },
    "momentum": {
        "name": ("动量", "Momentum"),
        "description": ("每月末买入过去 12 个月（不含最近 1 个月）涨幅最大的 5 只 ETF，各 20%；涨幅为负的不买，空位留作现金。",
                        "At each month-end, buys the 5 ETFs with the best 12-month return excluding the latest month, 20% each; ETFs with a negative return are skipped and their slot stays in cash."),
        "color": "#2a9d8f",
    },
    "classical_optimizer": {
        "name": ("经典优化器", "Classical optimizer"),
        "description": ("风险平价：每月末按过去一年波动率的倒数分配，波动越小配得越多，每只最多 25%。",
                        "Risk parity: at each month-end, weights by the inverse of each ETF's past-year volatility (calmer ETFs get more), at most 25% each."),
        "color": "#c4567a",
    },
    "equal_weight": {
        "name": ("等权", "Equal weight"),
        "description": ("把资金平均分给所有可交易的 ETF。每月末检查，偏离目标超过 5 个百分点才再平衡。",
                        "Splits capital equally across every tradable ETF. Checked at each month-end; rebalanced only when a weight drifts more than 5 points from target."),
        "color": "#d08a2e",
    },
    "spy_buy_hold": {
        "name": ("SPY 买入持有", "SPY buy & hold"),
        "description": ("全部买入标普 500（SPY）并持有。", "Buys the S&P 500 (SPY) and holds it."),
        "color": "#3d7fc4",
    },
    "cash": {
        "name": ("现金", "Cash"),
        "description": ("全部持有现金。用来对比的最低基准。",
                        "Holds only cash. The floor every strategy is compared with."),
        "color": "#8a96a3",
    },
}


def text(pair: tuple[str, str]) -> dict[str, str]:
    return {"zh": pair[0], "en": pair[1]}


def strategy_entries() -> list[dict]:
    return [{"id": sid, "name": text(s["name"]), "description": text(s["description"]),
             "color": s["color"]} for sid, s in STRATEGIES.items()]
