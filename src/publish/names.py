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
        "description": ("60% 标普 500（SPY）+ 40% 长期美债（TLT），每月再平衡。",
                        "60% S&P 500 (SPY) and 40% long Treasuries (TLT), rebalanced monthly."),
        "color": "#7a5bb5",
    },
    "momentum": {
        "name": ("动量", "Momentum"),
        "description": ("定期买入过去一段时间涨得最好的几只 ETF。",
                        "Periodically buys the ETFs with the strongest recent returns."),
        "color": "#2a9d8f",
    },
    "classical_optimizer": {
        "name": ("经典优化器", "Classical optimizer"),
        "description": ("用历史收益和波动做均值方差优化来分配。",
                        "Allocates by mean-variance optimization on historical returns and risk."),
        "color": "#c4567a",
    },
    "equal_weight": {
        "name": ("等权", "Equal weight"),
        "description": ("把资金平均分给所有可交易的 ETF，定期再平衡。",
                        "Splits capital equally across every tradable ETF, rebalanced periodically."),
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
