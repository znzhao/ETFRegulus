"""The seven asset categories the comparison report allocates across.

These are not new groupings invented for the report -- they are the `tradable:` groups
already declared in `config/universe.yaml`, plus the synthetic CASH leg, renamed for
presentation. Deriving them from the config rather than restating them is the point: a
ticker added to `universe.yaml` lands in a category automatically, and a ticker that
somehow belongs to none is an error rather than a silently missing slice of the portfolio.

    config key      report label
    -------------   ----------------------
    broad_equity    US Broad Equity
    sector_equity   US Sector Equity
    treasury        Interest Rate
    credit          Credit
    real_assets     Commodities
    international   International Equity
    (synthetic)     CASH
"""

from __future__ import annotations

from src.config.schema import UniverseConfig

#: Config group -> report label, in the order the report presents them. The order is
#: risk-descending-ish (equity, then rates, then credit, then real assets, then cash) so a
#: reader scans it the way an allocation is normally discussed.
CATEGORY_LABELS: dict[str, str] = {
    "broad_equity": "US Broad Equity",
    "sector_equity": "US Sector Equity",
    "treasury": "Interest Rate",
    "credit": "Credit",
    "real_assets": "Commodities",
    "international": "International Equity",
}

CASH_LABEL = "CASH"

#: Every column in an allocation table, in display order.
CATEGORIES: tuple[str, ...] = (*CATEGORY_LABELS.values(), CASH_LABEL)


def ticker_to_category(ucfg: UniverseConfig) -> dict[str, str]:
    """Map each tradable ticker to its report label.

    Raises if a ticker belongs to no group or to two: either would make the allocation
    columns fail to sum to 100%, and a silently dropped asset is exactly the kind of
    error a percentage table hides.
    """
    groups = ucfg.tradable if isinstance(ucfg.tradable, dict) else dict(ucfg.tradable)
    mapping: dict[str, str] = {}
    for key, tickers in groups.items():
        if key not in CATEGORY_LABELS:
            raise KeyError(
                f"universe group {key!r} has no report category. Add it to "
                f"CATEGORY_LABELS in src/evaluation/categories.py -- an unmapped group "
                f"would vanish from the allocation tables while they still summed to 100%."
            )
        for ticker in tickers:
            if ticker in mapping:
                raise ValueError(f"{ticker} appears in two universe groups")
            mapping[ticker] = CATEGORY_LABELS[key]

    missing = set(ucfg.tradable_tickers) - set(mapping)
    if missing:
        raise ValueError(f"tradable tickers with no category: {sorted(missing)}")
    return mapping
