"""Which products to forecast, and how products start.

- Variant replacement (variant_mapping.csv): before its switch_date, the old
  variant's sales count as the new variant's history (Parle G Rs5 -> Rs10),
  so the replacement starts with a real sales level instead of from zero.
  From the switch date the old variant is no longer forecast.
- Trimmed start: a product's history starts at its first sale (or its
  catalog launch_date, if earlier). The zero rows the export pads in before
  that are not demand and are dropped.
- Product list for a forecast date: products sold in the last
  FORECAST_ACTIVE_DAYS days, plus catalog items that are active and newly
  launched; minus anything delisted (active=0 or delist_date reached) or
  replaced.
- Cold start: a new catalog item with no sales yet starts at the average
  daily sales of products in the same category and price band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import FORECAST_ACTIVE_DAYS
from app.utils.logger import get_logger
from app.forecasting.data import SalesData

logger = get_logger(__name__)

# Rs price bands for the cold-start estimate (lower bounds).
PRICE_BANDS = [0, 5, 10, 20, 30, 50, 100]


def price_band(mrp) -> float:
    if mrp is None or pd.isna(mrp):
        return np.nan
    return float(max(b for b in PRICE_BANDS if mrp >= b))


def apply_variant_mapping(sales: SalesData) -> SalesData:
    """Move each old variant's sales before its switch date onto the new
    variant (summed with any of the new variant's own sales that day)."""

    mapping = sales.variant_mapping
    if mapping is None or mapping.empty:
        return sales

    daily = sales.daily.copy()
    switch = mapping.set_index("old_variant_id")
    old_ids = daily["variantId"].map(switch["new_variant_id"])
    before_switch = old_ids.notna() & (daily["date"] < daily["variantId"].map(switch["switch_date"]))
    moved = int(before_switch.sum())
    daily.loc[before_switch, "variantId"] = old_ids[before_switch].astype("int64")

    numeric = [c for c in ("demand", "bulkQuantity", "transactions") if c in daily.columns]
    agg = {c: "sum" for c in numeric}
    agg.update({c: "max" for c in ("isStockout", "bulkSource") if c in daily.columns})
    daily = daily.groupby(["variantId", "date"], as_index=False).agg(agg)
    if "transactions" in daily.columns:
        daily["transactions"] = daily["transactions"].replace(0, np.nan)

    products = sales.products.copy()
    for row in mapping.itertuples():
        if row.new_variant_id not in products.index and row.old_variant_id in products.index:
            products.loc[row.new_variant_id] = products.loc[row.old_variant_id]
    logger.info("Variant mapping: moved %d day(s) of history onto replacement variants", moved)
    return SalesData(daily, products, sales.catalog, mapping, sales.holidays)


def start_dates(sales: SalesData) -> pd.Series:
    """Per variant: first walk-in sale, or the catalog launch_date if that
    is earlier (zeros after a known launch are real demand)."""

    first_sale = sales.daily[sales.daily["demand"] > 0].groupby("variantId")["date"].min()
    if sales.catalog is not None and not sales.catalog.empty:
        launch = sales.catalog.set_index("variant_id")["launch_date"].reindex(first_sale.index)
        first_sale = first_sale.where(launch.isna() | (first_sale <= launch), launch)
    return first_sale


def products_with_catalog(products: pd.DataFrame, catalog: pd.DataFrame | None) -> pd.DataFrame:
    """Product attributes for everything that can be forecast, including
    catalog items that have not sold yet."""

    if catalog is None or catalog.empty:
        return products
    missing = catalog[~catalog["variant_id"].isin(products.index)].set_index("variant_id")
    extra = pd.DataFrame(
        {"productName": missing["product_name"], "category": missing["category"], "mrp": missing["mrp"]},
        index=missing.index.rename("variantId"),
    )
    return pd.concat([products, extra])


def trim_before_start(history: pd.DataFrame, starts: pd.Series) -> pd.DataFrame:
    """NaN out each product's padded rows before its start date; products
    that never sold are dropped from the training history."""

    keep = [v for v in history.columns if v in starts.index]
    history = history[keep]
    before = history.index.to_numpy()[:, None] < starts.reindex(history.columns).to_numpy()[None, :]
    return history.mask(before)


def recently_sold(history: pd.DataFrame, last_date: pd.Timestamp, days: int = FORECAST_ACTIVE_DAYS) -> set:
    window = history.loc[last_date - pd.Timedelta(days=days - 1) : last_date]
    return set(window.columns[(window.fillna(0) > 0).any(axis=0)])


def product_universe(
    history: pd.DataFrame,
    last_date: pd.Timestamp,
    day: pd.Timestamp,
    catalog: pd.DataFrame | None,
    mapping: pd.DataFrame | None,
    days: int = FORECAST_ACTIVE_DAYS,
) -> tuple[set, set]:
    """(products to forecast on `day`, of which new catalog items without
    sales history)."""

    universe = recently_sold(history, last_date, days)
    new_items = set()

    if catalog is not None and not catalog.empty:
        cat = catalog.set_index("variant_id")
        launched = cat["launch_date"].isna() | (cat["launch_date"] <= day)
        recent_launch = cat["launch_date"].notna() & (cat["launch_date"] > last_date - pd.Timedelta(days=days))
        candidates = cat.index[(cat["active"] == 1) & launched & recent_launch]
        for variant_id in candidates:
            if variant_id not in universe:
                universe.add(variant_id)
                if variant_id not in history.columns or history[variant_id].notna().sum() == 0:
                    new_items.add(variant_id)
        delisted = cat.index[(cat["active"] == 0) | (cat["delist_date"].notna() & (cat["delist_date"] <= day))]
        universe -= set(delisted)
        new_items -= set(delisted)

    if mapping is not None and not mapping.empty:
        replaced = set(mapping.loc[mapping["switch_date"] <= day, "old_variant_id"])
        universe -= replaced
        new_items -= replaced
    return universe, new_items


def cold_start_rates(
    new_items, history: pd.DataFrame, last_date: pd.Timestamp, products: pd.DataFrame, days: int = FORECAST_ACTIVE_DAYS
) -> dict:
    """Daily starting estimate for each new item: mean daily sales over the
    last `days` days of products in the same category AND price band; falls
    back to the category, then the price band, then all products."""

    window = history.loc[last_date - pd.Timedelta(days=days - 1) : last_date]
    rates = window.mean(axis=0, skipna=True).dropna()
    rates = rates[rates > 0]
    attrs = products.reindex(rates.index)
    peers = pd.DataFrame(
        {"rate": rates, "category": attrs["category"].astype("string"), "band": attrs["mrp"].map(price_band)}
    )

    estimates = {}
    for variant_id in new_items:
        info = products.loc[variant_id] if variant_id in products.index else pd.Series(dtype=object)
        category = info.get("category")
        band = price_band(info.get("mrp"))
        for mask in (
            (peers["category"] == category) & (peers["band"] == band),
            peers["category"] == category,
            peers["band"] == band,
        ):
            mask = mask.fillna(False)
            if mask.any():
                estimates[variant_id] = float(peers.loc[mask, "rate"].mean())
                break
        else:
            estimates[variant_id] = float(peers["rate"].median()) if len(peers) else 0.0
    return estimates
