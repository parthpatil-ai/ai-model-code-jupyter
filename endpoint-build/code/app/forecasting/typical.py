"""Typical-demand forecast: what a product normally sells, plus repeat orders.

Instead of a model trained to chase every spike, each product's daily
forecast is built from robust pieces of its own recent history. Every past
day is split into two parts:

  walk-in   what a normal day looks like: the day itself, or on a spike day
            (above `spike_multiplier` x the product's median sale day, and at
            least `spike_min_units`) the median sale day
  order     the rest of a spike day, treated as an order from a regular
            customer (Baked Samosa: ~230 units every ~3 weeks)

and the forecast is

  prediction = walk-in level x weekday factor + repeat orders

  walk-in level   mean walk-in units/day over the last `level_days`; rising
                  products use the last `rising_days`, products that sell on
                  few days use the last `long_days` (one sale must not swing
                  the level)
  weekday factor  the product's own weekday shape, pulled towards the
                  store's when the product has few units (the Sunday uplift)
  repeat orders   "rate":  the average order volume per day over the last
                           `order_days`, put on the weekdays orders usually
                           come (shrunk towards an even spread)
                  "timed": when there are >= `min_orders_for_timing` orders
                           with regular gaps, the typical order size on the
                           expected next order date (last order + median
                           gap); otherwise "rate"
                  "none":  no orders (walk-in only)

high_qty is a coverage level for stocking: the product's 75th-percentile
week over the last `high_weeks` weeks, spread over the days with the
weekday factor, never below the prediction.

A product is forecast when it sold in the last `active_days` days (or, if
`returning_min_days` is set, on at least that many days in the last
`long_days`: products that come and go); catalog rules (new, delisted, replaced items) and cold start come from
app/forecasting/universe.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.forecasting.data import SalesData
from app.forecasting.features import demand_matrix
from app.forecasting.universe import apply_variant_mapping, cold_start_rates, product_universe, products_with_catalog

ORDER_MODES = ("rate", "timed", "none")


@dataclass(frozen=True)
class TypicalConfig:
    level_days: int = 28
    rising_days: int = 14
    rising_ratio: float = 1.5
    long_days: int = 84
    infrequent_share: float = 0.15  # sells on fewer than 15% of days
    spike_multiplier: float = 3.0
    spike_min_units: float = 10.0
    weekday_days: int = 56
    weekday_shrink_units: float = 50.0
    orders: str = "rate"
    order_days: int = 56
    order_cluster_days: int = 2  # order days this close together are one order
    min_orders_for_timing: int = 3
    max_gap_cv: float = 0.5
    high_quantile: float = 0.75  # P90 covered 79% of demand but overstocked 165%
    high_weeks: int = 12
    active_days: int = 28
    returning_min_days: int | None = None  # off: it added ~80 phantom products a week
    include_bulk: bool = True  # bulk lines are stocked too

    def __post_init__(self):
        if self.orders not in ORDER_MODES:
            raise ValueError(f"orders must be one of {ORDER_MODES}, got {self.orders!r}")


def units_matrix(sales: SalesData, include_bulk: bool = True) -> pd.DataFrame:
    """dates x variantId units sold (walk-in, plus bulk lines when
    include_bulk); stock-out days and days before a product's first sale
    are NaN."""

    units = demand_matrix(sales, "demand")
    if include_bulk and sales.daily["bulkQuantity"].any():
        units = units + demand_matrix(sales, "bulkQuantity")
    sold = units.fillna(0) > 0
    first = sold.idxmax().where(sold.any())
    before = units.index.to_numpy()[:, None] < first.reindex(units.columns).to_numpy()[None, :]
    return units.mask(before | first.isna().to_numpy()[None, :])


def spike_thresholds(units: pd.DataFrame, config: TypicalConfig = TypicalConfig()) -> tuple[pd.Series, pd.Series]:
    """(median sale-day size, spike threshold) per product, from the last
    `long_days` days of `units`."""

    window = units.iloc[-config.long_days :]
    median_sale = window.where(window > 0).median()
    threshold = (median_sale * config.spike_multiplier).clip(lower=config.spike_min_units).fillna(config.spike_min_units)
    return median_sale.fillna(0.0), threshold


def split_walk_in_and_orders(units: pd.DataFrame, config: TypicalConfig = TypicalConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(walk-in, order) parts of every day; they add up to `units`."""

    median_sale, threshold = spike_thresholds(units, config)
    spike = units.gt(threshold, axis=1)
    walk_in = units.mask(spike, median_sale, axis=1)
    return walk_in, units - walk_in


def _walk_in_level(walk_in: pd.DataFrame, units: pd.DataFrame, config: TypicalConfig) -> pd.Series:
    level = walk_in.iloc[-config.level_days :].mean()
    recent = walk_in.iloc[-config.rising_days :].mean()
    before = walk_in.iloc[-2 * config.level_days : -config.level_days].mean()
    rising = (level >= 1) & (level >= config.rising_ratio * before)
    level = level.where(~rising, recent)

    long_window = units.iloc[-config.long_days :]
    sale_share = (long_window > 0).sum() / long_window.notna().sum().replace(0, np.nan)
    infrequent = sale_share < config.infrequent_share
    return level.where(~infrequent, walk_in.iloc[-config.long_days :].mean()).fillna(0.0)


def weekday_factors(walk_in: pd.DataFrame, config: TypicalConfig = TypicalConfig()) -> pd.DataFrame:
    """7 x variantId multipliers (mean 1 per product): the product's own
    weekday shape, shrunk towards the store's by its unit count."""

    window = walk_in.iloc[-config.weekday_days :]
    weekday = window.index.dayofweek
    store = window.sum(axis=1).groupby(weekday).mean()
    store = (store / store.mean()).reindex(range(7)).fillna(1.0)

    own = window.groupby(weekday).mean().reindex(range(7))
    own = own.div(own.mean().replace(0, np.nan), axis=1)
    store_shape = pd.DataFrame(np.repeat(store.to_numpy()[:, None], own.shape[1], axis=1), index=own.index, columns=own.columns)
    own = own.fillna(store_shape)
    units = window.sum()
    weight = units / (units + config.weekday_shrink_units)
    factors = own * weight + store_shape * (1 - weight)
    return factors.div(factors.mean(), axis=1).fillna(1.0)


def order_events(orders: pd.Series, cluster_days: int = 2) -> pd.Series:
    """One product's order days grouped into order events (date of the first
    day -> total units), merging days within `cluster_days` of each other."""

    days = orders[orders > 0]
    if days.empty:
        return days
    new_event = days.index.to_series().diff().dt.days.fillna(np.inf) > cluster_days
    sizes = days.groupby(new_event.cumsum().to_numpy()).sum()
    sizes.index = days.index[new_event.to_numpy()]
    return sizes


def _order_rates(orders: pd.DataFrame, config: TypicalConfig) -> tuple[pd.Series, pd.DataFrame]:
    """(order units per day, 7 x variantId weekday weights summing to 7)."""

    window = orders.iloc[-config.order_days :]
    rate = (window.sum() / window.notna().sum().replace(0, np.nan)).fillna(0.0)
    counts = (window > 0).groupby(window.index.dayofweek).sum().reindex(range(7)).fillna(0)
    weights = (counts + 1) / (counts.sum() + 7) * 7
    return rate, weights


def _timed_orders(orders: pd.Series, dates: pd.DatetimeIndex, config: TypicalConfig) -> pd.Series | None:
    """Units per forecast date from the expected next order(s), or None
    when the orders are too few or too irregular to time."""

    events = order_events(orders.iloc[-config.order_days :].dropna(), config.order_cluster_days)
    if len(events) < config.min_orders_for_timing:
        return None
    gaps = events.index.to_series().diff().dt.days.dropna()
    gap = float(gaps.median())
    if gap <= 0 or gaps.std() / gaps.mean() > config.max_gap_cv:
        return None
    size = float(events.median())
    out = pd.Series(0.0, index=dates)
    due = events.index[-1] + pd.Timedelta(days=round(gap))
    if due < dates[0]:
        if (dates[0] - due).days > gap:
            return out  # overdue by more than a full cycle: the customer has lapsed
        due = dates[0]
    while due <= dates[-1]:
        out[due] += size
        due += pd.Timedelta(days=round(gap))
    return out


def _high_week(units: pd.DataFrame, config: TypicalConfig) -> pd.Series:
    window = units.iloc[-7 * config.high_weeks :]
    blocks = np.arange(len(window))[::-1] // 7  # 7-day blocks ending on the last day
    weekly = window.groupby(blocks).sum(min_count=4)
    return weekly.quantile(config.high_quantile).fillna(0.0)


def typical_forecast(sales: SalesData, dates, config: TypicalConfig = TypicalConfig()) -> pd.DataFrame:
    """variantId, date, prediction (expected units), high_qty, walk_in,
    orders for every forecast product and date after `sales`."""

    dates = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(dates))))
    columns = ["variantId", "date", "prediction", "high_qty", "walk_in", "orders"]
    if len(dates) == 0:
        return pd.DataFrame(columns=columns)
    if dates[0] <= sales.end_date:
        raise ValueError(f"typical_forecast only forecasts after {sales.end_date.date()}")

    sales = apply_variant_mapping(sales)
    units = units_matrix(sales, config.include_bulk)
    units = units.loc[:, units.notna().any()]
    last = units.index[-1]
    walk_in, orders = split_walk_in_and_orders(units, config)
    level = _walk_in_level(walk_in, units, config)
    factors = weekday_factors(walk_in, config)
    rate, order_weights = _order_rates(orders, config)
    high_week = _high_week(units, config)

    long_window = units.iloc[-config.long_days :]
    returning = set(long_window.columns[(long_window > 0).sum() >= config.returning_min_days]) if config.returning_min_days else set()
    products = products_with_catalog(sales.products, sales.catalog)

    timed = {}
    if config.orders == "timed":
        for variant_id in rate.index[rate > 0]:
            timed[variant_id] = _timed_orders(orders[variant_id], dates, config)

    frames = []
    for day in dates:
        universe, new_items = product_universe(units, last, day, sales.catalog, sales.variant_mapping, config.active_days)
        allowed_returning, _ = product_universe(units, last, day, sales.catalog, sales.variant_mapping, config.long_days)
        ids = sorted((universe - new_items) | (returning & allowed_returning))
        ids = [v for v in ids if v in units.columns]
        dow = day.dayofweek
        walk = level.reindex(ids).to_numpy() * factors.loc[dow].reindex(ids).to_numpy()
        if config.orders == "none":
            order = np.zeros(len(ids))
        else:
            order = rate.reindex(ids).to_numpy() * order_weights.loc[dow].reindex(ids).to_numpy()
            if config.orders == "timed":
                order = np.array([timed[v][day] if timed.get(v) is not None else o for v, o in zip(ids, order)])
        prediction = walk + order
        high = np.maximum(prediction, high_week.reindex(ids).to_numpy() / 7 * factors.loc[dow].reindex(ids).to_numpy())
        frames.append(pd.DataFrame({"variantId": ids, "date": day, "prediction": prediction, "high_qty": high, "walk_in": walk, "orders": order}))

        if new_items:
            rates = cold_start_rates(new_items, units, last, products, config.active_days)
            start = np.array(list(rates.values()), dtype=float)
            frames.append(pd.DataFrame({"variantId": list(rates), "date": day, "prediction": start, "high_qty": start, "walk_in": start, "orders": 0.0}))

    out = pd.concat(frames, ignore_index=True)
    out["variantId"] = out["variantId"].astype("int64")
    return out[columns]
