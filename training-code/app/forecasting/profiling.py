"""Product profiles: typical demand level, spikiness, and a forecast group.

    python -m app.forecasting.profiling --data sales.csv --out evaluation_runs/profile

For every product, over the last `weeks` full weeks of the file:
  typical_per_day   average units/day with spike days capped (the "normal" level)
  mean_per_day      plain average units/day (spikes included)
  spike_days        days above the spike threshold = SPIKE_MULTIPLIER x the
                    product's median sale-day size (and >= SPIKE_MIN_UNITS)
  spike_share       share of the product's units that came from spike days
  zero_share        share of days with no sale
  weekly_cv         week-to-week variability (std / mean of weekly totals)
  trend             last 4 weeks' daily average / the 4 weeks before
and one group: dead, new, bulk, rising, spiky, slow or steady (see GROUP_RULES).

It also writes every spike day with what it coincides with (weekday,
store-wide spike day, late posting, month start/end, holiday), so we can
tell predictable spikes from random ones.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.forecasting.cleaning import CleaningConfig, find_late_posting_runs
from app.forecasting.data import SalesData, load_dataset

SPIKE_MULTIPLIER = 3.0
SPIKE_MIN_UNITS = 10
STORE_SPIKE_RATIO = 2.0  # store day >= 2x its trailing median = store-wide spike day

GROUP_RULES = """\
dead    no sale in the last 28 days
new     first sale within the last 28 days
bulk    >= 20 units per bill (transactions / isBulk) or, without bill data,
        sells on < 15% of days with a median sale day >= 50 units
rising  last 4 weeks >= 1.5x the 4 weeks before, at >= 1 unit/day
spiky   >= 30% of units from spike days
slow    < 0.5 units/day on average
steady  everything else"""


def _daily_matrix(sales: SalesData, column: str) -> pd.DataFrame:
    full = pd.date_range(sales.start_date, sales.end_date, freq="D")
    return sales.daily.pivot(index="date", columns="variantId", values=column).reindex(full).fillna(0.0)


def profile_products(sales: SalesData, weeks: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per product, one row per spike day)."""

    walk_in = _daily_matrix(sales, "demand")
    bulk = _daily_matrix(sales, "bulkQuantity")
    units = walk_in + bulk  # the system stocks both

    end = units.index[-1]
    last_sunday = end - pd.Timedelta(days=(end.dayofweek - 6) % 7)
    start = last_sunday - pd.Timedelta(weeks=weeks) + pd.Timedelta(days=1)
    window = units.loc[start:last_sunday]

    first_sale = units[units > 0].apply(lambda s: s.first_valid_index())
    store = units.sum(axis=1)
    store_median = store.rolling(28, min_periods=7).median().shift(1)
    store_spike_days = set(store.index[store >= STORE_SPIKE_RATIO * store_median])
    late_spike_days = {spike for _, spike in find_late_posting_runs(store, CleaningConfig())}
    holidays = set(pd.DatetimeIndex(sales.holidays["date"])) if sales.holidays is not None else set()

    rows, spikes = [], []
    for variant_id in units.columns:
        first = first_sale.get(variant_id)
        if first is None or pd.isna(first) or first > last_sunday:
            continue
        series = window.loc[max(start, first) :, variant_id]
        days = len(series)
        sale_days = series[series > 0]
        median_sale_day = float(sale_days.median()) if len(sale_days) else 0.0
        threshold = max(SPIKE_MULTIPLIER * median_sale_day, SPIKE_MIN_UNITS)
        is_spike = series > threshold
        capped = series.clip(upper=threshold)

        weekly = series.resample("W-SUN").sum()
        recent = units.loc[last_sunday - pd.Timedelta(days=27) : last_sunday, variant_id]
        before = units.loc[last_sunday - pd.Timedelta(days=55) : last_sunday - pd.Timedelta(days=28), variant_id]
        before_mean = before[before.index >= first].mean() if (before.index >= first).any() else np.nan

        bills = sales.daily[(sales.daily["variantId"] == variant_id) & (sales.daily["date"].between(start, last_sunday))]
        per_bill = (
            float((bills["demand"] + bills["bulkQuantity"]).sum() / bills["transactions"].sum())
            if "transactions" in bills and bills["transactions"].sum() > 0
            else np.nan
        )

        row = {
            "variantId": variant_id,
            "productName": sales.products["productName"].get(variant_id, ""),
            "category": sales.products["category"].get(variant_id, pd.NA),
            "first_sale": first.date(),
            "days": days,
            "units": float(series.sum()),
            "bulk_units": float(bulk.loc[series.index, variant_id].sum()),
            "mean_per_day": float(series.mean()),
            "typical_per_day": float(capped.mean()),
            "median_sale_day": median_sale_day,
            "spike_threshold": threshold,
            "spike_days": int(is_spike.sum()),
            "spike_share": float((series[is_spike] - threshold).sum() / series.sum()) if series.sum() else 0.0,
            "zero_share": float((series == 0).mean()),
            "weekly_cv": float(weekly.std() / weekly.mean()) if weekly.mean() > 0 else np.nan,
            "recent_per_day": float(recent[recent.index >= first].mean()),
            "trend": float(recent[recent.index >= first].mean() / before_mean) if before_mean and before_mean > 0 else np.nan,
            "units_per_bill": per_bill,
            "last_sale": series[series > 0].index.max().date() if (series > 0).any() else None,
        }
        row["group"] = _group(row, last_sunday, bulk_units=row["bulk_units"])
        rows.append(row)

        for day in series.index[is_spike]:
            spikes.append(
                {
                    "variantId": variant_id,
                    "productName": row["productName"],
                    "date": day.date(),
                    "weekday": day.day_name(),
                    "units": float(series[day]),
                    "typical_day": round(row["typical_per_day"], 1),
                    "times_typical": round(float(series[day]) / max(row["typical_per_day"], 0.1), 1),
                    "store_spike_day": day in store_spike_days,
                    "late_posting_day": day in late_spike_days,
                    "month_edge": day.day <= 3 or day.day >= 28,
                    "holiday": day in holidays,
                }
            )

    profile = pd.DataFrame(rows).sort_values("units", ascending=False).reset_index(drop=True)
    return profile, pd.DataFrame(spikes)


def _group(row: dict, last_sunday: pd.Timestamp, bulk_units: float) -> str:
    if row["last_sale"] is None or pd.Timestamp(row["last_sale"]) < last_sunday - pd.Timedelta(days=27):
        return "dead"
    if pd.Timestamp(row["first_sale"]) > last_sunday - pd.Timedelta(days=28):
        return "new"
    sale_day_share = 1 - row["zero_share"]
    if (pd.notna(row["units_per_bill"]) and row["units_per_bill"] >= 20) or bulk_units > 0.5 * row["units"]:
        return "bulk"
    if pd.isna(row["units_per_bill"]) and sale_day_share < 0.15 and row["median_sale_day"] >= 50:
        return "bulk"
    if pd.notna(row["trend"]) and row["trend"] >= 1.5 and row["recent_per_day"] >= 1:
        return "rising"
    if row["spike_share"] >= 0.30:
        return "spiky"
    if row["mean_per_day"] < 0.5:
        return "slow"
    return "steady"


def spike_patterns(spikes: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """What spike days coincide with, compared with what you'd expect by
    chance (1/7 of days are any given weekday, etc.)."""

    if spikes.empty:
        return pd.DataFrame()
    total = len(spikes)
    rows = [
        ("on a Sunday", (spikes["weekday"] == "Sunday").mean(), 1 / 7),
        ("on a Saturday", (spikes["weekday"] == "Saturday").mean(), 1 / 7),
        ("on a store-wide spike day", spikes["store_spike_day"].mean(), None),
        ("on a late-posting day", spikes["late_posting_day"].mean(), None),
        ("at month start/end (day 1-3, 28-31)", spikes["month_edge"].mean(), 7 / 30.4),
        ("on a holiday", spikes["holiday"].mean(), None),
    ]
    out = pd.DataFrame(rows, columns=["spike days...", "share", "expected by chance"])
    out["spike_days"] = total
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Profile every product: typical level, spikes, group.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--aux-dir")
    parser.add_argument("--weeks", type=int, default=12)
    parser.add_argument("--out", required=True, help="output folder")
    args = parser.parse_args(argv)

    sales = load_dataset(args.data, args.aux_dir)
    profile, spikes = profile_products(sales, args.weeks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    profile.to_csv(out / "product_profile.csv", index=False)
    spikes.to_csv(out / "spike_days.csv", index=False)
    patterns = spike_patterns(spikes, profile)
    patterns.to_csv(out / "spike_patterns.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_colwidth", 40):
        groups = profile.groupby("group").agg(products=("variantId", "size"), units=("units", "sum"))
        groups["share_of_units_%"] = (groups["units"] / groups["units"].sum() * 100).round(1)
        print(groups.sort_values("units", ascending=False).to_string())
        print("\n" + patterns.round(3).to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
