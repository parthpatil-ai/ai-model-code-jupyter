"""Spike handling for TRAINING data.

Three rules, applied in this order, each configurable (app/config.py, FORECAST_*):

1. Bulk lines: units on isBulk rows (or rows flagged by units per
   transaction, see app/forecasting/data.py) are not walk-in demand and are left out.
2. Late posting: when the whole store records zero sales for one or more
   days and the next day is a spike (>= spike_ratio x the median non-zero
   store day of the previous 28 days), the sales were posted late. Each
   product's spike-day units are spread evenly over the zero days + the
   spike day, so its total is unchanged.
3. Hampel cap per product: a day above median + k x 1.4826 x MAD of its
   reference days is lowered TO THE CAP (never to the median). Two
   references, by default the higher cap wins: the same weekday over the
   last 4 weeks, and the +/-7-day window. The cap never goes below a floor
   (the product's 99th percentile of non-zero days, or 3 x its 28-day mean),
   so slow sellers whose median is 0 are not capped to 0. Days listed in
   holidays.csv and stock-out days are never capped.

The original quantity is kept (`quantityOriginal` = walk-in + bulk as
loaded); the cleaned training value goes to `demandClean`. Every change is
logged (product, date, original, cleaned, rule) for review.

    python -m app.forecasting.cleaning --data sales.csv --out cleaning_log.csv
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import (
    FORECAST_CAP_FLOOR_MEAN_MULTIPLIER,
    FORECAST_CAP_FLOOR_QUANTILE,
    FORECAST_CAP_K,
    FORECAST_CAP_REFERENCE,
    FORECAST_CLEANING_ENABLED,
    FORECAST_LATE_POSTING_MIN_ZERO_DAYS,
    FORECAST_LATE_POSTING_SPIKE_RATIO,
)
from app.utils.logger import get_logger
from app.forecasting.data import SalesData, load_dataset
from app.forecasting.features import demand_matrix

logger = get_logger(__name__)

MAD_TO_SIGMA = 1.4826
LOG_COLUMNS = ["variantId", "productName", "date", "rule", "original", "cleaned", "change", "detail"]


@dataclass
class CleaningConfig:
    enabled: bool = FORECAST_CLEANING_ENABLED
    late_posting: bool = True
    late_posting_min_zero_days: int = FORECAST_LATE_POSTING_MIN_ZERO_DAYS
    late_posting_spike_ratio: float = FORECAST_LATE_POSTING_SPIKE_RATIO
    late_posting_lookback_days: int = 28
    cap: bool = True
    cap_k: float = FORECAST_CAP_K
    cap_reference: str = FORECAST_CAP_REFERENCE  # "weekday" | "window" | "both"
    cap_weekday_weeks: int = 4
    cap_window_days: int = 7
    cap_floor_quantile: float = FORECAST_CAP_FLOOR_QUANTILE
    cap_floor_mean_multiplier: float = FORECAST_CAP_FLOOR_MEAN_MULTIPLIER
    cap_floor_mean_days: int = 28
    skip_holidays: bool = True

    def __post_init__(self):
        if self.cap_reference not in ("weekday", "window", "both"):
            raise ValueError(f"cap_reference must be weekday, window or both, got {self.cap_reference!r}")


def clean_training_data(sales: SalesData, config: CleaningConfig | None = None) -> tuple[SalesData, pd.DataFrame]:
    """Return (sales with `quantityOriginal` and `demandClean` columns, change log).

    The input is not modified. With cleaning disabled, demandClean == demand
    (bulk lines are still left out).
    """

    config = config or CleaningConfig()
    daily = sales.daily.copy()
    daily["quantityOriginal"] = daily["demand"] + daily["bulkQuantity"]
    names = sales.products["productName"]
    log_frames = [_bulk_log(daily, names)]

    matrix = demand_matrix(sales, column="demand")
    if config.enabled and config.late_posting:
        matrix, frame = _spread_late_posting(matrix, sales.store_totals(), config)
        log_frames.append(frame)
    if config.enabled and config.cap:
        matrix, frame = _hampel_cap(matrix, sales.holidays, config)
        log_frames.append(frame)

    cleaned = matrix.stack(future_stack=True).rename("demandClean").reset_index()
    cleaned = cleaned[cleaned["demandClean"].notna()]
    daily = daily.merge(cleaned, on=["variantId", "date"], how="outer")
    for column, fill in (("demand", 0.0), ("bulkQuantity", 0.0), ("quantityOriginal", 0.0), ("isStockout", False), ("bulkSource", "")):
        daily[column] = daily[column].fillna(fill)
    daily["isStockout"] = daily["isStockout"].astype(bool)
    # stock-out days stay NaN-free here; demand_matrix masks them again
    daily["demandClean"] = daily["demandClean"].fillna(daily["demand"])
    daily = daily.sort_values(["variantId", "date"]).reset_index(drop=True)

    log = pd.concat([f for f in log_frames if not f.empty], ignore_index=True) if any(not f.empty for f in log_frames) else _empty_log()
    if not log.empty:
        log["productName"] = log["variantId"].map(names)
        log["change"] = log["cleaned"] - log["original"]
        log = log[LOG_COLUMNS].sort_values(["date", "variantId", "rule"]).reset_index(drop=True)
        summary = log.groupby("rule").agg(rows=("change", "size"), units=("change", "sum"))
        logger.info("Training-data cleaning changes:\n%s", summary.to_string())

    return SalesData(daily, sales.products, sales.catalog, sales.variant_mapping, sales.holidays), log


def _empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=LOG_COLUMNS)


def _bulk_log(daily: pd.DataFrame, names: pd.Series) -> pd.DataFrame:
    rows = daily[daily["bulkQuantity"] > 0]
    if rows.empty:
        return _empty_log()
    return pd.DataFrame(
        {
            "variantId": rows["variantId"].to_numpy(),
            "date": rows["date"].to_numpy(),
            "rule": "bulk_" + rows["bulkSource"].replace("", "isBulk").to_numpy(),
            "original": rows["quantityOriginal"].to_numpy(),
            "cleaned": rows["demand"].to_numpy(),
            "detail": [f"{b:g} bulk units left out of training" for b in rows["bulkQuantity"]],
        }
    )


def find_late_posting_runs(store_totals: pd.Series, config: CleaningConfig) -> list[tuple[pd.DatetimeIndex, pd.Timestamp]]:
    """[(zero days, spike day)] for every run of whole-store zero days that
    is immediately followed by a spike day."""

    totals = store_totals.sort_index()
    is_zero = totals.to_numpy() == 0
    runs, i, n = [], 0, len(totals)
    while i < n:
        if not is_zero[i]:
            i += 1
            continue
        j = i
        while j < n and is_zero[j]:
            j += 1
        if j < n and j - i >= config.late_posting_min_zero_days:
            before = totals.iloc[max(0, i - config.late_posting_lookback_days) : i]
            reference = before[before > 0].median()
            if pd.notna(reference) and reference > 0 and totals.iloc[j] >= config.late_posting_spike_ratio * reference:
                runs.append((totals.index[i:j], totals.index[j]))
        i = j
    return runs


def _spread_late_posting(matrix: pd.DataFrame, store_totals: pd.Series, config: CleaningConfig):
    matrix = matrix.copy()
    records = []
    for zero_days, spike_day in find_late_posting_runs(store_totals, config):
        days = zero_days.append(pd.DatetimeIndex([spike_day]))
        block = matrix.loc[days]
        spike = block.loc[spike_day]
        exists = block.notna()
        share = spike / exists.sum(axis=0).replace(0, np.nan)
        movers = spike[(spike > 0) & share.notna()].index
        detail = f"late posting: {zero_days[0].date()}..{zero_days[-1].date()} zero, {spike_day.date()} spike"
        for variant_id in movers:
            for day in days[exists[variant_id].to_numpy()]:
                before = float(block.at[day, variant_id])
                after = float(share[variant_id])
                matrix.at[day, variant_id] = after
                records.append((variant_id, day, "late_posting", before, after, detail))
        logger.info("Late posting: spread %s over %d day(s) for %d product(s)", spike_day.date(), len(days), len(movers))
    frame = pd.DataFrame(records, columns=["variantId", "date", "rule", "original", "cleaned", "detail"])
    return matrix, frame


def _median_and_mad(stack: np.ndarray, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Median and MAD along the first axis, NaN-aware; NaN where fewer than
    `min_count` reference values exist."""

    counts = np.sum(~np.isnan(stack), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        median = np.nanmedian(stack, axis=0)
        mad = np.nanmedian(np.abs(stack - median), axis=0)
    median[counts < min_count] = np.nan
    return median, mad


def cap_levels(matrix: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    """Cap for every (date, product); NaN where no cap applies."""

    caps = []
    if config.cap_reference in ("weekday", "both"):
        weeks = [matrix.shift(7 * w).to_numpy() for w in range(1, config.cap_weekday_weeks + 1)]
        median, mad = _median_and_mad(np.stack(weeks), min_count=2)
        caps.append(median + config.cap_k * MAD_TO_SIGMA * mad)
    if config.cap_reference in ("window", "both"):
        offsets = [o for o in range(-config.cap_window_days, config.cap_window_days + 1) if o != 0]
        window = [matrix.shift(-o).to_numpy() for o in offsets]
        median, mad = _median_and_mad(np.stack(window), min_count=3)
        caps.append(median + config.cap_k * MAD_TO_SIGMA * mad)
    reference_cap = np.fmax.reduce(caps) if len(caps) > 1 else caps[0]

    nonzero_quantile = matrix.where(matrix > 0).quantile(config.cap_floor_quantile).to_numpy()
    recent_mean = matrix.shift(1).rolling(config.cap_floor_mean_days, min_periods=7).mean().to_numpy()
    floor = np.fmax(nonzero_quantile[None, :], config.cap_floor_mean_multiplier * recent_mean)

    cap = np.fmax(reference_cap, floor)  # NaN only where neither part is defined
    return pd.DataFrame(cap, index=matrix.index, columns=matrix.columns)


def _hampel_cap(matrix: pd.DataFrame, holidays: pd.DataFrame | None, config: CleaningConfig):
    caps = cap_levels(matrix, config)
    over = (matrix > caps) & caps.notna()
    if config.skip_holidays and holidays is not None and not holidays.empty:
        over.loc[over.index.isin(holidays["date"])] = False

    rows, cols = np.nonzero(over.to_numpy())
    records = [
        (
            matrix.columns[c],
            matrix.index[r],
            "hampel_cap",
            float(matrix.iat[r, c]),
            float(caps.iat[r, c]),
            f"capped at {caps.iat[r, c]:.1f} (k={config.cap_k:g}, reference={config.cap_reference})",
        )
        for r, c in zip(rows, cols)
    ]
    cleaned = matrix.mask(over, caps)
    return cleaned, pd.DataFrame(records, columns=["variantId", "date", "rule", "original", "cleaned", "detail"])


def write_cleaning_log(log: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = log.copy()
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%d-%m-%Y")
        out[["original", "cleaned", "change"]] = out[["original", "cleaned", "change"]].round(2)
    out.to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show what the training-data cleaning changes, without training.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--aux-dir")
    parser.add_argument("--out", required=True, help="cleaning log CSV to write")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)

    _, log = clean_training_data(load_dataset(args.data, args.aux_dir))
    write_cleaning_log(log, args.out)
    with pd.option_context("display.width", 250, "display.max_colwidth", 45):
        print(log.groupby("rule").agg(rows=("change", "size"), units_changed=("change", "sum")).round(1).to_string())
        top = log.reindex(log["change"].abs().sort_values(ascending=False).index).head(args.top)
        print(f"\nTop {args.top} changes:\n{top.round(1).to_string(index=False)}")
    print(f"\nLog written to {args.out}")


if __name__ == "__main__":
    main()
