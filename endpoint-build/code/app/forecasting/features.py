"""Feature builder shared by training AND prediction.

Training rows and prediction rows come from one definition, so the two
paths cannot disagree (a one-day offset between them, "train/serve skew",
silently ruins a forecast):

    the row for (product, date t) describes day t itself
    (weekday/calendar of t) using only sales strictly before t
    (lag1 = sales on t-1, rolling7 = mean of t-7..t-1),
    and the training target of that row is the sales on day t.

Prediction for day t appends an empty row t to the history and calls the
same function, so the two paths cannot drift apart (see
tests/test_features.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.forecasting.data import SalesData

LAGS = (1, 2, 3, 7, 14)
ROLLING_WINDOWS = (7, 14, 28)
# Longest look-back any feature needs; prediction only has to pass this many
# trailing history rows to build_history_features.
MAX_LOOKBACK_DAYS = 28

CALENDAR_FEATURES = ["dayOfWeek", "dayOfMonth", "month", "weekOfYear", "isWeekend"]
HISTORY_FEATURES = [
    "lag1",
    "lag2",
    "lag3",
    "lag7",
    "lag14",
    "rolling7",
    "rolling14",
    "rollingStd7",
    "diff1",
    "diff7",
    "growth7",
]
# Extra history features used by the global model.
EXTRA_HISTORY_FEATURES = [
    "rolling28",
    "sameWeekdayMean4",  # mean of t-7, t-14, t-21, t-28: the baseline's own signal
    "saleDayShare28",  # share of the last 28 days with any sale
    "daysSinceLastSale",  # capped at MAX_LOOKBACK_DAYS + 1
]


def demand_matrix(sales: SalesData, column: str = "demand") -> pd.DataFrame:
    """dates x variantId walk-in demand (or `column`, e.g. demandClean).

    - a calendar day with no row for a product (after its first row) = 0 sold
    - a stock-out day = NaN (demand unknown, not "no demand")
    - days before a product's first row = NaN (it did not exist yet)
    """

    daily = sales.daily
    full_range = pd.date_range(sales.start_date, sales.end_date, freq="D")
    demand = daily.pivot(index="date", columns="variantId", values=column).reindex(full_range).fillna(0.0)

    stockout = daily.pivot(index="date", columns="variantId", values="isStockout").reindex(full_range)
    demand = demand.mask(stockout.fillna(False).astype(bool))

    first_seen = daily.groupby("variantId")["date"].min()
    before_start = demand.index.to_numpy()[:, None] < first_seen.reindex(demand.columns).to_numpy()[None, :]
    demand = demand.mask(before_start)

    demand.index.name = "date"
    demand.columns.name = "variantId"
    return demand.astype(float)


HOLIDAY_FEATURES = ["isHoliday", "daysToHoliday", "daysAfterHoliday"]
HOLIDAY_WINDOW_DAYS = 14


def holiday_features(dates, holidays=None) -> pd.DataFrame:
    """Per date: isHoliday (0/1), days until the next holiday and days since
    the last one, both capped at HOLIDAY_WINDOW_DAYS + 1 ("not near one").
    With no holidays.csv every date gets 0 / cap / cap."""

    dates = pd.DatetimeIndex(dates)
    cap = HOLIDAY_WINDOW_DAYS + 1
    out = pd.DataFrame({"isHoliday": 0, "daysToHoliday": cap, "daysAfterHoliday": cap}, index=dates)
    if holidays is None or len(holidays) == 0:
        return out

    marked = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(holidays))))
    values = marked.to_numpy()
    d = dates.to_numpy()
    nxt = np.searchsorted(values, d, side="left")  # first holiday >= date
    prev = np.searchsorted(values, d, side="right") - 1  # last holiday <= date
    to_next = np.where(nxt < len(values), (values[np.minimum(nxt, len(values) - 1)] - d) / np.timedelta64(1, "D"), cap)
    after_prev = np.where(prev >= 0, (d - values[np.maximum(prev, 0)]) / np.timedelta64(1, "D"), cap)
    out["isHoliday"] = (to_next == 0).astype(int)
    out["daysToHoliday"] = np.minimum(to_next, cap).astype(int)
    out["daysAfterHoliday"] = np.minimum(after_prev, cap).astype(int)
    return out


def calendar_features(dates) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    return pd.DataFrame(
        {
            "dayOfWeek": dates.dayofweek,
            "dayOfMonth": dates.day,
            "month": dates.month,
            "weekOfYear": dates.isocalendar().week.to_numpy().astype(int),
            "isWeekend": (dates.dayofweek >= 5).astype(int),
        },
        index=dates,
    )


def build_history_features(matrix: pd.DataFrame, target_dates=None) -> pd.DataFrame:
    """Long frame indexed by (date, variantId) with the lag/rolling features
    for each target date, computed only from matrix rows BEFORE that date.

    `matrix` is dates x variantId (see demand_matrix). Target dates must be
    rows of `matrix`; their own values are never read (a prediction row can
    be NaN).
    """

    target_dates = matrix.index if target_dates is None else pd.DatetimeIndex(target_dates)
    past = matrix.shift(1)  # row t now holds sales of t-1

    frames = {f"lag{k}": matrix.shift(k) for k in LAGS}
    for window in ROLLING_WINDOWS:
        frames[f"rolling{window}"] = past.rolling(window, min_periods=1).mean()
    frames["rollingStd7"] = past.rolling(7, min_periods=2).std()
    frames["sameWeekdayMean4"] = (
        pd.concat([matrix.shift(7 * w) for w in range(1, 5)], keys=range(4)).groupby(level=1).mean()
    ).reindex(matrix.index)
    sold = (past > 0).astype(float).where(past.notna())
    frames["saleDayShare28"] = sold.rolling(MAX_LOOKBACK_DAYS, min_periods=1).mean()
    frames["daysSinceLastSale"] = _days_since_last_sale(past)

    selected = {name: frame.loc[target_dates] for name, frame in frames.items()}
    long = pd.concat({name: frame.stack(future_stack=True) for name, frame in selected.items()}, axis=1)
    long.index.names = ["date", "variantId"]

    long["diff1"] = long["lag1"] - long["lag2"]
    long["diff7"] = long["lag7"] - long["lag14"]
    long["growth7"] = long["rolling7"] / (long["rolling14"] + 0.001)
    return long


def _days_since_last_sale(past: pd.DataFrame) -> pd.DataFrame:
    """Days between each row's date and the most recent earlier sale
    (`past` is already shifted by one day), capped at MAX_LOOKBACK_DAYS + 1
    so the training path and the 28-day prediction window agree."""

    position = pd.DataFrame(
        np.where(past.to_numpy() > 0, np.arange(len(past))[:, None], np.nan), index=past.index, columns=past.columns
    )
    last = position.ffill()
    days = np.arange(len(past))[:, None] - last.to_numpy() + 1
    days = np.where(np.isnan(days), MAX_LOOKBACK_DAYS + 1, np.minimum(days, MAX_LOOKBACK_DAYS + 1))
    return pd.DataFrame(days, index=past.index, columns=past.columns)


def build_features(matrix: pd.DataFrame, target_dates=None) -> pd.DataFrame:
    """History + calendar features, indexed by (date, variantId)."""

    features = build_history_features(matrix, target_dates)
    calendar = calendar_features(features.index.get_level_values("date"))
    for column in CALENDAR_FEATURES:
        features[column] = calendar[column].to_numpy()
    return features


def training_rows(matrix: pd.DataFrame) -> pd.DataFrame:
    """Every (product, day) with a known target: features describe day t
    from days < t; `target` is the demand on day t. Days before a product
    existed and stock-out days (NaN demand) are not training rows."""

    features = build_features(matrix)
    target = matrix.stack(future_stack=True)
    target.index.names = ["date", "variantId"]
    features["target"] = target.reindex(features.index).to_numpy()
    return features[features["target"].notna()]


def prediction_rows(history: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """Feature rows for `target_date`, the day right after the history ends.

    Uses exactly the same builder as training_rows, on the history's last
    MAX_LOOKBACK_DAYS rows plus an empty row for the target date.
    """

    target_date = pd.Timestamp(target_date)
    expected = history.index[-1] + pd.Timedelta(days=1)
    if target_date != expected:
        raise ValueError(f"prediction_rows expects the next day {expected.date()}, got {target_date.date()}")

    window = history.iloc[-MAX_LOOKBACK_DAYS:]
    extended = pd.concat([window, pd.DataFrame(np.nan, index=[target_date], columns=history.columns)])
    extended.index.name = history.index.name
    return build_features(extended, [target_date])
