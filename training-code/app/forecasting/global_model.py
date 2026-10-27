"""One global forecasting model for all products.

Instead of ~850 tiny models (one per product, ~200 rows each), one XGBoost
model learns from every product's history at once (~180k rows):

- objective "reg:tweedie": built for sales data that is mostly 0 with
  occasional positive counts; it predicts an expected value (mean), never
  negative, and does not pull slow sellers toward 0 like the old squared
  error + MAE-tuned setup did
- features: product (as a category), product category, price (mrp),
  weekday, plus the shared lag/rolling features. dayOfMonth, weekOfYear and
  month are gone -- with 7 months of data they mostly memorised which dates
  had spikes
- tuning: a small random search scored by item WAPE on time-ordered folds
  that replay the real use: train on data up to a Sunday, forecast the next
  7 days day by day, score the week
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.utils.logger import get_logger
from app.forecasting.features import (
    EXTRA_HISTORY_FEATURES,
    HISTORY_FEATURES,
    HOLIDAY_FEATURES,
    holiday_features,
    prediction_rows,
    training_rows,
)

logger = get_logger(__name__)

CALENDAR = ["dayOfWeek", "isWeekend"] + HOLIDAY_FEATURES
NUMERIC_FEATURES = CALENDAR + HISTORY_FEATURES + EXTRA_HISTORY_FEATURES + ["mrp"]
CATEGORICAL_FEATURES = ["product", "category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Chosen by tune() on cart 50 data up to 02-Aug-2026 (8 candidates, 3 weekly
# folds, mean item WAPE 88.2) -- before the first backtest week, so the
# backtest never scores weeks that were used for tuning.
DEFAULT_PARAMS = {
    "n_estimators": 700,
    "max_depth": 8,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0,
    "tweedie_variance_power": 1.1,
}

SEARCH_SPACE = {
    "n_estimators": [200, 400, 700],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 20],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.7, 0.9],
    "reg_lambda": [1.0, 5.0],
    "tweedie_variance_power": [1.1, 1.3, 1.5],
}


@dataclass
class TuningResult:
    best_params: dict
    trials: list = field(default_factory=list)  # (params, mean WAPE, per-fold WAPE)
    seconds: float = 0.0


def static_features(products: pd.DataFrame, variant_ids) -> pd.DataFrame:
    attrs = products.reindex(pd.Index(variant_ids, name="variantId"))
    return pd.DataFrame(
        {"mrp": attrs["mrp"].astype(float), "category": attrs["category"].astype("string").fillna("unknown")},
        index=attrs.index,
    )


class GlobalEngine:
    def __init__(self, params: dict | None = None, n_jobs: int = -1, random_state: int = 42, holidays=None):
        self.params = dict(params or DEFAULT_PARAMS)
        self.holidays = pd.DatetimeIndex([]) if holidays is None else pd.DatetimeIndex(holidays)
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model: XGBRegressor | None = None

    # -- design matrix ------------------------------------------------------

    def _design(self, rows: pd.DataFrame) -> pd.DataFrame:
        """rows: features indexed by (date, variantId) -> model input."""

        variant_ids = rows.index.get_level_values("variantId")
        static = self.static.reindex(variant_ids)
        X = rows[[c for c in NUMERIC_FEATURES if c not in HOLIDAY_FEATURES and c != "mrp"]].copy()
        calendar = holiday_features(rows.index.get_level_values("date"), self.holidays)
        for column in HOLIDAY_FEATURES:
            X[column] = calendar[column].to_numpy()
        X["mrp"] = static["mrp"].to_numpy()
        X["product"] = pd.Categorical(variant_ids, categories=self.product_categories)
        X["category"] = pd.Categorical(static["category"].to_numpy(), categories=self.category_categories)
        return X[FEATURES]

    def _new_regressor(self, params: dict) -> XGBRegressor:
        return XGBRegressor(
            **params,
            objective="reg:tweedie",
            tree_method="hist",
            enable_categorical=True,
            max_cat_to_onehot=1,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    # -- training / prediction ----------------------------------------------

    def fit(self, history: pd.DataFrame, products: pd.DataFrame, params: dict | None = None) -> "GlobalEngine":
        if params is not None:
            self.params = dict(params)
        self.static = static_features(products, history.columns)
        self.product_categories = list(history.columns)
        self.category_categories = sorted(set(self.static["category"]) | {"unknown"})

        rows = training_rows(history)
        X = self._design(rows)
        started = time.time()
        self.model = self._new_regressor(self.params).fit(X, rows["target"].to_numpy())
        logger.info(
            "Global model: %d rows, %d products, %d features, fitted in %.1fs",
            len(X),
            len(self.product_categories),
            X.shape[1],
            time.time() - started,
        )
        return self

    def one_step(self, history: pd.DataFrame, day: pd.Timestamp) -> pd.Series:
        rows = prediction_rows(history, day)
        values = self.model.predict(self._design(rows))
        return pd.Series(np.clip(values, 0.0, None), index=rows.index.get_level_values("variantId"))

    def forecast(self, history: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        """Expected values for consecutive `days` right after `history`."""

        history = history.copy()
        out = {}
        for day in days:
            values = self.one_step(history, day)
            history.loc[day] = values.reindex(history.columns)
            out[day] = values
        return pd.DataFrame(out).T  # dates x variantId


def week_wape(forecast: pd.DataFrame, actual: pd.DataFrame) -> float:
    """Item-level WAPE of one week: per-product weekly totals."""

    predicted = forecast.sum(axis=0)
    observed = actual.reindex(index=forecast.index, columns=forecast.columns).sum(axis=0, min_count=1).fillna(0.0)
    total = observed.sum()
    return float((predicted - observed).abs().sum() / total * 100) if total > 0 else np.nan


def tuning_folds(history: pd.DataFrame, n_folds: int) -> list[tuple[pd.Timestamp, pd.DatetimeIndex]]:
    """The last `n_folds` Mon-Sun weeks inside the training data, oldest
    first, as (cutoff Sunday, forecast week)."""

    end = history.index[-1]
    last_sunday = end - pd.Timedelta(days=(end.dayofweek - 6) % 7)
    folds = []
    for i in range(n_folds):
        week_end = last_sunday - pd.Timedelta(weeks=i)
        cutoff = week_end - pd.Timedelta(days=7)
        if cutoff - history.index[0] < pd.Timedelta(days=42):
            break
        folds.append((cutoff, pd.date_range(cutoff + pd.Timedelta(days=1), week_end, freq="D")))
    return list(reversed(folds))


def tune(
    history: pd.DataFrame,
    actual: pd.DataFrame,
    products: pd.DataFrame,
    n_iter: int = 8,
    n_folds: int = 3,
    n_jobs: int = -1,
    random_state: int = 42,
    holidays=None,
) -> TuningResult:
    """Random search over SEARCH_SPACE, scored by mean item WAPE over
    time-ordered folds. `history` is the (cleaned) training matrix;
    `actual` the uncleaned walk-in demand the forecast is judged on."""

    started = time.time()
    rng = np.random.default_rng(random_state)
    candidates = [dict(DEFAULT_PARAMS)] + [
        {key: values[rng.integers(len(values))] for key, values in SEARCH_SPACE.items()} for _ in range(n_iter - 1)
    ]
    folds = tuning_folds(history, n_folds)
    if not folds:
        logger.warning("Not enough history to tune; using default parameters")
        return TuningResult(dict(DEFAULT_PARAMS), seconds=time.time() - started)

    trials = []
    for params in candidates:
        scores = []
        for cutoff, week in folds:
            engine = GlobalEngine(params, n_jobs=n_jobs, random_state=random_state, holidays=holidays)
            engine.fit(history.loc[:cutoff], products)
            scores.append(week_wape(engine.forecast(history.loc[:cutoff], week), actual))
        trials.append((params, float(np.nanmean(scores)), scores))
        logger.info("tuning: WAPE %.1f %s", trials[-1][1], params)

    best = min(trials, key=lambda t: t[1])
    logger.info("tuning done in %.0fs: best WAPE %.1f with %s", time.time() - started, best[1], best[0])
    return TuningResult(best[0], trials, time.time() - started)
