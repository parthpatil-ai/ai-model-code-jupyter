"""The forecasting model behind /train-model, /predict and /evaluate-model.

    model = ForecastModel()
    model.load_auxiliary_files(folder)   # catalog / variant_mapping / holidays
    model.train(sales_df)                # or model.fit(SalesData)
    model.predict(cart_id, "05-10-2026") # [{"variant_id", "product_name", "quantity"}]

Pipeline:
  1. load + validate (forecasting/data.py): format errors with CSV line
     numbers, bulk lines kept apart, duplicates summed
  2. completeness check (forecasting/completeness.py): drops the partial last
     day, refuses data whose latest days are mostly missing
  3. training-data cleaning (forecasting/cleaning.py): late-posted spikes
     spread back over the zero days, Hampel cap per product; every change is
     kept in `cleaning_log`
  4. product list (forecasting/universe.py): replaced variants hand their
     history to the replacement, only recently sold / newly launched
     products are forecast, new items start at their category x price-band
     average
  5. forecast, by FORECAST_MODE:
     model    ONE global XGBoost model for all products
              (forecasting/global_model.py), Tweedie objective, features from
              forecasting/features.py, forecast day by day feeding back
              expected (unrounded) values
     typical  each product's typical walk-in level x weekday factor plus the
              average volume of its repeat orders (forecasting/typical.py);
              no model training
  6. output: whole units by default (FORECAST_ROUND_OUTPUT)
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from app.config import (
    FORECAST_MODE,
    FORECAST_HIGH_QUANTILE,
    FORECAST_N_JOBS,
    FORECAST_OUTPUT_HIGH_QTY,
    FORECAST_ROUND_OUTPUT,
    FORECAST_TUNE,
    FORECAST_TUNE_FOLDS,
    FORECAST_TUNE_ITERATIONS,
)
from app.forecasting.cleaning import CleaningConfig, clean_training_data
from app.forecasting.completeness import CompletenessConfig, CompletenessReport, check_completeness
from app.forecasting.data import DATE_FORMAT, SALES_REQUIRED_COLUMNS, SalesData, load_dataset_from_frame
from app.forecasting.features import demand_matrix
from app.forecasting.global_model import DEFAULT_PARAMS, FEATURES, GlobalEngine, TuningResult, tune
from app.forecasting.typical import TypicalConfig, typical_forecast
from app.forecasting.universe import (
    apply_variant_mapping,
    cold_start_rates,
    product_universe,
    products_with_catalog,
    start_dates,
    trim_before_start,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_FORECAST_HORIZON_DAYS = 730
FORECAST_MODES = ("model", "typical")


class ForecastModel:
    REQUIRED_COLUMNS = SALES_REQUIRED_COLUMNS
    features = FEATURES

    def __init__(
        self,
        completeness: CompletenessConfig | None = None,
        cleaning: CleaningConfig | None = None,
        round_output: bool = FORECAST_ROUND_OUTPUT,
        tune_params: bool = FORECAST_TUNE,
        tune_iterations: int = FORECAST_TUNE_ITERATIONS,
        tune_folds: int = FORECAST_TUNE_FOLDS,
        params: dict | None = None,
        n_jobs: int = FORECAST_N_JOBS,
        forecast_mode: str = FORECAST_MODE,
        typical: TypicalConfig | None = None,
        output_high_qty: bool = FORECAST_OUTPUT_HIGH_QTY,
    ):
        if forecast_mode not in FORECAST_MODES:
            raise ValueError(f"forecast_mode must be one of {FORECAST_MODES}, got {forecast_mode!r}")
        self.completeness = completeness or CompletenessConfig()
        self.cleaning = cleaning or CleaningConfig()
        self.round_output = round_output
        self.tune_params = tune_params
        self.tune_iterations = tune_iterations
        self.tune_folds = tune_folds
        self.params = dict(params or DEFAULT_PARAMS)
        self.n_jobs = n_jobs
        self.forecast_mode = forecast_mode
        self.typical = typical or TypicalConfig(high_quantile=FORECAST_HIGH_QUANTILE)
        self.output_high_qty = output_high_qty
        self.aux_dir: Path | None = None
        self.completeness_report: CompletenessReport | None = None
        self.cleaning_log: pd.DataFrame | None = None
        self.tuning: TuningResult | None = None
        self.training_seconds: float | None = None
        self.is_trained = False

    # -- API entry points ---------------------------------------------------

    def load_auxiliary_files(self, directory) -> None:
        """Called by model_service with the training file's folder, so
        catalog.csv / variant_mapping.csv / holidays.csv next to it are used."""

        self.aux_dir = Path(directory) if directory is not None else None

    def train(self, data: pd.DataFrame) -> None:
        self.fit(load_dataset_from_frame(data, self.aux_dir))

    def predict(self, cart_id: str, date: str) -> list[dict]:
        self._require_trained()
        target = pd.to_datetime(date, format=DATE_FORMAT)
        if target <= self.last_date:
            # A real one-day-ahead forecast from the history before that date,
            # not the recorded sales (echoing those back would make in-range
            # "predictions" look perfect).
            if target <= self.history.index[0]:
                raise ValueError(f"No history before {target.date()} to forecast from")
            frame = self._one_step(self.history.loc[: target - pd.Timedelta(days=1)], target)
        else:
            frame = self.forecast_frame([target])
        return self._records(frame, [date])[date]

    def predict_dates(self, cart_id: str, dates: list[str]) -> dict[str, list[dict]]:
        """Forecasts for several dates (used by /evaluate-model). Dates after
        the training data are forecast day by day from the last training
        day; dates inside it are forecast one day ahead from the history
        before them."""

        self._require_trained()
        parsed = [pd.to_datetime(d, format=DATE_FORMAT) for d in dates]
        future = [ts for ts in parsed if ts > self.last_date]
        frames = [self.forecast_frame(future)] if future else []
        for ts in sorted({ts for ts in parsed if self.history.index[0] < ts <= self.last_date}):
            frames.append(self._one_step(self.history.loc[: ts - pd.Timedelta(days=1)], ts))
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["variantId", "date", "prediction"])
        return self._records(combined, dates)

    # -- library entry points (used by the backtest) ------------------------

    def fit(self, sales: SalesData) -> "ForecastModel":
        started = time.time()
        sales, self.completeness_report = check_completeness(sales, self.completeness)
        self.sales = sales  # the typical forecast applies the variant mapping itself
        sales = apply_variant_mapping(sales)
        cleaned, self.cleaning_log = clean_training_data(sales, self.cleaning)
        self.start_dates = start_dates(sales)
        self.history = trim_before_start(demand_matrix(cleaned, column="demandClean"), self.start_dates)
        self.products = products_with_catalog(sales.products, sales.catalog)
        self.catalog = sales.catalog
        self.holidays = None if sales.holidays is None else pd.DatetimeIndex(sales.holidays["date"])
        self.variant_mapping = sales.variant_mapping
        self.last_date = sales.end_date

        if self.forecast_mode == "typical":
            self.engine = None
            self.training_seconds = round(time.time() - started, 1)
            logger.info("Typical-demand forecast ready in %.1fs (%s..%s)", self.training_seconds, self.history.index[0].date(), self.last_date.date())
            self.is_trained = True
            return self

        if self.tune_params:
            actual = demand_matrix(sales, column="demand")
            self.tuning = tune(
                self.history,
                actual,
                self.products,
                n_iter=self.tune_iterations,
                n_folds=self.tune_folds,
                n_jobs=self.n_jobs,
                holidays=self.holidays,
            )
            self.params = self.tuning.best_params

        self.engine = GlobalEngine(self.params, n_jobs=self.n_jobs, holidays=self.holidays).fit(self.history, self.products)
        self.training_seconds = round(time.time() - started, 1)
        logger.info("Model trained in %.1fs (%s..%s)", self.training_seconds, self.history.index[0].date(), self.last_date.date())
        self.is_trained = True
        return self

    def forecast_frame(self, dates) -> pd.DataFrame:
        """Expected units per (variantId, date) for dates after the training
        data, forecast day by day from the last training day.

        Each day's expected value (e.g. 0.4) is fed back as history for the
        next day as is. Rounding inside the loop would turn every slow
        seller's 0.4 into 0, and the 0 would then feed the next day's lags,
        so slow sellers collapse to 0 after a day or two."""

        self._require_trained()
        dates = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(dates))))
        if len(dates) == 0:
            return pd.DataFrame(columns=["variantId", "date", "prediction"])
        if dates[0] <= self.last_date:
            raise ValueError(f"forecast_frame only forecasts after {self.last_date.date()}")
        horizon = (dates[-1] - self.last_date).days
        if horizon > MAX_FORECAST_HORIZON_DAYS:
            raise ValueError(f"Refusing to forecast {horizon} days past the training data")

        if self.forecast_mode == "typical":
            return typical_forecast(self.sales, dates, self.typical)

        days = pd.date_range(self.last_date + pd.Timedelta(days=1), dates[-1], freq="D")
        wide = self.engine.forecast(self.history, days).loc[dates]

        frames = []
        for day in dates:
            universe, new_items = product_universe(
                self.history, self.last_date, day, self.catalog, self.variant_mapping
            )
            values = wide.loc[day]
            values = values[values.index.isin(universe)]
            frames.append(pd.DataFrame({"variantId": values.index.astype("int64"), "date": day, "prediction": values.to_numpy()}))
            if new_items:
                rates = cold_start_rates(new_items, self.history, self.last_date, self.products)
                frames.append(pd.DataFrame({"variantId": list(rates), "date": day, "prediction": list(rates.values())}))
        return pd.concat(frames, ignore_index=True)

    # -- internals ----------------------------------------------------------

    def _one_step(self, history: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
        if self.forecast_mode == "typical":
            return typical_forecast(self.sales.truncate(day - pd.Timedelta(days=1)), [day], self.typical)
        values = self.engine.one_step(history, day)
        return pd.DataFrame({"variantId": values.index.astype("int64"), "date": day, "prediction": values.to_numpy()})

    def _records(self, frame: pd.DataFrame, dates: list[str]) -> dict[str, list[dict]]:
        names = self.products["productName"]
        out = {d: [] for d in dates}
        lookup = {pd.to_datetime(d, format=DATE_FORMAT): d for d in dates}
        for row in frame.itertuples(index=False):
            key = lookup.get(pd.Timestamp(row.date))
            if key is None:
                continue
            record = {
                "variant_id": str(int(row.variantId)),
                "product_name": names.get(int(row.variantId), str(row.variantId)),
                "quantity": self._output_quantity(row.prediction),
            }
            if self.output_high_qty and hasattr(row, "high_qty"):
                record["high_quantity"] = self._output_quantity(row.high_qty)
            out[key].append(record)
        return out

    def _output_quantity(self, value: float) -> int | float:
        """API output: whole units by default; with
        round_output=False the expected value (2 decimals)."""

        value = max(float(value), 0.0)
        return int(round(value)) if self.round_output else round(value, 2)

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
