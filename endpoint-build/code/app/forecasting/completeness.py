"""Data completeness check, run before every training.

Sales exports are pulled while the latest days are still syncing in: the
21-Sep production model was trained on a file whose last 9 days held ~0-117
units/day instead of the real ~400-900. A model trained on that sees "no
demand" everywhere. This module:

1. drops the last day of the file (it is almost always partial), and
2. refuses to train if the store total over the most recent days is below
   `min_ratio` x the median daily total of the weeks before.

All thresholds are configurable (app/config.py, env vars FORECAST_*).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.config import (
    FORECAST_COMPLETENESS_MIN_RATIO,
    FORECAST_COMPLETENESS_RECENT_DAYS,
    FORECAST_COMPLETENESS_TRAILING_DAYS,
    FORECAST_DROP_LAST_DAY,
)
from app.utils.logger import get_logger
from app.forecasting.data import SalesData, TrainingRefusedError

logger = get_logger(__name__)


class DataIncompleteError(TrainingRefusedError):
    """Latest days of the training data look incomplete; refusing to train."""


@dataclass
class CompletenessConfig:
    drop_last_day: bool = FORECAST_DROP_LAST_DAY
    recent_days: int = FORECAST_COMPLETENESS_RECENT_DAYS
    trailing_days: int = FORECAST_COMPLETENESS_TRAILING_DAYS
    min_ratio: float = FORECAST_COMPLETENESS_MIN_RATIO
    enabled: bool = True


@dataclass
class CompletenessReport:
    dropped_day: pd.Timestamp | None
    recent_totals: dict = field(default_factory=dict)
    trailing_median: float | None = None
    ratio: float | None = None
    passed: bool = True
    skipped_reason: str | None = None


def check_completeness(sales: SalesData, config: CompletenessConfig | None = None) -> tuple[SalesData, CompletenessReport]:
    """Return (sales with the partial last day removed, report).

    Raises DataIncompleteError if the recent days' average store total is
    below `min_ratio` x the trailing median.
    """

    config = config or CompletenessConfig()
    if not config.enabled:
        return sales, CompletenessReport(dropped_day=None, skipped_reason="disabled")

    dropped = None
    if config.drop_last_day:
        dropped = sales.end_date
        sales = sales.truncate(dropped - pd.Timedelta(days=1))
        logger.info("Dropped last (possibly partial) day %s from training data", dropped.date())

    report = CompletenessReport(dropped_day=dropped)
    totals = sales.store_totals()
    needed = config.recent_days + 7
    if len(totals) < needed:
        report.skipped_reason = f"only {len(totals)} day(s) of data; need {needed} to judge completeness"
        logger.warning("Completeness check skipped: %s", report.skipped_reason)
        return sales, report

    recent = totals.iloc[-config.recent_days :]
    trailing = totals.iloc[-(config.recent_days + config.trailing_days) : -config.recent_days]
    median = float(trailing.median())
    report.recent_totals = {d.date().isoformat(): float(v) for d, v in recent.items()}
    report.trailing_median = median

    if median <= 0:
        report.skipped_reason = "trailing median is 0; cannot judge completeness"
        logger.warning("Completeness check skipped: %s", report.skipped_reason)
        return sales, report

    report.ratio = float(recent.mean() / median)
    if report.ratio < config.min_ratio:
        report.passed = False
        days = ", ".join(f"{d}: {v:.0f}" for d, v in report.recent_totals.items())
        raise DataIncompleteError(
            f"Training data looks incomplete: the last {config.recent_days} day(s) average "
            f"{recent.mean():.0f} units/day ({days}), which is {report.ratio:.0%} of the median "
            f"{median:.0f} units/day over the {len(trailing)} days before "
            f"(minimum {config.min_ratio:.0%}). Re-export once those days have fully synced, "
            f"or adjust FORECAST_COMPLETENESS_* in app/config.py."
        )

    logger.info(
        "Completeness check passed: last %d day(s) at %.0f%% of trailing median (%.0f units/day)",
        config.recent_days,
        report.ratio * 100,
        median,
    )
    return sales, report
