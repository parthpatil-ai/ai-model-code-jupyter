"""Central configuration for the Inventory Forecasting Engine."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

STORAGE_DIR = BASE_DIR / "storage"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
LOG_DIR = BASE_DIR / "logs"

# Sub-directory names created under storage/{cart_id}/
MODELS_SUBDIR = "models"
TRAINING_DATA_SUBDIR = "training_data"
PREDICTIONS_SUBDIR = "predictions"
INVENTORY_SUBDIR = "inventory"
OUTPUT_SUBDIR = "output"
EVALUATION_SUBDIR = "evaluation"

MODEL_FILENAME = "model.pkl"

DATE_FORMAT = "%d-%m-%Y"
DATE_FORMAT_LABEL = "DD-MM-YYYY"

CART_ID_PATTERN = r"^[A-Za-z0-9_-]+$"

INVENTORY_REQUIRED_COLUMNS = [
    "variant_id",
    "variant_name",
    "quantity",
]

PREDICTION_COLUMNS = [
    "variant_id",
    "product_name",
    "quantity",
]

RECOMMENDATION_COLUMNS = [
    "variant_id",
    "product_name",
    "store_id",
    "store_name",
    "store_inventory",
    "predicted_quantity",
    "recommended_quantity",
    "warehouse_inventory",
    "warehouse_status",
    "stn_quantity",
    "carried_over_quantity",
]

EVALUATION_COLUMNS = [
    "date",
    "variant_id",
    "product_name",
    "actual_quantity",
    "predicted_quantity",
    "error",
    "abs_error",
]

# ---------------------------------------------------------------------------
# Forecasting (app/forecasting/). Every setting can be overridden with an
# environment variable of the same name.
# ---------------------------------------------------------------------------

# How /train-model forecasts:
#   model    one global XGBoost model for all products (default)
#   typical  each product's typical walk-in level x weekday factor, plus the
#            average volume of its repeat (regular-customer / bulk) orders;
#            no model training, so /train-model takes seconds
FORECAST_MODE = os.environ.get("FORECAST_MODE", "model")

# Completeness check (forecasting/completeness.py): drop the last day of the
# training file, then refuse to train if the average store total over the
# most recent RECENT_DAYS days is below MIN_RATIO x the median daily total
# of the TRAILING_DAYS days before them.
FORECAST_DROP_LAST_DAY = os.environ.get("FORECAST_DROP_LAST_DAY", "1") == "1"
FORECAST_COMPLETENESS_RECENT_DAYS = int(os.environ.get("FORECAST_COMPLETENESS_RECENT_DAYS", "3"))
FORECAST_COMPLETENESS_TRAILING_DAYS = int(os.environ.get("FORECAST_COMPLETENESS_TRAILING_DAYS", "28"))
FORECAST_COMPLETENESS_MIN_RATIO = float(os.environ.get("FORECAST_COMPLETENESS_MIN_RATIO", "0.30"))

# Bulk lines (forecasting/data.py): without an isBulk column, a row with at
# least this many units per transaction is a bulk/B2B line.
FORECAST_BULK_UNITS_PER_TRANSACTION = float(os.environ.get("FORECAST_BULK_UNITS_PER_TRANSACTION", "20"))

# Training-data cleaning (forecasting/cleaning.py). Changes TRAINING data
# only; evaluation always scores against the recorded sales.
FORECAST_CLEANING_ENABLED = os.environ.get("FORECAST_CLEANING_ENABLED", "1") == "1"
#  - late posting: a run of >= MIN_ZERO_DAYS whole-store zero days followed by
#    a day >= SPIKE_RATIO x the median non-zero store day of the previous
#    28 days is spread evenly over the zero days + spike day, per product
FORECAST_LATE_POSTING_MIN_ZERO_DAYS = int(os.environ.get("FORECAST_LATE_POSTING_MIN_ZERO_DAYS", "1"))
FORECAST_LATE_POSTING_SPIKE_RATIO = float(os.environ.get("FORECAST_LATE_POSTING_SPIKE_RATIO", "3.0"))
#  - Hampel cap per product: values above median + K x 1.4826 x MAD (vs the
#    same weekday of the last 4 weeks AND the +/-7-day window; the higher cap
#    wins) are lowered to the cap. The cap never goes below the product's
#    FLOOR_QUANTILE of non-zero days (0.99: lower values trimmed normal peaks
#    of intermittent products) or FLOOR_MEAN_MULTIPLIER x its 28-day mean.
FORECAST_CAP_K = float(os.environ.get("FORECAST_CAP_K", "3.0"))
FORECAST_CAP_REFERENCE = os.environ.get("FORECAST_CAP_REFERENCE", "both")  # weekday | window | both
FORECAST_CAP_FLOOR_QUANTILE = float(os.environ.get("FORECAST_CAP_FLOOR_QUANTILE", "0.99"))
FORECAST_CAP_FLOOR_MEAN_MULTIPLIER = float(os.environ.get("FORECAST_CAP_FLOOR_MEAN_MULTIPLIER", "3.0"))

# Global model (forecasting/global_model.py): tune hyper-parameters on every
# /train-model run (random search of TUNE_ITERATIONS candidates, scored by
# item WAPE over TUNE_FOLDS time-ordered weekly folds), or use the defaults
# when FORECAST_TUNE=0. FORECAST_N_JOBS = XGBoost threads (-1 = all cores).
FORECAST_TUNE = os.environ.get("FORECAST_TUNE", "1") == "1"
FORECAST_TUNE_ITERATIONS = int(os.environ.get("FORECAST_TUNE_ITERATIONS", "8"))
FORECAST_TUNE_FOLDS = int(os.environ.get("FORECAST_TUNE_FOLDS", "3"))
FORECAST_N_JOBS = int(os.environ.get("FORECAST_N_JOBS", "-1"))

# Product list (forecasting/universe.py): forecast products sold in the last
# ACTIVE_DAYS days plus newly launched active catalog items; delisted and
# replaced variants are left out.
FORECAST_ACTIVE_DAYS = int(os.environ.get("FORECAST_ACTIVE_DAYS", "28"))

# Output: expected values are kept unrounded inside the forecast and rounded
# to whole units in /predict unless ROUND_OUTPUT=0. HIGH_QUANTILE is the
# coverage level of the optional high_quantity output (the product's P75
# week); OUTPUT_HIGH_QTY=1 adds it to each /predict record.
FORECAST_ROUND_OUTPUT = os.environ.get("FORECAST_ROUND_OUTPUT", "1") == "1"
FORECAST_HIGH_QUANTILE = float(os.environ.get("FORECAST_HIGH_QUANTILE", "0.75"))
FORECAST_OUTPUT_HIGH_QTY = os.environ.get("FORECAST_OUTPUT_HIGH_QTY", "0") == "1"

LOG_LEVEL = "INFO"
APP_NAME = "inventory-forecasting-engine"

# Maps cart_id -> human-readable store name, for reports/emails. Add entries
# here as more carts/stores come into scope.
STORE_NAME_MAP = {
    "50": "Khandwa Naka Store",
}

# Temporary scope-down: /ai-restocking only considers these variant_ids
# for now, ignoring everything else in the prediction/inventory data.
ACTIVE_VARIANT_IDS = [
    "520",
    "1552",
    "477",
    "863",
    "9002",
    "1299",
    "1487",
    "1300",
    "3269",
    "9001",
]

# Facility backend integration: /ai-restocking reads live cart inventory
# from here and forwards its restock recommendations here to auto-create a
# stock transfer note (STN).
FACILITY_API_BASE_URL = os.environ.get("FACILITY_API_BASE_URL", "http://13.200.182.203:3333")
FACILITY_CART_INVENTORY_ENDPOINT = "/cart/{cart_id}/inventory"
FACILITY_INVENTORY_TIMEOUT_SECONDS = 10
FACILITY_CREATE_STN_ENDPOINT = "/facility/create-stn"
FACILITY_STN_FROM_CART_ID = int(os.environ.get("FACILITY_STN_FROM_CART_ID", "1"))
FACILITY_STN_TIMEOUT_SECONDS = int(os.environ.get("FACILITY_STN_TIMEOUT_SECONDS", "20"))

# /ai-restocking checks a proposed STN's total worth (unit_price * quantity,
# summed across line items) against this before placing it -- see
# _fetch_maximum_worth / _apply_worth_cap in inventory_service.py.
FACILITY_MAXIMUM_STN_WORTH_ENDPOINT = "/stn/{cart_id}/maximum-worth"
FACILITY_MAXIMUM_WORTH_TIMEOUT_SECONDS = 10

# Leftover quantity trimmed from a worth-capped STN is persisted here (one
# file per cart, under storage/{cart_id}/output/) and folded into that
# cart's next /predict call by _consume_carry_over in prediction_service.py.
CARRY_OVER_FILENAME = "carry_over.csv"
CARRY_OVER_COLUMNS = ["variant_id", "product_name", "quantity"]

# Firebase auth used to obtain a short-lived ID token for the facility
# backend. Credentials come from env vars, never hardcoded here, since this
# file is committed to git.
FIREBASE_SIGN_IN_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "")
FACILITY_AUTH_EMAIL = os.environ.get("FACILITY_AUTH_EMAIL", "")
FACILITY_AUTH_PASSWORD = os.environ.get("FACILITY_AUTH_PASSWORD", "")
FACILITY_AUTH_TIMEOUT_SECONDS = 5
# Refresh the cached token this many seconds before its actual expiry.
FACILITY_AUTH_REFRESH_MARGIN_SECONDS = 60

# S3 archival for generated predictions. Auth is via the EC2 instance's
# attached IAM role -- boto3 picks that up automatically, no keys needed
# here. Region is left blank by default so boto3 resolves it itself
# (AWS_DEFAULT_REGION env var / ~/.aws/config on the instance); set
# S3_REGION explicitly only if that resolution doesn't work.
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "yatrikart-inventory-ai-dev-1")
S3_REGION = os.environ.get("S3_REGION", "")
S3_PREDICTION_KEY_PREFIX = "prediction/daily"
S3_REPLENISHMENT_KEY_PREFIX = "replenishment/daily"
