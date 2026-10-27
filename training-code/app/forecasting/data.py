"""Loader + validator for the cleaned training data format.

Main training CSV, one row per product per day:
    required: variantId, productName, date (DD-MM-YYYY), quantitySold
    optional: isBulk (0/1), isStockout (0/1), category, mrp,
              transactions (number of bills the units came from)

Bulk/B2B lines: if the file has isBulk, those rows are bulk. If it has no
isBulk but has transactions, a row with quantitySold / transactions >=
FORECAST_BULK_UNITS_PER_TRANSACTION (default 20) is treated as bulk.

Optional companion files, auto-discovered next to the training CSV:
    catalog.csv          variant_id, product_name, category, mrp,
                         launch_date, delist_date, active (0/1)
    variant_mapping.csv  old_variant_id, new_variant_id, switch_date
    holidays.csv         date, name

Everything downstream (completeness check, features, model, evaluation)
works from the normalized `SalesData` this module returns: one row per
(variantId, date) with walk-in demand, bulk quantity and a stock-out flag
kept in separate columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import FORECAST_BULK_UNITS_PER_TRANSACTION
from app.utils.logger import get_logger

logger = get_logger(__name__)

DATE_FORMAT = "%d-%m-%Y"

SALES_REQUIRED_COLUMNS = ["variantId", "productName", "date", "quantitySold"]
SALES_FLAG_COLUMNS = ["isBulk", "isStockout"]
SALES_ATTRIBUTE_COLUMNS = ["category", "mrp", "transactions"]

CATALOG_FILENAME = "catalog.csv"
CATALOG_COLUMNS = [
    "variant_id",
    "product_name",
    "category",
    "mrp",
    "launch_date",
    "delist_date",
    "active",
]

VARIANT_MAPPING_FILENAME = "variant_mapping.csv"
VARIANT_MAPPING_COLUMNS = ["old_variant_id", "new_variant_id", "switch_date"]

HOLIDAYS_FILENAME = "holidays.csv"
HOLIDAYS_COLUMNS = ["date", "name"]

_MAX_EXAMPLES = 5


class DataFormatError(ValueError):
    """Input file does not follow the agreed data format.

    Subclasses ValueError so the API layer's existing ValueError -> HTTP 400
    handling applies unchanged.
    """


class TrainingRefusedError(ValueError):
    """The input is well-formed but must not be trained on (for example the
    latest days look incomplete). The backtest records these as a refusal
    instead of a crash."""


@dataclass
class SalesData:
    """Normalized sales history for one cart.

    daily:    one row per (variantId, date) that appears in the source file.
              Columns: variantId (int), date (Timestamp), demand (walk-in
              units, bulk excluded), bulkQuantity, isStockout (bool),
              bulkSource ("" / "isBulk" / "transactions"), transactions.
    products: indexed by variantId. Columns: productName, category, mrp.
    """

    daily: pd.DataFrame
    products: pd.DataFrame
    catalog: pd.DataFrame | None = None
    variant_mapping: pd.DataFrame | None = None
    holidays: pd.DataFrame | None = None

    @property
    def start_date(self) -> pd.Timestamp:
        return self.daily["date"].min()

    @property
    def end_date(self) -> pd.Timestamp:
        return self.daily["date"].max()

    def truncate(self, last_date: pd.Timestamp) -> "SalesData":
        """History as it would have looked on `last_date` (inclusive)."""

        daily = self.daily[self.daily["date"] <= last_date].copy()
        products = self.products.loc[self.products.index.isin(daily["variantId"].unique())].copy()
        return SalesData(daily, products, self.catalog, self.variant_mapping, self.holidays)

    def store_totals(self) -> pd.Series:
        """All units (walk-in + bulk) per calendar day, gap days as 0."""

        totals = (self.daily["demand"] + self.daily["bulkQuantity"]).groupby(self.daily["date"]).sum()
        full_range = pd.date_range(self.start_date, self.end_date, freq="D")
        return totals.reindex(full_range, fill_value=0.0)


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _examples(df: pd.DataFrame, mask: pd.Series, column: str) -> str:
    """'line 12: "06-31-2026", line 40: "abc"' -- CSV line numbers (header = line 1)."""

    rows = df.loc[mask, column].head(_MAX_EXAMPLES)
    parts = [f'line {idx + 2}: "{value}"' for idx, value in rows.items()]
    more = int(mask.sum()) - len(parts)
    if more > 0:
        parts.append(f"... and {more} more")
    return ", ".join(parts)


def _require_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataFormatError(
            f"{source}: missing required column(s) {missing}. "
            f"Found columns: {list(df.columns)}. Expected at least: {required}."
        )


def _parse_dates(df: pd.DataFrame, column: str, source: str, allow_blank: bool = False) -> pd.Series:
    raw = df[column]
    blank = raw.isna() | (raw.astype(str).str.strip() == "")
    if blank.any() and not allow_blank:
        raise DataFormatError(f"{source}: column '{column}' has empty values ({_examples(df, blank, column)}).")

    parsed = pd.to_datetime(raw.where(~blank).astype("string").str.strip(), format=DATE_FORMAT, errors="coerce")
    bad = parsed.isna() & ~blank
    if bad.any():
        raise DataFormatError(
            f"{source}: column '{column}' must be dates in DD-MM-YYYY format (e.g. 21-09-2026). "
            f"Invalid values: {_examples(df, bad, column)}."
        )
    return parsed


def _parse_int(df: pd.DataFrame, column: str, source: str) -> pd.Series:
    numeric = pd.to_numeric(df[column], errors="coerce")
    bad = numeric.isna() | (numeric % 1 != 0)
    if bad.any():
        raise DataFormatError(
            f"{source}: column '{column}' must be a whole-number id. Invalid values: {_examples(df, bad, column)}."
        )
    return numeric.astype("int64")


def _parse_flag(df: pd.DataFrame, column: str, source: str, default: int | None = 0) -> pd.Series:
    raw = df[column]
    blank = raw.isna() | (raw.astype(str).str.strip() == "")
    if blank.any() and default is None:
        raise DataFormatError(f"{source}: column '{column}' has empty values ({_examples(df, blank, column)}).")

    numeric = pd.to_numeric(raw.where(~blank), errors="coerce")
    bad = ~blank & ~numeric.isin([0, 1])
    if bad.any():
        raise DataFormatError(
            f"{source}: column '{column}' must be 0 or 1. Invalid values: {_examples(df, bad, column)}."
        )
    return numeric.fillna(default if default is not None else 0).astype("int64")


def _parse_non_negative(df: pd.DataFrame, column: str, source: str, allow_blank: bool) -> pd.Series:
    raw = df[column]
    blank = raw.isna() | (raw.astype(str).str.strip() == "")
    if blank.any() and not allow_blank:
        raise DataFormatError(f"{source}: column '{column}' has empty values ({_examples(df, blank, column)}).")

    numeric = pd.to_numeric(raw.where(~blank), errors="coerce")
    bad = ~blank & (numeric.isna() | (numeric < 0))
    if bad.any():
        raise DataFormatError(
            f"{source}: column '{column}' must be a number >= 0. Invalid values: {_examples(df, bad, column)}."
        )
    return numeric.astype("float64")


_MRP_PATTERNS = [
    # "Mrp20", "MRP 10", "Mrp 5/-", "Rs10/-", "Rs 5/-", "₹10"
    re.compile(r"(?:mrp|rs\.?|₹)\s*[:.]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    # "10 MRP", "30 MRP", "Mrp10/-" already covered above
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:/-)?\s*mrp\b", re.IGNORECASE),
]


def parse_mrp_from_name(name: str) -> float:
    """Best-effort price from names like 'Crypto Water Mrp20' or 'Talc 10 MRP'.

    Only used when neither the training file nor catalog.csv carries an mrp.
    Returns NaN when no price can be read from the name.
    """

    if not isinstance(name, str):
        return np.nan
    for pattern in _MRP_PATTERNS:
        match = pattern.search(name)
        if match:
            return float(match.group(1))
    return np.nan


# --------------------------------------------------------------------------
# Main training file
# --------------------------------------------------------------------------


def validate_sales_frame(df: pd.DataFrame, source: str = "training file") -> pd.DataFrame:
    """Validate the raw training CSV and return it with typed columns.

    Raises DataFormatError with the offending CSV line numbers on any
    problem, so a bad export is caught before training starts.
    """

    if df is None or df.empty:
        raise DataFormatError(f"{source}: file has no data rows.")

    _require_columns(df, SALES_REQUIRED_COLUMNS, source)
    df = df.reset_index(drop=True)

    out = pd.DataFrame(index=df.index)
    out["variantId"] = _parse_int(df, "variantId", source)

    names = df["productName"].astype("string").str.strip()
    blank_name = names.isna() | (names == "")
    if blank_name.any():
        raise DataFormatError(
            f"{source}: column 'productName' has empty values ({_examples(df, blank_name, 'productName')})."
        )
    out["productName"] = names.astype(str)
    out["date"] = _parse_dates(df, "date", source)
    out["quantitySold"] = _parse_non_negative(df, "quantitySold", source, allow_blank=False)

    for flag in SALES_FLAG_COLUMNS:
        out[flag] = _parse_flag(df, flag, source) if flag in df.columns else 0
    out.attrs["has_isBulk"] = "isBulk" in df.columns
    out["transactions"] = (
        _parse_non_negative(df, "transactions", source, allow_blank=True) if "transactions" in df.columns else np.nan
    )

    out["category"] = (
        df["category"].astype("string").str.strip().replace("", pd.NA) if "category" in df.columns else pd.NA
    )
    out["mrp"] = _parse_non_negative(df, "mrp", source, allow_blank=True) if "mrp" in df.columns else np.nan

    return out


def flag_bulk_rows(raw: pd.DataFrame, units_per_transaction: float = FORECAST_BULK_UNITS_PER_TRANSACTION) -> pd.DataFrame:
    """Set isBulk / bulkSource on validated rows.

    An explicit isBulk column always wins. Without one, a row whose units per
    transaction reach `units_per_transaction` is bulk (e.g. 1,070 sauce
    sachets on one bill). Rows without a transaction count stay walk-in.
    """

    raw = raw.copy()
    raw["bulkSource"] = np.where(raw["isBulk"] == 1, "isBulk", "")
    if raw.attrs.get("has_isBulk", True) or raw["transactions"].isna().all():
        return raw

    per_bill = raw["quantitySold"] / raw["transactions"].where(raw["transactions"] > 0)
    by_transactions = per_bill >= units_per_transaction
    raw.loc[by_transactions, "isBulk"] = 1
    raw.loc[by_transactions, "bulkSource"] = "transactions"
    if by_transactions.any():
        logger.info(
            "Flagged %d row(s) as bulk: >= %s units per transaction",
            int(by_transactions.sum()),
            units_per_transaction,
        )
    return raw


def build_sales_data(
    raw: pd.DataFrame,
    catalog: pd.DataFrame | None = None,
    variant_mapping: pd.DataFrame | None = None,
    holidays: pd.DataFrame | None = None,
    bulk_units_per_transaction: float = FORECAST_BULK_UNITS_PER_TRANSACTION,
) -> SalesData:
    """Aggregate validated rows into one row per (variantId, date).

    Duplicate rows for the same product/day are SUMMED (not dropped):
    walk-in rows into `demand`, bulk rows into `bulkQuantity`. A day is a
    stock-out day if any of its rows says so.
    """

    if "transactions" not in raw.columns:
        raw = raw.assign(transactions=np.nan)
    raw = flag_bulk_rows(raw, bulk_units_per_transaction)

    duplicates = raw.duplicated(subset=["variantId", "date"], keep=False)
    if duplicates.any():
        logger.info(
            "Summing %d rows that share a (variantId, date) with another row", int(duplicates.sum())
        )

    walk_in = raw["quantitySold"].where(raw["isBulk"] == 0, 0.0)
    bulk = raw["quantitySold"].where(raw["isBulk"] == 1, 0.0)
    daily = (
        raw.assign(demand=walk_in, bulkQuantity=bulk)
        .groupby(["variantId", "date"], as_index=False)
        .agg(
            demand=("demand", "sum"),
            bulkQuantity=("bulkQuantity", "sum"),
            isStockout=("isStockout", "max"),
            bulkSource=("bulkSource", "max"),
            transactions=("transactions", lambda s: s.sum(min_count=1)),
        )
    )
    daily["isStockout"] = daily["isStockout"].astype(bool)
    daily = daily.sort_values(["variantId", "date"]).reset_index(drop=True)

    products = _product_attributes(raw, catalog)
    return SalesData(daily, products, catalog, variant_mapping, holidays)


def _product_attributes(raw: pd.DataFrame, catalog: pd.DataFrame | None) -> pd.DataFrame:
    """Per-variant name/category/mrp. Keyed on variantId -- a rename keeps
    one product, and two variants sharing a name stay two products.

    Priority for each attribute: latest non-empty value in the training
    file, then catalog.csv, then (mrp only) a price parsed from the name.
    """

    latest = raw.sort_values("date")
    grouped = latest.groupby("variantId")

    renamed = grouped["productName"].nunique()
    if (renamed > 1).any():
        logger.warning(
            "%d variant(s) appear under more than one productName; using the most recent name",
            int((renamed > 1).sum()),
        )

    products = pd.DataFrame(
        {
            "productName": grouped["productName"].last(),
            "category": grouped["category"].agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else pd.NA),
            "mrp": grouped["mrp"].agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
        }
    )
    products.index.name = "variantId"

    if catalog is not None and not catalog.empty:
        cat = catalog.set_index("variant_id")
        products["category"] = products["category"].fillna(cat["category"].reindex(products.index))
        products["mrp"] = products["mrp"].fillna(cat["mrp"].reindex(products.index))

    products["mrp"] = products["mrp"].fillna(products["productName"].map(parse_mrp_from_name))
    products["category"] = products["category"].astype("string")
    return products


def load_sales_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataFormatError(f"Training file not found: {path}")
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    except Exception as exc:  # pragma: no cover - pandas error text is informative enough
        raise DataFormatError(f"Could not read {path.name} as CSV: {exc}") from exc
    return validate_sales_frame(df, source=path.name)


# --------------------------------------------------------------------------
# Optional companion files
# --------------------------------------------------------------------------


def validate_catalog(df: pd.DataFrame, source: str = CATALOG_FILENAME) -> pd.DataFrame:
    _require_columns(df, CATALOG_COLUMNS, source)
    df = df.reset_index(drop=True)

    out = pd.DataFrame(index=df.index)
    out["variant_id"] = _parse_int(df, "variant_id", source)
    out["product_name"] = df["product_name"].astype("string").str.strip()
    out["category"] = df["category"].astype("string").str.strip().replace("", pd.NA)
    out["mrp"] = _parse_non_negative(df, "mrp", source, allow_blank=True)
    out["launch_date"] = _parse_dates(df, "launch_date", source, allow_blank=True)
    out["delist_date"] = _parse_dates(df, "delist_date", source, allow_blank=True)
    out["active"] = _parse_flag(df, "active", source, default=None)

    dup = out["variant_id"].duplicated(keep=False)
    if dup.any():
        raise DataFormatError(
            f"{source}: each variant_id must appear once. Duplicates: {_examples(out, dup, 'variant_id')}."
        )
    return out


def validate_variant_mapping(df: pd.DataFrame, source: str = VARIANT_MAPPING_FILENAME) -> pd.DataFrame:
    """Validate old -> new replacements and resolve chains (a->b, b->c
    becomes a->c and b->c), so every old id points at its final successor."""

    _require_columns(df, VARIANT_MAPPING_COLUMNS, source)
    df = df.reset_index(drop=True)

    out = pd.DataFrame(index=df.index)
    out["old_variant_id"] = _parse_int(df, "old_variant_id", source)
    out["new_variant_id"] = _parse_int(df, "new_variant_id", source)
    out["switch_date"] = _parse_dates(df, "switch_date", source)

    self_map = out["old_variant_id"] == out["new_variant_id"]
    if self_map.any():
        raise DataFormatError(
            f"{source}: a variant cannot replace itself ({_examples(out, self_map, 'old_variant_id')})."
        )

    conflict = out.groupby("old_variant_id")["new_variant_id"].transform("nunique") > 1
    if conflict.any():
        raise DataFormatError(
            f"{source}: an old_variant_id maps to more than one new_variant_id "
            f"({_examples(out, conflict, 'old_variant_id')})."
        )
    out = out.drop_duplicates(subset=["old_variant_id"]).reset_index(drop=True)

    successor = dict(zip(out["old_variant_id"], out["new_variant_id"]))
    final = {}
    for old in successor:
        seen = {old}
        current = successor[old]
        while current in successor:
            if current in seen:
                raise DataFormatError(f"{source}: variant mapping has a cycle involving variant {old}.")
            seen.add(current)
            current = successor[current]
        final[old] = current
    out["new_variant_id"] = out["old_variant_id"].map(final).astype("int64")
    return out


def validate_holidays(df: pd.DataFrame, source: str = HOLIDAYS_FILENAME) -> pd.DataFrame:
    _require_columns(df, HOLIDAYS_COLUMNS, source)
    df = df.reset_index(drop=True)
    out = pd.DataFrame({"date": _parse_dates(df, "date", source), "name": df["name"].astype("string")})
    return out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


_COMPANIONS = {
    CATALOG_FILENAME: validate_catalog,
    VARIANT_MAPPING_FILENAME: validate_variant_mapping,
    HOLIDAYS_FILENAME: validate_holidays,
}


def load_companion(directory: str | Path | None, filename: str) -> pd.DataFrame | None:
    """Load + validate an optional companion file; None if it isn't there."""

    if directory is None:
        return None
    path = Path(directory) / filename
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    validated = _COMPANIONS[filename](df, source=filename)
    logger.info("Loaded %s (%d rows)", path, len(validated))
    return validated


def load_dataset(training_path: str | Path, aux_dir: str | Path | None = None) -> SalesData:
    """Load the training CSV plus any companion files found in `aux_dir`
    (defaults to the training file's own folder)."""

    training_path = Path(training_path)
    aux_dir = training_path.parent if aux_dir is None else Path(aux_dir)
    raw = load_sales_csv(training_path)
    return build_sales_data(
        raw,
        catalog=load_companion(aux_dir, CATALOG_FILENAME),
        variant_mapping=load_companion(aux_dir, VARIANT_MAPPING_FILENAME),
        holidays=load_companion(aux_dir, HOLIDAYS_FILENAME),
    )


def load_dataset_from_frame(df: pd.DataFrame, aux_dir: str | Path | None = None) -> SalesData:
    """Same as load_dataset, for a DataFrame the API layer already read."""

    raw = validate_sales_frame(df.astype(str).replace({"nan": ""}), source="training file")
    return build_sales_data(
        raw,
        catalog=load_companion(aux_dir, CATALOG_FILENAME),
        variant_mapping=load_companion(aux_dir, VARIANT_MAPPING_FILENAME),
        holidays=load_companion(aux_dir, HOLIDAYS_FILENAME),
    )
