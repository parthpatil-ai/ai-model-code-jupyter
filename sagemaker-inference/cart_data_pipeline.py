#!/usr/bin/env python3
"""
cart_data_pipeline.py
=====================
One script for the whole cart data job:

  1. Fetch    rebuild `cart_item_daily_sales` for one cart from `sales_transactions` and read it back
  2. Build    clean the rows into sales.csv, catalog.csv, variant_mapping.csv, holidays.csv + report
  3. Test     check the files (totals, duplicates, stock-outs, ...) -> test_results.txt
  4. Save     everything goes to a local folder named after the cart, e.g. cart21/
  5. Upload   the four CSVs go to s3://<bucket>/raw/cart_<cartId>/ (replacing the previous run's files)

It combines fetch_and_build.py, cart_dataset_builder.py and check_export.py; the
cleaning logic is the same as cart_dataset_builder.py.

Usage
    python cart_data_pipeline.py --cart 21                      # Mongo -> build -> test -> local + S3
    python cart_data_pipeline.py --cart 21 --no-s3              # local only
    python cart_data_pipeline.py --cart 21 --skip-aggregate     # read cart_item_daily_sales as it is (no writes)
    python cart_data_pipeline.py --cart 21 --from-csv cart21/cart21_daily_sales_raw.csv   # no Mongo

Settings are read from the .env next to this script (git-ignored), then the project .env:
    MONGO_URI        mongodb://api:<password>@13.127.176.15:29100/yk_dev?authSource=admin&directConnection=true
    S3_BUCKET_NAME   default yatrikart-ml-prod-ai-stn
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION   (or an IAM role)

Requires: pip install pymongo pandas numpy boto3 python-dotenv
"""
import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    # this folder's .env (Mongo + S3 for this script) wins; the project .env only fills gaps
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# =============================================================================
# CONFIG
# =============================================================================
SOURCE = "sales_transactions"
TARGET = "cart_item_daily_sales"

# Extra filter on bills before unwinding, e.g. drop cancelled/refunded bills (spec rule 5).
# Adjust the field names to your schema, e.g. {"status": {"$nin": ["cancelled", "refunded"]}}
EXCLUDE_BILLS = {}

DEFAULT_BUCKET = "yatrikart-ml-prod-ai-stn"
DEFAULT_S3_PREFIX = "raw"          # SageMaker training reads raw/cart_<id>/

CONFIG = dict(
    seed=42,                      # makes the late-posting split reproducible
    # late-posting detection
    low_day_share=0.35,           # a day under 35% of normal units counts as partial/offline
    upload_breadth=1.5,           # upload day: >=1.5x the usual number of products sold ...
    upload_ratio=2.5,             # ... or each product at >=2.5x its usual quantity
    rolling_window=29,            # days used to define "normal" around each date
    max_offline_days=7,           # longer gaps are an incomplete export, not a late posting
    # bulk detection
    bulk_multiple=8,              # a row >= 8x the product's usual day is bulk ...
    bulk_min_units=40,            # ... and at least this many units
    bulk_min_units_water=100,     # water needs crate-size quantities
    supply_bulk_units=50,         # supplies/disposables: >= 50 units in a day is a bulk issue
    small_history_bulk_units=50,  # products with < 5 sale days: >= 50 units is bulk
    case_sizes=(12,),             # repeated identical multiples of this are standing orders
    # stock-out inference
    stockout_min_run=3,           # zero-day runs shorter than this are never stock-outs
    stockout_p=0.01,              # run is a stock-out if it has <1% chance of happening by luck
    # event days
    event_breadth=1.6,            # >=1.6x usual products sold ...
    event_bulk_units=250,         # ... plus >= 250 bulk units that day
    # catalog
    launch_grace_days=14,         # first sale within 14 days of data start: launch unknown
    delist_after_days=30,         # no sale for 30+ days and gap is abnormal -> delisted
    delist_hard_days=60,          # no sale for 60+ days -> delisted regardless
    # low-day report
    report_low_share=0.30,
)

# Late postings you already know about (optional). Auto-detection finds the rest.
# Format: (["upload day", ...], "first real day", "last real day")
KNOWN_LATE_POSTINGS = []

# Variants that changed from a packet to a single piece: {variantId: pieces per packet}.
# Rows whose name matches PACKET_NAME_PATTERN are multiplied into pieces.
PACKET_SIZES = {3902: 50, 4189: 50, 4190: 50}
PACKET_NAME_PATTERN = r"Mrp50$"

CATEGORY_RULES = [  # first match wins
    ("Supplies & Disposables", r"paper cup|stirrer|sipper lid|paper straw|coffee lid|^tb milk|coffee beans|disposal|diposal|nasta plates|zingy sauce|ketchup|\bccd\b|coolm|pos machine roll|green sauce|chilli sauce|sugar sachets|iced coffee glass"),
    ("Tobacco", r"gold flake|^flake |\b\d+n\b|classic rich|classic refined"),
    ("Household", r"detergent|\brin\b|\bvim\b"),
    ("General Merchandise", r"\bpen\b|fevistik|scissors|puppets|\bblocks\b|rubic|flashing ball|statue|locket|\bchain\b|bracelet|hair band|pony band|hanky|rain coat|umbrella|sunglasses|teddy|earring|breselet|envelope|rakhi|mobile cover|light ball|hair clutures"),
    ("Electronics & Accessories", r"earphone|charger|\bcable\b|portronics|power bank|earbuds|neckband|adapter|adaptor|kayroo|mobile holder"),
    ("Personal Care", r"shampoo|toothbrush|toothbursh|tooth paste|toothpaste|\bsoap\b|colgate|clinic plus|sunsilk|sanitary|face wash|face cream|beauty cream|patanjali|himalaya|sofy|ponds|parachute|perfume|vaseline|comb\b|hair oil|iodex|face mask|cotton mask|razor|gillette|condom|eyeliner|lip liner|wipes|nail paint|highlighter|zandu|mamy poko"),
    ("Ready-to-eat & Bakery", r"brawni|samosa|pattis|patties|vada pav|burger|sandwich|cuppa noodles"),
    ("Biscuits & Cookies", r"oreo|bournvita biscuit|chocobakes|bikis|milk vanilla wafer|happy happy|krack ?jack|jam-in"),
    ("Chocolates & Confectionery", r"luvit|whole almonds bar|milk bar|nestle|kismi|milk chocolate|dark bar|choco tube|cadbury|kitkat|snickers|5 star|coffee bar"),
    ("Dairy, Shakes & Coffee", r"smoodh|shake|lassi\b|\bkool\b|flavou?red milk|milk tp|trelish|chaas|buttermilk|kadhai doodh|cold coff|latte|cappuccino|espresso|\bcoffee\b|nescafe|\bmilk\b"),
    ("Snacks & Namkeen", r"chips|\blays\b|kurkure|bingo|doritos|cheetos|uncle|balaji|aakash|akash|haldiram|namkeen|bhujia|\bsev\b|gathiya|chivda|mixture|\bdal\b|peanut|popcorn|act ii|crax|nacho|cornitos|pringles|puffs|pufs|wheels|mad angles|pepitos|rollzone|zipzo|makhana|max protein|banana|rumbles|crunch|soya|matar|murmura|yellow diamond|\bmom\b|chana masala|pasta fingers|rite bite|peri peri|boondi|bondi|chana salted|bakarwadi|ujjaini sev"),
    ("Biscuits & Cookies", r"biscuit|cookie|britannia|good day|marie|jim jam|bourbon|monaco|krackjack|hide & seek|20-20|parle g\b|unibic|sunfeast|dark fantasy|wafer|waffer|nabati|bake|cream roll"),
    ("Chocolates & Confectionery", r"polo\b|gum\b|boomer|orbit|candy|toffee|eclair|pulse|mentos|jelly|lollipop|chocolate|choco|\bmints?\b|zykaa|laddu|chikki|kinder|rajni|pass-pass|tic tac|\beno\b|coconut premium bite|yumiloop|anar dana|gems"),
    ("Beverages", r"coconut water"),
    ("Water", r"\bwater\b|bisleri"),
    ("Beverages", r"cola|coca|pepsi|sprite|fanta|thums|limca|mazza|maaza|frooti|appy|slice|sting|redbull|red bull|monster|7up|mountain dew|nimbooz|lahori|paper boat|juice|drink|soda|jeera|shikanji|nimboo|aamras|iced tea|tropicana|mabu|glucon|energy|\btang\b|b natural|mirinda|schweppes|teazyy|chai|kinley|can\b"),
    ("Ready-to-eat & Bakery", r"puff|\broll\b|cake|muffin|bread|\bbun\b|noodle|maggi|cuppa|pizza|vada|kachori|quick "),
]

# Typical retail MRP (Rs) for products whose name carries no price.
MRP_FILL = {
    188: 80, 189: 35, 1073: 100, 1257: 20, 1309: 40, 1409: 100, 1414: 20, 1527: 40, 1573: 75,
    3469: 60, 3567: 40, 3568: 40, 3569: 40, 3571: 40, 3754: 150, 3756: 600, 3838: 1, 4026: 1,
    4027: 100, 4034: 20, 4035: 30, 4036: 20, 4038: 60, 4041: 15, 4042: 25, 4046: 70, 4047: 350,
    4049: 150, 4095: 110, 4121: 200, 4125: 50, 4126: 50, 4128: 40, 4132: 20, 4133: 30,
    4136: 299, 4137: 199, 4139: 999, 4140: 999, 4141: 799, 4142: 799, 4143: 999, 4144: 299,
    4145: 349, 4146: 299, 4148: 499, 4149: 499, 4150: 199, 4151: 199, 4152: 199, 4153: 299,
    4154: 299, 4162: 130, 21: 499, 79: 20, 1092: 300, 1120: 10, 1155: 60, 3471: 60, 3472: 25,
    3927: 120, 3928: 120, 4025: 2, 4048: 150, 4129: 40, 4130: 60, 4131: 60, 4214: 25, 4219: 30,
    4230: 40, 4330: 299, 4204: 10, 4205: 20, 4229: 40, 4231: 40, 4245: 40,
}

# National holidays / festivals (India, 2026). Moon-based dates can shift by a day.
NATIONAL_HOLIDAYS = [
    ("03-03-2026", "Holika Dahan"), ("04-03-2026", "Holi"), ("19-03-2026", "Gudi Padwa / Ugadi"),
    ("21-03-2026", "Eid ul-Fitr"), ("26-03-2026", "Ram Navami"), ("31-03-2026", "Mahavir Jayanti"),
    ("03-04-2026", "Good Friday"), ("14-04-2026", "Ambedkar Jayanti / Baisakhi"),
    ("01-05-2026", "Maharashtra Day / Labour Day / Buddha Purnima"), ("27-05-2026", "Eid al-Adha (Bakrid)"),
    ("26-06-2026", "Muharram"), ("15-08-2026", "Independence Day"), ("26-08-2026", "Milad-un-Nabi"),
    ("28-08-2026", "Raksha Bandhan"), ("04-09-2026", "Janmashtami"), ("14-09-2026", "Ganesh Chaturthi"),
    ("02-10-2026", "Gandhi Jayanti"), ("11-10-2026", "Navratri begins"), ("20-10-2026", "Dussehra"),
    ("08-11-2026", "Diwali"), ("24-11-2026", "Guru Nanak Jayanti"), ("25-12-2026", "Christmas"),
]
EVENT_LABEL = "Local event / catering day (inferred from sales)"
DATE_FMT = "%d-%m-%Y"
SALES_COLS = ["variantId", "productName", "date", "quantitySold", "isBulk", "isStockout", "category", "mrp"]
OUTPUT_FILES = ["sales.csv", "catalog.csv", "variant_mapping.csv", "holidays.csv"]


# =============================================================================
# STEP 1. FETCH FROM MONGODB
# =============================================================================
def daily_pipeline(cart_id, tz):
    return [
        {"$match": {"cartId": cart_id, **EXCLUDE_BILLS}},
        {"$unwind": "$items"},
        {"$match": {"items.name": {"$exists": True, "$ne": None}}},
        {"$group": {
            "_id": {
                "date": {"$dateToString": {"format": "%d-%m-%Y", "date": "$createdAt", "timezone": tz}},
                "cartId": "$cartId",
                "name": "$items.name",
            },
            "variantId": {"$first": "$items.variantId"},
            "sortDate": {"$min": {"$dateTrunc": {"date": "$createdAt", "unit": "day", "timezone": tz}}},
            "totalQuantity": {"$sum": "$items.quantity"},
        }},
        {"$sort": {"sortDate": 1, "_id.name": 1}},
        {"$project": {
            "_id": 0,
            "variantId": "$variantId",
            "quantitySold": "$totalQuantity",
            "date": "$_id.date",
            "cartId": "$_id.cartId",
            "productName": "$_id.name",
        }},
        {"$merge": {"into": TARGET, "on": ["date", "cartId", "productName"],
                    "whenMatched": "replace", "whenNotMatched": "insert"}},
    ]


def refresh_daily_sales(db, cart_id, tz):
    coll = db[TARGET]
    # unique key required by $merge; createIndex is a no-op if it already exists
    coll.create_index([("date", 1), ("cartId", 1), ("productName", 1)], unique=True)
    deleted = coll.delete_many({"cartId": cart_id}).deleted_count      # only this cart, not the whole collection
    db[SOURCE].aggregate(daily_pipeline(cart_id, tz), allowDiskUse=True)
    print(f"[mongo] cart {cart_id}: removed {deleted} old rows, rebuilt {TARGET} ({tz} days)")


def read_daily_sales(db, cart_id):
    docs = list(db[TARGET].find({"cartId": cart_id}, {"_id": 0}))
    if not docs:
        bills = db[SOURCE].count_documents({"cartId": cart_id})
        sys.exit(f"No rows in {TARGET} for cartId={cart_id} ({bills} bills in {SOURCE}). "
                 "Check the cart id or run without --skip-aggregate.")
    df = pd.DataFrame(docs)
    missing = {"variantId", "quantitySold", "date", "productName"} - set(df.columns)
    if missing:
        sys.exit(f"{TARGET} rows are missing fields: {sorted(missing)}")
    bad_vid = df["variantId"].isna().sum()
    if bad_vid:
        print(f"[warn] {bad_vid} rows have no variantId and were skipped")
        df = df[df["variantId"].notna()]
    df["variantId"] = df["variantId"].astype(int)
    q = pd.to_numeric(df["quantitySold"], errors="coerce").fillna(0)
    if (q != q.round()).any():
        print(f"[warn] {(q != q.round()).sum()} rows had fractional quantities; rounded")
    df["quantitySold"] = q.round().astype(int)
    print(f"[mongo] read {len(df):,} daily rows, {df['date'].nunique()} dates, {df['variantId'].nunique()} variants")
    return df[["variantId", "quantitySold", "date", "cartId", "productName"]]


def bill_totals(db, cart_id, tz):
    """Daily units per product counted straight from the bills, for the cross-check."""
    rows = [(b["createdAt"], b.get("status"), it.get("name"), it.get("quantity") or 0)
            for b in db[SOURCE].find({"cartId": cart_id, **EXCLUDE_BILLS},
                                     {"createdAt": 1, "status": 1, "items.name": 1, "items.quantity": 1})
            for it in b.get("items", [])]
    t = pd.DataFrame(rows, columns=["ts", "status", "name", "q"])
    t = t[t["name"].notna()]
    t["date"] = pd.to_datetime(t["ts"], utc=True).dt.tz_convert(tz).dt.strftime(DATE_FMT)
    return t


def fetch(a):
    from pymongo import MongoClient
    if not a.uri:
        sys.exit("Set MONGO_URI (environment or .env) or pass --uri")
    client = MongoClient(a.uri, serverSelectionTimeoutMS=15000)
    client.admin.command("ping")
    db = client[a.db] if a.db else client.get_default_database()
    print(f"[mongo] connected to {db.name}")
    if not a.skip_aggregate:
        refresh_daily_sales(db, a.cart, a.tz)
    df = read_daily_sales(db, a.cart)
    bills = bill_totals(db, a.cart, a.tz)
    client.close()
    return df, bills


# =============================================================================
# STEP 2. BUILD (same logic as cart_dataset_builder.py)
# =============================================================================
def category(name):
    s = name.lower()
    for cat, pat in CATEGORY_RULES:
        if re.search(pat, s):
            return cat
    return "Other"


def parse_mrp(name):
    s = name.lower().replace("!0", "10")
    for pat in (r"(?:mrp|rs\.?)\s*\.?\s*(\d+(?:\.\d+)?)", r"(\d+(?:\.\d+)?)\s*mrp",
                r"(\d+(?:\.\d+)?)\s*/-", r"(\d+\.\d+)\s*paisa"):
        m = re.search(pat, s)
        if m:
            return float(m.group(1))
    return None


def mrp_for(variant_id, name):
    m = parse_mrp(name)
    return MRP_FILL.get(variant_id) if m is None else m


def fmt(d):
    return pd.Timestamp(d).strftime(DATE_FMT)


def load_raw(path, today, rep):
    raw = pd.read_csv(path)
    aliases = {"quantitySold": ["quantitySold", "totalQuantity", "quantity", "qty"],
               "productName": ["productName", "name", "product_name"],
               "variantId": ["variantId", "variant_id"], "date": ["date"]}
    cols = {}
    for std, opts in aliases.items():
        hit = next((c for c in opts if c in raw.columns), None)
        if hit is None:
            sys.exit(f"Input is missing a '{std}' column (looked for {opts})")
        cols[hit] = std
    raw = raw.rename(columns=cols)[list(cols.values())]
    raw["d"] = pd.to_datetime(raw["date"], format=DATE_FMT, errors="coerce")
    bad = raw["d"].isna().sum()
    if bad:
        rep["warnings"].append(f"{bad} rows had unreadable dates and were skipped")
        raw = raw[raw["d"].notna()]
    neg = (raw["quantitySold"] < 0).sum()
    if neg:
        rep["warnings"].append(f"{neg} negative rows (returns) found; netted against same product/day")
    raw = (raw.groupby(["variantId", "d", "productName"], as_index=False)["quantitySold"].sum())
    raw["quantitySold"] = raw["quantitySold"].clip(lower=0)
    raw["orig"] = (raw["productName"].str.replace("\xa0", " ", regex=False)
                   .str.replace(r"\s+", " ", regex=True).str.strip())
    canon = raw.sort_values("d").groupby("variantId")["orig"].last()
    rep["renamed"] = int((raw.groupby("variantId")["orig"].nunique() > 1).sum())

    # packet -> piece
    old = raw["variantId"].isin(PACKET_SIZES) & raw["orig"].str.contains(PACKET_NAME_PATTERN)
    rep["packet"] = (int(old.sum()), int(raw.loc[old, "quantitySold"].sum()))
    raw.loc[old, "quantitySold"] *= raw.loc[old, "variantId"].map(PACKET_SIZES)
    raw.loc[old, "orig"] = raw.loc[old, "variantId"].map(canon)
    rep["packet"] += (int(raw.loc[old, "quantitySold"].sum()),)

    # keep complete days only
    today = pd.Timestamp(today)
    dropped = raw[raw["d"] >= today]
    rep["dropped_today"] = (int(len(dropped)), int(dropped["quantitySold"].sum()))
    data = raw[raw["d"] < today].copy()
    return data, canon, raw          # raw (incl. today) is used for catalog & mapping evidence


def daily_profile(data, days):
    pmed = data.groupby("variantId")["quantitySold"].median()
    x = data.assign(r=data["quantitySold"] / data["variantId"].map(pmed))
    t = (x.groupby("d").agg(units=("quantitySold", "sum"), prods=("variantId", "size"), ratio=("r", "median"))
         .reindex(days))
    w = CONFIG["rolling_window"]
    t["norm_units"] = t["units"].rolling(w, center=True, min_periods=5).median()
    t["norm_prods"] = t["prods"].rolling(w, center=True, min_periods=5).median()
    t["low"] = t["units"].isna() | (t["units"] < CONFIG["low_day_share"] * t["norm_units"])
    return t


def detect_late_postings(t, days, rep):
    blocks = []
    big = (t["prods"] >= CONFIG["upload_breadth"] * t["norm_prods"]) | (t["ratio"] >= CONFIG["upload_ratio"])
    low = t["low"].values
    i = 1
    while i < len(days):
        if big.iloc[i] and not low[i] and low[i - 1]:
            j = i
            while j > 0 and low[j - 1]:
                j -= 1
            if j == 0:            # gap at the very start of the data is not a late posting
                i += 1
                continue
            if i - j > CONFIG["max_offline_days"]:
                rep["warnings"].append(f"{i - j} offline/low days before {fmt(days[i])} - too long to be a late "
                                       "posting; re-export that period")
                i += 1
                continue
            k = i
            while k + 1 < len(days) and t["ratio"].iloc[k + 1] >= CONFIG["upload_ratio"] and not low[k + 1]:
                k += 1
            blocks.append(([days[u] for u in range(i, k + 1)], days[j], days[k]))
            i = k + 1
        else:
            i += 1
    for ups, a, b in KNOWN_LATE_POSTINGS:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if not any(x[1] <= a <= x[2] for x in blocks):
            blocks.append(([pd.Timestamp(u) for u in ups], a, b))
    return sorted(blocks, key=lambda x: x[1])


def build_sales_rows(data, canon, blocks, rep):
    rng = np.random.default_rng(CONFIG["seed"])
    upload_to_block = {u: pd.date_range(a, b) for ups, a, b in blocks for u in ups}
    block_days = {d for _, a, b in blocks for d in pd.date_range(a, b)}
    ref = data[~data["d"].isin(block_days)]
    stats = {k: (len(g), g["quantitySold"].median(), g["quantitySold"].tolist()) for k, g in ref.groupby("variantId")}

    def split(vid, q, nd, cat):
        per_day = q / nd
        if cat == "Supplies & Disposables":
            return (0, q, "supply/catering order") if per_day >= CONFIG["supply_bulk_units"] else (q, 0, "")
        n, med, vals = stats.get(vid, (0, 0, []))
        if n < 5:
            return (0, q, "large order, little history") if per_day >= CONFIG["small_history_bulk_units"] else (q, 0, "")
        if nd == 1 and q >= 48 and any(q % c == 0 for c in CONFIG["case_sizes"]) and vals.count(q) >= 3:
            return 0, q, "repeated case-size standing order"
        min_q = CONFIG["bulk_min_units_water"] if cat == "Water" else CONFIG["bulk_min_units"]
        if q >= min_q and per_day >= CONFIG["bulk_multiple"] * med:
            walk = int(np.ceil(med)) * nd
            return walk, q - walk, f">= {CONFIG['bulk_multiple']}x usual day ({med:g})"
        return q, 0, ""

    rows, bulk_log, moved = [], [], {}
    for r in data.itertuples():
        rng_days = upload_to_block.get(r.d)
        nd = len(rng_days) if rng_days is not None else 1
        walk, bulk, why = split(r.variantId, int(r.quantitySold), nd, category(canon[r.variantId]))
        m = mrp_for(r.variantId, r.orig)
        if bulk:
            rows.append((r.variantId, r.d, bulk, 1, m))
            bulk_log.append((canon[r.variantId], r.d, bulk, why))
        if walk and nd == 1:
            rows.append((r.variantId, r.d, walk, 0, m))
        elif walk:
            share = np.full(nd, walk // nd)
            share[rng.choice(nd, walk % nd, replace=False)] += 1
            rows += [(r.variantId, d, int(u), 0, m) for d, u in zip(rng_days, share) if u]
            moved[r.d] = moved.get(r.d, 0) + walk
    s = pd.DataFrame(rows, columns=["variantId", "date", "quantitySold", "isBulk", "mrp"])
    s = s.groupby(["variantId", "date", "isBulk"], as_index=False).agg(quantitySold=("quantitySold", "sum"), mrp=("mrp", "first"))
    rep["moved"] = moved
    rep["bulk_log"] = sorted(bulk_log, key=lambda x: -x[2])
    return s


def infer_stockouts(s, days, blocks):
    vids = sorted(s["variantId"].unique())
    w = (s[s.isBulk == 0].pivot_table(index="variantId", columns="date", values="quantitySold", aggfunc="sum")
         .reindex(index=vids, columns=days).fillna(0))
    synthetic = {d for _, a, b in blocks for d in pd.date_range(a, b) if d != b}   # block = one observation
    obs = [d for d in days if d not in synthetic]
    col = {d: i for i, d in enumerate(days)}
    X = w[obs].values > 0
    S = np.zeros((len(vids), len(days)), dtype=np.int8)
    N = len(obs)
    for v in range(len(vids)):
        x = X[v]
        nz = x.sum()
        if nz < 3:
            continue
        i = 0
        while i < N:
            if x[i]:
                i += 1
                continue
            j = i
            while j < N and not x[j]:
                j += 1
            L = j - i
            if L >= CONFIG["stockout_min_run"] and (1 - nz / (N - L)) ** L < CONFIG["stockout_p"]:
                S[v, [col[d] for d in obs[i:j]]] = 1
            i = j
        for _, a, b in blocks:
            S[v, [col[d] for d in pd.date_range(a, b)]] = S[v, col[b]]
    S[w.values > 0] = 0
    return pd.DataFrame(S, index=w.index, columns=days), w


def build_catalog(raw_all, canon, stock, end, start):
    first = raw_all.groupby("variantId")["d"].min()
    last = raw_all.groupby("variantId")["d"].max()
    cat = pd.DataFrame({"variant_id": canon.index, "product_name": canon.values})
    cat["category"] = cat["product_name"].map(category)
    cat["mrp"] = [mrp_for(v, n) for v, n in zip(cat.variant_id, cat.product_name)]
    f, l = cat.variant_id.map(first), cat.variant_id.map(last)
    grace = start + pd.Timedelta(days=CONFIG["launch_grace_days"])
    cat["launch_date"] = np.where(f > grace, f.dt.strftime(DATE_FMT), "")
    trailing_out = cat.variant_id.map(lambda v: v in stock.index and stock.loc[v, end] == 1)
    gone = (l <= end - pd.Timedelta(days=CONFIG["delist_after_days"])) & (
        trailing_out | (l <= end - pd.Timedelta(days=CONFIG["delist_hard_days"])))
    cat["delist_date"] = np.where(gone, (l + pd.Timedelta(days=1)).dt.strftime(DATE_FMT), "")
    cat["active"] = (~gone).astype(int)
    return cat


def detect_replacements(raw_all, canon):
    base = lambda x: re.sub(r"\b(mrp|rs\.?)\s*\d+(\.\d+)?\s*(/-)?|\d+\s*/-|\bmrp\b|\d+\s*ml\b|[^a-z ]", "",
                            x.lower()).replace("  ", " ").strip()
    g = raw_all.groupby("variantId")["d"]
    v = pd.DataFrame({"name": canon, "first": g.min(), "last": g.max(), "n": g.size()})
    v["base"] = v["name"].map(base)
    out = []
    for _, grp in v.groupby("base"):
        for i in grp.index:
            for j in grp.index:
                a, c = grp.loc[i], grp.loc[j]
                if a["last"] < c["first"] and (c["first"] - a["last"]).days <= 60 and a.n >= 5 and c.n >= 3:
                    out.append((i, j, fmt(c["first"]), a["name"], c["name"]))
    return out


def build_holidays(daily, bulk_by_day, block_days):
    hol = pd.DataFrame(NATIONAL_HOLIDAYS, columns=["date", "name"])
    ev = [d for d in daily.index if d not in block_days
          and daily.loc[d, "products"] >= CONFIG["event_breadth"] * daily.loc[d, "norm_prods"]
          and bulk_by_day.get(d, 0) >= CONFIG["event_bulk_units"]]
    hol = pd.concat([hol, pd.DataFrame({"date": [fmt(d) for d in ev], "name": EVENT_LABEL})])
    hol["_k"] = pd.to_datetime(hol["date"], format=DATE_FMT)
    return hol.sort_values("_k").drop(columns="_k"), ev


def build(raw_path, out, today, start_date):
    """Writes the four CSVs to `out` and returns the report text."""
    rep = {"warnings": []}
    data, canon, raw_all = load_raw(raw_path, today, rep)
    start = pd.Timestamp(start_date) if start_date else data["d"].min()
    data = data[data["d"] >= start]
    end = data["d"].max()
    days = pd.date_range(start, end)

    profile = daily_profile(data, days)
    blocks = detect_late_postings(profile, days, rep)
    gaps = profile["units"].isna()
    runs = (gaps != gaps.shift()).cumsum()[gaps]
    for _, g in runs.groupby(runs):
        if len(g) > CONFIG["max_offline_days"]:
            rep["warnings"].append(f"EXPORT LOOKS INCOMPLETE: no rows at all {fmt(g.index[0])} to {fmt(g.index[-1])} "
                                   f"({len(g)} days) - these days will read as 0 sales")
    s = build_sales_rows(data, canon, blocks, rep)
    assert s["quantitySold"].sum() == data["quantitySold"].sum(), "late-posting spread changed totals"

    stock, walk = infer_stockouts(s, days, blocks)
    so = stock.stack()
    so = so[so == 1].reset_index()
    so.columns = ["variantId", "date", "isStockout"]
    so["quantitySold"], so["isBulk"] = 0, 0
    so["mrp"] = [mrp_for(v, canon[v]) for v in so["variantId"]]
    s["isStockout"] = 0
    sales = pd.concat([s, so], ignore_index=True)
    sales["productName"] = sales["variantId"].map(canon)
    sales["category"] = sales["productName"].map(category)
    sales = sales.sort_values(["date", "variantId", "isBulk"])

    # checks the spec requires
    assert not sales.duplicated(["variantId", "date", "isBulk"]).any(), "duplicate rows"
    assert (sales["quantitySold"] >= 0).all(), "negative quantity"
    assert sales["quantitySold"].sum() == data["quantitySold"].sum(), "totals changed"
    missing_days = set(days) - set(sales.loc[sales.quantitySold > 0, "date"])

    f = sales[SALES_COLS].copy()
    f["date"] = f["date"].dt.strftime(DATE_FMT)
    f.to_csv(out / "sales.csv", index=False, encoding="utf-8", float_format="%g")

    catalog = build_catalog(raw_all, canon, stock, end, start)
    catalog.to_csv(out / "catalog.csv", index=False, encoding="utf-8", float_format="%g")

    maps = detect_replacements(raw_all, canon)
    pd.DataFrame([m[:3] for m in maps], columns=["old_variant_id", "new_variant_id", "switch_date"]) \
        .to_csv(out / "variant_mapping.csv", index=False)

    daily = sales.groupby(["date", "isBulk"])["quantitySold"].sum().unstack(fill_value=0).reindex(days, fill_value=0)
    daily.columns = ["walkin", "bulk"] if 1 in daily.columns else ["walkin"]
    if "bulk" not in daily:
        daily["bulk"] = 0
    daily["products"] = (walk > 0).sum().reindex(days).fillna(0)
    daily["norm_prods"] = daily["products"].rolling(CONFIG["rolling_window"], center=True, min_periods=5).median()
    block_days = {d for _, a, b in blocks for d in pd.date_range(a, b)}
    hol, events = build_holidays(daily, daily["bulk"].to_dict(), block_days)
    hol.to_csv(out / "holidays.csv", index=False, encoding="utf-8")

    return make_report(sales, data, days, blocks, rep, daily, stock, catalog, maps, events, missing_days, start)


def make_report(sales, data, days, blocks, rep, daily, stock, catalog, maps, events, missing_days, start):
    L = []
    add = L.append
    tot = daily["walkin"] + daily["bulk"]
    normal = tot[tot > 0].median()
    add("=" * 70)
    add("CART DATASET REPORT")
    add("=" * 70)
    add(f"Date range      : {fmt(days[0])} to {fmt(days[-1])} ({len(days)} days)")
    add(f"Rows / products : {len(sales):,} rows, {sales.variantId.nunique()} products")
    add(f"Units           : {int(sales.quantitySold.sum()):,} total, {int(sales[sales.isBulk == 1].quantitySold.sum()):,} bulk "
        f"({int((sales.isBulk == 1).sum())} bulk rows)")
    if rep["dropped_today"][0]:
        add(f"Excluded        : today's partial day ({rep['dropped_today'][0]} rows, {rep['dropped_today'][1]} units)")
    if rep["packet"][0]:
        add(f"Unit conversion : {rep['packet'][0]} packet rows -> pieces ({rep['packet'][1]} -> {rep['packet'][2]} units)")
    add(f"Variants renamed: {rep['renamed']} (current name used, same id kept)")

    add("\nDAYS WITH 0 SALES OR UNDER {:.0%} OF A NORMAL DAY (normal = {:,.0f} units)".format(CONFIG["report_low_share"], normal))
    block_days = {d for _, a, b in blocks for d in pd.date_range(a, b)}
    low = tot[tot < CONFIG["report_low_share"] * normal]
    if low.empty and not missing_days:
        add("  none")
    for d, u in low.items():
        why = ("first day of the data (cart started recording)" if d == start else
               "inside a late-posting block" if d in block_days else "no cause found in the data")
        add(f"  {fmt(d)} ({d.day_name()[:3]}): {int(u)} units - {why}")

    add("\nLATE-POSTING FIXES (walk-in units spread evenly; bulk stays on the upload day)")
    if not blocks:
        add("  none detected")
    for ups, a, b in blocks:
        moved = sum(rep["moved"].get(u, 0) for u in ups)
        add(f"  {', '.join(fmt(u) for u in ups)} -> {fmt(a)} to {fmt(b)}: {moved:,} units")

    add("\nSTOCK-OUTS")
    add(f"  Method: a run of >= {CONFIG['stockout_min_run']} zero days is marked when, given how often the product")
    add(f"  normally sells, a gap that long has < {CONFIG['stockout_p']:.0%} chance of happening by luck.")
    add("  Days before first sale / after last sale count as 'not on the cart'.")
    add(f"  Marked: {int(stock.values.sum()):,} product-days across {int((stock.sum(axis=1) > 0).sum())} products")

    add("\n20 LARGEST BULK ROWS")
    for n, d, u, why in rep["bulk_log"][:20]:
        add(f"  {u:>6,}  {fmt(d)}  {n}  [{why}]")

    add("\nCATALOG / MAPPING / HOLIDAYS")
    add(f"  catalog: {len(catalog)} products, {int(catalog.active.sum())} active, "
        f"{int((catalog.launch_date != '').sum())} launch dates, {int((catalog.delist_date != '').sum())} delisted")
    for o, n, sd, on, nn in maps:
        add(f"  replaced: {o} {on}  ->  {n} {nn} from {sd}")
    add(f"  event days added to holidays: {', '.join(fmt(e) for e in events) or 'none'}")

    add("\nCHECK / UNSURE")
    estimated = catalog.variant_id.isin(MRP_FILL.keys()) & catalog.product_name.map(parse_mrp).isna()
    add(f"  - {int(estimated.sum())} MRPs filled from typical retail prices (no price in the name)")
    add(f"  - {int((catalog.category == 'Other').sum())} products could not be categorised ('Other')")
    add("  - transactions column omitted: the export has no bill counts")
    add("  - moon-based holiday dates can shift by a day; add local festivals for the cart's city")
    for w_ in rep["warnings"]:
        add(f"  - {w_}")
    return "\n".join(L)


# =============================================================================
# STEP 3. TEST THE OUTPUT
# =============================================================================
class Results:
    def __init__(self):
        self.lines, self.failed = [], False

    def check(self, ok, msg, bad="FAIL"):
        status = "PASS" if ok else bad
        self.failed |= status == "FAIL"
        self.lines.append(f"[{status}] {msg}")

    def note(self, msg):
        self.lines.append(f"       {msg}")


def test_output(out, raw_path, cart, bills, today, start):
    res = Results()
    res.lines += [f"TEST RESULTS - cart {cart}  ({pd.Timestamp.now():%d-%m-%Y %H:%M})", "=" * 70]
    for name in OUTPUT_FILES:
        res.check((out / name).exists(), f"{name} exists")
    raw = pd.read_csv(raw_path)
    s = pd.read_csv(out / "sales.csv")
    cat = pd.read_csv(out / "catalog.csv")
    hol = pd.read_csv(out / "holidays.csv")
    res.check(list(s.columns) == SALES_COLS, f"sales.csv columns {list(s.columns)}")

    # Mongo daily table vs the bills
    if bills is None:
        res.check(False, "Mongo cross-check skipped (no Mongo data in this run)", "WARN")
    else:
        mine = bills.groupby(["date", "name"])["q"].sum().rename("expected").reset_index()
        m = mine.merge(raw.rename(columns={"productName": "name"}), on=["date", "name"], how="outer")
        diff = int((m["expected"].fillna(-1) != m["quantitySold"].fillna(-1)).sum())
        res.check(diff == 0, f"Mongo daily table matches {SOURCE}: {len(m):,} product-days, "
                             f"{int(bills['q'].sum()):,} units, {diff} mismatches")
        other = bills.groupby("status")["q"].sum().drop("COMPLETED", errors="ignore")
        res.check(other.sum() == 0, f"only COMPLETED bills counted ({int(other.sum()):,} units from other "
                                    "statuses; see EXCLUDE_BILLS)", "WARN")
        for st, u in other.items():
            res.note(f"{st}: {int(u):,} units")

    # totals and rows
    # expected = raw units after the build's own adjustments: packets -> pieces, days before --start
    # and from --today on dropped
    rd = pd.to_datetime(raw["date"], format=DATE_FMT)
    name = raw["productName"].str.replace("\xa0", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
    packet = raw["variantId"].isin(PACKET_SIZES) & name.str.contains(PACKET_NAME_PATTERN)
    units = raw["quantitySold"].where(~packet, raw["quantitySold"] * raw["variantId"].map(PACKET_SIZES))
    kept = (rd < pd.Timestamp(today)) & (rd >= (pd.Timestamp(start) if start else rd.min()))
    expected = int(units[kept].sum())
    res.check(expected == s["quantitySold"].sum(),
              f"total units kept: raw {int(raw['quantitySold'].sum()):,} -> expected {expected:,} -> "
              f"sales {int(s['quantitySold'].sum()):,}")
    if packet.any():
        res.note(f"{int(packet.sum())} packet rows converted to pieces "
                 f"({int(raw.loc[packet, 'quantitySold'].sum()):,} -> {int(units[packet].sum()):,} units)")
    if (~kept).any():
        res.note(f"{int((~kept).sum())} rows outside the date window ({int(units[~kept].sum()):,} units) excluded")
    dup = int(s.duplicated(["variantId", "date", "isBulk"]).sum())
    res.check(dup == 0, f"no duplicate (variantId, date, isBulk) rows ({dup} found)")
    res.check((s["quantitySold"] >= 0).all(), "no negative quantities")
    bad = int(((s["isStockout"] == 1) & (s["quantitySold"] > 0)).sum())
    res.check(bad == 0, f"no stock-out day with sales ({bad} found)")
    res.check(set(s["variantId"]) == set(cat["variant_id"]),
              f"catalog covers every product in sales ({cat['variant_id'].nunique()} vs {s['variantId'].nunique()})")

    # data quality
    s["d"] = pd.to_datetime(s["date"], format=DATE_FMT)
    rd = pd.to_datetime(raw["date"], format=DATE_FMT)
    first, last = rd.groupby(raw["variantId"]).min(), rd.groupby(raw["variantId"]).max()
    so = s[s["isStockout"] == 1]
    outside = int(((so["d"] < so["variantId"].map(first)) | (so["d"] > so["variantId"].map(last))).sum())
    res.check(outside == 0, f"stock-outs inside product life only: {len(so):,} stock-out rows, "
                            f"{outside:,} ({outside / max(len(so), 1):.0%}) before first / after last sale", "WARN")
    data_years = set(s["d"].dt.year)
    hol_years = set(pd.to_datetime(hol["date"], format=DATE_FMT).dt.year)
    res.check(data_years <= hol_years, f"holidays cover every data year (data {sorted(data_years)}, "
                                       f"holidays {sorted(hol_years)})", "WARN")
    miss = int(cat["mrp"].isna().sum())
    res.check(miss == 0, f"every product has an MRP ({miss} missing, {int(s['mrp'].isna().sum()):,} sales rows)", "WARN")
    other = int((cat["category"] == "Other").sum())
    res.check(other == 0, f"every product categorised ({other} of {len(cat)} are 'Other')", "WARN")

    # merged from data_validation.py: calendar completeness, recent zero-sales anomaly, catalog uniqueness
    daily_all = s.groupby("d")["quantitySold"].sum().sort_index()
    if len(daily_all) >= 2:
        full_days = pd.date_range(daily_all.index.min(), daily_all.index.max(), freq="D")
        missing_cal = full_days.difference(daily_all.index)
        res.check(len(missing_cal) == 0,
                  f"{len(missing_cal)} calendar dates have no sales rows at all; "
                  "verify genuine zero-sales vs missing export", "WARN")
    if len(daily_all) >= 29:
        recent = daily_all.tail(3)
        baseline = daily_all.iloc[:-3].tail(28).median()
        anomaly = baseline > 0 and (recent == 0).any()
        res.check(not anomaly,
                  f"no zero-sales anomaly in latest 3 days vs 28-day baseline median {baseline:g}", "WARN")
    dup_cat = int(cat["variant_id"].duplicated().sum())
    res.check(dup_cat == 0, f"catalog.csv has no duplicate variant_id ({dup_cat} found)")

    sold = s[s["quantitySold"] > 0]
    res.lines += ["", "SUMMARY",
                  f"  date range   : {s['d'].min():%d-%m-%Y} to {s['d'].max():%d-%m-%Y}",
                  f"  products     : {s['variantId'].nunique()} ({int(cat['active'].sum())} active)",
                  f"  sales rows   : {len(sold):,} with sales, {len(so):,} stock-out rows",
                  f"  units        : {int(s['quantitySold'].sum()):,} ({int(s.loc[s.isBulk == 1, 'quantitySold'].sum()):,} bulk)",
                  f"  result       : {'FAIL' if res.failed else 'PASS (see WARN lines)'}"]
    return res


# =============================================================================
# STEP 5. UPLOAD TO S3
# =============================================================================
def upload(out, bucket, prefix, cart):
    """Only the four training files go up; the raw pull, report and test results stay local."""
    import boto3
    folder = f"{prefix.strip('/')}/cart_{cart}/"
    client = boto3.client("s3", region_name=os.environ.get("S3_REGION") or None)
    for name in OUTPUT_FILES:
        client.upload_file(str(out / name), bucket, folder + name)
        print(f"[s3] {name:<20} -> s3://{bucket}/{folder}{name}")
    return f"s3://{bucket}/{folder}"


# =============================================================================
# MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Fetch a cart's sales from MongoDB, build the forecasting CSVs, "
                                            "test them and save them locally and to S3.")
    p.add_argument("--cart", type=int, required=True, help="cartId, e.g. 21")
    p.add_argument("--uri", default=os.environ.get("MONGO_URI"), help="Mongo URI (default: $MONGO_URI)")
    p.add_argument("--db", default=None, help="database name (default: the one in the URI)")
    p.add_argument("--tz", default="Asia/Kolkata", help="timezone that defines a sales day")
    p.add_argument("--out", default=None, help="local output folder (default: cart<ID> next to this script)")
    p.add_argument("--today", default=None, help="YYYY-MM-DD; this day and later are excluded (default: today in --tz)")
    p.add_argument("--start", default=None, help="YYYY-MM-DD first day to include")
    p.add_argument("--skip-aggregate", action="store_true", help="do not rebuild cart_item_daily_sales (read-only)")
    p.add_argument("--from-csv", default=None, help="build from this raw daily CSV instead of MongoDB")
    p.add_argument("--no-s3", action="store_true", help="save locally only")
    p.add_argument("--bucket", default=os.environ.get("S3_BUCKET_NAME", DEFAULT_BUCKET), help="S3 bucket")
    p.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX, help=f"S3 folder (default: {DEFAULT_S3_PREFIX})")
    a = p.parse_args()

    now = datetime.now(ZoneInfo(a.tz))
    today = a.today or now.strftime("%Y-%m-%d")
    out = Path(a.out) if a.out else Path(__file__).resolve().parent / f"cart{a.cart}"
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"cart{a.cart}_daily_sales_raw.csv"

    # 1. fetch
    print(f"\n== 1. FETCH  cart {a.cart}")
    bills = None
    if a.from_csv:
        df = pd.read_csv(a.from_csv)
        print(f"[file] read {len(df):,} rows from {a.from_csv}")
    else:
        df, bills = fetch(a)
    df.to_csv(raw_path, index=False)
    print(f"[file] raw snapshot -> {raw_path}")

    # 2. build
    print("\n== 2. BUILD")
    report = build(raw_path, out, today, a.start)
    (out / f"cart{a.cart}_report.txt").write_text(report, encoding="utf-8")
    print(report)

    # 3. test
    print("\n== 3. TEST")
    res = test_output(out, raw_path, a.cart, bills, today, a.start)
    (out / "test_results.txt").write_text("\n".join(res.lines) + "\n", encoding="utf-8")
    print("\n".join(res.lines))

    # 4. local
    print(f"\n== 4. SAVED LOCALLY  {out}")
    for f in sorted(out.iterdir()):
        if f.is_file():
            print(f"  {f.name:<32} {f.stat().st_size / 1024:>10,.0f} KB")

    # 5. s3
    if a.no_s3:
        print("\n== 5. S3  skipped (--no-s3)")
    elif res.failed:
        print("\n== 5. S3  skipped: tests FAILED, fix them before uploading")
    else:
        print("\n== 5. S3")
        try:
            where = upload(out, a.bucket, a.s3_prefix, a.cart)
            print(f"[s3] uploaded to {where}")
        except Exception as exc:
            print(f"[s3] upload FAILED: {exc}")
            sys.exit(2)
    sys.exit(1 if res.failed else 0)


if __name__ == "__main__":
    main()
