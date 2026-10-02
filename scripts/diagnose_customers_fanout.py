"""
Confirms (or rules out) whether staging.customers_clean has customers
with more than one distinct (customer_state, customer_city) pair, which
would make `SELECT DISTINCT customer_unique_id, customer_state,
customer_city` in load_base_tables() return multiple rows for the same
customer -- fanning out every downstream merge in build_snapshot().

Also checks whether this has already silently duplicated rows in
analytics.customer_snapshot_features (the stored TRAINING data), which
would mean this isn't just a scoring-script issue.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://cassie:cassie@localhost:5432/cassie")
engine = create_engine(DATABASE_URL)

print("=== Checking staging.customers_clean for multiple state/city per customer ===")
customers = pd.read_sql(
    "SELECT DISTINCT customer_unique_id, customer_state, customer_city FROM staging.customers_clean",
    engine,
)
print(f"Total rows from DISTINCT query: {len(customers)}")
print(f"Distinct customer_unique_id values: {customers['customer_unique_id'].nunique()}")

dupe_ids = customers["customer_unique_id"][customers["customer_unique_id"].duplicated(keep=False)]
if len(dupe_ids):
    print(f"\n{dupe_ids.nunique()} customers have MORE THAN ONE (state, city) pair.")
    sample_id = dupe_ids.iloc[0]
    print(f"Sample -- all rows for customer_unique_id = {sample_id!r}:")
    print(customers[customers["customer_unique_id"] == sample_id].to_string(index=False))
else:
    print("\nNo fan-out here -- every customer has exactly one (state, city) pair. Not the cause.")

print("\n=== Checking analytics.customer_snapshot_features (TRAINING data) for duplicate rows ===")
train_features = pd.read_sql(
    "SELECT customer_unique_id, snapshot_date FROM analytics.customer_snapshot_features",
    engine,
)
dupe_pairs = train_features.duplicated(subset=["customer_unique_id", "snapshot_date"], keep=False)
n_dupe_rows = dupe_pairs.sum()
print(f"Total rows: {len(train_features)}")
print(f"Rows that are duplicates of (customer_unique_id, snapshot_date): {n_dupe_rows}")
if n_dupe_rows:
    print(
        f"\n{train_features[dupe_pairs]['customer_unique_id'].nunique()} distinct customers "
        "have duplicate rows within at least one snapshot -- this means the TRAINING "
        "data stored in the DB already has this fan-out baked in, not just the scoring run."
    )
else:
    print("No duplicate (customer_unique_id, snapshot_date) pairs in stored training data.")
