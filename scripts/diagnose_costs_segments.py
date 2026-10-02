"""
Quick diagnostic -- run this once, paste the output back.
Finds which of raw.customer_costs / raw.customer_segments has more than
one row per customer_unique_id, and shows a sample so we can see WHY
(different columns per row? a time/version column? true duplicates?).
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://cassie:cassie@localhost:5432/cassie")
engine = create_engine(DATABASE_URL)

for table in ["raw.customer_costs", "raw.customer_segments"]:
    print(f"\n=== {table} ===")
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
    print("Columns:", list(df.columns))
    print("Total rows:", len(df))
    print("Distinct customer_unique_id:", df["customer_unique_id"].nunique())

    dupe_ids = df["customer_unique_id"][df["customer_unique_id"].duplicated(keep=False)]
    if len(dupe_ids):
        print(f"{dupe_ids.nunique()} customer_unique_id values appear more than once.")
        sample_id = dupe_ids.iloc[0]
        print(f"\nSample -- all rows for customer_unique_id = {sample_id!r}:")
        print(df[df["customer_unique_id"] == sample_id].to_string(index=False))
    else:
        print("No duplicates -- this table is one row per customer.")
