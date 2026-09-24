"""
Phase 7 follow-up — null-rate check on analytics.customer_snapshot_features.

Prints the % null for every column, broken out by split, so Phase 8
picks imputation strategies from real numbers instead of assumptions.

Usage:
    python scripts/check_feature_nulls.py
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://cassie:cassie@localhost:5432/cassie")

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)


def main():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM analytics.customer_snapshot_features", engine)

    print(f"Total rows: {len(df)}\n")

    null_pct = (df.isnull().mean() * 100).round(2).sort_values(ascending=False)
    print("=== Null % by column (all splits combined) ===")
    print(null_pct[null_pct > 0])

    print("\n=== Null % by column, split by train/validation/test ===")
    for split in ["train", "validation", "test"]:
        sub = df[df["split"] == split]
        sub_null = (sub.isnull().mean() * 100).round(2).sort_values(ascending=False)
        sub_null = sub_null[sub_null > 0]
        if len(sub_null):
            print(f"\n--- {split} ({len(sub)} rows) ---")
            print(sub_null)

    # A specific sanity check: null purchase_gap_* should exactly match
    # lifetime_orders == 1 (can't compute a gap with only one purchase)
    single_order_pct = (df["lifetime_orders"] == 1).mean() * 100
    gap_null_pct = df["purchase_gap_mean"].isnull().mean() * 100
    print(f"\nSanity check: {single_order_pct:.2f}% of rows have lifetime_orders == 1, "
          f"{gap_null_pct:.2f}% have null purchase_gap_mean — these should match closely.")


if __name__ == "__main__":
    main()
