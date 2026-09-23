"""
Load the Olist CSVs into raw.* tables, then build staging.* from them.

Usage:
    python scripts/load_raw_data.py

Expects:
    - Postgres running (docker compose up -d db)
    - DATABASE_URL in .env (see .env.example)
    - CSVs already downloaded into data/raw/ (see
      docs/data_download_instructions.md)
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://cassie:cassie@localhost:5432/cassie")
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
SQL_DIR = Path(__file__).parent.parent / "sql" / "schema"

TABLES = [
    ("olist_customers_dataset.csv", "olist_customers", []),
    ("olist_orders_dataset.csv", "olist_orders", [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]),
    ("olist_order_items_dataset.csv", "olist_order_items", ["shipping_limit_date"]),
    ("olist_order_payments_dataset.csv", "olist_order_payments", []),
    ("olist_order_reviews_dataset.csv", "olist_order_reviews", [
        "review_creation_date", "review_answer_timestamp",
    ]),
    ("olist_products_dataset.csv", "olist_products", []),
    ("olist_sellers_dataset.csv", "olist_sellers", []),
    ("olist_geolocation_dataset.csv", "olist_geolocation", []),
    ("product_category_name_translation.csv", "product_category_translation", []),
]


def run_sql_file(engine, path: Path):
    raw_sql = path.read_text(encoding="utf-8")
    stripped_lines = []
    for line in raw_sql.splitlines():
        idx = line.find("--")
        stripped_lines.append(line[:idx] if idx != -1 else line)
    sql_no_comments = "\n".join(stripped_lines)

    with engine.begin() as conn:
        for statement in sql_no_comments.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
    print(f"  ran {path.name}")


def main():
    engine = create_engine(DATABASE_URL)

    print("1. Creating raw schema + tables...")
    run_sql_file(engine, SQL_DIR / "01_raw_schema.sql")

    print("2. Loading CSVs into raw.*...")
    for filename, table, date_cols in TABLES:
        csv_path = RAW_DIR / filename
        if not csv_path.exists():
            print(f"  SKIPPED {filename} - not found in {RAW_DIR}. "
                  f"See docs/data_download_instructions.md.")
            continue
        df = pd.read_csv(csv_path, parse_dates=date_cols)
        df.to_sql(table, engine, schema="raw", if_exists="append", index=False, method="multi", chunksize=5000)
        print(f"  loaded {len(df):>8} rows -> raw.{table}")

    print("3. Creating staging schema + tables (from raw)...")
    run_sql_file(engine, SQL_DIR / "02_staging_schema.sql")

    print(r"Done. Verify with: docker compose exec db psql -U cassie -d cassie -c '\dt raw.*'")


if __name__ == "__main__":
    main()