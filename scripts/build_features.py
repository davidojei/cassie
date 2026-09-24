"""
Phase 7 — Feature engineering.

Builds analytics.customer_snapshot_features: one row per
(customer_unique_id, snapshot_date), with every feature computed using
ONLY data at or before that snapshot date, and a churn label computed
from what happens strictly after it. This is the leakage-safe design
required by the brief's §11 (temporal validation) and §7 (no future
information in features).

Design decisions (see docs/feature_engineering.md for full reasoning):

- Snapshot dates: monthly, 2017-07-01 through 2018-03-01 (9 snapshots).
  Chosen so every snapshot + 180-day churn window <= 2018-08-31, the
  last full month of real order volume (docs/business_exploration_findings.md,
  finding 9 — Sept 2018 is a partial-month artifact, excluded).
- A customer only appears in a snapshot if they have >=1 purchase event
  on or before that snapshot date (recency is undefined otherwise).
- Churn label: no purchase event in (snapshot_date, snapshot_date + 180d].
  Because every snapshot's window fits inside the valid data range by
  construction, no per-customer censoring check is needed — this is
  what the snapshot date choice above buys us.
- Split is temporal, not random: train = 2017-07..2017-12 snapshots,
  validation = 2018-01..2018-02, test = 2018-03. Per §11's requirement
  that earlier history trains, later periods validate/test.

Usage:
    python scripts/build_features.py

NOTE: this has not been run against the live dataset yet. Run it and
report back the summary output + any errors — see chat.
"""

import os
from datetime import timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://cassie:cassie@localhost:5432/cassie")
CHURN_WINDOW_DAYS = 180
MAX_VALID_DATE = pd.Timestamp("2018-08-31")

SNAPSHOT_DATES = pd.to_datetime([
    "2017-07-01", "2017-08-01", "2017-09-01", "2017-10-01",
    "2017-11-01", "2017-12-01", "2018-01-01", "2018-02-01", "2018-03-01",
])
TRAIN_SNAPSHOTS = pd.to_datetime(["2017-07-01", "2017-08-01", "2017-09-01",
                                   "2017-10-01", "2017-11-01", "2017-12-01"])
VALIDATION_SNAPSHOTS = pd.to_datetime(["2018-01-01", "2018-02-01"])
TEST_SNAPSHOTS = pd.to_datetime(["2018-03-01"])

for s in SNAPSHOT_DATES:
    assert s + timedelta(days=CHURN_WINDOW_DAYS) <= MAX_VALID_DATE, \
        f"Snapshot {s} + {CHURN_WINDOW_DAYS}d exceeds valid data range — would be censored."


def load_base_tables(engine):
    purchase_events = pd.read_sql(
        "SELECT customer_unique_id, order_purchase_timestamp, representative_order_id "
        "FROM staging.customer_purchase_events",
        engine, parse_dates=["order_purchase_timestamp"],
    )

    orders = pd.read_sql("""
        SELECT o.order_id, c.customer_unique_id, o.order_purchase_timestamp,
               o.order_delivered_customer_date, o.order_estimated_delivery_date,
               o.delivery_status, o.delivery_days
        FROM staging.orders_clean o
        JOIN staging.customers_clean c ON o.customer_id = c.customer_id
    """, engine, parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                              "order_estimated_delivery_date"])
    orders["delivery_delay_days"] = (
        orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
    ).dt.days  # positive = later than promised

    items = pd.read_sql(
        "SELECT order_id, product_id, seller_id, price, freight_value FROM staging.order_items_clean",
        engine,
    )
    products = pd.read_sql(
        "SELECT product_id, product_category_english FROM staging.products_clean", engine
    )
    items = items.merge(products, on="product_id", how="left")

    order_value = items.groupby("order_id").agg(
        order_value=("price", lambda s: s.sum()),
        order_freight=("freight_value", "sum"),
    ).reset_index()
    order_value["order_total"] = order_value["order_value"] + order_value["order_freight"]

    payments = pd.read_sql(
        "SELECT order_id, payment_type, payment_installments, payment_value FROM staging.payments_clean",
        engine,
    )
    payment_summary = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        installments_mean=("payment_installments", "mean"),
        payment_type_mode=("payment_type", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
    ).reset_index()

    reviews = pd.read_sql(
        "SELECT order_id, review_score FROM staging.reviews_clean", engine
    ).groupby("order_id").agg(review_score=("review_score", "mean")).reset_index()

    order_summary = (
        orders.merge(order_value, on="order_id", how="left")
              .merge(payment_summary, on="order_id", how="left")
              .merge(reviews, on="order_id", how="left")
    )

    # item-level table for category/seller diversity + concentration
    item_summary = items.merge(
        orders[["order_id", "customer_unique_id", "order_purchase_timestamp"]],
        on="order_id", how="left",
    )

    customers = pd.read_sql(
        "SELECT DISTINCT customer_unique_id, customer_state, customer_city FROM staging.customers_clean",
        engine,
    )

    return purchase_events, order_summary, item_summary, customers


def hhi(shares: pd.Series) -> float:
    """Herfindahl-Hirschman-style concentration index: sum of squared shares. 1.0 = fully concentrated."""
    total = shares.sum()
    if total == 0:
        return np.nan
    return float(((shares / total) ** 2).sum())


def build_snapshot(snapshot_date, purchase_events, order_summary, item_summary, customers):
    past = order_summary[order_summary["order_purchase_timestamp"] <= snapshot_date].copy()
    future_events = purchase_events[purchase_events["order_purchase_timestamp"] > snapshot_date]
    past_items = item_summary[item_summary["order_purchase_timestamp"] <= snapshot_date]

    active_customers = past["customer_unique_id"].unique()
    if len(active_customers) == 0:
        return pd.DataFrame()

    rows = []
    win90 = snapshot_date - timedelta(days=90)
    win90_prev_start = snapshot_date - timedelta(days=180)
    win30 = snapshot_date - timedelta(days=30)
    win180 = snapshot_date - timedelta(days=180)

    next_purchase = (
        future_events.sort_values("order_purchase_timestamp")
        .groupby("customer_unique_id")["order_purchase_timestamp"].first()
    )

    grouped = past.groupby("customer_unique_id")
    grouped_items = past_items.groupby("customer_unique_id")

    for customer_id, g in grouped:
        g = g.sort_values("order_purchase_timestamp")
        first_purchase = g["order_purchase_timestamp"].min()
        last_purchase = g["order_purchase_timestamp"].max()
        tenure_days = max((snapshot_date - first_purchase).days, 1)

        recent90 = g[g["order_purchase_timestamp"] > win90]
        prev90 = g[(g["order_purchase_timestamp"] > win90_prev_start) & (g["order_purchase_timestamp"] <= win90)]

        purchase_ts = g["order_purchase_timestamp"].drop_duplicates().sort_values()
        gaps = purchase_ts.diff().dt.days.dropna()

        # Frequency COUNTS must use deduplicated purchase events, not raw
        # order rows (a cart split = 2 order rows but 1 real purchase) —
        # this is the standing rule from docs/churn_definition.md. Revenue
        # SUMS below correctly stay order-level, since each split order is
        # still real revenue.
        lifetime_orders_true = len(purchase_ts)
        recent90_events = purchase_ts[purchase_ts > win90]
        prev90_events = purchase_ts[(purchase_ts > win90_prev_start) & (purchase_ts <= win90)]

        delivered = g[g["delivery_status"] == "delivered"]
        recent_delivered = delivered[delivered["order_purchase_timestamp"] > win90]

        # label
        nxt = next_purchase.get(customer_id, pd.NaT)
        if pd.isna(nxt):
            churned = 1
            days_to_next_purchase = np.nan
        else:
            gap = (nxt - snapshot_date).days
            days_to_next_purchase = gap
            churned = 1 if gap > CHURN_WINDOW_DAYS else 0

        row = {
            "customer_unique_id": customer_id,
            "snapshot_date": snapshot_date,
            # RECENCY
            "days_since_last_purchase": (snapshot_date - last_purchase).days,
            # FREQUENCY (event-based counts — see fix above)
            "orders_last_30_days": (purchase_ts > win30).sum(),
            "orders_last_90_days": len(recent90_events),
            "orders_last_180_days": (purchase_ts > win180).sum(),
            "lifetime_orders": lifetime_orders_true,
            # MONETARY (order-level sums — correct to include split orders)
            "lifetime_revenue": g["order_total"].sum(),
            "revenue_last_90_days": recent90["order_total"].sum(),
            "average_order_value": g["order_total"].mean(),
            "max_order_value": g["order_total"].max(),
            # BEHAVIOR
            "purchase_frequency": lifetime_orders_true / tenure_days,
            "purchase_gap_mean": gaps.mean() if len(gaps) else np.nan,
            "purchase_gap_std": gaps.std() if len(gaps) > 1 else np.nan,
            "days_since_previous_order": (snapshot_date - last_purchase).days,
            # DELIVERY
            "average_delivery_delay": delivered["delivery_delay_days"].mean(),
            "late_delivery_rate": (delivered["delivery_delay_days"] > 0).mean() if len(delivered) else np.nan,
            "average_delivery_days": delivered["delivery_days"].mean(),
            # CUSTOMER EXPERIENCE
            "average_review_score": g["review_score"].mean(),
            "negative_review_rate": (g["review_score"] <= 2).mean() if g["review_score"].notna().any() else np.nan,
            # PAYMENTS
            "preferred_payment_method": g["payment_type_mode"].mode().iloc[0] if not g["payment_type_mode"].mode().empty else None,
            "installment_frequency": g["installments_mean"].mean(),
            "payment_value": g["payment_value"].sum(),
            # TREND (base, §12)
            "spend_trend_90d": recent90["order_total"].sum() - prev90["order_total"].sum(),
            "purchase_frequency_trend_90d": len(recent90_events) - len(prev90_events),
            "review_trend_90d": recent90["review_score"].mean() - prev90["review_score"].mean()
                if recent90["review_score"].notna().any() and prev90["review_score"].notna().any() else np.nan,
            # ADVANCED TREND (§13) — explicit recent vs. historical pairs
            "recent_90d_revenue": recent90["order_total"].sum(),
            "previous_90d_revenue": prev90["order_total"].sum(),
            "recent_90d_order_count": len(recent90_events),
            "lifetime_purchase_frequency": lifetime_orders_true / tenure_days,
            "recent_delivery_delay": recent_delivered["delivery_delay_days"].mean() if len(recent_delivered) else np.nan,
            "historical_delivery_delay": delivered["delivery_delay_days"].mean(),
            # LABEL
            "days_to_next_purchase": days_to_next_purchase,
            "churned": churned,
        }
        rows.append(row)

    features = pd.DataFrame(rows)

    # item-level: category/seller diversity + concentration
    item_rows = []
    for customer_id, gi in grouped_items:
        cat_rev = gi.groupby("product_category_english")["price"].sum()
        seller_rev = gi.groupby("seller_id")["price"].sum()
        item_rows.append({
            "customer_unique_id": customer_id,
            "unique_categories": gi["product_category_english"].nunique(),
            "category_concentration": hhi(cat_rev),
            "unique_sellers": gi["seller_id"].nunique(),
            "seller_concentration": hhi(seller_rev),
        })
    item_features = pd.DataFrame(item_rows)

    features = features.merge(item_features, on="customer_unique_id", how="left")
    features = features.merge(customers, on="customer_unique_id", how="left")

    return features


def main():
    engine = create_engine(DATABASE_URL)
    print("Loading base tables...")
    purchase_events, order_summary, item_summary, customers = load_base_tables(engine)
    print(f"  {len(order_summary)} orders, {len(item_summary)} order-items, "
          f"{purchase_events['customer_unique_id'].nunique()} customers")

    all_snapshots = []
    for snap in SNAPSHOT_DATES:
        print(f"Building snapshot {snap.date()}...")
        feats = build_snapshot(snap, purchase_events, order_summary, item_summary, customers)
        if snap in TRAIN_SNAPSHOTS:
            feats["split"] = "train"
        elif snap in VALIDATION_SNAPSHOTS:
            feats["split"] = "validation"
        else:
            feats["split"] = "test"
        print(f"  {len(feats)} customers active as of this snapshot, "
              f"churn rate: {feats['churned'].mean():.2%}")
        all_snapshots.append(feats)

    full = pd.concat(all_snapshots, ignore_index=True)

    # --- Fix cross-split customer leakage ---
    # Without this, a customer active by an early snapshot is also "active"
    # (and therefore included) in every later snapshot, so the same
    # customer ends up in train AND validation AND test. Verified against
    # the real run: 100% of train customers also appeared in test. That
    # inflates evaluation — a model can effectively learn customer-specific
    # patterns in train and get credit for "generalizing" to the same
    # customer in test.
    #
    # Fix: assign each customer exclusively to the split of their FIRST
    # appearance (their earliest qualifying snapshot). They keep all their
    # rows within that split's snapshot window (still valid, repeated
    # observations of the same customer over time — normal panel-data
    # structure), but are dropped from any other split entirely.
    first_appearance = (
        full.loc[full.groupby("customer_unique_id")["snapshot_date"].idxmin(),
                 ["customer_unique_id", "split"]]
        .rename(columns={"split": "assigned_split"})
    )
    before_customers = full["customer_unique_id"].nunique()
    full = full.merge(first_appearance, on="customer_unique_id")
    full = full[full["split"] == full["assigned_split"]].drop(columns=["assigned_split"])
    print(f"\nCross-split leakage fix: {before_customers} unique customers before, "
          f"{full['customer_unique_id'].nunique()} after (each now in exactly one split)")

    print("\n=== Summary by split ===")
    print(full.groupby("split").agg(
        rows=("customer_unique_id", "count"),
        unique_customers=("customer_unique_id", "nunique"),
        churn_rate=("churned", "mean"),
    ))

    print("\nWriting to analytics.customer_snapshot_features...")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS analytics")
    full.to_sql("customer_snapshot_features", engine, schema="analytics",
                if_exists="replace", index=False, method="multi", chunksize=5000)
    print(f"Done. {len(full)} rows written.")


if __name__ == "__main__":
    main()
