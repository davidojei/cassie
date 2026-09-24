import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "scripts"))
import pandas as pd
import numpy as np
from build_features import build_snapshot, SNAPSHOT_DATES, hhi

# --- synthetic data mimicking the real schema shapes ---
customers = pd.DataFrame({
    "customer_unique_id": ["c1", "c2", "c3", "c4"],
    "customer_state": ["SP", "RJ", "SP", "MG"],
    "customer_city": ["sao paulo", "rio", "sao paulo", "bh"],
})

# c1: repeat buyer, 3 orders spread out. c2: one-time buyer. c3: buyer with an order AFTER snapshot (tests label).
# c4: CART-SPLIT case — 2 orders at the exact same timestamp (tests the
# frequency-count fix: lifetime_orders should be 1, not 2).
orders_raw = [
    ("o1", "c1", "2017-01-10", "2017-01-20", "2017-01-18", "delivered", 10),
    ("o2", "c1", "2017-04-15", "2017-04-25", "2017-04-22", "delivered", 10),
    ("o3", "c1", "2017-06-20", "2017-07-02", "2017-06-28", "delivered", 12),
    ("o4", "c2", "2017-03-01", "2017-03-15", "2017-03-10", "delivered", 14),
    ("o5", "c3", "2017-02-01", "2017-02-10", "2017-02-08", "delivered", 9),
    ("o6", "c3", "2018-01-15", "2018-01-25", "2018-01-20", "delivered", 10),  # future purchase after snapshot
    ("o7", "c4", "2017-05-01", "2017-05-11", "2017-05-09", "delivered", 10),
    ("o8", "c4", "2017-05-01", "2017-05-11", "2017-05-09", "delivered", 10),  # same timestamp as o7 -- cart split
]
orders = pd.DataFrame(orders_raw, columns=[
    "order_id", "customer_unique_id", "order_purchase_timestamp",
    "order_delivered_customer_date", "order_estimated_delivery_date",
    "delivery_status", "delivery_days",
])
for c in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
    orders[c] = pd.to_datetime(orders[c])
orders["delivery_delay_days"] = (orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]).dt.days

order_summary = orders.copy()
order_summary["order_total"] = [150.0, 200.0, 90.0, 300.0, 60.0, 120.0, 80.0, 45.0]
order_summary["payment_value"] = order_summary["order_total"]
order_summary["installments_mean"] = [1, 2, 1, 3, 1, 1, 1, 1]
order_summary["payment_type_mode"] = ["credit_card"] * 8
order_summary["review_score"] = [5, 4, 3, 2, 5, 4, 4, 4]

item_summary = pd.DataFrame({
    "order_id": ["o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8"],
    "product_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
    "seller_id": ["s1", "s2", "s1", "s3", "s4", "s4", "s5", "s6"],
    "price": [140.0, 190.0, 85.0, 290.0, 55.0, 110.0, 75.0, 40.0],
    "freight_value": [10.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 5.0],
    "product_category_english": ["toys", "toys", "electronics", "furniture", "beauty", "beauty", "sports", "sports"],
})
item_summary = item_summary.merge(
    order_summary[["order_id", "customer_unique_id", "order_purchase_timestamp"]],
    on="order_id",
)

purchase_events = order_summary[["customer_unique_id", "order_purchase_timestamp"]].drop_duplicates().copy()
purchase_events["representative_order_id"] = "n/a"  # not used by build_snapshot

# snapshot at 2017-07-01: c1 active (3 orders), c2 active (1 order, no future -> churned),
# c3 active (1 order as of snapshot, but has a future purchase in 2018-01 -> gap = 198 days -> churned=1 since >180)
snap = pd.Timestamp("2017-07-01")
result = build_snapshot(snap, purchase_events, order_summary, item_summary, customers)

print(result[["customer_unique_id", "lifetime_orders", "days_since_last_purchase",
              "average_order_value", "unique_categories", "category_concentration",
              "days_to_next_purchase", "churned"]].to_string(index=False))

assert set(result["customer_unique_id"]) == {"c1", "c2", "c3", "c4"}, "should include all 4 customers active by snapshot"
assert result.loc[result.customer_unique_id == "c1", "lifetime_orders"].iloc[0] == 3
assert result.loc[result.customer_unique_id == "c2", "churned"].iloc[0] == 1, "c2 never buys again -> churned"
c3_gap = result.loc[result.customer_unique_id == "c3", "days_to_next_purchase"].iloc[0]
print(f"\nc3 gap to next purchase: {c3_gap} days (should be ~198)")
assert result.loc[result.customer_unique_id == "c3", "churned"].iloc[0] == 1, "c3's gap (198d) > 180d -> churned"

# The actual fix under test: c4 has 2 ORDER ROWS at the same timestamp
# (a cart split) but only 1 real PURCHASE EVENT. lifetime_orders must be 1.
c4_lifetime_orders = result.loc[result.customer_unique_id == "c4", "lifetime_orders"].iloc[0]
c4_revenue = result.loc[result.customer_unique_id == "c4", "average_order_value"].iloc[0]  # avg is over both order rows
print(f"\nc4 (cart-split, 2 order rows, same timestamp): lifetime_orders = {c4_lifetime_orders} (should be 1, NOT 2)")
assert c4_lifetime_orders == 1, "BUG: cart-split orders are being double-counted as 2 purchases"

print("\nALL SMOKE TEST ASSERTIONS PASSED (including cart-split frequency-count fix)")
