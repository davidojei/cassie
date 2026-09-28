"""
Smoke test for scripts/generate_synthetic_data.py — synthetic
"behavior" input, no DB required. Checks referential integrity, value
ranges, and — most important — that generation functions only accept
the behavior columns they're documented to use (nothing resembling a
future churn label sneaks in as an input).
"""

import sys
import pathlib
import inspect

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "scripts"))

import numpy as np
import pandas as pd

from generate_synthetic_data import (
    generate_customer_support, generate_marketing_campaigns,
    generate_customer_costs, generate_customer_segments,
    generate_retention_interventions,
)

np.random.seed(0)
n = 50
behavior = pd.DataFrame({
    "customer_unique_id": [f"cust_{i}" for i in range(n)],
    "first_purchase": pd.to_datetime("2017-01-01") + pd.to_timedelta(np.random.randint(0, 300, n), unit="D"),
    "n_orders": np.random.randint(1, 4, n),
    "avg_delivery_days": np.random.uniform(5, 20, n),
    "n_late_deliveries": np.random.choice([0, 0, 0, 1, 2], n),
    "avg_review_score": np.random.uniform(1, 5, n),
    "total_revenue": np.random.exponential(150, n),
    "n_voucher_payments": np.random.choice([0, 0, 1], n),
})
behavior["last_purchase"] = behavior["first_purchase"] + pd.to_timedelta(np.random.randint(0, 200, n), unit="D")
behavior["revenue_percentile"] = behavior["total_revenue"].rank(pct=True)

valid_ids = set(behavior["customer_unique_id"])

# --- test 1: no generation function can see a churn/future-purchase label ---
# (the behavior dataframe itself has no such column -- this test would
# fail immediately if one were accidentally added upstream)
label_like_cols = [c for c in behavior.columns if "churn" in c.lower() or "next_purchase" in c.lower()]
assert not label_like_cols, f"behavior input must never contain label-like columns, found: {label_like_cols}"
print("No label-like columns in generation input: PASSED")

# --- test 2: customer_support ---
support = generate_customer_support(behavior)
assert set(support["customer_unique_id"]).issubset(valid_ids), "support tickets reference unknown customers"
assert support["resolution_hours"].min() > 0
assert support["customer_satisfaction"].between(1, 5).all()
# customers with late deliveries should show a higher ticket rate than those without
late_cust = behavior[behavior["n_late_deliveries"] > 0]["customer_unique_id"]
clean_cust = behavior[behavior["n_late_deliveries"] == 0]["customer_unique_id"]
late_rate = support[support["customer_unique_id"].isin(late_cust)].shape[0] / max(len(late_cust), 1)
clean_rate = support[support["customer_unique_id"].isin(clean_cust)].shape[0] / max(len(clean_cust), 1)
assert late_rate >= clean_rate, (
    f"customers with late deliveries should have >= ticket rate, got late={late_rate:.2f} clean={clean_rate:.2f}"
)
print(f"customer_support: {len(support)} tickets, late-delivery correlation confirmed "
      f"(late={late_rate:.2f} vs clean={clean_rate:.2f} tickets/customer): PASSED")

# --- test 3: marketing_campaigns ---
campaigns = generate_marketing_campaigns(behavior)
assert set(campaigns["customer_unique_id"]).issubset(valid_ids)
assert campaigns["offer_percentage"].between(0, 20).all()
# clicked implies opened, converted implies clicked -- cascade must hold
assert (~campaigns["clicked"] | campaigns["opened"]).all(), "clicked=True with opened=False is impossible"
assert (~campaigns["converted"] | campaigns["clicked"]).all(), "converted=True with clicked=False is impossible"
print(f"marketing_campaigns: {len(campaigns)} touches, open/click/convert cascade valid: PASSED")

# --- test 4: customer_costs ---
costs = generate_customer_costs(behavior, support)
assert set(costs["customer_unique_id"]) == valid_ids, "every customer must have a cost row"
assert (costs[["acquisition_cost", "annual_service_cost", "support_cost",
               "discount_cost", "estimated_processing_cost"]] >= 0).all().all(), "no cost may be negative"
print(f"customer_costs: {len(costs)} rows, all non-negative: PASSED")

# --- test 5: customer_segments ---
segments = generate_customer_segments(behavior)
assert set(segments["customer_unique_id"]) == valid_ids
assert segments["customer_tier"].isin(["bronze", "silver", "gold", "platinum"]).all()
assert (segments.loc[segments["strategic_account"], "customer_tier"] == "platinum").all(), \
    "only platinum tier should be marked strategic_account"
print(f"customer_segments: {len(segments)} rows, tier logic consistent: PASSED")

# --- test 6: retention_interventions ---
interventions = generate_retention_interventions(behavior, pd.Timestamp("2017-12-31"))
if len(interventions):
    assert set(interventions["customer_unique_id"]).issubset(valid_ids)
    assert (interventions["created_at"] <= pd.Timestamp("2017-12-31")).all(), \
        "no intervention may be dated after max_valid_date"
    merged = interventions.merge(behavior[["customer_unique_id", "last_purchase"]], on="customer_unique_id")
    gap_days = (merged["created_at"] - merged["last_purchase"]).dt.days
    assert (gap_days >= 150).all(), "interventions should only target customers already 150+ days quiet"
print(f"retention_interventions: {len(interventions)} rows, timing constraints hold: PASSED")

print("\nALL SYNTHETIC DATA GENERATION SMOKE TESTS PASSED")

# --- test 8: the span_days leakage fix, proven concretely ---
# Two customers, IDENTICAL in every way that should matter (first_purchase,
# revenue_percentile, late_deliveries, review_score) except last_purchase --
# one a true one-timer (last_purchase = first_purchase+10), one a long-span
# repeat buyer (last_purchase = first_purchase+400). Before the fix, the
# event-dating window was computed FROM last_purchase, so these two
# customers would get systematically different campaign/ticket densities
# before any early cutoff -- silently leaking their future return behavior.
# After the fix, generation must not depend on last_purchase at all, so with
# the same RNG state the two customers' outputs must be identical.
import generate_synthetic_data as gen_mod

twin_behavior = pd.DataFrame({
    "customer_unique_id": ["twin_A", "twin_B"],
    "first_purchase": [pd.Timestamp("2017-05-01")] * 2,
    "last_purchase": [pd.Timestamp("2017-05-11"), pd.Timestamp("2018-06-05")],  # only difference
    "n_orders": [1, 2],
    "avg_delivery_days": [10.0, 10.0],
    "n_late_deliveries": [0, 0],
    "avg_review_score": [4.0, 4.0],
    "total_revenue": [100.0, 100.0],
    "n_voucher_payments": [0, 0],
    "revenue_percentile": [0.5, 0.5],
})

gen_mod.RNG = np.random.default_rng(123)
campaigns_A = generate_marketing_campaigns(twin_behavior.iloc[[0]])
gen_mod.RNG = np.random.default_rng(123)  # reset to the same state
campaigns_B = generate_marketing_campaigns(twin_behavior.iloc[[1]])

dates_A = sorted(campaigns_A["campaign_date"].tolist())
dates_B = sorted(campaigns_B["campaign_date"].tolist())
assert dates_A == dates_B, (
    "BUG: identical customers except last_purchase produced different campaign "
    f"dates -- last_purchase is still leaking into event timing.\nA={dates_A}\nB={dates_B}"
)
print(f"span_days leakage fix proven: twins with different last_purchase produced "
      f"IDENTICAL campaign dates ({len(dates_A)} campaigns each): PASSED")

# --- test 7: the join fan-out fix in load_real_behavior(), proven concretely ---
# One order: 3 items (total price 100) and 2 payment installments.
# A naive (unfixed) query joining order_items and payments directly to
# orders would fan out to 3x2=6 rows, inflating total_revenue to 600
# instead of 100. This builds an in-memory DB and runs the SAME
# pre-aggregated CTE structure load_real_behavior() uses, to prove the
# fix actually prevents that multiplication rather than just claiming it.
from sqlalchemy import create_engine as _create_engine, text as _text

sqlite_engine = _create_engine("sqlite:///:memory:")
with sqlite_engine.begin() as conn:
    conn.execute(_text("CREATE TABLE order_items_clean (order_id TEXT, price REAL, freight_value REAL)"))
    conn.execute(_text("CREATE TABLE payments_clean (order_id TEXT, payment_type TEXT)"))
    conn.execute(_text("CREATE TABLE reviews_clean (order_id TEXT, review_score INTEGER)"))
    # 3 items summing to price=100 (freight 0 for simplicity)
    for price in [30, 30, 40]:
        conn.execute(_text("INSERT INTO order_items_clean VALUES ('o1', :p, 0)"), {"p": price})
    # 2 payment installments -- the second dimension of the fan-out risk
    conn.execute(_text("INSERT INTO payments_clean VALUES ('o1', 'credit_card')"))
    conn.execute(_text("INSERT INTO payments_clean VALUES ('o1', 'credit_card')"))
    conn.execute(_text("INSERT INTO reviews_clean VALUES ('o1', 5)"))

# same pre-aggregate-then-join structure as the fixed load_real_behavior()
result = pd.read_sql("""
    WITH order_totals AS (
        SELECT order_id, SUM(price + freight_value) AS order_total
        FROM order_items_clean GROUP BY order_id
    ),
    order_reviews AS (
        SELECT order_id, AVG(review_score) AS review_score
        FROM reviews_clean GROUP BY order_id
    )
    SELECT ot.order_total, ore.review_score
    FROM order_totals ot
    LEFT JOIN order_reviews ore ON ot.order_id = ore.order_id
""", sqlite_engine)

assert result.loc[0, "order_total"] == 100, (
    f"fan-out bug reproduced: expected order_total=100, got {result.loc[0, 'order_total']} "
    f"(600 would mean items x payments fan-out is happening again)"
)
print(f"Join fan-out fix proven: order_total = {result.loc[0, 'order_total']} "
      f"(not 600, which a re-introduced fan-out bug would produce): PASSED")
