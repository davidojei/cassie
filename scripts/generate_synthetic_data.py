"""
Generates the 5 synthetic enterprise tables scheduled since Phase 1/3
(docs/data_dictionary.md, sql/schema/01_raw_schema.sql) and never built:
customer_support, marketing_campaigns, customer_costs, customer_segments,
retention_interventions.

CRITICAL DESIGN RULE, followed throughout this file: every synthetic
field is generated from REAL, already-observed behavior (delivery
lateness, review scores, spend, purchase recency) -- NEVER from the
future churn label. Wiring synthetic data to the label would manufacture
a result instead of discovering one. If this data ends up helping the
churn model later, it should be because these signals genuinely
correlate with behavior the model already has weak access to, not
because the answer was hidden inside the input.

See docs/synthetic_data_generation.md for every assumption and
correlation choice explained, and for honest expectations about what
this can and can't be expected to do for the churn model.

Usage:
    python scripts/generate_synthetic_data.py
"""

import os
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://cassie:cassie@localhost:5432/cassie")
SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic"
RNG = np.random.default_rng(42)  # fixed seed -- reproducible, documented


def load_real_behavior(engine) -> pd.DataFrame:
    """Per-customer real behavior summary -- the ONLY inputs synthetic
    generation is allowed to depend on.

    IMPORTANT: order_items and payments are pre-aggregated to one row
    per order_id in CTEs BEFORE joining to orders. Joining them directly
    at the orders level (as an earlier version of this query did) causes
    a fan-out -- an order with 3 items and 2 payment installments would
    produce 3x2=6 duplicated rows, silently multiplying total_revenue
    and n_late_deliveries by 6x for that order. Same category of bug as
    Phase 3's cart-split double-counting and Phase 7's order-vs-event
    counting fix -- joining a one-to-many relation without pre-
    aggregating first, showing up a third time in this project."""
    df = pd.read_sql("""
        WITH order_totals AS (
            SELECT order_id, SUM(price + freight_value) AS order_total
            FROM staging.order_items_clean
            GROUP BY order_id
        ),
        order_reviews AS (
            SELECT order_id, AVG(review_score) AS review_score
            FROM staging.reviews_clean
            GROUP BY order_id
        ),
        order_vouchers AS (
            SELECT order_id, MAX(CASE WHEN payment_type = 'voucher' THEN 1 ELSE 0 END) AS has_voucher
            FROM staging.payments_clean
            GROUP BY order_id
        )
        SELECT
            c.customer_unique_id,
            MIN(o.order_purchase_timestamp) AS first_purchase,
            MAX(o.order_purchase_timestamp) AS last_purchase,
            COUNT(DISTINCT o.order_id) AS n_orders,
            AVG(o.delivery_days) AS avg_delivery_days,
            SUM(CASE WHEN o.late_delivery_outlier THEN 1 ELSE 0 END) AS n_late_deliveries,
            AVG(ore.review_score) AS avg_review_score,
            SUM(COALESCE(ot.order_total, 0)) AS total_revenue,
            SUM(COALESCE(ov.has_voucher, 0)) AS n_voucher_payments
        FROM staging.customers_clean c
        JOIN staging.orders_clean o ON c.customer_id = o.customer_id
        LEFT JOIN order_totals ot ON o.order_id = ot.order_id
        LEFT JOIN order_reviews ore ON o.order_id = ore.order_id
        LEFT JOIN order_vouchers ov ON o.order_id = ov.order_id
        GROUP BY c.customer_unique_id
    """, engine, parse_dates=["first_purchase", "last_purchase"])
    df["total_revenue"] = df["total_revenue"].fillna(0)
    df["avg_review_score"] = df["avg_review_score"].fillna(df["avg_review_score"].median())
    df["revenue_percentile"] = df["total_revenue"].rank(pct=True)
    return df


def generate_customer_support(behavior: pd.DataFrame) -> pd.DataFrame:
    """Ticket probability rises with late-delivery count and falls with
    review score -- a realistic, non-circular pattern (complaints follow
    bad delivery/product experiences, not future purchases)."""
    rows = []
    ISSUE_TYPES = ["delivery_delay", "product_defect", "billing", "return_request", "other"]

    for _, c in behavior.iterrows():
        base_rate = 0.08
        late_effect = min(c["n_late_deliveries"] * 0.15, 0.5)
        review_effect = max(0, (3.5 - c["avg_review_score"]) * 0.08)
        ticket_prob = min(base_rate + late_effect + review_effect, 0.85)

        n_tickets = min(RNG.poisson(ticket_prob * 1.5), 4)  # capped so no customer gets an implausible ticket count

        for _ in range(n_tickets):
            span_days = max((c["last_purchase"] - c["first_purchase"]).days, 1)
            created_at = c["first_purchase"] + timedelta(days=int(RNG.uniform(0, span_days + 30)))

            issue_weights = [0.45, 0.2, 0.15, 0.15, 0.05] if c["n_late_deliveries"] > 0 else [0.15, 0.3, 0.2, 0.25, 0.1]
            issue_type = RNG.choice(ISSUE_TYPES, p=issue_weights)
            priority = RNG.choice(["low", "medium", "high"],
                                   p=[0.5, 0.35, 0.15] if issue_type != "billing" else [0.3, 0.4, 0.3])
            resolution_hours = float(RNG.lognormal(mean=2.5 if priority == "high" else 3.2, sigma=0.6))
            resolved = bool(RNG.random() < (0.92 if resolution_hours < 48 else 0.75))
            escalated = bool((not resolved) and RNG.random() < 0.4)
            satisfaction = int(np.clip(
                RNG.normal(4.0 - (resolution_hours / 100) - (0 if resolved else 1.5), 0.8), 1, 5
            ))

            rows.append({
                "ticket_id": str(uuid.uuid4()),
                "customer_unique_id": c["customer_unique_id"],
                "created_at": created_at,
                "issue_type": issue_type,
                "priority": priority,
                "resolution_hours": round(resolution_hours, 1),
                "resolved": resolved,
                "customer_satisfaction": satisfaction,
                "escalated": escalated,
            })
    return pd.DataFrame(rows)


def generate_marketing_campaigns(behavior: pd.DataFrame) -> pd.DataFrame:
    """Campaign FREQUENCY scales with customer value tier (realistic --
    businesses market more to valuable segments). Open/click/convert
    rates are independent random cascades, NOT tied to future purchase,
    to avoid smuggling the label in through the back door."""
    rows = []
    CHANNELS = ["email", "sms", "push"]
    TYPES = ["newsletter", "promotion", "win_back", "product_update"]

    for _, c in behavior.iterrows():
        n_campaigns = int(RNG.poisson(2 + c["revenue_percentile"] * 4))
        for _ in range(n_campaigns):
            span_days = max((c["last_purchase"] - c["first_purchase"]).days, 1) + 60
            campaign_date = c["first_purchase"] + timedelta(days=int(RNG.uniform(0, span_days)))
            channel = RNG.choice(CHANNELS, p=[0.6, 0.25, 0.15])
            campaign_type = RNG.choice(TYPES)
            offer_pct = float(RNG.choice([0, 5, 10, 15, 20], p=[0.4, 0.2, 0.2, 0.15, 0.05]))

            opened = bool(RNG.random() < 0.32)
            clicked = bool(opened and RNG.random() < 0.22)
            converted = bool(clicked and RNG.random() < 0.12)
            cost = {"email": 0.05, "sms": 0.08, "push": 0.02}[channel]

            rows.append({
                "campaign_id": str(uuid.uuid4()),
                "customer_unique_id": c["customer_unique_id"],
                "campaign_date": campaign_date,
                "channel": channel,
                "campaign_type": campaign_type,
                "offer_percentage": offer_pct,
                "opened": opened,
                "clicked": clicked,
                "converted": converted,
                "campaign_cost": cost,
            })
    return pd.DataFrame(rows)


def generate_customer_costs(behavior: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    """Costs derived from legitimate real relationships: support_cost
    from actual generated ticket volume/resolution time, discount_cost
    from actual voucher-payment usage, acquisition_cost as a modest
    random draw (no real acquisition-channel data exists to derive it
    from -- documented as the one largely-unconstrained field)."""
    ticket_cost = support.groupby("customer_unique_id")["resolution_hours"].sum().mul(1.5)  # $1.50/hr support labor, illustrative rate
    df = behavior[["customer_unique_id", "total_revenue", "n_voucher_payments"]].copy()
    df["support_cost"] = df["customer_unique_id"].map(ticket_cost).fillna(0).round(2)
    df["discount_cost"] = (df["n_voucher_payments"] * RNG.uniform(8, 25, size=len(df))).round(2)
    df["acquisition_cost"] = RNG.uniform(15, 60, size=len(df)).round(2)  # no real channel data -- documented assumption
    df["annual_service_cost"] = (df["total_revenue"] * 0.03 + df["support_cost"] * 0.5).round(2)
    df["estimated_processing_cost"] = (df["total_revenue"] * 0.015).round(2)
    return df[["customer_unique_id", "acquisition_cost", "annual_service_cost",
               "support_cost", "discount_cost", "estimated_processing_cost"]]


def generate_customer_segments(behavior: pd.DataFrame) -> pd.DataFrame:
    """Segment/tier derived directly from real revenue percentile --
    exactly how a real business would define these, not arbitrary."""
    df = behavior[["customer_unique_id", "revenue_percentile"]].copy()

    def tier(p):
        if p >= 0.99: return "platinum"
        if p >= 0.90: return "gold"
        if p >= 0.60: return "silver"
        return "bronze"

    df["customer_tier"] = df["revenue_percentile"].apply(tier)
    df["segment"] = df["customer_tier"].map({
        "platinum": "high_value", "gold": "high_value",
        "silver": "standard", "bronze": "low_value",
    })
    df["strategic_account"] = df["customer_tier"] == "platinum"
    manager_pool = [f"manager_{i}" for i in range(1, 8)]
    df["account_manager"] = df.apply(
        lambda r: RNG.choice(manager_pool) if r["customer_tier"] in ("platinum", "gold") else None, axis=1
    )
    return df[["customer_unique_id", "segment", "customer_tier", "account_manager", "strategic_account"]]


def generate_retention_interventions(behavior: pd.DataFrame, max_valid_date: pd.Timestamp) -> pd.DataFrame:
    """Simulates a basic real-world ops process: customers who went
    notably quiet (>150 days since last purchase, as of some point
    safely before the dataset's end) sometimes got a retention offer.
    success_probability reflects a modest, realistic win-back rate --
    NOT tied to whether they actually returned in the real data, since
    that would leak the label directly."""
    rows = []
    TYPES = ["discount_offer", "personal_outreach", "free_shipping", "loyalty_points"]

    for _, c in behavior.iterrows():
        days_quiet = (max_valid_date - c["last_purchase"]).days
        if days_quiet < 150:
            continue
        if RNG.random() > 0.15:  # only a fraction of eligible lapsed customers actually got targeted
            continue

        created_at = c["last_purchase"] + timedelta(days=int(RNG.uniform(150, min(days_quiet, 250))))
        if created_at > max_valid_date:
            continue

        intervention_type = RNG.choice(TYPES, p=[0.4, 0.15, 0.3, 0.15])
        cost = {"discount_offer": 25, "personal_outreach": 12, "free_shipping": 15, "loyalty_points": 8}[intervention_type]
        success_probability = round(float(RNG.uniform(0.08, 0.22)), 3)  # realistic modest win-back rate, not label-derived
        outcome = RNG.choice(["success", "no_response", "declined"], p=[success_probability, 0.6, 0.4 - success_probability])

        rows.append({
            "intervention_id": str(uuid.uuid4()),
            "customer_unique_id": c["customer_unique_id"],
            "intervention_type": intervention_type,
            "cost": cost,
            "success_probability": success_probability,
            "created_at": created_at,
            "outcome": outcome,
        })
    return pd.DataFrame(rows)


def main():
    engine = create_engine(DATABASE_URL)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading real per-customer behavior (the only allowed input)...")
    behavior = load_real_behavior(engine)
    print(f"  {len(behavior)} customers")

    print("Generating customer_support...")
    support = generate_customer_support(behavior)
    print(f"  {len(support)} tickets")

    print("Generating marketing_campaigns...")
    campaigns = generate_marketing_campaigns(behavior)
    print(f"  {len(campaigns)} campaign touches")

    print("Generating customer_costs...")
    costs = generate_customer_costs(behavior, support)
    print(f"  {len(costs)} customers")

    print("Generating customer_segments...")
    segments = generate_customer_segments(behavior)
    print(f"  {len(segments)} customers")

    print("Generating retention_interventions...")
    interventions = generate_retention_interventions(behavior, pd.Timestamp("2018-08-31"))
    print(f"  {len(interventions)} interventions")

    tables = {
        "customer_support": support,
        "marketing_campaigns": campaigns,
        "customer_costs": costs,
        "customer_segments": segments,
        "retention_interventions": interventions,
    }

    print("\nWriting CSVs to data/synthetic/...")
    for name, df in tables.items():
        df.to_csv(SYNTHETIC_DIR / f"{name}.csv", index=False)

    print("Loading into raw.* (truncating first -- these tables' DDL already exists from Phase 3)...")
    with engine.begin() as conn:
        for name in tables:
            conn.execute(text(f"TRUNCATE raw.{name}"))
    for name, df in tables.items():
        df.to_sql(name, engine, schema="raw", if_exists="append", index=False, method="multi", chunksize=5000)
        print(f"  loaded raw.{name}: {len(df)} rows")

    print("\nDone.")


if __name__ == "__main__":
    main()
