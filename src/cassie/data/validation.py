"""
Data quality checks, run directly against the loaded `raw` schema in
Postgres. Each check is a single SQL query returning a violation count.

Severity follows docs/data_quality_plan.md: `critical` checks should
halt a staging rebuild in CI; `warning` checks are logged but don't
block anything. Thresholds and dispositions here match the reasoning
already worked out in docs/data_quality_report.md and
docs/churn_definition.md — this module doesn't re-derive them, it
encodes them so they run every time the pipeline runs, not just once
in a notebook.
"""

from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    severity: str          # "critical" | "warning"
    description: str
    violation_count: int

    @property
    def passed(self) -> bool:
        return self.violation_count == 0


CHECKS = [
    {
        "name": "duplicate_pk_orders",
        "severity": "critical",
        "description": "Duplicate order_id in raw.olist_orders",
        "sql": "SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM raw.olist_orders",
    },
    {
        "name": "duplicate_pk_products",
        "severity": "critical",
        "description": "Duplicate product_id in raw.olist_products",
        "sql": "SELECT COUNT(*) - COUNT(DISTINCT product_id) FROM raw.olist_products",
    },
    {
        "name": "duplicate_pk_sellers",
        "severity": "critical",
        "description": "Duplicate seller_id in raw.olist_sellers",
        "sql": "SELECT COUNT(*) - COUNT(DISTINCT seller_id) FROM raw.olist_sellers",
    },
    {
        "name": "orphan_fk_order_items",
        "severity": "critical",
        "description": "order_items rows whose order_id has no matching order",
        "sql": """
            SELECT COUNT(*) FROM raw.olist_order_items oi
            LEFT JOIN raw.olist_orders o ON oi.order_id = o.order_id
            WHERE o.order_id IS NULL
        """,
    },
    {
        "name": "orphan_fk_payments",
        "severity": "critical",
        "description": "payments rows whose order_id has no matching order",
        "sql": """
            SELECT COUNT(*) FROM raw.olist_order_payments p
            LEFT JOIN raw.olist_orders o ON p.order_id = o.order_id
            WHERE o.order_id IS NULL
        """,
    },
    {
        "name": "orphan_fk_reviews",
        "severity": "critical",
        "description": "reviews rows whose order_id has no matching order",
        "sql": """
            SELECT COUNT(*) FROM raw.olist_order_reviews r
            LEFT JOIN raw.olist_orders o ON r.order_id = o.order_id
            WHERE o.order_id IS NULL
        """,
    },
    {
        "name": "negative_price",
        "severity": "critical",
        "description": "order_items.price < 0",
        "sql": "SELECT COUNT(*) FROM raw.olist_order_items WHERE price < 0",
    },
    {
        "name": "negative_freight_value",
        "severity": "critical",
        "description": "order_items.freight_value < 0",
        "sql": "SELECT COUNT(*) FROM raw.olist_order_items WHERE freight_value < 0",
    },
    {
        "name": "negative_payment_value",
        "severity": "critical",
        "description": "payments.payment_value < 0",
        "sql": "SELECT COUNT(*) FROM raw.olist_order_payments WHERE payment_value < 0",
    },
    {
        "name": "delivered_before_purchased",
        "severity": "critical",
        "description": "order_delivered_customer_date earlier than order_purchase_timestamp",
        "sql": """
            SELECT COUNT(*) FROM raw.olist_orders
            WHERE order_delivered_customer_date < order_purchase_timestamp
        """,
    },
    {
        "name": "future_purchase_timestamp",
        "severity": "critical",
        "description": "order_purchase_timestamp later than now()",
        "sql": "SELECT COUNT(*) FROM raw.olist_orders WHERE order_purchase_timestamp > NOW()",
    },
    {
        "name": "review_score_out_of_range",
        "severity": "critical",
        "description": "review_score outside 1-5",
        "sql": "SELECT COUNT(*) FROM raw.olist_order_reviews WHERE review_score NOT BETWEEN 1 AND 5",
    },
    {
        "name": "customer_identity_consistency",
        "severity": "critical",
        "description": "customer_id mapping to more than one customer_unique_id "
                        "(would break the customer_unique_id-as-persistent-identity rule)",
        "sql": """
            SELECT COUNT(*) FROM (
                SELECT customer_id FROM raw.olist_customers
                GROUP BY customer_id
                HAVING COUNT(DISTINCT customer_unique_id) > 1
            ) t
        """,
    },
    {
        "name": "late_delivery_outliers",
        "severity": "warning",
        "description": "Deliveries past the 99th-percentile threshold (46 days) — "
                        "see docs/data_quality_report.md for why 46, not the IQR "
                        "formula's 28.5, was chosen",
        "sql": """
            SELECT COUNT(*) FROM raw.olist_orders
            WHERE EXTRACT(DAY FROM (order_delivered_customer_date - order_purchase_timestamp)) > 46
        """,
    },
    {
        "name": "same_timestamp_order_pairs",
        "severity": "warning",
        "description": "Same customer_unique_id with >1 order at the exact same "
                        "purchase timestamp — cart-split candidates, corrected for "
                        "in staging.customer_purchase_events",
        "sql": """
            SELECT COUNT(*) FROM (
                SELECT c.customer_unique_id, o.order_purchase_timestamp
                FROM raw.olist_orders o
                JOIN raw.olist_customers c ON o.customer_id = c.customer_id
                GROUP BY c.customer_unique_id, o.order_purchase_timestamp
                HAVING COUNT(*) > 1
            ) t
        """,
    },
    {
        "name": "uncategorized_products",
        "severity": "warning",
        "description": "Product categories with no match in the translation table",
        "sql": """
            SELECT COUNT(*) FROM raw.olist_products p
            LEFT JOIN raw.product_category_translation t
                ON p.product_category_name = t.product_category_name
            WHERE p.product_category_name IS NOT NULL AND t.product_category_name IS NULL
        """,
    },
]


def run_all_checks(engine) -> list[CheckResult]:
    """Run every check in CHECKS against the given SQLAlchemy engine."""
    from sqlalchemy import text

    results = []
    with engine.connect() as conn:
        for check in CHECKS:
            count = conn.execute(text(check["sql"])).scalar()
            results.append(CheckResult(
                name=check["name"],
                severity=check["severity"],
                description=check["description"],
                violation_count=count,
            ))
    return results


def any_critical_failed(results: list[CheckResult]) -> bool:
    return any(r.severity == "critical" and not r.passed for r in results)
