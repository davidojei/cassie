# Data Quality Report

Generated from `notebooks/01_data_audit.ipynb`, run against the full
Olist dataset (9 files, downloaded via Kaggle). Numbers below are exact
(directly computed), not estimated.

## Summary

The dataset is clean. No critical-severity failures. Two structural
findings that changed how downstream analysis is built (purchase-event
deduplication, delivery-outlier method), and one framing finding that
changes what "churn" even means for this business.

## Row counts

| Table | Rows | Columns |
|---|---|---|
| customers | 99,441 | 5 |
| orders | 99,441 | 8 |
| order_items | 112,650 | 7 |
| payments | 103,886 | 5 |
| reviews | 99,224 | 7 |
| products | 32,951 | 9 |
| sellers | 3,095 | 4 |
| geolocation | 1,000,163 | 5 |
| category_translation | 71 | 2 |

Date range: **2016-09-04 to 2018-10-17** (~2.1 years).

## Critical checks — all pass

- Duplicate primary keys: 0 (orders, products, sellers)
- Orphan foreign keys: 0 (order_items, payments, reviews all resolve to
  a real order_id)
- Negative financial values: 0 (price, freight_value, payment_value)
- Orders delivered before purchase: 0
- Future purchase timestamps: 0
- Out-of-range review scores: 0

## Warning-tier findings — disposition decided

| Finding | Disposition |
|---|---|
| `order_delivered_customer_date` null in 2.98% of orders, `order_delivered_carrier_date` null in 1.79%, `order_approved_at` null in 0.16% | Kept, not dropped — these are undelivered/cancelled/in-transit orders, a real business state. Flag as `delivery_status != delivered`, don't impute a date. |
| `review_comment_title` null in 88.3%, `review_comment_message` null in 58.7% | Kept — most reviews are score-only. Expected, not a quality issue. |
| `product_category_name` and related fields null in 1.85% of products | Kept, labeled `category: unknown` — too small to drop. |
| `payment_value == 0` in 9 rows | Kept — almost certainly voucher/full-discount payments. Note in `analytics.customer_revenue_metrics`. |
| 2 product categories absent from English translation (`pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos`) | Kept, translated manually in the feature pipeline rather than dropping those products. |

## Delivery-duration threshold — formal method tried, rejected in favor of a practical one

Distribution: mean 12.1 days, median 10, 99th percentile 46 days, max
209 days, 0 negative values.

**Tried:** the standard IQR outlier rule (Q3 + 1.5×IQR). Q1=6, Q3=15,
IQR=9, giving a threshold of **28.5 days** — which flags **5.05% of all
orders (5,025)** as "outliers."

**Rejected:** 1-in-20 orders is not rare or unusual for this business —
it's routine variance in a right-skewed distribution (long tail of slow
deliveries is normal for e-commerce logistics across a country the size
of Brazil). IQR assumes roughly symmetric spread; applied blindly to a
skewed variable it over-flags ordinary slowness as anomalous. Keeping
this result would mean "investigate 1 in 20 orders," which isn't
actionable.

**Decision:** use the 99th percentile (**46 days**) as the practical
outlier threshold — genuinely rare (~1% of orders), the point past
which a delivery is unusual enough to be worth operational attention,
not just "slower than average." The IQR attempt and its rejection are
kept in this report rather than deleted, since the reasoning (a
textbook method producing a technically correct but practically useless
answer) is worth having on record.

## Structural finding: cart-splitting inflates naive repeat-purchase counts

Olist assigns a separate `order_id` per checkout; a cart split across
multiple sellers can produce multiple `order_id`s at or near the same
timestamp.

- Naive repeat-buyer count (>1 `customer_id` per `customer_unique_id`,
  raw orders): 2,997 customers (3.12%)
- Corrected repeat-buyer count (distinct purchase events — collapsing
  same-timestamp orders): **2,740 customers (2.85%)**
- Of 999 zero-day-gap order pairs, 292 (29.2%) share the exact same
  timestamp — clear cart splits.
- Cross-checked against `seller_id`: **62.4% of zero-gap pairs involve
  completely different sellers**, confirming cart-splitting-by-seller
  as a real mechanism. The remaining 37.6% are not explained by seller
  difference — open item, see `docs/churn_definition.md`.

**Decision:** all downstream frequency/recency features use distinct
`(customer_unique_id, order_purchase_timestamp)` purchase events, not
raw order rows. This is the definition of "a purchase" for the whole
project — documented here, in `docs/data_dictionary.md`, and enforced
in Phase 7 feature engineering.

## Framing finding: this is an overwhelmingly one-time-purchase business

**97.15% of customers never place a second order**, even after the
cart-split correction. This is the most important fact from the audit —
it reframes "churn" from "active user went quiet" to "identify the rare
customer likely to return, and the rarer high-value one among them."
Named as a risk in `docs/16_risk_register.md` (to be written): a model
evaluated on overall accuracy alone will look great by always
predicting "will not return," which is business-useless.

## Churn window — resolved with exact numbers

See `docs/churn_definition.md` for the full decision. Summary: 180-day
window, chosen using exact (not interpolated) coverage computed on the
purchase-event gap distribution — 82.31% coverage, 540 genuine repeat
gaps excluded, versus 66.72%/1,016 at 90 days and 91.48%/260 at 270
days.
