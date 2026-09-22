# Data Quality Plan

Automated checks run as part of the `staging` load (Phase 4), before
anything reaches `analytics`. Checks are split into **critical** (fail
the pipeline) and **warning** (log and continue, surface in the
data-quality report).

## Checks

| Check | Tables | Severity |
|---|---|---|
| Nulls in required fields (PKs, FKs, timestamps used for churn windows) | all | Critical |
| Duplicate primary keys | `orders`, `customers`, `products`, `sellers` | Critical |
| Invalid/impossible dates (e.g. delivery before purchase, future timestamps) | `orders` | Critical |
| Negative financial values (`price`, `freight_value`, `payment_value`) | `order_items`, `payments` | Critical |
| Impossible delivery durations (negative, or absurdly long outliers) | `orders` | Warning → investigate distribution before deciding threshold |
| Orphan foreign keys (`order_id` with no matching order, etc.) | `order_items`, `payments`, `reviews` | Critical |
| Inconsistent customer identities (`customer_id` reused across `customer_unique_id`, or vice versa in a way that breaks the 1:many assumption) | `customers` | Critical |
| Duplicate orders (same customer, near-identical items/timestamp) | `orders` | Warning |
| `review_score` out of 1–5 range | `reviews` | Critical |
| Category codes in `products` with no match in the translation table | `products` | Warning |

## Process

1. Run checks in `src/cassie/data/validation.py`, invoked by
   `scripts/run_quality_checks.py`.
2. Critical failures halt the `staging` build (exit non-zero,
   CI-friendly).
3. Every run writes a report to `docs/12_data_quality_report.md`
   (generated, not hand-written) with: rows checked, rows failed per
   check, % of dataset affected, and the disposition (dropped / flagged
   / kept-with-caveat) for each failure category.
4. Any row dropped or corrected is logged with a reason — nothing is
   silently discarded.

## Resolved from the Phase 2 audit (see `docs/data_quality_report.md` for full detail)

- **Delivery duration:** no negative values exist; nothing is
  "impossible." Formal IQR method (Q3 + 1.5×IQR = 28.5 days) was tried
  and rejected — it flags 5.05% of orders, too many to be an actionable
  "outlier" signal on this right-skewed distribution. Using the 99th
  percentile instead: flag (don't drop) anything over 46 days as an
  outlier for review. See `docs/data_quality_report.md` for the full
  reasoning.
- **Missing `order_delivered_customer_date`** (2.98% of orders): kept,
  not dropped or imputed — these are legitimately undelivered/in-transit
  orders. Represented as `delivery_status != delivered`, not a null to
  clean up.
- **All other critical checks passed with zero violations** on the real
  dataset: no duplicate PKs, no orphan FKs, no negative financial
  values, no delivered-before-purchased orders, no future timestamps,
  no out-of-range review scores.
- **New finding not anticipated in the original plan:** cart-splitting
  (multiple `order_id`s per real checkout) inflates naive repeat-purchase
  counts by ~9%. All downstream frequency features must use distinct
  `(customer_unique_id, order_purchase_timestamp)` purchase events, not
  raw order rows — see `docs/data_quality_report.md` and
  `docs/churn_definition.md`.
