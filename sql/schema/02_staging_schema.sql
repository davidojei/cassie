-- Schema: staging
-- Cleaned, typed, deterministic. No business logic (churn labels,
-- revenue-at-risk, etc.) — that's `analytics`/`business`. Every
-- disposition here traces to a specific line in
-- docs/data_quality_report.md — nothing is dropped or flagged without
-- a documented reason.

CREATE SCHEMA IF NOT EXISTS staging;

-- ============================================================
-- staging.customers_clean
-- Pass-through from raw; customer_unique_id is the persistent identity
-- (see docs/data_dictionary.md "customer identity rule" — customer_id
-- is per-order, never use it for grouping).
-- ============================================================

CREATE TABLE staging.customers_clean AS
SELECT
    customer_id,
    customer_unique_id,
    customer_zip_code_prefix,
    customer_city,
    customer_state
FROM raw.olist_customers;

ALTER TABLE staging.customers_clean ADD PRIMARY KEY (customer_id);
CREATE INDEX idx_staging_customers_unique_id ON staging.customers_clean (customer_unique_id);

-- ============================================================
-- staging.orders_clean
-- Adds: delivery_status (instead of imputing missing delivery dates —
-- see data_quality_report.md, 2.98% null delivered_customer_date is a
-- real business state, not bad data), delivery_days,
-- late_delivery_outlier (99th percentile = 46 days; the formal IQR
-- threshold of 28.5 days was tried and rejected as over-flagging routine
-- variance — see data_quality_report.md).
-- ============================================================

CREATE TABLE staging.orders_clean AS
SELECT
    order_id,
    customer_id,
    order_status,
    order_purchase_timestamp,
    order_approved_at,
    order_delivered_carrier_date,
    order_delivered_customer_date,
    order_estimated_delivery_date,
    CASE
        WHEN order_delivered_customer_date IS NOT NULL THEN 'delivered'
        WHEN order_status = 'canceled' THEN 'canceled'
        ELSE 'in_progress_or_undelivered'
    END AS delivery_status,
    EXTRACT(DAY FROM (order_delivered_customer_date - order_purchase_timestamp))::INTEGER AS delivery_days
FROM raw.olist_orders;

ALTER TABLE staging.orders_clean ADD PRIMARY KEY (order_id);
CREATE INDEX idx_staging_orders_customer_id ON staging.orders_clean (customer_id);
CREATE INDEX idx_staging_orders_purchase_ts ON staging.orders_clean (order_purchase_timestamp);

ALTER TABLE staging.orders_clean ADD COLUMN late_delivery_outlier BOOLEAN;
UPDATE staging.orders_clean SET late_delivery_outlier = (delivery_days > 46);

-- ============================================================
-- staging.customer_purchase_events
-- THE key output of Phase 2: distinct (customer_unique_id,
-- order_purchase_timestamp) pairs. This corrects for cart-splitting
-- (see docs/churn_definition.md — 62.4% of same-timestamp order pairs
-- are confirmed different-seller cart splits). Every downstream
-- frequency/recency/churn feature must be built from this table, NOT
-- from raw order counts.
-- ============================================================

CREATE TABLE staging.customer_purchase_events AS
SELECT DISTINCT
    c.customer_unique_id,
    o.order_purchase_timestamp,
    -- keep one representative order_id per event for traceability/joins
    MIN(o.order_id) AS representative_order_id
FROM staging.orders_clean o
JOIN staging.customers_clean c ON o.customer_id = c.customer_id
GROUP BY c.customer_unique_id, o.order_purchase_timestamp;

CREATE INDEX idx_purchase_events_customer ON staging.customer_purchase_events (customer_unique_id);
CREATE INDEX idx_purchase_events_ts ON staging.customer_purchase_events (order_purchase_timestamp);

-- ============================================================
-- staging.order_items_clean / payments_clean / reviews_clean
-- No quality issues found (see data_quality_report.md) — pass-through
-- with consistent naming, kept as separate staging tables rather than
-- reading raw.* directly downstream, so `analytics` always reads from
-- `staging`, never `raw`.
-- ============================================================

CREATE TABLE staging.order_items_clean AS
SELECT * FROM raw.olist_order_items;
ALTER TABLE staging.order_items_clean ADD PRIMARY KEY (order_id, order_item_id);
CREATE INDEX idx_staging_items_order_id ON staging.order_items_clean (order_id);

CREATE TABLE staging.payments_clean AS
SELECT * FROM raw.olist_order_payments;
ALTER TABLE staging.payments_clean ADD PRIMARY KEY (order_id, payment_sequential);
CREATE INDEX idx_staging_payments_order_id ON staging.payments_clean (order_id);

CREATE TABLE staging.reviews_clean AS
SELECT * FROM raw.olist_order_reviews;
CREATE INDEX idx_staging_reviews_order_id ON staging.reviews_clean (order_id);

-- ============================================================
-- staging.products_clean
-- Nulls in category/dimension fields (1.85%) kept, category coalesced
-- to 'unknown' rather than dropped (see data_quality_report.md).
-- Two untranslated categories (pc_gamer,
-- portateis_cozinha_e_preparadores_de_alimentos) mapped manually.
-- ============================================================

CREATE TABLE staging.products_clean AS
SELECT
    p.product_id,
    COALESCE(t.product_category_name_english,
             CASE p.product_category_name
                 WHEN 'pc_gamer' THEN 'pc_gamer'
                 WHEN 'portateis_cozinha_e_preparadores_de_alimentos' THEN 'kitchen_appliances_food_prep'
                 ELSE 'unknown'
             END) AS product_category_english,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm
FROM raw.olist_products p
LEFT JOIN raw.product_category_translation t
    ON p.product_category_name = t.product_category_name;

ALTER TABLE staging.products_clean ADD PRIMARY KEY (product_id);

CREATE TABLE staging.sellers_clean AS
SELECT * FROM raw.olist_sellers;
ALTER TABLE staging.sellers_clean ADD PRIMARY KEY (seller_id);
