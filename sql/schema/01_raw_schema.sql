-- Schema: raw
-- Untouched loads, one table per source file. Column names/types kept
-- close to source. No joins, no derived columns, no deduplication here
-- (deduplication happens in staging — see the cart-split finding in
-- docs/data_quality_report.md).

CREATE SCHEMA IF NOT EXISTS raw;

-- ============================================================
-- Olist source tables
-- ============================================================

CREATE TABLE raw.olist_customers (
    customer_id             TEXT NOT NULL,
    customer_unique_id      TEXT NOT NULL,
    customer_zip_code_prefix TEXT,
    customer_city           TEXT,
    customer_state          TEXT,
    PRIMARY KEY (customer_id)
);
CREATE INDEX idx_raw_customers_unique_id ON raw.olist_customers (customer_unique_id);

CREATE TABLE raw.olist_orders (
    order_id                       TEXT NOT NULL,
    customer_id                    TEXT NOT NULL,
    order_status                   TEXT,
    order_purchase_timestamp       TIMESTAMP,
    order_approved_at              TIMESTAMP,
    order_delivered_carrier_date   TIMESTAMP,
    order_delivered_customer_date  TIMESTAMP,
    order_estimated_delivery_date  TIMESTAMP,
    PRIMARY KEY (order_id)
);
CREATE INDEX idx_raw_orders_customer_id ON raw.olist_orders (customer_id);
CREATE INDEX idx_raw_orders_purchase_ts ON raw.olist_orders (order_purchase_timestamp);

CREATE TABLE raw.olist_order_items (
    order_id        TEXT NOT NULL,
    order_item_id   INTEGER NOT NULL,
    product_id      TEXT NOT NULL,
    seller_id       TEXT NOT NULL,
    shipping_limit_date TIMESTAMP,
    price           NUMERIC(12, 2),
    freight_value   NUMERIC(12, 2),
    PRIMARY KEY (order_id, order_item_id)
);
CREATE INDEX idx_raw_order_items_order_id ON raw.olist_order_items (order_id);
CREATE INDEX idx_raw_order_items_product_id ON raw.olist_order_items (product_id);
CREATE INDEX idx_raw_order_items_seller_id ON raw.olist_order_items (seller_id);

CREATE TABLE raw.olist_order_payments (
    order_id            TEXT NOT NULL,
    payment_sequential  INTEGER NOT NULL,
    payment_type        TEXT,
    payment_installments INTEGER,
    payment_value       NUMERIC(12, 2),
    PRIMARY KEY (order_id, payment_sequential)
);
CREATE INDEX idx_raw_payments_order_id ON raw.olist_order_payments (order_id);

CREATE TABLE raw.olist_order_reviews (
    review_id               TEXT NOT NULL,
    order_id                TEXT NOT NULL,
    review_score            SMALLINT,
    review_comment_title    TEXT,
    review_comment_message  TEXT,
    review_creation_date    TIMESTAMP,
    review_answer_timestamp TIMESTAMP,
    PRIMARY KEY (review_id, order_id)
);
CREATE INDEX idx_raw_reviews_order_id ON raw.olist_order_reviews (order_id);

CREATE TABLE raw.olist_products (
    product_id                  TEXT NOT NULL,
    product_category_name       TEXT,
    product_name_lenght         NUMERIC,
    product_description_lenght  NUMERIC,
    product_photos_qty          NUMERIC,
    product_weight_g            NUMERIC,
    product_length_cm           NUMERIC,
    product_height_cm           NUMERIC,
    product_width_cm            NUMERIC,
    PRIMARY KEY (product_id)
);

CREATE TABLE raw.olist_sellers (
    seller_id               TEXT NOT NULL,
    seller_zip_code_prefix  TEXT,
    seller_city             TEXT,
    seller_state            TEXT,
    PRIMARY KEY (seller_id)
);

CREATE TABLE raw.olist_geolocation (
    geolocation_zip_code_prefix TEXT,
    geolocation_lat              NUMERIC,
    geolocation_lng              NUMERIC,
    geolocation_city             TEXT,
    geolocation_state            TEXT
    -- no PK: legitimately many rows per zip prefix, source data
);
CREATE INDEX idx_raw_geolocation_zip ON raw.olist_geolocation (geolocation_zip_code_prefix);

CREATE TABLE raw.product_category_translation (
    product_category_name          TEXT NOT NULL,
    product_category_name_english  TEXT,
    PRIMARY KEY (product_category_name)
);

-- ============================================================
-- Synthetic enterprise tables (Phase 2/6 generation — schema reserved
-- now, populated later; see docs/data_dictionary.md §2)
-- ============================================================

CREATE TABLE raw.customer_support (
    ticket_id            TEXT NOT NULL,
    customer_unique_id   TEXT NOT NULL,
    created_at            TIMESTAMP,
    issue_type            TEXT,
    priority               TEXT,
    resolution_hours       NUMERIC,
    resolved               BOOLEAN,
    customer_satisfaction  SMALLINT,
    escalated              BOOLEAN,
    PRIMARY KEY (ticket_id)
);
CREATE INDEX idx_raw_support_customer ON raw.customer_support (customer_unique_id);

CREATE TABLE raw.marketing_campaigns (
    campaign_id          TEXT NOT NULL,
    customer_unique_id   TEXT NOT NULL,
    campaign_date         TIMESTAMP,
    channel                TEXT,
    campaign_type          TEXT,
    offer_percentage       NUMERIC,
    opened                 BOOLEAN,
    clicked                BOOLEAN,
    converted              BOOLEAN,
    campaign_cost          NUMERIC(12, 2),
    PRIMARY KEY (campaign_id, customer_unique_id)
);
CREATE INDEX idx_raw_campaigns_customer ON raw.marketing_campaigns (customer_unique_id);

CREATE TABLE raw.customer_costs (
    customer_unique_id       TEXT NOT NULL,
    acquisition_cost          NUMERIC(12, 2),
    annual_service_cost       NUMERIC(12, 2),
    support_cost               NUMERIC(12, 2),
    discount_cost               NUMERIC(12, 2),
    estimated_processing_cost   NUMERIC(12, 2),
    PRIMARY KEY (customer_unique_id)
);

CREATE TABLE raw.customer_segments (
    customer_unique_id    TEXT NOT NULL,
    segment                 TEXT,
    customer_tier            TEXT,
    account_manager           TEXT,
    strategic_account         BOOLEAN,
    PRIMARY KEY (customer_unique_id)
);

CREATE TABLE raw.retention_interventions (
    intervention_id       TEXT NOT NULL,
    customer_unique_id    TEXT NOT NULL,
    intervention_type       TEXT,
    cost                     NUMERIC(12, 2),
    success_probability      NUMERIC,
    created_at                TIMESTAMP,
    outcome                   TEXT,
    PRIMARY KEY (intervention_id)
);
CREATE INDEX idx_raw_interventions_customer ON raw.retention_interventions (customer_unique_id);
