# Cassie — Customer Analytics & Strategic Scenario Intelligence Engine

Cassie is a decision-support platform for a fictional retailer, **Meridian
Commerce Group**, built to answer one executive question with evidence:

> Why are customers leaving, which customers are most important to save,
> how much revenue is at risk, what is driving the risk, and which
> intervention should management prioritize?

Cassie is **not** a churn dashboard and **not** "chat with your CSV." It is
a pipeline: raw business data → analysis → customer risk → explanation →
financial impact → scenario analysis → business recommendation → decision.

Built as a portfolio project demonstrating business analysis, data
analytics, SQL, BI, ML, decision intelligence, and AI/LLM systems —
targeted at Riskgratis Technologies (Nigerian ERP / business-process /
data & analytics consulting).

## Status

**Phase 1 — repository + environment.** Data audit, schema, and models not
yet built. See `docs/23_implementation_roadmap.md` (to be written once
phases begin) and the phase list below.

## Architecture at a glance

- **SQL (PostgreSQL)** is the source of analytical truth — `raw` →
  `staging` → `analytics` → `ml` → `business` schemas.
- **ML models** (Logistic Regression → Random Forest → XGBoost) are an
  analytical engine, not the end product.
- **Business-rule engine** turns model output into financial impact
  (revenue at risk, intervention ROI, what-if scenarios) via
  deterministic Python functions, unit tested.
- **LLM** is the natural-language interface only. It calls tools/functions
  for every number it reports and never invents a business result.

## Dataset

Primary: [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(~100k orders; customers, orders, products, sellers, order items,
payments, reviews, delivery, geolocation), optionally joined with the
Olist Marketing Funnel dataset. Not redistributed in this repo — see
`docs/data_download_instructions.md`.

A synthetic enterprise layer (support tickets, marketing campaigns,
customer costs, segments, interventions, business policies) is added on
top, clearly labeled as synthetic — see `docs/data_dictionary.md`.

**Important:** all retention/churn analysis uses `customer_unique_id`
(the persistent customer identity), never `customer_id` (which is
per-order). Documented in `docs/data_dictionary.md`.

## Repository layout

See `docs/repo_structure.md` for the full tree and what belongs where.

## Phases

1. Repository + environment *(this commit)*
2. Download and inspect Olist data
3. Database schema
4. Data quality
5. Business exploration
6. Define churn
7. Feature engineering
8. Baseline model (Logistic Regression)
9. XGBoost + model comparison
10. SHAP
11. Customer value + revenue at risk
12. Scenario engine
13. FastAPI
14. Dashboard
15. LLM tools
16. Business Analyst mode
17. RAG
18. UAT
19. LLM evaluation
20. Deployment
21. Documentation
22. Final executive presentation

## Setup

```bash
cp .env.example .env        # fill in DB + LLM credentials, never commit .env
pip install -e .             # or: poetry install / uv sync, see pyproject.toml
docker compose up -d db      # PostgreSQL
```

## License

TBD.
