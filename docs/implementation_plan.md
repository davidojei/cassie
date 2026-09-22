# Initial Implementation Plan

Working principle (from the project brief, §42): build → test → inspect
output → explain what was discovered → commit → continue. Nothing moves
to the next phase without that loop closing, and nothing is claimed
(a metric, a finding, a business number) before it's actually produced.

## Phase-by-phase, near-term detail

**Phase 1 — Repository + environment** *(this delivery)*
Directory skeleton, `README.md`, `pyproject.toml`, `.env.example`,
`docker-compose.yml` stub for Postgres, data dictionary, schema
proposal, data-quality plan, business-question backlog. Stop and review
before Phase 2.

**Phase 2 — Download and inspect Olist data**
Download per `data_download_instructions.md`. `notebooks/01_data_audit.ipynb`:
row counts, date ranges, null rates per column, cardinality of keys,
sanity-check the `customer_id` vs `customer_unique_id` relationship
concretely (how many orders per unique customer, in reality). Output
feeds directly into finalizing the data-quality plan's thresholds.

**Phase 3 — Database schema**
Stand up Postgres (docker-compose), write real DDL for `raw`/`staging`
per the schema proposal, load raw CSVs as-is.

**Phase 4 — Data quality**
Implement and run the checks in `data_quality_plan.md` against the
loaded `raw` tables; produce `docs/12_data_quality_report.md`; build
`staging` from what passes.

**Phase 5 — Business exploration**
`notebooks/02_business_exploration.ipynb`: repeat-purchase behavior,
inter-purchase gaps, revenue concentration, review-score distribution —
the inputs Phase 6 needs to pick a churn window defensibly rather than
by assumption.

**Phase 6 — Define churn**
`notebooks/03_churn_definition.ipynb`: evaluate 90/120/180/270-day
inactivity windows against Phase 5's purchase-gap distribution, document
the chosen definition and the rejected alternatives with reasoning
(§7).

**Phase 7 onward**
Feature engineering → baseline model → XGBoost/comparison → SHAP →
customer value/revenue-at-risk → scenario engine → FastAPI → dashboard
→ LLM tools → Business Analyst mode → RAG → UAT → LLM evaluation →
deployment → documentation → executive presentation, each following the
same build/test/inspect/explain/commit loop, each phase's design
decisions documented in the corresponding `docs/` file as they're made
rather than retrofitted at the end.

## What's deliberately not decided yet

- Exact churn window (Phase 6, needs real data)
- Exact feature list beyond the brief's minimum set (Phase 7, may grow
  once Phase 5 surfaces something worth encoding)
- dbt adoption (brief allows deferring it — default to deferring unless
  Phase 3/4 SQL gets unwieldy without it)
- LLM provider/model choice for Phase 15 (not blocking earlier phases)

## Immediate next step

Your call: proceed to Phase 2 (data audit) once you've downloaded the
Olist CSVs into `data/raw/`, or review/adjust anything in Phase 1's
output first.
