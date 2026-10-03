# Phase 13 — FastAPI Service

## What this phase does

Wraps Phase 11's scored customer data and Phase 12's scenario engine
behind HTTP endpoints — the interface Phase 15 (LLM tool-calling) and
Phase 14 (dashboard) will call, instead of importing Python modules
directly.

## Key scope decision: serves cached results, no live model inference

Reads `outputs/customer_revenue_at_risk.csv` at startup and serves it
from memory; calls `scenario_engine.py`'s functions on demand. Does
**not** re-run `build_snapshot()` or `predict_proba()` per request —
scoring a brand-new hypothetical customer isn't in scope here. If a
later phase needs that, it's a different endpoint.

Data is loaded once at startup (96k rows is small, re-reading per
request would be pointless I/O). If `score_customer_value.py` reruns
while the API is live, responses stay stale until `POST
/admin/reload-data` is called or the process restarts — deliberate,
no silent auto-reload. `/health` reports row count and the CSV's
last-modified time so staleness is visible, not hidden.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | row count, data last-modified time |
| GET | `/customers` | paginated, filterable by `risk_tier`/`segment`/`min_revenue_at_risk` |
| GET | `/customers/{id}` | single record, 404 if not found |
| POST | `/scenarios/campaign-roi` | wraps `campaign_roi()` |
| POST | `/scenarios/budget-allocation` | wraps `budget_allocation()` |
| POST | `/scenarios/risk-tier-whatif` | wraps `risk_tier_whatif()` |
| POST | `/admin/reload-data` | re-reads the CSV without restarting |

Interactive docs at `/docs` once running.

## Caveats travel with every monetary response

Every response touching `revenue_at_risk` or ROI includes
`model_caveat` (and `retention_effect_caveat` where relevant) as
explicit response fields — same discipline as Phases 11-12. An LLM
tool-caller in Phase 15 gets the caveat in the same payload as the
number, not as a separate thing it could drop.

## Error handling — two different status codes, by design

- **422**: Pydantic schema validation fails before the handler runs
  (e.g. `cost_per_customer <= 0`, caught by `Field(gt=0)`).
- **400**: a business-logic error raised *inside*
  `scenario_engine.py`'s functions (invalid tier name, mismatched
  cutoffs/labels, missing `n`/`tier`/`segment` for the chosen
  selection type).
- **404**: unknown `customer_unique_id`.
- **503**: scored CSV missing entirely — message points at
  `score_customer_value.py`, not a raw file-not-found trace.

This was caught during testing, not assumed: a first manual test run
expected 400 everywhere and had to be corrected once the actual status
codes were observed.

## Testing

`tests/unit/test_api.py` — 20 tests against a synthetic fixture CSV
written to a temp directory (not the real production data), using
FastAPI's `TestClient`. All 20 pass.

**Gotcha worth keeping in the test file's docstring**: `TestClient`
must be used as a context manager (`with TestClient(app) as client:`)
for the lifespan startup event to actually run and load data. Without
`with`, `/health` silently reports `data_loaded: False` even when the
fixture file exists — this bit the first manual verification pass
before the real test suite existed, and is an easy mistake to repeat
without the note.

## Bug fixed this phase

`CustomerRecord`'s `class Config: extra = "allow"` used Pydantic V1's
deprecated config style — caught via a `PydanticDeprecatedSince20`
warning during test runs, not left in. Fixed to `model_config =
ConfigDict(extra="allow")`. Full suite rerun afterward to confirm
nothing broke.

## What's NOT in this phase

- No auth (local portfolio demo, not deployed) — revisit in Phase 20
  if it becomes relevant there.
- No live model inference endpoint.
- Scenario 4 (churn-window what-if) isn't wired up — it wasn't built
  in Phase 12 either.

## Next phase

Phase 14: dashboard.
