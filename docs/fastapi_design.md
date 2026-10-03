# FastAPI Service — Design (Phase 13)

## What this phase does

Wraps Phase 11's scored customer data and Phase 12's scenario engine
behind HTTP endpoints — the layer Phase 15 (LLM tool-calling) and
Phase 14 (dashboard) will actually call, rather than importing Python
modules directly.

## Scope decision: serves cached results, does not run live model inference

This API reads `outputs/customer_revenue_at_risk.csv` (Phase 11's
output) and calls `scenario_engine.py`'s functions (Phase 12) on
demand. It does **not** re-run `build_snapshot()` or the model's
`predict_proba()` per request — scoring a new customer profile live
isn't in scope here. Rebuilding scores means rerunning
`score_customer_value.py` and restarting/reloading the API, same as
any batch-scored system. Flagging this explicitly because it's an
assumption, not a stated requirement — if Phase 15's LLM tools need to
score a *hypothetical* customer profile on the fly (not one already in
the CSV), that's a different endpoint this phase doesn't build.

## Data loading strategy

Loaded into memory **once at startup** (FastAPI lifespan), not
per-request — 96k rows is small enough that in-memory serving is
simple and fast, and re-reading the CSV on every request would add
needless disk I/O for no benefit, since the data doesn't change
between `score_customer_value.py` runs.

Trade-off made explicit: if someone reruns the scoring script while
the API is live, responses stay stale until either the process
restarts or `POST /admin/reload-data` is called. No file-watcher /
auto-reload — added complexity this phase doesn't need, and silent
auto-reload could change answers mid-session in a way that's harder to
reason about than an explicit reload call.

`GET /health` reports row count and the scored CSV's last-modified
time, so staleness is visible rather than hidden.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | row count, CSV last-modified time, basic liveness |
| GET | `/customers` | paginated list, filterable by `risk_tier`, `segment`, `min_revenue_at_risk` |
| GET | `/customers/{customer_unique_id}` | single customer record, 404 if not found |
| POST | `/scenarios/campaign-roi` | wraps `campaign_roi()` |
| POST | `/scenarios/budget-allocation` | wraps `budget_allocation()` |
| POST | `/scenarios/risk-tier-whatif` | wraps `risk_tier_whatif()` |
| POST | `/admin/reload-data` | re-reads the CSV from disk without restarting |

## Caveats travel with every monetary response — same discipline as Phases 11-12

Every response touching `revenue_at_risk`, `expected_revenue_saved`,
or ROI includes `model_caveat` and, where relevant,
`retention_effect_caveat` as explicit response fields, not just
something a human happened to print in a docstring. A consumer of this
API (including an LLM tool-caller in Phase 15) should not be able to
get the number without also getting the caveat in the same payload.

## Error handling

- Scenario functions already raise `ValueError` for bad input (invalid
  tier, zero cost, mismatched cutoffs/labels) — caught and returned as
  `400` with the original message, not swallowed or turned into a
  generic 500.
- If the scored CSV is missing entirely (scoring script never run),
  `/health` and every data-dependent endpoint return `503` with a
  message pointing at `score_customer_value.py`, not a raw file-not-
  found stack trace.
- Unknown customer ID: `404`, not an empty `200`.

## Response models

Pydantic models for every request/response — gives automatic
validation (e.g. `retention_effect_pct` bounds) and free OpenAPI docs
at `/docs`, which doubles as documentation for Phase 15's tool
definitions later.

## What's NOT in this phase

- No auth — this is a local portfolio demo, not a deployed multi-tenant
  service. Phase 20 (deployment) is the natural place to revisit this
  if it matters there.
- No live model inference endpoint (see scope decision above).
- Scenario 4 (churn-window what-if) isn't wired up — it wasn't built
  in Phase 12 either.

## Testing

`tests/unit/test_api.py` using FastAPI's `TestClient` against a small
fixture CSV (same philosophy as Phase 12's tests — not the real
production data), covering: health check, customer lookup (found +
404), each scenario endpoint (success + validation-error → 400), and
missing-data → 503.
