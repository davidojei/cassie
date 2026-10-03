"""
scripts/scenario_engine.py

Phase 12: Scenario engine (scenarios 1-3; churn-window what-if deferred
to a separate orchestration script per docs/scenario_engine_design.md).

Consumes outputs/customer_revenue_at_risk.csv (Phase 11). Every
function here is both CLI-callable (for quick checks) and importable
(for the Phase 13 API / Phase 14 dashboard) -- this phase is business
logic only, no UI assumptions baked in.

retention_effect_pct is ALWAYS a user-supplied assumption, never
estimated from retention_interventions data -- see design doc for why.
Every output restates it explicitly, alongside the model's own
ROC-AUC caveat, so a downstream number can never get separated from
either caveat.

Usage examples:
    python scripts/scenario_engine.py campaign-roi --select top_n --n 500 \\
        --cost-per-customer 15 --effect moderate

    python scripts/scenario_engine.py campaign-roi --select risk_tier --tier high \\
        --cost-per-customer 15 --effect 0.25

    python scripts/scenario_engine.py budget-allocation --budget 50000 \\
        --cost-per-customer 15 --effect conservative

    python scripts/scenario_engine.py risk-tier-whatif --cutoffs 0 0.4 0.7 1.0 \\
        --labels low medium high --method absolute
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REVENUE_AT_RISK_PATH = Path(__file__).parent.parent / "outputs" / "customer_revenue_at_risk.csv"

MODEL_CAVEAT = (
    "Estimated based on a model with limited predictive power "
    "(ROC-AUC ~0.55-0.65) - treat as a rough prioritization signal "
    "for where to focus retention effort, not a precise forecast."
)

RETENTION_EFFECT_CAVEAT = (
    "retention_effect_pct is a user-supplied assumption, not derived "
    "from this project's data or any external benchmark -- treat ROI "
    "figures as directional, not promised outcomes."
)

# Illustrative starting points only -- NOT derived from this project's
# data or any external benchmark. See docs/scenario_engine_design.md.
RETENTION_EFFECT_PRESETS = {
    "conservative": 0.10,
    "moderate": 0.20,
    "optimistic": 0.35,
}


def load_scored_customers() -> pd.DataFrame:
    if not REVENUE_AT_RISK_PATH.exists():
        raise FileNotFoundError(
            f"{REVENUE_AT_RISK_PATH} not found -- run scripts/score_customer_value.py first."
        )
    return pd.read_csv(REVENUE_AT_RISK_PATH)


def resolve_retention_effect(effect: str) -> float:
    """Accepts either a preset name or a raw float string like '0.27'."""
    if effect in RETENTION_EFFECT_PRESETS:
        return RETENTION_EFFECT_PRESETS[effect]
    try:
        value = float(effect)
    except ValueError:
        raise ValueError(
            f"--effect must be a preset ({list(RETENTION_EFFECT_PRESETS)}) or a number between 0 and 1, got {effect!r}"
        )
    if not (0 < value <= 1):
        raise ValueError(f"retention_effect_pct must be in (0, 1], got {value}")
    return value


def select_targets(df: pd.DataFrame, selection_type: str, **kwargs) -> pd.DataFrame:
    if selection_type == "top_n":
        n = kwargs["n"]
        return df.sort_values("revenue_at_risk", ascending=False).head(n).copy()
    elif selection_type == "risk_tier":
        tier = kwargs["tier"]
        matched = df[df["risk_tier"] == tier].copy()
        if matched.empty:
            raise ValueError(f"No customers with risk_tier={tier!r}. Valid values: {df['risk_tier'].unique().tolist()}")
        return matched
    elif selection_type == "segment":
        segment = kwargs["segment"]
        if "segment" not in df.columns:
            raise ValueError("No 'segment' column in the scored customer data.")
        matched = df[df["segment"] == segment].copy()
        if matched.empty:
            raise ValueError(f"No customers with segment={segment!r}. Valid values: {df['segment'].unique().tolist()}")
        return matched
    else:
        raise ValueError(f"Unknown selection_type: {selection_type!r}")


def _validate_cost_per_customer(cost_per_customer: float) -> None:
    if cost_per_customer <= 0:
        raise ValueError(f"cost_per_customer must be > 0, got {cost_per_customer}")


def compute_expected_revenue_saved(df: pd.DataFrame, retention_effect_pct: float) -> pd.DataFrame:
    out = df.copy()
    out["expected_revenue_saved"] = (
        out["customer_value"] * out["predicted_churn_probability"] * retention_effect_pct
    )
    return out


def campaign_roi(
    df: pd.DataFrame,
    selection_type: str,
    cost_per_customer: float,
    retention_effect_pct: float,
    **selection_kwargs,
):
    _validate_cost_per_customer(cost_per_customer)
    targeted = select_targets(df, selection_type, **selection_kwargs)
    targeted = compute_expected_revenue_saved(targeted, retention_effect_pct)

    n_targeted = len(targeted)
    total_cost = n_targeted * cost_per_customer
    total_revenue_saved = float(targeted["expected_revenue_saved"].sum())
    net_benefit = total_revenue_saved - total_cost
    roi_pct = (net_benefit / total_cost) if total_cost > 0 else np.nan

    summary = {
        "scenario": "campaign_roi",
        "selection_type": selection_type,
        "selection_params": selection_kwargs,
        "retention_effect_pct": retention_effect_pct,
        "cost_per_customer": cost_per_customer,
        "n_targeted": n_targeted,
        "total_campaign_cost": total_cost,
        "total_expected_revenue_saved": round(total_revenue_saved, 2),
        "net_expected_benefit": round(net_benefit, 2),
        "roi_pct": round(roi_pct, 4) if pd.notna(roi_pct) else None,
        "model_caveat": MODEL_CAVEAT,
        "retention_effect_caveat": RETENTION_EFFECT_CAVEAT,
    }

    # --- Smoke test ---
    assert np.isfinite(total_revenue_saved), "total_expected_revenue_saved is not finite."
    assert np.isfinite(total_cost), "total_campaign_cost is not finite."
    assert n_targeted <= len(df), "Targeted more customers than exist in the scored dataset."

    return summary, targeted


def budget_allocation(
    df: pd.DataFrame,
    total_budget: float,
    cost_per_customer: float,
    retention_effect_pct: float,
):
    _validate_cost_per_customer(cost_per_customer)
    scored = compute_expected_revenue_saved(df, retention_effect_pct)
    scored["expected_revenue_saved_per_dollar"] = scored["expected_revenue_saved"] / cost_per_customer
    scored = scored.sort_values("expected_revenue_saved_per_dollar", ascending=False)

    max_targetable = int(total_budget // cost_per_customer)
    selected = scored.head(max_targetable).copy()

    n_targeted = len(selected)
    total_spent = n_targeted * cost_per_customer
    total_revenue_saved = float(selected["expected_revenue_saved"].sum())
    net_benefit = total_revenue_saved - total_spent

    # Marginal-value curve: does more budget still pay off, or have we
    # already captured the available opportunity? Shown so the chosen
    # budget isn't presented as if it were obviously the right amount.
    curve_rows = []
    for multiplier in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        b = total_budget * multiplier
        k = min(int(b // cost_per_customer), len(scored))
        sub = scored.head(k)
        sub_revenue_saved = float(sub["expected_revenue_saved"].sum())
        curve_rows.append({
            "budget": round(b, 2),
            "n_targeted": k,
            "total_revenue_saved": round(sub_revenue_saved, 2),
            "net_benefit": round(sub_revenue_saved - k * cost_per_customer, 2),
        })
    curve = pd.DataFrame(curve_rows)

    summary = {
        "scenario": "budget_allocation",
        "total_budget": total_budget,
        "cost_per_customer": cost_per_customer,
        "retention_effect_pct": retention_effect_pct,
        "n_targeted": n_targeted,
        "total_spent": total_spent,
        "total_expected_revenue_saved": round(total_revenue_saved, 2),
        "net_expected_benefit": round(net_benefit, 2),
        "model_caveat": MODEL_CAVEAT,
        "retention_effect_caveat": RETENTION_EFFECT_CAVEAT,
    }

    # --- Smoke test ---
    assert n_targeted <= max_targetable, "Targeted more customers than the budget covers."
    assert total_spent <= total_budget, "Total spent exceeds the stated budget."
    assert np.isfinite(total_revenue_saved), "total_expected_revenue_saved is not finite."

    return summary, selected, curve


def risk_tier_whatif(
    df: pd.DataFrame,
    cutoffs: list,
    labels: list,
    method: str = "quantile",
):
    """
    method='quantile': cutoffs are fractions (e.g. [0, 0.5, 0.8, 1.0]),
        same style as Phase 11's current tiers.
    method='absolute': cutoffs are literal probability thresholds
        (e.g. [0, 0.4, 0.7, 1.0]), business-meaningful rather than
        data-relative.
    """
    if len(labels) != len(cutoffs) - 1:
        raise ValueError(f"Need {len(cutoffs) - 1} labels for {len(cutoffs)} cutoffs, got {len(labels)}.")

    out = df.copy()
    if method == "quantile":
        # duplicates="drop" matches the handling already used in Phase 11's
        # cold-start check -- without it, a repeated value at a cutoff point
        # throws an unhandled pandas error instead of degrading gracefully.
        # If edges get dropped, fewer bins than requested are produced and
        # `labels` will mismatch -- surfaced as a clear error rather than
        # a silent misalignment.
        try:
            out["risk_tier_scenario"] = pd.qcut(
                out["predicted_churn_probability"], q=cutoffs, labels=labels, duplicates="drop"
            )
        except ValueError as e:
            raise ValueError(
                f"Could not create {len(labels)} quantile bins from cutoffs {cutoffs} -- "
                f"likely duplicate bin edges in the data. Try fewer/different cutoffs. ({e})"
            )
    elif method == "absolute":
        out["risk_tier_scenario"] = pd.cut(
            out["predicted_churn_probability"], bins=cutoffs, labels=labels, include_lowest=True
        )
    else:
        raise ValueError(f"method must be 'quantile' or 'absolute', got {method!r}")

    summary = out.groupby("risk_tier_scenario", observed=True).agg(
        n_customers=("customer_unique_id", "count"),
        total_revenue_at_risk=("revenue_at_risk", "sum"),
        mean_predicted_churn_probability=("predicted_churn_probability", "mean"),
    ).reset_index()

    # How many customers moved tiers vs. the current Phase 11 tiers --
    # useful to see the practical effect of changing the cutoffs.
    movement = None
    if "risk_tier" in out.columns:
        movement = pd.crosstab(out["risk_tier"], out["risk_tier_scenario"], margins=True)

    return out, summary, movement


def save_scenario_result(name: str, summary: dict, detail_df: pd.DataFrame = None) -> None:
    """
    Persists a scenario run, matching the project's existing convention
    (artifacts/reports/ for summary JSON, outputs/ for row-level CSV --
    see artifacts/reports/baseline_model_metrics.json from Phase 8,
    outputs/customer_revenue_at_risk.csv from Phase 11). Without this,
    a scenario result that ends up in a report or presentation has no
    record of how it was produced.
    """
    reports_dir = Path(__file__).parent.parent / "artifacts" / "reports"
    outputs_dir = Path(__file__).parent.parent / "outputs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary_path = reports_dir / f"scenario_{name}_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved scenario summary to {summary_path}")

    if detail_df is not None:
        detail_path = outputs_dir / f"scenario_{name}_{timestamp}_detail.csv"
        detail_df.to_csv(detail_path, index=False)
        print(f"Saved scenario detail to {detail_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 12 scenario engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("campaign-roi")
    p1.add_argument("--select", choices=["top_n", "risk_tier", "segment"], required=True)
    p1.add_argument("--n", type=int, help="for --select top_n")
    p1.add_argument("--tier", type=str, help="for --select risk_tier")
    p1.add_argument("--segment", type=str, help="for --select segment")
    p1.add_argument("--cost-per-customer", type=float, required=True)
    p1.add_argument("--effect", type=str, required=True, help="preset name or a number in (0,1]")

    p2 = sub.add_parser("budget-allocation")
    p2.add_argument("--budget", type=float, required=True)
    p2.add_argument("--cost-per-customer", type=float, required=True)
    p2.add_argument("--effect", type=str, required=True)

    p3 = sub.add_parser("risk-tier-whatif")
    p3.add_argument("--cutoffs", type=float, nargs="+", required=True)
    p3.add_argument("--labels", type=str, nargs="+", required=True)
    p3.add_argument("--method", choices=["quantile", "absolute"], default="quantile")

    args = parser.parse_args()
    df = load_scored_customers()

    if args.command == "campaign-roi":
        effect = resolve_retention_effect(args.effect)
        kwargs = {}
        if args.select == "top_n":
            if args.n is None:
                parser.error("--select top_n requires --n")
            kwargs["n"] = args.n
        elif args.select == "risk_tier":
            if args.tier is None:
                parser.error("--select risk_tier requires --tier")
            kwargs["tier"] = args.tier
        elif args.select == "segment":
            if args.segment is None:
                parser.error("--select segment requires --segment")
            kwargs["segment"] = args.segment

        summary, targeted = campaign_roi(df, args.select, args.cost_per_customer, effect, **kwargs)
        print(json.dumps(summary, indent=2, default=str))
        save_scenario_result(f"campaign_roi_{args.select}", summary, targeted)

    elif args.command == "budget-allocation":
        effect = resolve_retention_effect(args.effect)
        summary, selected, curve = budget_allocation(df, args.budget, args.cost_per_customer, effect)
        print(json.dumps(summary, indent=2, default=str))
        print("\n--- Marginal value curve (what if budget were different) ---")
        print(curve.to_string(index=False))
        summary["marginal_value_curve"] = curve.to_dict(orient="records")
        save_scenario_result("budget_allocation", summary, selected)

    elif args.command == "risk-tier-whatif":
        out, summary, movement = risk_tier_whatif(df, args.cutoffs, args.labels, args.method)
        print("\n--- New tier summary ---")
        print(summary.to_string(index=False))
        if movement is not None:
            print("\n--- Movement vs. current Phase 11 tiers (rows=old, cols=new) ---")
            print(movement.to_string())
        whatif_summary = {
            "scenario": "risk_tier_whatif",
            "cutoffs": args.cutoffs,
            "labels": args.labels,
            "method": args.method,
            "tier_summary": summary.to_dict(orient="records"),
            "model_caveat": MODEL_CAVEAT,
        }
        save_scenario_result("risk_tier_whatif", whatif_summary, out)


if __name__ == "__main__":
    main()
