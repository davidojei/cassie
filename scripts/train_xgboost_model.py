"""
Phase 9 — XGBoost, compared directly against the Phase 8 baseline.

Reuses the EXACT same feature set, preprocessing, and train/validation/
test split as scripts/train_baseline_model.py (imported, not
re-implemented) — the only thing that differs between the two models is
the model architecture itself. That's deliberate: if the two are
evaluated on different feature representations, a performance
difference can't be attributed to the model, only to the confound.

Class imbalance handled via sample_weight computed the same way
sklearn's class_weight="balanced" does internally (see
compute_sample_weight below) — same imbalance-handling *methodology* as
the baseline, not two different strategies that would also confound
the comparison.

Usage:
    python scripts/train_xgboost_model.py
"""

import json
import sys
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sqlalchemy import create_engine
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from train_baseline_model import (  # noqa: E402
    DATABASE_URL, NUMERIC_FEATURES, CATEGORICAL_FEATURES, INDICATOR_SPECS,
    LABEL_COLUMN, load_data, build_preprocessor, evaluate,
)


def build_xgb_model() -> XGBClassifier:
    # Round 1 (docs/model_comparison_round1_overfit.md) used 200 trees,
    # depth 4, no regularization, no early stopping -- it overfit badly
    # (train/validation ROC-AUC gap of 0.34, vs the baseline's 0.08).
    # This version constrains capacity and stops training against
    # validation performance directly, rather than a fixed tree count.
    return XGBClassifier(
        n_estimators=1000,          # upper bound only -- early stopping picks the real number
        max_depth=3,                 # shallower trees, less room to memorize noise
        learning_rate=0.03,          # slower learning, needs more rounds but generalizes better
        subsample=0.8,                # each tree sees 80% of rows
        colsample_bytree=0.8,         # each tree sees 80% of features
        reg_alpha=0.5,                  # L1 regularization
        reg_lambda=2.0,                 # L2 regularization
        min_child_weight=10,             # require more evidence before a further split
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )


def stratify_by_history(df, pipeline, feature_cols, split_name: str):
    """Direct test of the Phase 8 cold-start hypothesis: does performance
    differ between customers with only 1 lifetime order (no repeat-purchase
    signal at all) vs 2+ (some real history)? Not left as speculation."""
    from sklearn.metrics import roc_auc_score

    print(f"\n--- Cold-start check on {split_name}: ROC-AUC by history depth ---")
    for label, mask in [
        ("lifetime_orders == 1 (no history)", df["lifetime_orders"] == 1),
        ("lifetime_orders >= 2 (some history)", df["lifetime_orders"] >= 2),
    ]:
        subset = df[mask]
        if subset[LABEL_COLUMN].nunique() < 2:
            print(f"  {label}: n={len(subset)}, only one class present — ROC-AUC undefined")
            continue
        proba = pipeline.predict_proba(subset[feature_cols])[:, 1]
        auc = roc_auc_score(subset[LABEL_COLUMN], proba)
        print(f"  {label}: n={len(subset)}, ROC-AUC={auc:.4f}")


def main():
    engine = create_engine(DATABASE_URL)
    print("Loading analytics.customer_snapshot_features...")
    df = load_data(engine)

    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + list(INDICATOR_SPECS.keys())
    X_train, y_train = train[feature_cols], train[LABEL_COLUMN]
    X_val, y_val = validation[feature_cols], validation[LABEL_COLUMN]
    X_test, y_test = test[feature_cols], test[LABEL_COLUMN]

    print(f"\nTrain: {len(X_train)} rows, churn rate {y_train.mean():.2%}")
    print(f"Validation: {len(X_val)} rows, churn rate {y_val.mean():.2%}")
    print(f"Test: {len(X_test)} rows, churn rate {y_test.mean():.2%}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    print("\nFitting preprocessor on train, transforming train + validation "
          "(validation needed pre-transformed for early stopping's eval_set)...")
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    print("Fitting XGBoost with early stopping against validation "
          "(test is never touched during fitting or stopping)...")
    model = build_xgb_model()
    model.fit(
        X_train_transformed, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val_transformed, y_val)],
        verbose=False,
    )
    print(f"Early stopping selected {model.best_iteration + 1} trees "
          f"(of {model.n_estimators} allowed)")

    # Reassemble as a Pipeline from the already-fitted pieces so evaluate()
    # (which expects a full pipeline taking raw feature columns) works
    # identically to the baseline script -- this does NOT refit anything,
    # sklearn Pipelines just call transform/predict on fitted steps in order.
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    results = {
        "train": evaluate(pipeline, X_train, y_train, "TRAIN"),
        "validation": evaluate(pipeline, X_val, y_val, "VALIDATION"),
        "test": evaluate(pipeline, X_test, y_test, "TEST (report only, do not tune against this)"),
    }

    # Direct test of the cold-start hypothesis from docs/baseline_model_results.md
    stratify_by_history(validation, pipeline, feature_cols, "VALIDATION")
    stratify_by_history(test, pipeline, feature_cols, "TEST")

    artifacts_dir = Path(__file__).parent.parent / "artifacts" / "models"
    reports_dir = Path(__file__).parent.parent / "artifacts" / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, artifacts_dir / "xgboost_model.joblib")
    with open(reports_dir / "xgboost_model_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nModel saved to {artifacts_dir / 'xgboost_model.joblib'}")
    print(f"Metrics saved to {reports_dir / 'xgboost_model_metrics.json'}")


if __name__ == "__main__":
    main()
