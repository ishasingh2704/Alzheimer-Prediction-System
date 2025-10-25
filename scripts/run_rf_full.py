#!/usr/bin/env python3
"""
Full hyperparameter grid search for Random Forest.

Usage:
  python scripts/run_rf_full.py

This script loads preprocessed pickles from `outputs/preprocessed/`, runs
GridSearchCV over a full (but sensible) RandomForest parameter grid, saves the
best model to `models/`, writes evaluation plots and a plain-language summary
to `outputs/models/`, and appends the results to `outputs/models/model_comparison.json`.

Warning: this can be compute- and time-intensive. It uses all available CPU cores by default.
"""
import argparse
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)


def load_data(prep_dir: Path):
    X_train = pd.read_pickle(prep_dir / "X_train.pkl")
    X_val = pd.read_pickle(prep_dir / "X_val.pkl")
    X_test = pd.read_pickle(prep_dir / "X_test.pkl")
    y_train = pd.read_pickle(prep_dir / "y_train.pkl")
    y_val = pd.read_pickle(prep_dir / "y_val.pkl")
    y_test = pd.read_pickle(prep_dir / "y_test.pkl")
    # Ensure y objects are pandas Series for concat and metric compatibility
    import numpy as _np
    if isinstance(y_train, _np.ndarray):
        y_train = pd.Series(y_train)
    if isinstance(y_val, _np.ndarray):
        y_val = pd.Series(y_val)
    if isinstance(y_test, _np.ndarray):
        y_test = pd.Series(y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_and_save(model, X, y, prefix, out_dir: Path):
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)) if probs is not None else None,
    }

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({prefix})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = out_dir / f"confusion_matrix_{prefix}.png"
    plt.savefig(cm_path)
    plt.close()

    # ROC curve
    if probs is not None:
        disp = RocCurveDisplay.from_predictions(y, probs)
        plt.title(f"ROC Curve ({prefix})")
        roc_path = out_dir / f"roc_curve_{prefix}.png"
        plt.savefig(roc_path)
        plt.close()
    else:
        roc_path = None

    return metrics, str(cm_path), str(roc_path) if roc_path is not None else None


def generate_layman_summary(name: str, metrics: dict, top_features: list) -> str:
    lines = [f"Model: {name}", ""]
    lines.append(
        f"Overall accuracy: {metrics['accuracy']:.3f} — the percentage of total cases the model predicted correctly."
    )
    lines.append(
        f"Sensitivity (recall): {metrics['recall']:.3f} — how well the model detects positive cases (true positives)."
    )
    lines.append(
        f"Precision: {metrics['precision']:.3f} — how often a positive prediction is correct."
    )
    if metrics.get("roc_auc") is not None:
        lines.append(
            f"ROC AUC: {metrics['roc_auc']:.3f} — overall ability to separate positive and negative cases."
        )
    lines.append("")
    lines.append("Top features influencing predictions:")
    for feat, imp in top_features:
        lines.append(f" - {feat}: importance {imp:.3f} (higher = more influence)")
    lines.append("")
    lines.append(
        "Interpretation guide: High sensitivity means fewer missed positive cases. High precision means fewer false positives."
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", default="outputs/preprocessed", help="Path to preprocessed pickles")
    parser.add_argument("--models-dir", default="models", help="Where to save trained models")
    parser.add_argument("--out-dir", default="outputs/models", help="Where to write plots and summaries")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    prep_dir = Path(args.preprocessed_dir)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(prep_dir)

    param_grid = {
        "n_estimators": [200, 500, 1000],
        "max_depth": [None, 10, 20, 30],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2],
        "bootstrap": [True],
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=args.n_jobs)

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    gs = GridSearchCV(rf, param_grid, cv=cv, scoring="roc_auc", n_jobs=args.n_jobs, verbose=2)
    print("Starting GridSearchCV for Random Forest (this may take a while)...")
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    print("Best params:", gs.best_params_)
    # Refit on train+val for production readiness
    X_retrain = pd.concat([X_train, X_val], axis=0)
    y_retrain = pd.concat([y_train, y_val], axis=0)
    best.fit(X_retrain, y_retrain)

    # Save model
    model_path = models_dir / "random_forest_best_full.pkl"
    joblib.dump(best, model_path)

    # Feature importances
    importances = best.feature_importances_
    feat_names = X_train.columns.tolist()
    feat_imp = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    top20 = feat_imp[:20]

    # Save importance plot
    plt.figure(figsize=(8, 6))
    names = [n for n, _ in top20][::-1]
    imps = [v for _, v in top20][::-1]
    plt.barh(range(len(names)), imps, color="C2")
    plt.yticks(range(len(names)), names)
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.tight_layout()
    fi_path = out_dir / "feature_importances_rf_top20.png"
    plt.savefig(fi_path)
    plt.close()

    # Evaluate
    val_metrics, val_cm, val_roc = evaluate_and_save(best, X_val, y_val, "validation", out_dir)
    test_metrics, test_cm, test_roc = evaluate_and_save(best, X_test, y_test, "test", out_dir)

    # Human summary
    summary = generate_layman_summary("Random Forest (full)", test_metrics, top20[:10])
    summary_path = out_dir / "random_forest_summary.txt"
    summary_path.write_text(summary)

    # Append to comparison JSON
    comp_path = out_dir / "model_comparison.json"
    record = {
        "model": "random_forest",
        "params": gs.best_params_ if hasattr(gs, "best_params_") else {},
        "validation": val_metrics,
        "test": test_metrics,
        "model_path": str(model_path),
        "feature_importance_plot": str(fi_path),
    }
    # Read existing comparison file robustly (accept list or dict)
    if comp_path.exists():
        try:
            data = json.loads(comp_path.read_text())
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []
    data.append(record)
    comp_path.write_text(json.dumps(data, indent=2))

    print("Done. Artifacts written to:")
    print(" - model:", model_path)
    print(" - summary:", summary_path)
    print(" - comparison JSON:", comp_path)


if __name__ == "__main__":
    main()
