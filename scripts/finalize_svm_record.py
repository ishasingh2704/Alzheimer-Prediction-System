"""
Create or append a record for the SVM run to outputs/models/model_comparison.json
by inspecting saved artifacts (model file, summary, feature importance plot) so we
can avoid re-running the heavy GridSearch if it already completed and saved outputs.

Usage:
  python scripts/finalize_svm_record.py
"""
from pathlib import Path
import json
import re

OUT_DIR = Path("outputs") / "models"
MODELS_DIR = Path("models")
comp_path = OUT_DIR / "model_comparison.json"
model_path = MODELS_DIR / "svm_best_full.pkl"
summary_path = OUT_DIR / "svm_summary.txt"
fi_path = OUT_DIR / "feature_importances_svm_top20.png"

record = {
    "model": "svm",
    "params": {},
    "validation": None,
    "test": None,
    "model_path": str(model_path) if model_path.exists() else None,
    "feature_importance_plot": str(fi_path) if fi_path.exists() else None,
}

# Try to parse test metrics from summary file if it exists
if summary_path.exists():
    txt = summary_path.read_text()
    # patterns like: Overall accuracy: 0.941 — text
    def find_float(prefix):
        m = re.search(rf"{re.escape(prefix)}\s*([0-9]*\.?[0-9]+)", txt)
        return float(m.group(1)) if m else None

    accuracy = find_float("Overall accuracy:")
    recall = find_float("Sensitivity (recall):")
    precision = find_float("Precision:")
    roc_auc = find_float("ROC AUC:")
    test = {}
    if accuracy is not None:
        test["accuracy"] = accuracy
    if precision is not None:
        test["precision"] = precision
    if recall is not None:
        test["recall"] = recall
    if roc_auc is not None:
        test["roc_auc"] = roc_auc
    if test:
        record["test"] = test

# Append robustly to comparison file (handle dict or list)
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

# avoid duplicate entries (match by model_path)
existing_paths = {d.get('model_path') for d in data if isinstance(d, dict)}
if record.get('model_path') not in existing_paths:
    data.append(record)
    comp_path.write_text(json.dumps(data, indent=2))
    print('Appended SVM record to', comp_path)
else:
    print('SVM record already present in', comp_path)

print('Model file exists:', model_path.exists())
print('Summary exists:', summary_path.exists())
print('Feature importance exists:', fi_path.exists())
