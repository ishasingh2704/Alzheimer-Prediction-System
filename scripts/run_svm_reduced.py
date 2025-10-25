"""Run a reduced SVM GridSearch to validate SVM pipeline.
Saves model, plots, and metrics to outputs/models/ and models/.
"""
from pathlib import Path
import time
import json
import joblib
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)

# Paths
ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = ROOT / 'outputs' / 'preprocessed'
MODELS_OUT_DIR = ROOT / 'outputs' / 'models'
MODELS_OUT_DIR.mkdir(parents=True, exist_ok=True)
ROOT_MODELS_DIR = ROOT / 'models'
ROOT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Files
files = {
    'X_train': PREPROCESSED_DIR / 'X_train.pkl',
    'X_val': PREPROCESSED_DIR / 'X_val.pkl',
    'X_test': PREPROCESSED_DIR / 'X_test.pkl',
    'y_train': PREPROCESSED_DIR / 'y_train.pkl',
    'y_val': PREPROCESSED_DIR / 'y_val.pkl',
    'y_test': PREPROCESSED_DIR / 'y_test.pkl',
}

missing = [k for k, p in files.items() if not p.exists()]
if missing:
    print('ERROR: Missing preprocessed files:', missing)
    print('Please run scripts/run_preprocessing_quick.py or notebooks/02_data_preprocessing.ipynb to generate them.')
    sys.exit(1)

# Load
print('Loading preprocessed datasets...')
X_train = joblib.load(files['X_train'])
X_val = joblib.load(files['X_val'])
X_test = joblib.load(files['X_test'])
y_train = joblib.load(files['y_train'])
y_val = joblib.load(files['y_val'])
y_test = joblib.load(files['y_test'])
print('Loaded shapes:', X_train.shape, X_val.shape, X_test.shape)

# Evaluation helper
def evaluate_model(model, X, y, prefix='val'):
    y_pred = model.predict(X)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X)[:, 1]
        except Exception:
            y_proba = None
    elif hasattr(model, 'decision_function'):
        try:
            y_proba = model.decision_function(X)
        except Exception:
            y_proba = None

    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y, y_proba))
        except Exception:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix — {prefix}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = MODELS_OUT_DIR / f'confusion_matrix_{prefix}.png'
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close()

    # ROC curve
    roc_path = None
    if y_proba is not None:
        try:
            fpr, tpr, _ = roc_curve(y, y_proba)
            plt.figure(figsize=(6,5))
            plt.plot(fpr, tpr, label=f'AUC = {metrics.get("roc_auc"):.3f}' if metrics.get('roc_auc') else 'ROC')
            plt.plot([0,1],[0,1],'k--', alpha=0.6)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve — {prefix}')
            plt.legend(loc='lower right')
            roc_path = MODELS_OUT_DIR / f'roc_curve_{prefix}.png'
            plt.tight_layout()
            plt.savefig(roc_path, dpi=200, bbox_inches='tight')
            plt.close()
        except Exception:
            roc_path = None

    metrics['classification_report'] = classification_report(y, y_pred, zero_division=0, output_dict=True)
    metrics['confusion_matrix_path'] = str(cm_path)
    metrics['roc_curve_path'] = str(roc_path) if roc_path is not None else None
    return metrics

# Reduced SVM grid
param_grid = {
    'C': [0.1, 1],
    'kernel': ['linear','rbf'],
    'gamma': ['scale', 0.1]
}

print('Param grid:', param_grid)

svc = SVC(probability=True, random_state=42)
grid = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

print('Starting GridSearchCV for SVM...')
start = time.time()
grid.fit(X_train, y_train)
end = time.time()
print(f'GridSearchCV completed in {(end-start)/60:.2f} minutes')
print('Best params:', grid.best_params_)
print('Best CV score:', grid.best_score_)

best_model = grid.best_estimator_

# Evaluate
print('Evaluating on validation set...')
val_metrics = evaluate_model(best_model, X_val, y_val, prefix='svm_validation')
print('Validation metrics (summary):', {k: v for k, v in val_metrics.items() if k in ['accuracy','precision','recall','f1','roc_auc']})

print('Evaluating on test set...')
test_metrics = evaluate_model(best_model, X_test, y_test, prefix='svm_test')
print('Test metrics (summary):', {k: v for k, v in test_metrics.items() if k in ['accuracy','precision','recall','f1','roc_auc']})

# Save model
model_path = ROOT_MODELS_DIR / 'svm_best_reduced.pkl'
joblib.dump(best_model, model_path)
print('Saved best model to', model_path)

# Append results
results_path = MODELS_OUT_DIR / 'model_comparison.json'
all_results = {}
if results_path.exists():
    try:
        with open(results_path, 'r') as f:
            all_results = json.load(f)
    except Exception:
        all_results = {}

model_name = 'SVM_reduced_grid'
all_results[model_name] = {
    'best_params': grid.best_params_,
    'best_cv_score': grid.best_score_,
    'validation_metrics': val_metrics,
    'test_metrics': test_metrics,
    'saved_model_path': str(model_path),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}

with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print('Appended metrics to', results_path)
print('Done.')
