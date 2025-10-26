"""Test SHAP import and produce a small explanation for the Random Forest model.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / 'outputs' / 'preprocessed'
MODELS = ROOT / 'models'
OUT = ROOT / 'outputs' / 'models'

# Load test data
X_test_path = PRE / 'X_test.pkl'
y_test_path = PRE / 'y_test.pkl'
if not X_test_path.exists():
    raise FileNotFoundError(f"Missing {X_test_path}")
X_test = joblib.load(X_test_path)
y_test = joblib.load(y_test_path)

# Load Random Forest
rf_path = MODELS / 'random_forest_best_reduced.pkl'
if rf_path.exists():
    print('Loading Random Forest model from', rf_path)
    rf = joblib.load(rf_path)
else:
    rf = None
    print('Random Forest model not found at', rf_path)

# Load SVM
svm_path = MODELS / 'svm_best_reduced.pkl'
if svm_path.exists():
    print('Loading SVM model from', svm_path)
    svm = joblib.load(svm_path)
else:
    svm = None
    print('SVM model not found at', svm_path)

# Test SHAP
try:
    import shap
    print('SHAP version:', shap.__version__)
    # Use a small background sample
    X_small = X_test.iloc[:50] if hasattr(X_test, 'iloc') else X_test[:50]
    if rf is not None:
        print('Creating TreeExplainer for RF...')
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_small)
        print('Computed SHAP values for RF; sample shape:', [s.shape for s in shap_values] if isinstance(shap_values, list) else shap_values.shape)
    if svm is not None:
        print('Creating KernelExplainer for SVM (may be slow)...')
        # use a tiny background
        X_bg = X_small.iloc[:20] if hasattr(X_small, 'iloc') else X_small[:20]
        try:
            expl = shap.KernelExplainer(svm.predict_proba, X_bg)
            sv_svm = expl.shap_values(X_small.iloc[:5])
            print('Computed SHAP (Kernel) for SVM on 5 samples')
        except Exception as e:
            print('SVM SHAP (Kernel) failed:', e)
except Exception as e:
    print('Error importing or running SHAP:', e)

print('Test complete')
