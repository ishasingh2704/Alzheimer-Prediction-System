## Model Report — Random Forest & SVM

Date: 2025-10-25

This document summarizes how the two production-ready models were created, what data and preprocessing were used, how hyperparameters were tuned, and how to interpret their outputs (including a short layman-friendly example). Artifacts (models, plots, plain-language summaries) are saved under `outputs/models/` and `models/`.

---

## 1 — Summary

- Models produced: Random Forest (ensemble tree classifier) and SVM (Support Vector Classifier with RBF kernel).
- Task: binary classification (predicts whether a subject is positive for the target diagnosis — e.g., Alzheimer’s vs not). The dataset used is `data/alzheimer_dataset.csv` (2149 rows × 35 columns before preprocessing).
- Purpose: produce accurate and interpretable predictions and provide layman-readable explanations and plots for non-technical stakeholders.

## 2 — Data & preprocessing

- Raw file: `data/alzheimer_dataset.csv` (approx. 2149 rows, 35 columns).
- Preprocessing steps (implemented in `scripts/run_preprocessing_quick.py` and `notebooks/02_data_preprocessing.ipynb`):
  - Missing-value imputation: median for numeric fields, mode for categorical fields.
  - Categorical encoding: label encoding for binary categories, one-hot encoding for multi-class where appropriate.
  - Feature engineering: derived features such as AgeGroup, BMI category, cholesterol risk score, and a simple symptom-count feature (implementation details in `notebooks/02_data_preprocessing.ipynb`).
  - Scaling: numerical features scaled with StandardScaler inside model pipelines (SVM) or kept as-is for Random Forest.

- Preprocessed artifacts (pickles):
  - `outputs/preprocessed/X_train.pkl` (1504 rows × ~43 features)
  - `outputs/preprocessed/X_val.pkl` (322 rows × ~43 features)
  - `outputs/preprocessed/X_test.pkl` (323 rows × ~43 features)
  - `outputs/preprocessed/y_train.pkl`, `y_val.pkl`, `y_test.pkl`

Notes: the shapes reported above are from the quick preprocessing run. If you re-run preprocessing the exact counts will match your final data after feature engineering.

## 3 — Data splits and how they were used

- Train / Validation / Test split used during development:
  - Train: 70% (approx. 1504 rows)
  - Validation: 15% (approx. 322 rows)
  - Test: 15% (approx. 323 rows)

- Training workflow:
 1. GridSearchCV performed on the training set (with Stratified K-Fold CV, default 5 folds) to find best hyperparameters.
 2. After selecting best hyperparameters, the model is re-trained (refit) on Train + Validation for better final performance before saving.
 3. Final evaluation metrics are reported on the held-out Test set.

## 4 — Models and hyperparameter tuning

### Random Forest (full-grid)

- Implementation: `sklearn.ensemble.RandomForestClassifier` (random_state=42). GridSearchCV used to select hyperparameters.
- Full-grid parameters searched (representative):
  - n_estimators: [200, 500, 1000]
  - max_depth: [None, 10, 20, 30]
  - max_features: ['sqrt', 'log2']
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2]
  - bootstrap: [True]

- Best parameters found (final run):
  - bootstrap: True
  - max_depth: 20
  - max_features: 'sqrt'
  - min_samples_leaf: 1
  - min_samples_split: 2
  - n_estimators: 500

### SVM (full-grid)

- Implementation: `sklearn.svm.SVC` inside a `sklearn.pipeline.Pipeline` with `StandardScaler` and `probability=True`.
- Full-grid parameters searched (representative):
  - kernel: ['linear', 'rbf', 'poly']
  - C: [0.1, 1, 10, 100]
  - gamma: ['scale', 'auto', 0.01, 0.001] (for RBF/poly)
  - degree: [2, 3] (for poly)

- Best parameters found (final run):
  - svc__kernel: 'rbf'
  - svc__C: 1
  - svc__gamma: 'scale'

## 5 — Evaluation results (final Test metrics)

These are the final metrics saved after re-fitting best models on Train+Validation and evaluating on Test.

- Random Forest (Test set):
  - Accuracy: 0.9412
  - Precision: 0.9130
  - Recall (Sensitivity): 0.9211
  - F1 score: 0.9170
  - ROC AUC: 0.9505

- SVM (Test set):
  - Accuracy: 0.8390
  - Precision: 0.7583
  - Recall (Sensitivity): 0.7982
  - F1 score: 0.7778
  - ROC AUC: 0.9033

Files with evaluation artifacts (in `outputs/models/`):
- `model_comparison.json` — aggregated records for each model.
- `random_forest_summary.txt`, `svm_summary.txt` — plain-language summaries.
- `feature_importances_rf_top20.png`, `feature_importances_svm_top20.png` — top features plots.
- `confusion_matrix_validation.png`, `confusion_matrix_test.png` and `roc_curve_*.png` — performance plots.

## 6 — What the models do and what data they use

- Both models are binary classifiers: given a subject's features (demographics, clinical measurements, derived features), they output either:
  - A predicted label (0/1 or 'No'/'Yes'), and
  - A probability score (model's confidence that the subject is positive).

- Example input features (after preprocessing) include: age, education, BMI, cholesterol level, presence/absence of specific symptoms, cognitive test scores, and the engineered features mentioned earlier.

## 7 — How to interpret model outputs (layman-friendly)

- Prediction and probability:
  - If the model outputs label = 1 and probability = 0.92, read this as "the model predicts this person is positive with high confidence (92%)." It does not mean a definitive diagnosis — it's a risk prediction that should be used alongside clinical judgement.

- Confusion matrix (example):
  - True Positive (TP): cases the model labeled positive and are actually positive.
  - False Negative (FN): cases the model missed (actual positive but predicted negative) — important because missed positive cases may be critical.
  - Precision: of all people predicted positive, how many truly are positive (high precision = few false alarms).
  - Recall / Sensitivity: of all truly positive people, how many did the model find (high recall = few missed cases).

- ROC AUC: measures the model's ability to separate positive vs negative across thresholds (0.5 = random, 1.0 = perfect). Higher is better.

## 8 — Example interpretation (concrete)

- Example subject (values simplified):
  - Age: 72, BMI: 26.1, Cholesterol: 230 mg/dL, SymptomCount: 4, MemoryTestScore: 22

- Random Forest output (example):
  - Predicted label: 1 (Positive)
  - Probability: 0.88
  - Interpretation (layman): "Model estimates an 88% chance this person has a positive diagnosis. The model uses top features such as memory score, age group, cholesterol risk, and symptom count to make this judgment. Consider a follow-up clinical assessment." 

- SVM output (example):
  - Predicted label: 1 (Positive)
  - Probability: 0.81
  - Interpretation (layman): "Model estimates an 81% chance of a positive result; it is slightly less confident than Random Forest for this case. Use this to prioritize further testing or specialist referrals."

Notes on thresholds: by default, probability >= 0.5 => label 1. For operational use you can raise/lower the threshold depending on whether you value fewer false negatives (raise recall by lowering threshold) or fewer false positives (raise precision by increasing threshold).

## 9 — How to reproduce the runs

1. Make sure the virtual environment is active (`.venv`).
2. Create preprocessing outputs (if not present):

```powershell
.\.venv\Scripts\python.exe scripts\run_preprocessing_quick.py
```

3. Run the full SVM training (already run in this session):

```powershell
.\.venv\Scripts\python.exe scripts\run_svm_full.py
```

4. Run the full Random Forest training (already run in this session):

```powershell
.\.venv\Scripts\python.exe scripts\run_rf_full.py
```

5. Open `notebooks/03_svm_model.ipynb` to view the side-by-side comparison and plain-language summaries.

## 10 — Implementation details (code-level explanation)

This section gives a concise, code-oriented explanation of how each model is constructed and trained. The full training logic is implemented in `scripts/run_svm_full.py` and `scripts/run_rf_full.py` (see those files for complete implementation). Below are the key pieces and the exact building blocks used so you can trace behavior precisely.

- Common utilities (both scripts):
  - `load_data(prep_dir)` — reads preprocessed pickles from `outputs/preprocessed/` and coerces label arrays to `pandas.Series` for safe concatenation and metric functions.
  - `evaluate_and_save(model, X, y, prefix, out_dir)` — computes accuracy/precision/recall/F1/ROC-AUC (when probabilities are available), saves confusion matrix and ROC curve images to `outputs/models/`, and returns a metrics dict.
  - `generate_layman_summary(name, metrics, top_features)` — turns numeric results into a plain-language summary saved as `*_summary.txt`.

- SVM pipeline (exact construction used):

  - Pipeline definition (exact):

    Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True, random_state=42))])

  - GridSearchCV over kernels and hyperparameters (representative grid):
    - linear kernel: C in [0.1, 1, 10, 100]
    - rbf kernel: C in [0.1, 1, 10, 100], gamma in ["scale", "auto", 0.01, 0.001]
    - poly kernel: C in [0.1, 1, 10], degree in [2, 3], gamma in ["scale", "auto", 0.01]

  - After GridSearchCV picks the best estimator on the training set, the selected pipeline is re-fit on Train+Validation and saved to `models/svm_best_full.pkl`.
  - Feature importance: for linear kernel the SVM coefficients are used; otherwise permutation importance is computed on the validation set (`sklearn.inspection.permutation_importance`).

- Random Forest (exact construction used):

  - Estimator instantiation (exact):

    RandomForestClassifier(random_state=42, n_jobs=args.n_jobs)

  - GridSearchCV parameter grid (representative):
    - n_estimators: [200, 500, 1000]
    - max_depth: [None, 10, 20, 30]
    - max_features: ["sqrt", "log2"]
    - min_samples_split: [2, 5, 10]
    - min_samples_leaf: [1, 2]
    - bootstrap: [True]

  - After GridSearchCV selects the best RandomForest on the training set, that estimator is re-fit on Train+Validation and saved to `models/random_forest_best_full.pkl`.
  - Feature importance: `estimator.feature_importances_` is used to rank features and a top-20 plot is saved.

- Cross-validation and scoring
  - Both scripts use `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` by default and `GridSearchCV(..., scoring='roc_auc')` to optimize AUC across folds.

- Reproducible production step
  - After hyperparameter search finishes, both scripts concatenate Train + Validation and re-fit the selected estimator on that combined set for the final model. This final model is what gets saved in `models/` and used for Test-set evaluation.

If you'd like, I can embed short, runnable code snippets (the exact pipeline + fit/evaluate call) into this report so a reviewer can copy-paste the minimal training example directly from `model_report.md`. Tell me if you prefer full runnable snippets (I can add them) or to keep the report as high-level provenance and link directly to the scripts.

## 11 — Limitations and next steps

- Limitations:
  - Models are only as good as the data and preprocessing; biases in the dataset may affect predictions.
  - SVM is slower for larger datasets and can be sensitive to feature scaling and outliers.
  - The Random Forest is more interpretable (feature importances, SHAP values were tested) and performed better on the test set in this run.

- Suggested next steps:
  - Run SHAP summary plots for the Random Forest to provide per-feature directional explanations (already validated with `scripts/test_shap_models.py`).
  - Calibrate probabilities (e.g., using isotonic or Platt scaling) if well-calibrated probabilities are required.
  - Create a short HTML/PDF report combining `*_summary.txt` and the key plots for stakeholder distribution. I can produce this automatically.

---

If you want me to restore the file into git history or commit it now, tell me and I will proceed.
## Model Report — Random Forest & SVM

Date: 2025-10-25

This document summarizes how the two production-ready models were created, what data and preprocessing were used, how hyperparameters were tuned, and how to interpret their outputs (including a short layman-friendly example). Artifacts (models, plots, plain-language summaries) are saved under `outputs/models/` and `models/`.

---

## 1 — Summary

- Models produced: Random Forest (ensemble tree classifier) and SVM (Support Vector Classifier with RBF kernel).
- Task: binary classification (predicts whether a subject is positive for the target diagnosis — e.g., Alzheimer’s vs not). The dataset used is `data/alzheimer_dataset.csv` (2149 rows × 35 columns before preprocessing).
- Purpose: produce accurate and interpretable predictions and provide layman-readable explanations and plots for non-technical stakeholders.

## 2 — Data & preprocessing

- Raw file: `data/alzheimer_dataset.csv` (approx. 2149 rows, 35 columns).
- Preprocessing steps (implemented in `scripts/run_preprocessing_quick.py` and `notebooks/02_data_preprocessing.ipynb`):
  - Missing-value imputation: median for numeric fields, mode for categorical fields.
  - Categorical encoding: label encoding for binary categories, one-hot encoding for multi-class where appropriate.
  - Feature engineering: derived features such as AgeGroup, BMI category, cholesterol risk score, and a simple symptom-count feature (implementation details in `notebooks/02_data_preprocessing.ipynb`).
  - Scaling: numerical features scaled with StandardScaler inside model pipelines (SVM) or kept as-is for Random Forest.
## 10 — Implementation details (code-level explanation)

- Preprocessed artifacts (pickles):
  - `outputs/preprocessed/X_train.pkl` (1504 rows × ~43 features)
  - `outputs/preprocessed/X_val.pkl` (322 rows × ~43 features)
  - `outputs/preprocessed/X_test.pkl` (323 rows × ~43 features)
  - `outputs/preprocessed/y_train.pkl`, `y_val.pkl`, `y_test.pkl`

Notes: the shapes reported above are from the quick preprocessing run. If you re-run preprocessing the exact counts will match your final data after feature engineering.

## 3 — Data splits and how they were used

- Train / Validation / Test split used during development:
  - Train: 70% (approx. 1504 rows)
  - Validation: 15% (approx. 322 rows)
  - Test: 15% (approx. 323 rows)

- Training workflow:
  1. GridSearchCV performed on the training set (with Stratified K-Fold CV, default 5 folds) to find best hyperparameters.
  2. After selecting best hyperparameters, the model is re-trained (refit) on Train + Validation for better final performance before saving.
  3. Final evaluation metrics are reported on the held-out Test set.

## 4 — Models and hyperparameter tuning

### Random Forest (full-grid)

- Implementation: `sklearn.ensemble.RandomForestClassifier` (random_state=42). GridSearchCV used to select hyperparameters.
- Full-grid parameters searched (representative):
  - n_estimators: [200, 500, 1000]
  - max_depth: [None, 10, 20, 30]
  - max_features: ['sqrt', 'log2']
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2]
  - bootstrap: [True]

- Best parameters found (final run):
  - bootstrap: True
  - max_depth: 20
  - max_features: 'sqrt'
  - min_samples_leaf: 1
  - min_samples_split: 2
  - n_estimators: 500

### SVM (full-grid)

- Implementation: `sklearn.svm.SVC` inside a `sklearn.pipeline.Pipeline` with `StandardScaler` and `probability=True`.
- Full-grid parameters searched (representative):
  - kernel: ['linear', 'rbf', 'poly']
  - C: [0.1, 1, 10, 100]
  - gamma: ['scale', 'auto', 0.01, 0.001] (for RBF/poly)
  - degree: [2, 3] (for poly)

- Best parameters found (final run):
  - svc__kernel: 'rbf'
  - svc__C: 1
  - svc__gamma: 'scale'

## 5 — Evaluation results (final Test metrics)

These are the final metrics saved after re-fitting best models on Train+Validation and evaluating on Test.

- Random Forest (Test set):
  - Accuracy: 0.9412
  - Precision: 0.9130
  - Recall (Sensitivity): 0.9211
  - F1 score: 0.9170
  - ROC AUC: 0.9505

- SVM (Test set):
  - Accuracy: 0.8390
  - Precision: 0.7583
  - Recall (Sensitivity): 0.7982
  - F1 score: 0.7778
  - ROC AUC: 0.9033

Files with evaluation artifacts (in `outputs/models/`):
- `model_comparison.json` — aggregated records for each model.
- `random_forest_summary.txt`, `svm_summary.txt` — plain-language summaries.
- `feature_importances_rf_top20.png`, `feature_importances_svm_top20.png` — top features plots.
- `confusion_matrix_validation.png`, `confusion_matrix_test.png` and `roc_curve_*.png` — performance plots.

## 6 — What the models do and what data they use

- Both models are binary classifiers: given a subject's features (demographics, clinical measurements, derived features), they output either:
  - A predicted label (0/1 or 'No'/'Yes'), and
  - A probability score (model's confidence that the subject is positive).

- Example input features (after preprocessing) include: age, education, BMI, cholesterol level, presence/absence of specific symptoms, cognitive test scores, and the engineered features mentioned earlier.

## 7 — How to interpret model outputs (layman-friendly)

- Prediction and probability:
  - If the model outputs label = 1 and probability = 0.92, read this as "the model predicts this person is positive with high confidence (92%)." It does not mean a definitive diagnosis — it's a risk prediction that should be used alongside clinical judgement.

- Confusion matrix (example):
  - True Positive (TP): cases the model labeled positive and are actually positive.
  - False Negative (FN): cases the model missed (actual positive but predicted negative) — important because missed positive cases may be critical.
  - Precision: of all people predicted positive, how many truly are positive (high precision = few false alarms).
  - Recall / Sensitivity: of all truly positive people, how many did the model find (high recall = few missed cases).

- ROC AUC: measures the model's ability to separate positive vs negative across thresholds (0.5 = random, 1.0 = perfect). Higher is better.

## 8 — Example interpretation (concrete)

- Example subject (values simplified):
  - Age: 72, BMI: 26.1, Cholesterol: 230 mg/dL, SymptomCount: 4, MemoryTestScore: 22

- Random Forest output (example):
  - Predicted label: 1 (Positive)
  - Probability: 0.88
  - Interpretation (layman): "Model estimates an 88% chance this person has a positive diagnosis. The model uses top features such as memory score, age group, cholesterol risk, and symptom count to make this judgment. Consider a follow-up clinical assessment." 

- SVM output (example):
  - Predicted label: 1 (Positive)
  - Probability: 0.81
  - Interpretation (layman): "Model estimates an 81% chance of a positive result; it is slightly less confident than Random Forest for this case. Use this to prioritize further testing or specialist referrals."

Notes on thresholds: by default, probability >= 0.5 => label 1. For operational use you can raise/lower the threshold depending on whether you value fewer false negatives (raise recall by lowering threshold) or fewer false positives (raise precision by increasing threshold).

## 9 — How to reproduce the runs

1. Make sure the virtual environment is active (`.venv`).
2. Create preprocessing outputs (if not present):

```powershell
.\.venv\Scripts\python.exe scripts\run_preprocessing_quick.py
```

3. Run the full SVM training (already run in this session):

```powershell
.\.venv\Scripts\python.exe scripts\run_svm_full.py
```

4. Run the full Random Forest training (already run in this session):

```powershell
.\.venv\Scripts\python.exe scripts\run_rf_full.py
```

5. Open `notebooks/03_svm_model.ipynb` to view the side-by-side comparison and plain-language summaries.

## 10 — Limitations and next steps

- Limitations:
  - Models are only as good as the data and preprocessing; biases in the dataset may affect predictions.
  - SVM is slower for larger datasets and can be sensitive to feature scaling and outliers.
  - The Random Forest is more interpretable (feature importances, SHAP values were tested) and performed better on the test set in this run.

- Suggested next steps:
  - Run SHAP summary plots for the Random Forest to provide per-feature directional explanations (already validated with `scripts/test_shap_models.py`).
  - Calibrate probabilities (e.g., using isotonic or Platt scaling) if well-calibrated probabilities are required.
  - Create a short HTML/PDF report combining `*_summary.txt` and the key plots for stakeholder distribution. I can produce this automatically.

---


If you want, I can now generate a single HTML report (printer-friendly) that bundles both plain-language summaries, the confusion matrices, ROC curves and top-20 feature plots — would you like me to create that? If yes, tell me whether you prefer PDF or HTML.
