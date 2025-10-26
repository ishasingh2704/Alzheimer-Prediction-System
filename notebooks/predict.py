import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)

def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def load_label_encoder(encoder_path):
    encoder_path = Path(encoder_path)
    if encoder_path.exists():
        return joblib.load(encoder_path)
    return None

def evaluate_predictions(y_true, y_pred, y_prob=None):
    """Compute evaluation metrics when ground truth exists."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro", zero_division=0)),
        "recall": float(recall_score(
            y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro", zero_division=0)),
        "f1": float(f1_score(
            y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro", zero_division=0))
    }

    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None

    return metrics

def predict(model, input_csv, output_csv=None, label_encoder=None):
    df = pd.read_csv(input_csv)
    df_out = df.copy()

    # Detect target/ground-truth column if present
    target_col = None
    for col in ["Diagnosis", "Target", "Label"]:
        if col in df.columns:
            target_col = col
            break

    # Drop target column for prediction
    if target_col:
        X = df.drop(columns=[target_col])
    else:
        X = df

    preds = model.predict(X)

    # Decode if label encoder is available
    if label_encoder:
        preds_decoded = label_encoder.inverse_transform(preds)
    else:
        preds_decoded = preds

    df_out["Predicted_Diagnosis"] = preds_decoded

    # Evaluate if ground truth exists
    if target_col:
        y_true = df[target_col]
        if label_encoder and (y_true.dtype == object or not np.issubdtype(y_true.dtype, np.number)):
            y_true = label_encoder.transform(y_true)

        y_prob = None
        try:
            y_prob = model.predict_proba(X)[:, 1]
        except Exception:
            pass

        metrics = evaluate_predictions(y_true, preds, y_prob)
        print("\nEvaluation metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_true, preds, zero_division=0))
    else:
        print("No ground truth found — skipping evaluation.")

    # Save results
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_csv, index=False)
        print(f"\n✅ Predictions saved to: {output_csv}")

        # Show first 5 predictions directly
        print(f"\n🔍 Sample predictions (first 5 of {len(df_out)} rows):")
        print(df_out[["Predicted_Diagnosis"]].head().to_string(index=False))
    else:
        print(f"\n🔍 Sample predictions (first 5 of {len(df_out)} rows):")
        print(df_out[["Predicted_Diagnosis"]].head().to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Run predictions using trained logistic regression model.")
    parser.add_argument("--model", default="outputs/models/logistic_pipeline.joblib",
                        help="Path to trained model (.joblib)")
    parser.add_argument("--data", required=True, help="Path to input CSV for prediction")
    parser.add_argument("--output", default="outputs/models/predictions.csv",
                        help="Path to save predictions")
    parser.add_argument("--encoder", default="outputs/models/label_encoder_target.joblib",
                        help="Path to label encoder (optional)")
    args = parser.parse_args()

    model = load_model(args.model)
    label_encoder = load_label_encoder(args.encoder)
    predict(model, args.data, args.output, label_encoder)

if __name__ == "__main__":
    main()
