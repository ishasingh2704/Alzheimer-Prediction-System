"""
Comprehensive Model Evaluation Script for Alzheimer Prediction System

This script performs comprehensive evaluation of all trained models including:
1. Test set evaluation with multiple metrics
2. Cross-validation analysis
3. Feature importance analysis
4. Error analysis
5. Model comparison and recommendation
6. PDF report generation
"""

import os
import sys
import json
import time
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ML libraries
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, matthews_corrcoef,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.inspection import permutation_importance

# PDF generation
try:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
except ImportError:
    print("Warning: PDF generation may not work properly")

# Suppress warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, project_root=None):
        """Initialize the evaluator with project paths"""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.models_dir = self.project_root / "models"
        self.outputs_dir = self.project_root / "outputs"
        self.models_output_dir = self.outputs_dir / "models"
        self.preprocessed_dir = self.outputs_dir / "preprocessed"
        self.evaluation_dir = self.outputs_dir / "evaluation"
        
        # Create evaluation directory
        self.evaluation_dir.mkdir(exist_ok=True)
        
        # Model paths and metadata
        self.models = {}
        self.model_metadata = {}
        self.results = defaultdict(dict)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_models(self):
        """Load all available trained models"""
        print("Loading trained models...")
        
        # Define model paths
        model_paths = {
            'Decision Tree': self.models_output_dir / "decision_tree_model.pkl",
            'Random Forest (Reduced)': self.models_dir / "random_forest_best_reduced.pkl",
            'Random Forest (Full)': self.models_dir / "random_forest_best_full.pkl",
            'SVM (Reduced)': self.models_dir / "svm_best_reduced.pkl",
            'SVM (Full)': self.models_dir / "svm_best_full.pkl",
        }
        
        # Check for additional models
        xgboost_path = self.models_output_dir / "xgboost_model.pkl"
        if xgboost_path.exists():
            model_paths['XGBoost'] = xgboost_path
            
        logistic_path = self.models_output_dir / "logistic_regression_model.pkl"
        if logistic_path.exists():
            model_paths['Logistic Regression'] = logistic_path
        
        # Load existing models
        for name, path in model_paths.items():
            if path.exists():
                try:
                    self.models[name] = joblib.load(path)
                    print(f"✓ Loaded {name}")
                except Exception as e:
                    print(f"✗ Failed to load {name}: {e}")
            else:
                print(f"✗ Model not found: {path}")
        
        print(f"Successfully loaded {len(self.models)} models")
        return len(self.models) > 0
    
    def load_data(self):
        """Load preprocessed datasets"""
        print("Loading preprocessed data...")
        
        try:
            # Load from CSV files to ensure consistent feature names
            data_files = {
                'X_train': 'X_train.csv',
                'X_val': 'X_val.csv', 
                'X_test': 'X_test.csv',
                'y_train': 'y_train.csv',
                'y_val': 'y_val.csv',
                'y_test': 'y_test.csv'
            }
            
            for name, filename in data_files.items():
                csv_path = self.preprocessed_dir / filename
                if csv_path.exists():
                    data = pd.read_csv(csv_path)
                    # For target variables, extract the first column if it's a DataFrame
                    if name.startswith('y_') and isinstance(data, pd.DataFrame):
                        data = data.iloc[:, 0] if data.shape[1] == 1 else data
                    setattr(self, name, data)
                else:
                    # Fallback to pickle
                    pkl_path = self.preprocessed_dir / filename.replace('.csv', '.pkl')
                    if pkl_path.exists():
                        data = joblib.load(pkl_path)
                        # Convert numpy arrays to pandas objects if needed
                        if isinstance(data, np.ndarray):
                            if name.startswith('X_'):
                                # For feature matrices, create DataFrame with generic column names
                                data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
                            else:
                                # For target vectors, create Series
                                data = pd.Series(data)
                        setattr(self, name, data)
                    else:
                        raise FileNotFoundError(f"Neither {csv_path} nor {pkl_path} found")
            
            # Combine train and validation for cross-validation
            if isinstance(self.X_train, pd.DataFrame):
                self.X_full = pd.concat([self.X_train, self.X_val], axis=0, ignore_index=True)
            else:
                self.X_full = np.vstack([self.X_train, self.X_val])
            
            if isinstance(self.y_train, pd.Series):
                self.y_full = pd.concat([self.y_train, self.y_val], axis=0, ignore_index=True)
            else:
                self.y_full = np.concatenate([self.y_train, self.y_val])
            
            print("✓ Data loaded successfully")
            print(f"  Training set: {self.X_train.shape}")
            print(f"  Validation set: {self.X_val.shape}")
            print(f"  Test set: {self.X_test.shape}")
            print(f"  Full dataset: {self.X_full.shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load data: {e}")
            return False
    
    def evaluate_test_set(self):
        """Comprehensive test set evaluation"""
        print("\n" + "="*60)
        print("PERFORMING TEST SET EVALUATION")
        print("="*60)
        
        # Store predictions for ensemble analysis
        all_predictions = {}
        all_probabilities = {}
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Handle feature name mismatches
                X_test_for_model = self._prepare_data_for_model(model, self.X_test)
                
                # Time prediction
                start_time = time.time()
                y_pred = model.predict(X_test_for_model)
                prediction_time = time.time() - start_time
                
                # Get prediction probabilities
                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test_for_model)[:, 1]
                    elif hasattr(model, 'decision_function'):
                        y_proba = model.decision_function(X_test_for_model)
                    else:
                        y_proba = y_pred.astype(float)
                except:
                    y_proba = y_pred.astype(float)
                
                # Store predictions
                all_predictions[model_name] = y_pred
                all_probabilities[model_name] = y_proba
                
                # Calculate metrics
                metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
                metrics['prediction_time'] = prediction_time
                
                self.results[model_name]['test_metrics'] = metrics
                
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1-Score: {metrics['f1']:.4f}")
                print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
                print(f"  Prediction time: {prediction_time:.4f}s")
                
            except Exception as e:
                print(f"  ✗ Failed to evaluate {model_name}: {e}")
                continue
        
        # Save predictions
        if all_predictions:
            pred_df = pd.DataFrame(all_predictions)
            pred_df['y_true'] = self.y_test if isinstance(self.y_test, np.ndarray) else self.y_test.values
            pred_df.to_csv(self.evaluation_dir / "test_predictions.csv", index=False)
            
            # Generate visualizations only if we have predictions
            self._generate_confusion_matrices()
            self._generate_roc_curves(all_probabilities)
            self._generate_pr_curves(all_probabilities)
        else:
            print("No models could be evaluated successfully!")
    
    def _prepare_data_for_model(self, model, X_data):
        """Prepare data to match model's expected features"""
        # For models that don't have feature_names_in_, just return the data
        if not hasattr(model, 'feature_names_in_'):
            return X_data
        
        expected_features = model.feature_names_in_
        
        # If X_data has the same features, return as is
        if hasattr(X_data, 'columns') and len(expected_features) == len(X_data.columns):
            if all(feat in X_data.columns for feat in expected_features):
                return X_data[expected_features]
        
        # Try to match features or use subset
        if hasattr(X_data, 'columns'):
            available_features = [feat for feat in expected_features if feat in X_data.columns]
            if available_features:
                print(f"    Warning: Using {len(available_features)}/{len(expected_features)} matching features")
                return X_data[available_features]
        
        # If we can't match features, try to use the data as-is (for models trained on different preprocessing)
        print(f"    Warning: Feature mismatch. Expected {len(expected_features)}, got {X_data.shape[1]}")
        return X_data
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate comprehensive metrics"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'mcc': matthews_corrcoef(y_true, y_pred),
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        # ROC-AUC (handle edge cases)
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = 0.5
        
        # Average Precision
        try:
            metrics['avg_precision'] = average_precision_score(y_true, y_proba)
        except:
            metrics['avg_precision'] = metrics['precision']
        
        return metrics
    
    def _generate_confusion_matrices(self):
        """Generate confusion matrix heatmaps for all models"""
        print("\nGenerating confusion matrices...")
        
        successful_models = [name for name in self.models.keys() 
                           if name in self.results and 'test_metrics' in self.results[name]]
        
        if not successful_models:
            print("No successful models to generate confusion matrices for")
            return
        
        n_models = len(successful_models)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for idx, model_name in enumerate(successful_models):
            model = self.models[model_name]
            
            try:
                X_test_for_model = self._prepare_data_for_model(model, self.X_test)
                y_pred = model.predict(X_test_for_model)
                cm = confusion_matrix(self.y_test, y_pred)
                
                ax = axes[idx]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['No Alzheimer', 'Alzheimer'],
                           yticklabels=['No Alzheimer', 'Alzheimer'])
                ax.set_title(f'{model_name}\nConfusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                
            except Exception as e:
                print(f"Could not generate confusion matrix for {model_name}: {e}")
                axes[idx].text(0.5, 0.5, f'Error: {model_name}', 
                              ha='center', va='center', transform=axes[idx].transAxes)
        
        # Hide unused subplots
        for idx in range(len(successful_models), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.evaluation_dir / "confusion_matrices_all.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_roc_curves(self, all_probabilities):
        """Generate ROC curves for all models"""
        print("Generating ROC curves...")
        
        plt.figure(figsize=(12, 8))
        
        for model_name, y_proba in all_probabilities.items():
            try:
                fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                auc = roc_auc_score(self.y_test, y_proba)
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{model_name} (AUC = {auc:.3f})')
            except Exception as e:
                print(f"Warning: Could not plot ROC for {model_name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.evaluation_dir / "roc_curves_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_pr_curves(self, all_probabilities):
        """Generate Precision-Recall curves for all models"""
        print("Generating Precision-Recall curves...")
        
        plt.figure(figsize=(12, 8))
        
        for model_name, y_proba in all_probabilities.items():
            try:
                precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
                avg_precision = average_precision_score(self.y_test, y_proba)
                plt.plot(recall, precision, linewidth=2,
                        label=f'{model_name} (AP = {avg_precision:.3f})')
            except Exception as e:
                print(f"Warning: Could not plot PR curve for {model_name}: {e}")
        
        # Baseline (random classifier)
        pos_ratio = np.sum(self.y_test) / len(self.y_test)
        plt.axhline(y=pos_ratio, color='k', linestyle='--', linewidth=1,
                   label=f'Random Classifier (AP = {pos_ratio:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.evaluation_dir / "pr_curves_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_cross_validation(self):
        """Perform 10-fold stratified cross-validation"""
        print("\n" + "="*60)
        print("PERFORMING CROSS-VALIDATION ANALYSIS")
        print("="*60)
        
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nCross-validating {model_name}...")
            
            try:
                # Prepare data for this model
                X_full_for_model = self._prepare_data_for_model(model, self.X_full)
                
                scores = cross_validate(model, X_full_for_model, self.y_full, 
                                      cv=cv, scoring=scoring, n_jobs=-1)
                
                cv_results[model_name] = {
                    metric: {
                        'mean': scores[f'test_{metric}'].mean(),
                        'std': scores[f'test_{metric}'].std(),
                        'scores': scores[f'test_{metric}']
                    }
                    for metric in scoring
                }
                
                self.results[model_name]['cv_metrics'] = cv_results[model_name]
                
                print(f"  Accuracy: {cv_results[model_name]['accuracy']['mean']:.4f} ± {cv_results[model_name]['accuracy']['std']:.4f}")
                print(f"  F1-Score: {cv_results[model_name]['f1']['mean']:.4f} ± {cv_results[model_name]['f1']['std']:.4f}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
        
        # Generate box plots
        if cv_results:
            self._generate_cv_boxplots(cv_results)
        
        return cv_results
    
    def _generate_cv_boxplots(self, cv_results):
        """Generate box plots for cross-validation results"""
        print("Generating cross-validation box plots...")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            data = []
            labels = []
            
            for model_name in cv_results.keys():
                if metric in cv_results[model_name]:
                    data.append(cv_results[model_name][metric]['scores'])
                    labels.append(model_name)
            
            if data:
                box_plot = ax.boxplot(data, labels=labels, patch_artist=True)
                
                # Color the boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_title(f'{metric.replace("_", " ").title()} - Cross Validation')
                ax.set_ylabel('Score')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplot
        axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.evaluation_dir / "cv_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_feature_importance(self):
        """Analyze feature importance across models"""
        print("\n" + "="*60)
        print("ANALYZING FEATURE IMPORTANCE")
        print("="*60)
        
        importance_data = {}
        
        for model_name, model in self.models.items():
            print(f"\nAnalyzing {model_name}...")
            
            try:
                # Extract feature importance based on model type
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    importance = model.feature_importances_
                    method = "Built-in Importance"
                    
                elif hasattr(model, 'coef_'):
                    # Linear models
                    importance = np.abs(model.coef_[0])
                    method = "Coefficient Magnitude"
                    
                else:
                    # Use permutation importance for other models
                    print(f"  Using permutation importance (this may take a while)...")
                    perm_importance = permutation_importance(
                        model, self.X_test, self.y_test, 
                        n_repeats=10, random_state=42, n_jobs=-1
                    )
                    importance = perm_importance.importances_mean
                    method = "Permutation Importance"
                
                # Get feature names
                if hasattr(self.X_test, 'columns'):
                    feature_names = self.X_test.columns.tolist()
                else:
                    feature_names = [f'Feature_{i}' for i in range(len(importance))]
                
                importance_data[model_name] = {
                    'importance': importance,
                    'features': feature_names,
                    'method': method
                }
                
                print(f"  ✓ Extracted importance using {method}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
        
        # Average importance across tree-based models
        self._generate_feature_importance_plots(importance_data)
        
        # Save feature importance data
        self._save_feature_importance_report(importance_data)
    
    def _generate_feature_importance_plots(self, importance_data):
        """Generate feature importance visualizations"""
        print("Generating feature importance plots...")
        
        # Individual model plots
        n_models = len(importance_data)
        if n_models == 0:
            print("No feature importance data available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, (model_name, data) in enumerate(importance_data.items()):
            if idx >= 6:
                break
                
            ax = axes[idx]
            
            # Get top 20 features
            importance = data['importance']
            features = data['features']
            
            # Sort by importance
            indices = np.argsort(importance)[-20:]
            top_importance = importance[indices]
            
            # Ensure we don't go out of bounds
            valid_indices = [i for i in indices if i < len(features)]
            top_features = [features[i] for i in valid_indices]
            top_importance = importance[valid_indices]
            
            # Create horizontal bar plot
            bars = ax.barh(range(len(top_importance)), top_importance)
            ax.set_yticks(range(len(top_importance)))
            ax.set_yticklabels(top_features, fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name}\n({data["method"]})', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Color bars
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_importance)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        # Hide unused subplots
        for idx in range(len(importance_data), 6):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.evaluation_dir / "feature_importance_individual.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Average importance across tree-based models
        self._generate_averaged_importance_plot(importance_data)
    
    def _generate_averaged_importance_plot(self, importance_data):
        """Generate averaged feature importance plot"""
        tree_models = {}
        
        # Identify tree-based models for averaging
        for model_name, data in importance_data.items():
            if data['method'] == "Built-in Importance":
                tree_models[model_name] = data
        
        if len(tree_models) < 2:
            print("Not enough tree-based models for averaging")
            return
        
        print("Generating averaged feature importance plot...")
        
        # Find models with the same number of features for averaging
        feature_counts = {}
        for model_name, data in tree_models.items():
            n_features = len(data['features'])
            if n_features not in feature_counts:
                feature_counts[n_features] = []
            feature_counts[n_features].append((model_name, data))
        
        # Use the group with the most models
        largest_group = max(feature_counts.values(), key=len)
        
        if len(largest_group) < 2:
            print("Cannot average: no models have matching feature sets")
            return
        
        # Average importance across models with same features
        first_model = largest_group[0][1]
        all_features = first_model['features']
        avg_importance = np.zeros(len(all_features))
        
        for model_name, data in largest_group:
            avg_importance += data['importance']
        
        avg_importance /= len(largest_group)
        
        # Plot top 20 averaged features
        indices = np.argsort(avg_importance)[-20:]
        valid_indices = [i for i in indices if i < len(all_features)]
        top_avg_importance = avg_importance[valid_indices]
        top_features = [all_features[i] for i in valid_indices]
        
        if len(top_features) == 0:
            print("No valid features to plot")
            return
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_avg_importance)), top_avg_importance)
        plt.yticks(range(len(top_avg_importance)), top_features)
        plt.xlabel('Average Importance Score')
        plt.title(f'Top {len(top_features)} Features - Averaged Across Tree-Based Models', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Color bars
        colors = plt.cm.plasma(np.linspace(0, 1, len(top_avg_importance)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(self.evaluation_dir / "feature_importance_averaged.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_feature_importance_report(self, importance_data):
        """Save detailed feature importance report"""
        print("Saving feature importance report...")
        
        report = []
        report.append("# Feature Importance Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for model_name, data in importance_data.items():
            report.append(f"## {model_name}")
            report.append(f"Method: {data['method']}")
            report.append("")
            
            # Sort features by importance
            importance = data['importance']
            features = data['features']
            sorted_indices = np.argsort(importance)[::-1]
            
            report.append("| Rank | Feature | Importance |")
            report.append("|------|---------|------------|")
            
            for rank, idx in enumerate(sorted_indices[:20], 1):
                if idx < len(features):  # Ensure we don't go out of bounds
                    feature = features[idx]
                    imp_score = importance[idx]
                    report.append(f"| {rank} | {feature} | {imp_score:.6f} |")
            
            report.append("")
        
        # Save report
        with open(self.evaluation_dir / "feature_importance_report.md", 'w') as f:
            f.write('\n'.join(report))
    
    def perform_error_analysis(self):
        """Analyze prediction errors and patterns"""
        print("\n" + "="*60)
        print("PERFORMING ERROR ANALYSIS")
        print("="*60)
        
        # Get predictions from best performing model
        best_model_name = self._get_best_model()
        best_model = self.models[best_model_name]
        
        print(f"Analyzing errors for best model: {best_model_name}")
        
        y_pred = best_model.predict(self.X_test)
        
        # Identify misclassified samples
        misclassified = self.y_test != y_pred
        false_positives = (self.y_test == 0) & (y_pred == 1)
        false_negatives = (self.y_test == 1) & (y_pred == 0)
        
        print(f"Total misclassified: {misclassified.sum()}")
        print(f"False positives: {false_positives.sum()}")
        print(f"False negatives: {false_negatives.sum()}")
        
        # Analyze feature distributions for errors
        self._analyze_error_patterns(false_positives, false_negatives)
        
        # Save error analysis
        error_df = pd.DataFrame({
            'y_true': self.y_test,
            'y_pred': y_pred,
            'misclassified': misclassified,
            'false_positive': false_positives,
            'false_negative': false_negatives
        })
        
        error_df.to_csv(self.evaluation_dir / "error_analysis.csv", index=False)
    
    def _analyze_error_patterns(self, false_positives, false_negatives):
        """Analyze patterns in prediction errors"""
        print("Analyzing error patterns...")
        
        if not hasattr(self.X_test, 'columns'):
            print("Cannot analyze error patterns: feature names not available")
            return
        
        # Select important features for analysis
        feature_cols = self.X_test.columns[:10]  # Use first 10 features
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, feature in enumerate(feature_cols):
            if idx >= 10:
                break
                
            ax = axes[idx]
            
            # Plot distributions
            correct_predictions = ~(false_positives | false_negatives)
            
            if correct_predictions.sum() > 0:
                ax.hist(self.X_test.loc[correct_predictions, feature], 
                       alpha=0.7, label='Correct', bins=20, density=True)
            
            if false_positives.sum() > 0:
                ax.hist(self.X_test.loc[false_positives, feature], 
                       alpha=0.7, label='False Positive', bins=20, density=True)
            
            if false_negatives.sum() > 0:
                ax.hist(self.X_test.loc[false_negatives, feature], 
                       alpha=0.7, label='False Negative', bins=20, density=True)
            
            ax.set_title(f'{feature}', fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.evaluation_dir / "error_pattern_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_best_model(self):
        """Get the best performing model based on weighted scoring"""
        if not self.results:
            return list(self.models.keys())[0]
        
        weighted_scores = {}
        
        for model_name in self.models.keys():
            if 'test_metrics' in self.results[model_name]:
                metrics = self.results[model_name]['test_metrics']
                
                # Weighted scoring: F1 (40%) + ROC-AUC (30%) + Accuracy (20%) + Training Time (10%)
                score = (
                    metrics.get('f1', 0) * 0.4 +
                    metrics.get('roc_auc', 0) * 0.3 +
                    metrics.get('accuracy', 0) * 0.2 +
                    (1 - min(metrics.get('prediction_time', 0), 1)) * 0.1  # Invert time (faster is better)
                )
                
                weighted_scores[model_name] = score
        
        if weighted_scores:
            return max(weighted_scores, key=weighted_scores.get)
        else:
            return list(self.models.keys())[0]
    
    def generate_comparison_table(self):
        """Generate comprehensive model comparison table"""
        print("\n" + "="*60)
        print("GENERATING MODEL COMPARISON TABLE")
        print("="*60)
        
        # Prepare comparison data
        comparison_data = []
        
        for model_name in self.models.keys():
            row = {'Model': model_name}
            
            # Test metrics
            if 'test_metrics' in self.results[model_name]:
                test_metrics = self.results[model_name]['test_metrics']
                row.update({
                    'Accuracy': f"{test_metrics.get('accuracy', 0):.4f}",
                    'Precision': f"{test_metrics.get('precision', 0):.4f}",
                    'Recall': f"{test_metrics.get('recall', 0):.4f}",
                    'F1-Score': f"{test_metrics.get('f1', 0):.4f}",
                    'ROC-AUC': f"{test_metrics.get('roc_auc', 0):.4f}",
                    'Specificity': f"{test_metrics.get('specificity', 0):.4f}",
                    'MCC': f"{test_metrics.get('mcc', 0):.4f}",
                    'Prediction Time (s)': f"{test_metrics.get('prediction_time', 0):.4f}"
                })
            
            # Cross-validation metrics
            if 'cv_metrics' in self.results[model_name]:
                cv_metrics = self.results[model_name]['cv_metrics']
                row.update({
                    'CV Accuracy': f"{cv_metrics.get('accuracy', {}).get('mean', 0):.4f} ± {cv_metrics.get('accuracy', {}).get('std', 0):.4f}",
                    'CV F1-Score': f"{cv_metrics.get('f1', {}).get('mean', 0):.4f} ± {cv_metrics.get('f1', {}).get('std', 0):.4f}"
                })
            
            comparison_data.append(row)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save as CSV
        comparison_df.to_csv(self.evaluation_dir / "model_comparison_table.csv", index=False)
        
        # Generate markdown table
        self._generate_markdown_table(comparison_df)
        
        print("✓ Model comparison table generated")
        
        return comparison_df
    
    def _generate_markdown_table(self, df):
        """Generate formatted markdown table"""
        markdown_lines = []
        markdown_lines.append("# Model Comparison Table")
        markdown_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_lines.append("")
        
        # Table header
        headers = df.columns.tolist()
        markdown_lines.append("| " + " | ".join(headers) + " |")
        markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Table rows
        for _, row in df.iterrows():
            row_str = "| " + " | ".join(str(row[col]) for col in headers) + " |"
            markdown_lines.append(row_str)
        
        markdown_lines.append("")
        
        # Best model recommendation
        best_model = self._get_best_model()
        markdown_lines.append(f"## Recommendation")
        markdown_lines.append(f"**Best performing model:** {best_model}")
        markdown_lines.append("")
        markdown_lines.append("*Based on weighted scoring: F1-score (40%) + ROC-AUC (30%) + Accuracy (20%) + Training Time (10%)*")
        
        # Save markdown
        with open(self.evaluation_dir / "model_comparison_table.md", 'w') as f:
            f.write('\n'.join(markdown_lines))
    
    def generate_pdf_report(self):
        """Generate comprehensive PDF report"""
        print("\n" + "="*60)
        print("GENERATING PDF REPORT")
        print("="*60)
        
        try:
            with PdfPages(self.evaluation_dir / "final_evaluation_report.pdf") as pdf:
                # Title page
                self._add_title_page(pdf)
                
                # Model comparison table
                self._add_comparison_table_page(pdf)
                
                # Add visualization pages
                viz_files = [
                    ("confusion_matrices_all.png", "Confusion Matrices"),
                    ("roc_curves_comparison.png", "ROC Curves Comparison"),
                    ("pr_curves_comparison.png", "Precision-Recall Curves"),
                    ("cv_boxplots.png", "Cross-Validation Results"),
                    ("feature_importance_individual.png", "Feature Importance (Individual)"),
                    ("feature_importance_averaged.png", "Feature Importance (Averaged)"),
                    ("error_pattern_analysis.png", "Error Pattern Analysis")
                ]
                
                for filename, title in viz_files:
                    file_path = self.evaluation_dir / filename
                    if file_path.exists():
                        self._add_visualization_page(pdf, file_path, title)
                
                # Summary page
                self._add_summary_page(pdf)
            
            print("✓ PDF report generated successfully")
            
        except Exception as e:
            print(f"✗ Failed to generate PDF report: {e}")
            print("Individual visualizations and data files are still available")
    
    def _add_title_page(self, pdf):
        """Add title page to PDF"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.8, "Alzheimer Prediction System", 
                horizontalalignment='center', fontsize=24, fontweight='bold')
        
        ax.text(0.5, 0.7, "Comprehensive Model Evaluation Report", 
                horizontalalignment='center', fontsize=18)
        
        # Details
        ax.text(0.5, 0.5, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                horizontalalignment='center', fontsize=12)
        
        ax.text(0.5, 0.4, f"Models evaluated: {len(self.models)}", 
                horizontalalignment='center', fontsize=12)
        
        ax.text(0.5, 0.3, f"Test samples: {len(self.y_test)}", 
                horizontalalignment='center', fontsize=12)
        
        # Best model
        best_model = self._get_best_model()
        ax.text(0.5, 0.2, f"Recommended model: {best_model}", 
                horizontalalignment='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _add_comparison_table_page(self, pdf):
        """Add model comparison table to PDF"""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        # Load comparison data
        try:
            comparison_df = pd.read_csv(self.evaluation_dir / "model_comparison_table.csv")
            
            # Create table
            table_data = []
            for _, row in comparison_df.iterrows():
                table_data.append([str(row[col]) for col in comparison_df.columns])
            
            table = ax.table(cellText=table_data,
                           colLabels=comparison_df.columns.tolist(),
                           cellLoc='center',
                           loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(comparison_df.columns)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax.set_title("Model Comparison Table", fontsize=16, fontweight='bold', pad=20)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not load comparison table: {e}", 
                   horizontalalignment='center', fontsize=12)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _add_visualization_page(self, pdf, image_path, title):
        """Add visualization page to PDF"""
        try:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            # Load and display image
            import matplotlib.image as mpimg
            img = mpimg.imread(image_path)
            ax.imshow(img)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Could not add {title} to PDF: {e}")
    
    def _add_summary_page(self, pdf):
        """Add summary page to PDF"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Summary text
        summary_text = []
        summary_text.append("EVALUATION SUMMARY")
        summary_text.append("=" * 50)
        summary_text.append("")
        
        # Best model details
        best_model = self._get_best_model()
        if best_model in self.results and 'test_metrics' in self.results[best_model]:
            metrics = self.results[best_model]['test_metrics']
            summary_text.append(f"RECOMMENDED MODEL: {best_model}")
            summary_text.append(f"  • Accuracy: {metrics.get('accuracy', 0):.4f}")
            summary_text.append(f"  • F1-Score: {metrics.get('f1', 0):.4f}")
            summary_text.append(f"  • ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            summary_text.append(f"  • Precision: {metrics.get('precision', 0):.4f}")
            summary_text.append(f"  • Recall: {metrics.get('recall', 0):.4f}")
        
        summary_text.append("")
        summary_text.append("KEY FINDINGS:")
        summary_text.append(f"  • {len(self.models)} models were evaluated")
        summary_text.append(f"  • Test set size: {len(self.y_test)} samples")
        summary_text.append(f"  • 10-fold cross-validation performed")
        summary_text.append(f"  • Feature importance analysis completed")
        summary_text.append(f"  • Error pattern analysis conducted")
        
        # Display text
        ax.text(0.1, 0.9, '\n'.join(summary_text), 
                verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def run_complete_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("=" * 60)
        print("ALZHEIMER PREDICTION SYSTEM - MODEL EVALUATION")
        print("=" * 60)
        print(f"Evaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load models and data
        if not self.load_models():
            print("✗ Failed to load models. Exiting.")
            return False
        
        if not self.load_data():
            print("✗ Failed to load data. Exiting.")
            return False
        
        # Perform evaluations
        try:
            self.evaluate_test_set()
            self.perform_cross_validation()
            self.analyze_feature_importance()
            self.perform_error_analysis()
            self.generate_comparison_table()
            self.generate_pdf_report()
            
            print("\n" + "=" * 60)
            print("EVALUATION COMPLETED SUCCESSFULLY")
            print("=" * 60)
            
            # Print best model recommendation
            best_model = self._get_best_model()
            print(f"\n🏆 RECOMMENDED MODEL: {best_model}")
            
            if best_model in self.results and 'test_metrics' in self.results[best_model]:
                metrics = self.results[best_model]['test_metrics']
                print(f"   • Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"   • F1-Score: {metrics.get('f1', 0):.4f}")
                print(f"   • ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            
            print(f"\n📁 Results saved to: {self.evaluation_dir}")
            print(f"📊 PDF report: {self.evaluation_dir / 'final_evaluation_report.pdf'}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main execution function"""
    # Set project root
    project_root = Path(__file__).parent
    
    # Initialize evaluator
    evaluator = ModelEvaluator(project_root)
    
    # Run complete evaluation
    success = evaluator.run_complete_evaluation()
    
    if success:
        print("\n✅ All evaluations completed successfully!")
    else:
        print("\n❌ Evaluation completed with errors.")
    
    return success


if __name__ == "__main__":
    main()