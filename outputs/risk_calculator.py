"""
Alzheimer's Disease Risk Score Calculator
==========================================

This module provides a comprehensive risk scoring system for Alzheimer's disease
prediction based on machine learning models and clinical risk factors.

Author: AI Assistant
Date: 2025-10-28
Version: 1.0
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Any, Optional
import warnings

class AlzheimerRiskCalculator:
    """
    Comprehensive Alzheimer's disease risk assessment calculator
    """

    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.model_loaded = True

    def calculate_risk_score(self, patient_data: pd.Series, 
                           feature_names: Optional[List[str]] = None) -> Dict[str, Any]:

        if feature_names is None:
            feature_names = patient_data.index.tolist()

        patient_features = patient_data[feature_names].values.reshape(1, -1)
        base_probability = self.model.predict_proba(patient_features)[0, 1]
        base_score = base_probability * 10

        adjustments = []
        adjustment_score = 0

        # Age > 75
        if 'Age' in patient_data.index and patient_data.get('Age', 0) > 75:
            adjustment_score += 0.5
            adjustments.append("Advanced age (>75 years): +0.5")

        # MMSE < 24
        if 'MMSE' in patient_data.index and patient_data.get('MMSE', 30) < 24:
            adjustment_score += 1.0
            adjustments.append("Cognitive impairment (MMSE<24): +1.0")

        # Family history
        family_history_cols = [col for col in patient_data.index 
                             if 'family' in col.lower() or 'genetic' in col.lower()]
        if any(patient_data.get(col, 0) == 1 for col in family_history_cols):
            adjustment_score += 0.5
            adjustments.append("Family history of Alzheimer's: +0.5")

        # Cardiovascular risk
        cv_risk_cols = [col for col in patient_data.index if any(term in col.lower() 
                       for term in ['hypertension', 'diabetes', 'cholesterol', 'heart', 'cardiovascular'])]
        cv_risk_count = sum(patient_data.get(col, 0) for col in cv_risk_cols)
        if cv_risk_count >= 2:
            adjustment_score += 0.5
            adjustments.append(f"Multiple cardiovascular risks ({cv_risk_count}): +0.5")

        # High symptom count
        symptom_cols = [col for col in patient_data.index if any(term in col.lower() 
                       for term in ['symptom', 'complaint', 'problem'])]
        symptom_count = sum(patient_data.get(col, 0) for col in symptom_cols)
        if symptom_count > 3:
            adjustment_score += 0.5
            adjustments.append(f"High symptom count ({symptom_count}): +0.5")

        final_score = min(base_score + adjustment_score, 10.0)

        if final_score <= 3:
            risk_level = "Low Risk"
            risk_color = "green"
        elif final_score <= 6:
            risk_level = "Moderate Risk"
            risk_color = "yellow"
        else:
            risk_level = "High Risk"
            risk_color = "red"

        feature_importance = self._calculate_feature_importance(patient_features, feature_names)
        recommendations = self._generate_recommendations(patient_data, final_score, adjustments)

        return {
            'risk_score': round(final_score, 2),
            'base_score': round(base_score, 2),
            'adjustment_score': round(adjustment_score, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'base_probability': round(base_probability, 4),
            'adjustments': adjustments,
            'top_features': feature_importance[:5],
            'recommendations': recommendations
        }

    def _calculate_feature_importance(self, patient_features: np.ndarray, 
                                    feature_names: List[str]) -> List[Tuple[str, float]]:
        feature_importance = []

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_contributions = [(feature_names[i], importances[i] * patient_features[0, i]) 
                                   for i in range(len(feature_names))]
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            feature_importance = feature_contributions
        elif hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_[0] if hasattr(self.model.coef_[0], '__len__') else self.model.coef_
            feature_contributions = [(feature_names[i], coefficients[i] * patient_features[0, i]) 
                                   for i in range(len(feature_names))]
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            feature_importance = feature_contributions

        return feature_importance

    def _generate_recommendations(self, patient_data: pd.Series, 
                                risk_score: float, adjustments: List[str]) -> List[str]:
        recommendations = []

        if risk_score >= 7:
            recommendations += [
                "🔴 Seek immediate medical evaluation for comprehensive cognitive assessment",
                "📋 Consider referral to neurology or memory specialist",
                "🧠 Undergo detailed neuropsychological testing"
            ]
        elif risk_score >= 4:
            recommendations += [
                "🟡 Schedule enhanced cognitive screening within 6 months",
                "📊 Monitor cognitive function more frequently",
                "🏥 Discuss with primary care physician"
            ]
        else:
            recommendations += [
                "🟢 Continue regular health check-ups",
                "🧠 Maintain current cognitive health practices"
            ]

        recommendations += [
            "🏃‍♀️ Engage in regular physical exercise (150+ minutes/week)",
            "🧘‍♀️ Practice cognitive stimulation activities (puzzles, reading, social interaction)",
            "🥗 Follow Mediterranean diet rich in omega-3 fatty acids",
            "😴 Maintain good sleep hygiene (7-8 hours/night)",
            "🚭 Avoid smoking and limit alcohol consumption"
        ]

        if any("cardiovascular" in adj.lower() for adj in adjustments):
            recommendations += [
                "❤️ Focus on cardiovascular health management",
                "💊 Ensure optimal blood pressure and cholesterol control"
            ]

        if any("mmse" in adj.lower() for adj in adjustments):
            recommendations += [
                "🧠 Consider cognitive training programs",
                "📚 Engage in mentally stimulating activities daily"
            ]

        return recommendations

    def batch_calculate(self, patients_df: pd.DataFrame, 
                       feature_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        results = []
        for i in range(len(patients_df)):
            patient = patients_df.iloc[i]
            result = self.calculate_risk_score(patient, feature_names)
            result['patient_id'] = f"Patient_{i+1:03d}"
            results.append(result)
        return results

def load_risk_calculator(model_path: str = '../outputs/models/best_model.pkl') -> AlzheimerRiskCalculator:
    return AlzheimerRiskCalculator(model_path)

def format_risk_report(result: Dict[str, Any], patient_id: str = None) -> str:
    if patient_id is None:
        patient_id = result.get('patient_id', 'Unknown')

    report = f"""
Alzheimer's Risk Assessment Report
Patient ID: {patient_id}
Assessment Date: 2025-10-28

RISK SCORE: {result['risk_score']:.2f}/10 ({result['risk_level']})
Base Model Score: {result['base_score']:.2f}/10
Clinical Adjustments: +{result['adjustment_score']:.2f}
Model Confidence: {result['base_probability']:.1%}

Clinical Adjustments Applied:
"""

    if result['adjustments']:
        for adjustment in result['adjustments']:
            report += f"- {adjustment}\n"
    else:
        report += "- No clinical risk factor adjustments applied\n"

    report += "\nTop Contributing Factors:\n"

    if result['top_features']:
        for i, (feature, contribution) in enumerate(result['top_features'], 1):
            direction = "↑" if contribution > 0 else "↓"
            report += f"{i}. {feature}: {direction} {abs(contribution):.3f}\n"

    report += "\nRecommendations:\n"
    for i, rec in enumerate(result['recommendations'][:5], 1):
        report += f"{i}. {rec}\n"

    return report

if __name__ == "__main__":
    calculator = load_risk_calculator()
    print("Alzheimer's Risk Calculator loaded successfully!")
