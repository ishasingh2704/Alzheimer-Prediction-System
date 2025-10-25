"""
Advanced Alzheimer's Dataset Analysis
This script performs advanced statistical analysis and risk factor identification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_output_dir():
    """Create analysis outputs directory if it doesn't exist"""
    if not os.path.exists('analysis_outputs'):
        os.makedirs('analysis_outputs')

def load_dataset():
    """Load the Alzheimer's dataset from CSV file"""
    try:
        df = pd.read_csv('data/alzheimer_dataset.csv')
        print("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print("Error: alzheimer_dataset.csv not found in data/ directory")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def encode_categorical_for_correlation(df):
    """Encode categorical variables for correlation analysis"""
    df_encoded = df.copy()
    
    # Binary encoding for categorical variables
    categorical_mappings = {}
    
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            unique_values = df_encoded[col].unique()
            if len(unique_values) <= 10:  # Only encode if reasonable number of categories
                if len(unique_values) == 2:
                    # Binary encoding
                    mapping = {unique_values[0]: 0, unique_values[1]: 1}
                    df_encoded[col] = df_encoded[col].map(mapping)
                    categorical_mappings[col] = mapping
                else:
                    # Label encoding for multiple categories
                    mapping = {val: idx for idx, val in enumerate(unique_values)}
                    df_encoded[col] = df_encoded[col].map(mapping)
                    categorical_mappings[col] = mapping
    
    return df_encoded, categorical_mappings

def feature_correlation_analysis(df):
    """Perform comprehensive feature correlation analysis"""
    print("="*50)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*50)
    
    # Encode categorical variables for correlation
    df_encoded, mappings = encode_categorical_for_correlation(df)
    
    results = {}
    
    # 1. Correlation with Diagnosis
    if 'Diagnosis' in df_encoded.columns:
        diagnosis_corr = df_encoded.corr()['Diagnosis'].abs().sort_values(ascending=False)
        diagnosis_corr = diagnosis_corr.drop('Diagnosis')  # Remove self-correlation
        
        print("Top 15 Features Correlated with Diagnosis:")
        print("-" * 45)
        for feature, corr in diagnosis_corr.head(15).items():
            print(f"{feature}: {corr:.4f}")
        
        results['diagnosis_correlation'] = diagnosis_corr
        
        # Visualize top correlations with diagnosis
        plt.figure(figsize=(12, 8))
        top_features = diagnosis_corr.head(15)
        plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Absolute Correlation with Diagnosis')
        plt.title('Top 15 Features Correlated with Diagnosis')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('analysis_outputs/diagnosis_correlation_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Highly correlated feature pairs
    correlation_matrix = df_encoded.corr()
    
    # Find pairs with correlation > 0.7
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = abs(correlation_matrix.iloc[i, j])
            if corr_value > 0.7:
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                high_corr_pairs.append((feature1, feature2, corr_value))
    
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nHighly Correlated Feature Pairs (|r| > 0.7):")
    print("-" * 50)
    if high_corr_pairs:
        for feature1, feature2, corr in high_corr_pairs:
            print(f"{feature1} - {feature2}: {corr:.4f}")
    else:
        print("No feature pairs with correlation > 0.7 found.")
    
    results['high_correlation_pairs'] = high_corr_pairs
    
    # 3. Feature importance ranking
    if 'Diagnosis' in df_encoded.columns:
        feature_importance = diagnosis_corr.to_dict()
        results['feature_importance_ranking'] = feature_importance
    
    return results

def group_analysis(df):
    """Compare feature means across diagnosis groups with statistical tests"""
    print("="*50)
    print("GROUP ANALYSIS BY DIAGNOSIS")
    print("="*50)
    
    if 'Diagnosis' not in df.columns:
        print("Diagnosis column not found!")
        return None
    
    results = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove Diagnosis if it's numerical
    if 'Diagnosis' in numerical_cols:
        numerical_cols.remove('Diagnosis')
    
    diagnosis_groups = df['Diagnosis'].unique()
    results['diagnosis_groups'] = diagnosis_groups
    
    # Group means comparison
    group_means = df.groupby('Diagnosis')[numerical_cols].mean()
    print("Mean values by Diagnosis group:")
    print(group_means)
    
    results['group_means'] = group_means
    
    # Statistical tests
    statistical_results = []
    
    for col in numerical_cols:
        if df[col].notna().sum() > 0:  # Only test if there are non-null values
            groups_data = [df[df['Diagnosis'] == group][col].dropna() for group in diagnosis_groups]
            
            # Remove empty groups
            groups_data = [group for group in groups_data if len(group) > 0]
            
            if len(groups_data) >= 2:
                if len(groups_data) == 2:
                    # T-test for two groups
                    stat, p_value = ttest_ind(groups_data[0], groups_data[1])
                    test_type = "T-test"
                else:
                    # ANOVA for multiple groups
                    stat, p_value = f_oneway(*groups_data)
                    test_type = "ANOVA"
                
                significant = p_value < 0.05
                statistical_results.append({
                    'feature': col,
                    'test_type': test_type,
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': significant
                })
    
    # Sort by p-value
    statistical_results.sort(key=lambda x: x['p_value'])
    
    print(f"\nStatistical Tests Results (sorted by p-value):")
    print("-" * 60)
    for result in statistical_results[:15]:  # Show top 15
        significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
        print(f"{result['feature']}: {result['test_type']} p={result['p_value']:.6f} {significance}")
    
    results['statistical_tests'] = statistical_results
    
    # Visualize key differences
    significant_features = [r['feature'] for r in statistical_results if r['significant']][:8]
    
    if significant_features:
        n_features = len(significant_features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(significant_features):
            if i < len(axes):
                group_means[feature].plot(kind='bar', ax=axes[i], color='skyblue')
                axes[i].set_title(f'{feature} by Diagnosis')
                axes[i].set_xlabel('Diagnosis')
                axes[i].set_ylabel(f'Mean {feature}')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(significant_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('analysis_outputs/group_differences_significant_features.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def lifestyle_risk_analysis(df):
    """Analyze lifestyle risk factors"""
    print("="*50)
    print("LIFESTYLE RISK FACTOR ANALYSIS")
    print("="*50)
    
    lifestyle_factors = ['Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']
    available_factors = [col for col in lifestyle_factors if col in df.columns]
    
    if not available_factors or 'Diagnosis' not in df.columns:
        print("Required columns not found!")
        return None
    
    results = {}
    
    # Calculate risk ratios for each lifestyle factor
    for factor in available_factors:
        if df[factor].dtype == 'object' or df[factor].nunique() <= 10:
            # Categorical analysis
            crosstab = pd.crosstab(df[factor], df['Diagnosis'], normalize='index') * 100
            print(f"\n{factor} - Diagnosis Distribution (%):")
            print(crosstab.round(2))
            
            # Chi-square test
            contingency_table = pd.crosstab(df[factor], df['Diagnosis'])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            results[factor] = {
                'crosstab': crosstab,
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    # Visualize lifestyle factors
    if available_factors:
        n_factors = len(available_factors)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, factor in enumerate(available_factors):
            if i < len(axes):
                crosstab = pd.crosstab(df[factor], df['Diagnosis'], normalize='index') * 100
                crosstab.plot(kind='bar', ax=axes[i], stacked=False)
                axes[i].set_title(f'{factor} by Diagnosis')
                axes[i].set_xlabel(factor)
                axes[i].set_ylabel('Percentage')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].legend(title='Diagnosis')
        
        # Hide unused subplots
        for i in range(len(available_factors), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('analysis_outputs/lifestyle_risk_factors.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def medical_history_analysis(df):
    """Analyze medical history risk factors"""
    print("="*50)
    print("MEDICAL HISTORY RISK FACTOR ANALYSIS")
    print("="*50)
    
    medical_factors = ['FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 
                      'Depression', 'HeadInjury', 'Hypertension']
    available_factors = [col for col in medical_factors if col in df.columns]
    
    if not available_factors or 'Diagnosis' not in df.columns:
        print("Required columns not found!")
        return None
    
    results = {}
    
    # Calculate risk ratios for each medical factor
    for factor in available_factors:
        crosstab = pd.crosstab(df[factor], df['Diagnosis'], normalize='index') * 100
        print(f"\n{factor} - Diagnosis Distribution (%):")
        print(crosstab.round(2))
        
        # Chi-square test
        contingency_table = pd.crosstab(df[factor], df['Diagnosis'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        results[factor] = {
            'crosstab': crosstab,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Visualize medical history factors
    if available_factors:
        n_factors = len(available_factors)
        n_cols = 3
        n_rows = (n_factors + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, factor in enumerate(available_factors):
            if i < len(axes):
                crosstab = pd.crosstab(df[factor], df['Diagnosis'], normalize='index') * 100
                crosstab.plot(kind='bar', ax=axes[i], stacked=False)
                axes[i].set_title(f'{factor} by Diagnosis')
                axes[i].set_xlabel(factor)
                axes[i].set_ylabel('Percentage')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].legend(title='Diagnosis')
        
        # Hide unused subplots
        for i in range(len(available_factors), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('analysis_outputs/medical_history_risk_factors.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def cognitive_symptoms_analysis(df):
    """Analyze cognitive symptoms and their relationship with diagnosis"""
    print("="*50)
    print("COGNITIVE SYMPTOMS ANALYSIS")
    print("="*50)
    
    cognitive_factors = ['MMSE', 'MemoryComplaints', 'Confusion', 'Disorientation', 'Forgetfulness']
    available_factors = [col for col in cognitive_factors if col in df.columns]
    
    if not available_factors or 'Diagnosis' not in df.columns:
        print("Required columns not found!")
        return None
    
    results = {}
    
    # Analyze each cognitive factor
    for factor in available_factors:
        if df[factor].dtype in ['int64', 'float64']:
            # Numerical analysis
            group_stats = df.groupby('Diagnosis')[factor].agg(['mean', 'std', 'count'])
            print(f"\n{factor} Statistics by Diagnosis:")
            print(group_stats.round(3))
            
            # ANOVA test
            groups = [df[df['Diagnosis'] == diagnosis][factor].dropna() for diagnosis in df['Diagnosis'].unique()]
            groups = [group for group in groups if len(group) > 0]
            
            if len(groups) >= 2:
                if len(groups) == 2:
                    stat, p_value = ttest_ind(groups[0], groups[1])
                    test_type = "T-test"
                else:
                    stat, p_value = f_oneway(*groups)
                    test_type = "ANOVA"
                
                results[factor] = {
                    'group_stats': group_stats,
                    'test_type': test_type,
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        else:
            # Categorical analysis
            crosstab = pd.crosstab(df[factor], df['Diagnosis'], normalize='index') * 100
            print(f"\n{factor} - Diagnosis Distribution (%):")
            print(crosstab.round(2))
            
            # Chi-square test
            contingency_table = pd.crosstab(df[factor], df['Diagnosis'])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            results[factor] = {
                'crosstab': crosstab,
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    # Visualize cognitive symptoms
    if available_factors:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, factor in enumerate(available_factors[:6]):
            if i < len(axes):
                if df[factor].dtype in ['int64', 'float64']:
                    # Box plot for numerical
                    sns.boxplot(data=df, x='Diagnosis', y=factor, ax=axes[i])
                    axes[i].set_title(f'{factor} Distribution by Diagnosis')
                else:
                    # Bar plot for categorical
                    crosstab = pd.crosstab(df[factor], df['Diagnosis'], normalize='index') * 100
                    crosstab.plot(kind='bar', ax=axes[i], stacked=False)
                    axes[i].set_title(f'{factor} by Diagnosis')
                    axes[i].set_ylabel('Percentage')
                
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(available_factors), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('analysis_outputs/cognitive_symptoms_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def generate_insights_document(correlation_results, group_results, lifestyle_results, 
                             medical_results, cognitive_results):
    """Generate comprehensive insights document"""
    
    insights = f"""# Advanced Alzheimer's Analysis Insights

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This document presents key findings from advanced statistical analysis of the Alzheimer's dataset, 
focusing on feature correlations, group differences, and risk factor identification.

## 1. Feature Correlation Insights

### Top Predictive Features
"""

    if correlation_results and 'diagnosis_correlation' in correlation_results:
        top_features = correlation_results['diagnosis_correlation'].head(10)
        insights += "\nThe following features show the strongest correlation with Alzheimer's diagnosis:\n\n"
        for i, (feature, corr) in enumerate(top_features.items(), 1):
            insights += f"{i}. **{feature}**: {corr:.4f}\n"
    
    if correlation_results and 'high_correlation_pairs' in correlation_results:
        insights += f"\n### Highly Correlated Feature Pairs\n\n"
        if correlation_results['high_correlation_pairs']:
            insights += "The following feature pairs show high correlation (>0.7):\n\n"
            for feature1, feature2, corr in correlation_results['high_correlation_pairs'][:10]:
                insights += f"- **{feature1}** ↔ **{feature2}**: {corr:.4f}\n"
        else:
            insights += "No feature pairs with correlation >0.7 were found.\n"

    insights += f"""

## 2. Group Analysis Insights

### Statistically Significant Differences
"""

    if group_results and 'statistical_tests' in group_results:
        significant_features = [r for r in group_results['statistical_tests'] if r['significant']][:10]
        if significant_features:
            insights += "The following features show statistically significant differences across diagnosis groups:\n\n"
            for result in significant_features:
                significance_level = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*"
                insights += f"- **{result['feature']}**: {result['test_type']} (p={result['p_value']:.6f}) {significance_level}\n"

    insights += f"""

## 3. Lifestyle Risk Factor Insights

### Key Findings
"""

    if lifestyle_results:
        insights += "Analysis of lifestyle factors revealed:\n\n"
        for factor, results in lifestyle_results.items():
            if results['significant']:
                insights += f"- **{factor}**: Statistically significant association with diagnosis (p={results['p_value']:.4f})\n"

    insights += f"""

## 4. Medical History Risk Factor Insights

### Key Findings
"""

    if medical_results:
        insights += "Analysis of medical history factors revealed:\n\n"
        for factor, results in medical_results.items():
            if results['significant']:
                insights += f"- **{factor}**: Statistically significant association with diagnosis (p={results['p_value']:.4f})\n"

    insights += f"""

## 5. Cognitive Symptoms Insights

### Key Findings
"""

    if cognitive_results:
        insights += "Analysis of cognitive symptoms revealed:\n\n"
        for factor, results in cognitive_results.items():
            if results['significant']:
                insights += f"- **{factor}**: Statistically significant association with diagnosis (p={results['p_value']:.4f})\n"

    insights += f"""

## 6. Clinical Implications

### Risk Stratification
Based on the analysis, the following factors appear most important for risk assessment:

"""

    if correlation_results and 'diagnosis_correlation' in correlation_results:
        top_risk_factors = correlation_results['diagnosis_correlation'].head(5)
        for i, (feature, corr) in enumerate(top_risk_factors.items(), 1):
            insights += f"{i}. {feature} (correlation: {corr:.3f})\n"

    insights += f"""

### Recommendations for Clinical Practice

1. **Screening Priority**: Focus on the top correlated features for early screening
2. **Risk Assessment**: Consider lifestyle and medical history factors in risk models
3. **Monitoring**: Track cognitive symptoms for disease progression
4. **Prevention**: Address modifiable lifestyle risk factors

## 7. Statistical Notes

- Significance level: p < 0.05
- Multiple comparisons: Consider Bonferroni correction for multiple testing
- Effect sizes: Correlation coefficients and group differences should be interpreted alongside clinical significance

## 8. Generated Visualizations

The following visualizations support these findings:

1. `diagnosis_correlation_ranking.png` - Feature importance ranking
2. `group_differences_significant_features.png` - Group comparisons
3. `lifestyle_risk_factors.png` - Lifestyle factor analysis
4. `medical_history_risk_factors.png` - Medical history analysis
5. `cognitive_symptoms_analysis.png` - Cognitive symptoms analysis

---
*Generated by Advanced Alzheimer's Analysis Script*
"""

    # Save the insights document
    with open('analysis_outputs/advanced_analysis_insights.md', 'w') as f:
        f.write(insights)
    
    print("Advanced insights document saved to: analysis_outputs/advanced_analysis_insights.md")

def main():
    """Main function to run the advanced analysis"""
    print("Starting Advanced Alzheimer's Dataset Analysis...")
    print("="*70)
    
    # Create output directory
    create_output_dir()
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return
    
    # Perform advanced analyses
    print("\n1. Feature Correlation Analysis...")
    correlation_results = feature_correlation_analysis(df)
    
    print("\n2. Group Analysis...")
    group_results = group_analysis(df)
    
    print("\n3. Lifestyle Risk Factor Analysis...")
    lifestyle_results = lifestyle_risk_analysis(df)
    
    print("\n4. Medical History Analysis...")
    medical_results = medical_history_analysis(df)
    
    print("\n5. Cognitive Symptoms Analysis...")
    cognitive_results = cognitive_symptoms_analysis(df)
    
    print("\n6. Generating Insights Document...")
    generate_insights_document(correlation_results, group_results, lifestyle_results,
                             medical_results, cognitive_results)
    
    print("\n" + "="*70)
    print("Advanced Analysis Complete!")
    print("All visualizations and insights saved to: analysis_outputs/")

if __name__ == "__main__":
    main()