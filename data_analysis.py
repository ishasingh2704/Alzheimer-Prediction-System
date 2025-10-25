"""
Alzheimer's Dataset Exploratory Data Analysis
This script performs comprehensive EDA on the Alzheimer's dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def basic_info_analysis(df):
    """Perform basic dataset information analysis"""
    print("="*50)
    print("DATASET BASIC INFORMATION")
    print("="*50)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumn Names:")
    print(df.columns.tolist())
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nDataset Info:")
    df.info()
    
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict()
    }

def statistical_summary(df):
    """Generate statistical summary for numerical features"""
    print("="*50)
    print("STATISTICAL SUMMARY")
    print("="*50)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print("Numerical columns:", numerical_cols.tolist())
    
    summary = df[numerical_cols].describe()
    print("\nStatistical Summary:")
    print(summary)
    
    return summary

def missing_values_analysis(df):
    """Analyze missing values and create heatmap"""
    print("="*50)
    print("MISSING VALUES ANALYSIS")
    print("="*50)
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    print("Missing Values Summary:")
    print(missing_df)
    
    # Create missing values heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig('analysis_outputs/missing_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return missing_df

def target_variable_analysis(df):
    """Analyze the target variable (Diagnosis) distribution"""
    print("="*50)
    print("TARGET VARIABLE ANALYSIS")
    print("="*50)
    
    if 'Diagnosis' in df.columns:
        diagnosis_counts = df['Diagnosis'].value_counts()
        diagnosis_percent = df['Diagnosis'].value_counts(normalize=True) * 100
        
        print("Diagnosis Distribution:")
        for diagnosis, count in diagnosis_counts.items():
            percent = diagnosis_percent[diagnosis]
            print(f"{diagnosis}: {count} ({percent:.2f}%)")
        
        # Plot diagnosis distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        diagnosis_counts.plot(kind='bar', color='skyblue')
        plt.title('Diagnosis Distribution (Count)')
        plt.xlabel('Diagnosis')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.pie(diagnosis_counts.values, labels=diagnosis_counts.index, autopct='%1.1f%%')
        plt.title('Diagnosis Distribution (Percentage)')
        
        plt.tight_layout()
        plt.savefig('analysis_outputs/diagnosis_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return diagnosis_counts
    else:
        print("Diagnosis column not found!")
        return None

def age_diagnosis_analysis(df):
    """Analyze age distribution by diagnosis"""
    if 'Age' in df.columns and 'Diagnosis' in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Box plot
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x='Diagnosis', y='Age')
        plt.title('Age Distribution by Diagnosis (Box Plot)')
        plt.xticks(rotation=45)
        
        # Violin plot
        plt.subplot(1, 2, 2)
        sns.violinplot(data=df, x='Diagnosis', y='Age')
        plt.title('Age Distribution by Diagnosis (Violin Plot)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('analysis_outputs/age_by_diagnosis.png', dpi=300, bbox_inches='tight')
        plt.close()

def demographic_analysis(df):
    """Analyze gender and ethnicity distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gender distribution
    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts()
        axes[0, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Gender Distribution')
        
        # Gender by Diagnosis
        if 'Diagnosis' in df.columns:
            pd.crosstab(df['Gender'], df['Diagnosis']).plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Gender Distribution by Diagnosis')
            axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Ethnicity distribution
    if 'Ethnicity' in df.columns:
        ethnicity_counts = df['Ethnicity'].value_counts()
        ethnicity_counts.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Ethnicity Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Ethnicity by Diagnosis
        if 'Diagnosis' in df.columns:
            pd.crosstab(df['Ethnicity'], df['Diagnosis']).plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Ethnicity Distribution by Diagnosis')
            axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('analysis_outputs/demographic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def correlation_analysis(df):
    """Create correlation heatmap for numerical features"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 1:
        plt.figure(figsize=(16, 12))
        correlation_matrix = df[numerical_cols].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, fmt='.2f')
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        plt.savefig('analysis_outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return correlation_matrix

def distribution_plots(df):
    """Create distribution plots for key numerical features"""
    key_features = ['BMI', 'MMSE', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
                   'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides']
    
    available_features = [col for col in key_features if col in df.columns]
    
    if available_features:
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes
        
        for i, feature in enumerate(available_features):
            if i < len(axes):
                df[feature].hist(bins=30, ax=axes[i], alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('analysis_outputs/numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

def categorical_plots(df):
    """Create count plots for categorical features"""
    categorical_features = ['Smoking', 'Diabetes', 'Depression', 'CardiovascularDisease',
                          'Hypertension', 'FamilyHistoryAlzheimers', 'HeadInjury']
    
    available_features = [col for col in categorical_features if col in df.columns]
    
    if available_features:
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes
        
        for i, feature in enumerate(available_features):
            if i < len(axes):
                df[feature].value_counts().plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Count')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('analysis_outputs/categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

def cognitive_analysis(df):
    """Analyze cognitive features by diagnosis"""
    cognitive_features = ['MMSE', 'MemoryComplaints', 'Confusion', 'FunctionalAssessment']
    available_features = [col for col in cognitive_features if col in df.columns]
    
    if available_features and 'Diagnosis' in df.columns:
        n_features = len(available_features)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features[:4]):
            if i < 4:
                sns.boxplot(data=df, x='Diagnosis', y=feature, ax=axes[i])
                axes[i].set_title(f'{feature} by Diagnosis')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('analysis_outputs/cognitive_features_by_diagnosis.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_summary_report(df, basic_info, summary_stats, missing_values, diagnosis_dist):
    """Generate a comprehensive summary report in markdown format"""
    report = f"""# Alzheimer's Dataset Analysis Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview

- **Total Records**: {basic_info['shape'][0]:,}
- **Total Features**: {basic_info['shape'][1]}
- **Dataset Size**: {df.memory_usage().sum() / 1024**2:.2f} MB

## Data Types Summary

"""

    # Add data types summary
    dtype_summary = df.dtypes.value_counts()
    for dtype, count in dtype_summary.items():
        report += f"- **{dtype}**: {count} columns\n"

    report += f"""

## Missing Values Summary

"""
    if not missing_values.empty:
        report += "| Column | Missing Count | Missing Percentage |\n"
        report += "|--------|---------------|--------------------|\n"
        for col, row in missing_values.iterrows():
            report += f"| {col} | {row['Missing Count']} | {row['Missing Percentage']:.2f}% |\n"
    else:
        report += "No missing values found in the dataset.\n"

    if diagnosis_dist is not None:
        report += f"""

## Target Variable (Diagnosis) Distribution

"""
        for diagnosis, count in diagnosis_dist.items():
            percentage = (count / diagnosis_dist.sum()) * 100
            report += f"- **{diagnosis}**: {count:,} ({percentage:.2f}%)\n"

    report += f"""

## Key Insights

1. **Dataset Size**: The dataset contains {basic_info['shape'][0]:,} records with {basic_info['shape'][1]} features.

2. **Data Quality**: {'No missing values detected.' if missing_values.empty else f'{len(missing_values)} columns have missing values.'}

3. **Target Distribution**: {'Balanced' if diagnosis_dist is not None and diagnosis_dist.std() / diagnosis_dist.mean() < 0.5 else 'Imbalanced'} distribution across diagnosis categories.

## Generated Visualizations

The following visualizations have been saved to the `analysis_outputs/` folder:

1. `missing_values_heatmap.png` - Missing values pattern
2. `diagnosis_distribution.png` - Target variable distribution
3. `age_by_diagnosis.png` - Age distribution by diagnosis
4. `demographic_analysis.png` - Gender and ethnicity analysis
5. `correlation_heatmap.png` - Feature correlations
6. `numerical_distributions.png` - Distribution of key numerical features
7. `categorical_distributions.png` - Distribution of categorical features
8. `cognitive_features_by_diagnosis.png` - Cognitive features by diagnosis

## Recommendations

1. **Data Preprocessing**: {'No preprocessing needed for missing values.' if missing_values.empty else 'Handle missing values before modeling.'}
2. **Feature Engineering**: Consider creating derived features from existing ones.
3. **Model Selection**: {'Consider ensemble methods for balanced classification.' if diagnosis_dist is not None and diagnosis_dist.std() / diagnosis_dist.mean() < 0.5 else 'Use techniques for imbalanced classification.'}

---
*Report generated by Alzheimer's Detection System Data Analysis Script*
"""

    # Save the report
    with open('analysis_outputs/analysis_summary_report.md', 'w') as f:
        f.write(report)
    
    print("Summary report saved to: analysis_outputs/analysis_summary_report.md")

def main():
    """Main function to run the complete analysis"""
    print("Starting Alzheimer's Dataset Analysis...")
    print("="*60)
    
    # Create output directory
    create_output_dir()
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return
    
    # Perform analyses
    print("\n1. Basic Information Analysis...")
    basic_info = basic_info_analysis(df)
    
    print("\n2. Statistical Summary...")
    summary_stats = statistical_summary(df)
    
    print("\n3. Missing Values Analysis...")
    missing_values = missing_values_analysis(df)
    
    print("\n4. Target Variable Analysis...")
    diagnosis_dist = target_variable_analysis(df)
    
    print("\n5. Creating Visualizations...")
    print("   - Age by Diagnosis...")
    age_diagnosis_analysis(df)
    
    print("   - Demographic Analysis...")
    demographic_analysis(df)
    
    print("   - Correlation Analysis...")
    correlation_analysis(df)
    
    print("   - Distribution Plots...")
    distribution_plots(df)
    
    print("   - Categorical Plots...")
    categorical_plots(df)
    
    print("   - Cognitive Features Analysis...")
    cognitive_analysis(df)
    
    print("\n6. Generating Summary Report...")
    generate_summary_report(df, basic_info, summary_stats, missing_values, diagnosis_dist)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("All visualizations saved to: analysis_outputs/")
    print("Summary report saved to: analysis_outputs/analysis_summary_report.md")

if __name__ == "__main__":
    main()
