"""
Alzheimer's Dataset Preprocessing Script
This script performs comprehensive data preprocessing including cleaning, feature engineering,
encoding, scaling, and splitting for the Alzheimer's detection system.
"""

import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_filename = f"preprocessing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def create_output_dirs():
    """Create necessary output directories"""
    dirs = ['preprocessed_data', 'models', 'preprocessing_outputs']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            logging.info(f"Created directory: {dir_name}")

def load_dataset():
    """Load the Alzheimer's dataset"""
    try:
        df = pd.read_csv('data/alzheimer_dataset.csv')
        logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error("alzheimer_dataset.csv not found in data/ directory")
        return None
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None

def data_cleaning(df):
    """Perform comprehensive data cleaning"""
    logging.info("Starting data cleaning...")
    
    # Store original shape
    original_shape = df.shape
    logging.info(f"Original dataset shape: {original_shape}")
    
    # 1. Remove unnecessary columns
    columns_to_remove = ['PatientID', 'DoctorInCharge']
    existing_cols_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    if existing_cols_to_remove:
        df = df.drop(columns=existing_cols_to_remove)
        logging.info(f"Removed columns: {existing_cols_to_remove}")
    
    # 2. Handle missing values
    logging.info("Handling missing values...")
    
    # Check for missing values
    missing_before = df.isnull().sum()
    missing_cols = missing_before[missing_before > 0]
    
    if not missing_cols.empty:
        logging.info(f"Found missing values in {len(missing_cols)} columns")
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Median imputation for numerical features
        if numerical_cols:
            numerical_imputer = SimpleImputer(strategy='median')
            df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
            
            # Save the imputer
            with open('models/numerical_imputer.pkl', 'wb') as f:
                pickle.dump(numerical_imputer, f)
            
            logging.info(f"Applied median imputation to {len(numerical_cols)} numerical columns")
        
        # Mode imputation for categorical features
        if categorical_cols:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
            
            # Save the imputer
            with open('models/categorical_imputer.pkl', 'wb') as f:
                pickle.dump(categorical_imputer, f)
            
            logging.info(f"Applied mode imputation to {len(categorical_cols)} categorical columns")
    else:
        logging.info("No missing values found")
    
    # 3. Handle outliers using IQR method
    outlier_columns = ['BMI', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
                      'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides']
    
    existing_outlier_cols = [col for col in outlier_columns if col in df.columns]
    
    outlier_info = {}
    for col in existing_outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        # Cap outliers instead of removing them
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        outliers_after = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        outlier_info[col] = {
            'outliers_before': outliers_before,
            'outliers_after': outliers_after,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        if outliers_before > 0:
            logging.info(f"{col}: Capped {outliers_before} outliers (bounds: {lower_bound:.2f} - {upper_bound:.2f})")
    
    logging.info(f"Data cleaning completed. Final shape: {df.shape}")
    
    return df, outlier_info

def feature_engineering(df):
    """Create new features from existing ones"""
    logging.info("Starting feature engineering...")
    
    original_features = df.columns.tolist()
    
    # 1. Age groups
    if 'Age' in df.columns:
        df['AgeGroup'] = pd.cut(df['Age'], 
                               bins=[0, 50, 65, 100], 
                               labels=['Young', 'Middle', 'Elderly'])
        logging.info("Created AgeGroup feature")
    
    # 2. BMI categories
    if 'BMI' in df.columns:
        df['BMICategory'] = pd.cut(df['BMI'],
                                  bins=[0, 18.5, 25, 30, 100],
                                  labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        logging.info("Created BMICategory feature")
    
    # 3. Blood Pressure categories
    if 'SystolicBP' in df.columns and 'DiastolicBP' in df.columns:
        def categorize_bp(systolic, diastolic):
            if systolic < 120 and diastolic < 80:
                return 'Normal'
            elif systolic < 130 and diastolic < 80:
                return 'Elevated'
            else:
                return 'Hypertension'
        
        df['BPCategory'] = df.apply(lambda row: categorize_bp(row['SystolicBP'], row['DiastolicBP']), axis=1)
        logging.info("Created BPCategory feature")
    
    # 4. Cholesterol risk score
    cholesterol_cols = ['CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides']
    available_chol_cols = [col for col in cholesterol_cols if col in df.columns]
    
    if len(available_chol_cols) >= 3:
        # Normalize cholesterol values and create risk score
        chol_risk_score = 0
        
        if 'CholesterolTotal' in df.columns:
            chol_risk_score += (df['CholesterolTotal'] > 200).astype(int)  # High total cholesterol
        
        if 'CholesterolLDL' in df.columns:
            chol_risk_score += (df['CholesterolLDL'] > 130).astype(int)  # High LDL
        
        if 'CholesterolHDL' in df.columns:
            chol_risk_score += (df['CholesterolHDL'] < 40).astype(int)  # Low HDL
        
        if 'CholesterolTriglycerides' in df.columns:
            chol_risk_score += (df['CholesterolTriglycerides'] > 150).astype(int)  # High triglycerides
        
        df['CholesterolRiskScore'] = chol_risk_score
        logging.info("Created CholesterolRiskScore feature")
    
    # 5. Cognitive decline score
    cognitive_cols = ['MMSE', 'MemoryComplaints', 'Confusion', 'Disorientation', 'Forgetfulness']
    available_cog_cols = [col for col in cognitive_cols if col in df.columns]
    
    if len(available_cog_cols) >= 3:
        cognitive_score = 0
        
        if 'MMSE' in df.columns:
            # Lower MMSE indicates cognitive decline
            cognitive_score += (df['MMSE'] < 24).astype(int)  # MMSE < 24 suggests cognitive impairment
        
        # For binary cognitive symptoms (assuming 1 = present, 0 = absent)
        for col in ['MemoryComplaints', 'Confusion', 'Disorientation', 'Forgetfulness']:
            if col in df.columns:
                if df[col].dtype == 'object':
                    # Convert Yes/No to 1/0
                    df[col] = df[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 1: 1, 0: 0})
                cognitive_score += df[col].fillna(0)
        
        df['CognitiveDeclineScore'] = cognitive_score
        logging.info("Created CognitiveDeclineScore feature")
    
    # 6. Lifestyle risk score
    lifestyle_cols = ['Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']
    available_lifestyle_cols = [col for col in lifestyle_cols if col in df.columns]
    
    if len(available_lifestyle_cols) >= 3:
        lifestyle_risk = 0
        
        # Higher values generally indicate better lifestyle (except smoking and alcohol)
        if 'Smoking' in df.columns:
            # Convert to binary if needed
            if df['Smoking'].dtype == 'object':
                df['Smoking'] = df['Smoking'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 1: 1, 0: 0})
            lifestyle_risk += df['Smoking'].fillna(0)
        
        if 'AlcoholConsumption' in df.columns:
            # Assuming higher values = more consumption = higher risk
            lifestyle_risk += (df['AlcoholConsumption'] > df['AlcoholConsumption'].median()).astype(int)
        
        if 'PhysicalActivity' in df.columns:
            # Lower physical activity = higher risk
            lifestyle_risk += (df['PhysicalActivity'] < df['PhysicalActivity'].median()).astype(int)
        
        if 'DietQuality' in df.columns:
            # Lower diet quality = higher risk
            lifestyle_risk += (df['DietQuality'] < df['DietQuality'].median()).astype(int)
        
        if 'SleepQuality' in df.columns:
            # Lower sleep quality = higher risk
            lifestyle_risk += (df['SleepQuality'] < df['SleepQuality'].median()).astype(int)
        
        df['LifestyleRiskScore'] = lifestyle_risk
        logging.info("Created LifestyleRiskScore feature")
    
    # 7. Medical history risk score
    medical_cols = ['FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 
                   'Depression', 'HeadInjury', 'Hypertension']
    available_medical_cols = [col for col in medical_cols if col in df.columns]
    
    if len(available_medical_cols) >= 3:
        medical_risk = 0
        
        for col in available_medical_cols:
            if df[col].dtype == 'object':
                # Convert Yes/No to 1/0
                df[col] = df[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 1: 1, 0: 0})
            medical_risk += df[col].fillna(0)
        
        df['MedicalHistoryRiskScore'] = medical_risk
        logging.info("Created MedicalHistoryRiskScore feature")
    
    new_features = [col for col in df.columns if col not in original_features]
    logging.info(f"Feature engineering completed. Created {len(new_features)} new features: {new_features}")
    
    return df

def encode_features(df):
    """Encode categorical features"""
    logging.info("Starting feature encoding...")
    
    # Store encoders
    encoders = {}
    
    # 1. Identify binary and multi-class categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variable from encoding (will handle separately)
    if 'Diagnosis' in categorical_cols:
        categorical_cols.remove('Diagnosis')
    
    binary_features = []
    multiclass_features = []
    
    for col in categorical_cols:
        unique_values = df[col].nunique()
        if unique_values == 2:
            binary_features.append(col)
        elif unique_values > 2:
            multiclass_features.append(col)
    
    # 2. Label encode binary features
    for col in binary_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[f'{col}_labelencoder'] = le
        logging.info(f"Label encoded binary feature: {col}")
    
    # 3. One-hot encode multi-class features
    for col in multiclass_features:
        # Store the original categories for inverse transform BEFORE dropping
        encoders[f'{col}_categories'] = df[col].unique().tolist()
        
        # Create dummy variables
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
        
        logging.info(f"One-hot encoded multi-class feature: {col}")
    
    # 4. Encode target variable
    if 'Diagnosis' in df.columns:
        le_target = LabelEncoder()
        df['Diagnosis'] = le_target.fit_transform(df['Diagnosis'].astype(str))
        encoders['target_labelencoder'] = le_target
        logging.info(f"Encoded target variable: {le_target.classes_}")
    
    # Save all encoders
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    logging.info(f"Feature encoding completed. Saved {len(encoders)} encoders")
    
    return df, encoders

def scale_features(df):
    """Scale numerical features"""
    logging.info("Starting feature scaling...")
    
    scalers = {}
    
    # Features for StandardScaler
    standard_scale_features = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
                              'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
                              'MMSE', 'FunctionalAssessment', 'ADL']
    
    # Features for MinMaxScaler
    minmax_scale_features = ['AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']
    
    # Filter for existing columns
    existing_standard_features = [col for col in standard_scale_features if col in df.columns]
    existing_minmax_features = [col for col in minmax_scale_features if col in df.columns]
    
    # Apply StandardScaler
    if existing_standard_features:
        standard_scaler = StandardScaler()
        df[existing_standard_features] = standard_scaler.fit_transform(df[existing_standard_features])
        scalers['standard_scaler'] = standard_scaler
        scalers['standard_features'] = existing_standard_features
        logging.info(f"Applied StandardScaler to {len(existing_standard_features)} features")
    
    # Apply MinMaxScaler
    if existing_minmax_features:
        minmax_scaler = MinMaxScaler()
        df[existing_minmax_features] = minmax_scaler.fit_transform(df[existing_minmax_features])
        scalers['minmax_scaler'] = minmax_scaler
        scalers['minmax_features'] = existing_minmax_features
        logging.info(f"Applied MinMaxScaler to {len(existing_minmax_features)} features")
    
    # Save scalers
    with open('models/scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    
    logging.info("Feature scaling completed")
    
    return df, scalers

def split_data(df):
    """Split data into train, validation, and test sets"""
    logging.info("Starting data splitting...")
    
    if 'Diagnosis' not in df.columns:
        logging.error("Target variable 'Diagnosis' not found!")
        return None, None, None, None, None, None
    
    # Separate features and target
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']
    
    # First split: 70% train, 30% temp (15% val + 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Second split: split temp into val and test (50-50 of temp = 15% each of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logging.info(f"Data split completed:")
    logging.info(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    logging.info(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
    logging.info(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    
    # Check class distribution
    train_dist = y_train.value_counts(normalize=True)
    val_dist = y_val.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    
    logging.info("Class distribution:")
    logging.info(f"  Train: {train_dist.to_dict()}")
    logging.info(f"  Val: {val_dist.to_dict()}")
    logging.info(f"  Test: {test_dist.to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance"""
    logging.info("Checking for class imbalance...")
    
    class_counts = y_train.value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    logging.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 2:  # Apply SMOTE if imbalance ratio > 2
        logging.info("Applying SMOTE to balance classes...")
        
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Log new class distribution
        new_class_counts = pd.Series(y_train_balanced).value_counts()
        logging.info(f"After SMOTE - Class distribution: {new_class_counts.to_dict()}")
        
        # Save SMOTE object
        with open('models/smote.pkl', 'wb') as f:
            pickle.dump(smote, f)
        
        return X_train_balanced, y_train_balanced, True
    else:
        logging.info("Classes are reasonably balanced. SMOTE not applied.")
        return X_train, y_train, False

def save_datasets(X_train, X_val, X_test, y_train, y_val, y_test):
    """Save preprocessed datasets"""
    logging.info("Saving preprocessed datasets...")
    
    # Combine features and target for saving
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Save as CSV
    train_data.to_csv('preprocessed_data/train_data.csv', index=False)
    val_data.to_csv('preprocessed_data/val_data.csv', index=False)
    test_data.to_csv('preprocessed_data/test_data.csv', index=False)
    
    # Save feature names
    feature_names = X_train.columns.tolist()
    with open('preprocessed_data/feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    logging.info("Datasets saved successfully:")
    logging.info(f"  Train: preprocessed_data/train_data.csv")
    logging.info(f"  Validation: preprocessed_data/val_data.csv")
    logging.info(f"  Test: preprocessed_data/test_data.csv")
    logging.info(f"  Features: {len(feature_names)} features saved")

def generate_preprocessing_report(original_df, final_df, outlier_info, encoders, scalers, 
                                smote_applied, log_filename):
    """Generate comprehensive preprocessing report"""
    logging.info("Generating preprocessing report...")
    
    report = f"""# Data Preprocessing Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Transformation Summary

### Original Dataset
- **Shape**: {original_df.shape}
- **Features**: {original_df.shape[1]}
- **Samples**: {original_df.shape[0]:,}

### Final Dataset
- **Shape**: {final_df.shape}
- **Features**: {final_df.shape[1]}
- **Samples**: {final_df.shape[0]:,}

### Changes
- **Features Added**: {final_df.shape[1] - original_df.shape[1]}
- **Samples Changed**: {final_df.shape[0] - original_df.shape[0]:,}

## Data Cleaning

### Missing Values
"""
    
    original_missing = original_df.isnull().sum()
    final_missing = final_df.isnull().sum()
    
    if original_missing.sum() > 0:
        report += f"- **Original Missing Values**: {original_missing.sum():,} total\n"
        report += f"- **Final Missing Values**: {final_missing.sum():,} total\n"
    else:
        report += "- No missing values found in original dataset\n"
    
    ### Outlier Handling
    if outlier_info:
        report += "\n### Outlier Handling (IQR Method)\n\n"
        for col, info in outlier_info.items():
            if info['outliers_before'] > 0:
                report += f"- **{col}**: {info['outliers_before']} outliers capped (bounds: {info['lower_bound']:.2f} - {info['upper_bound']:.2f})\n"
    
    ## Feature Engineering
    engineered_features = [col for col in final_df.columns if col not in original_df.columns and col != 'Diagnosis']
    if engineered_features:
        report += f"\n## Feature Engineering\n\nCreated {len(engineered_features)} new features:\n\n"
        for feature in engineered_features:
            report += f"- {feature}\n"
    
    ## Encoding
    if encoders:
        report += f"\n## Feature Encoding\n\n"
        label_encoded = [key for key in encoders.keys() if 'labelencoder' in key]
        onehot_encoded = [key for key in encoders.keys() if 'categories' in key]
        
        if label_encoded:
            report += f"### Label Encoded Features ({len(label_encoded)})\n"
            for encoder in label_encoded:
                feature_name = encoder.replace('_labelencoder', '')
                report += f"- {feature_name}\n"
        
        if onehot_encoded:
            report += f"\n### One-Hot Encoded Features ({len(onehot_encoded)})\n"
            for encoder in onehot_encoded:
                feature_name = encoder.replace('_categories', '')
                report += f"- {feature_name}\n"
    
    ## Scaling
    if scalers:
        report += f"\n## Feature Scaling\n\n"
        if 'standard_features' in scalers:
            report += f"### StandardScaler Applied ({len(scalers['standard_features'])} features)\n"
            for feature in scalers['standard_features']:
                report += f"- {feature}\n"
        
        if 'minmax_features' in scalers:
            report += f"\n### MinMaxScaler Applied ({len(scalers['minmax_features'])} features)\n"
            for feature in scalers['minmax_features']:
                report += f"- {feature}\n"
    
    ## Data Splitting
    report += f"\n## Data Splitting\n\n"
    report += f"- **Training Set**: 70% of data\n"
    report += f"- **Validation Set**: 15% of data\n"
    report += f"- **Test Set**: 15% of data\n"
    report += f"- **Stratification**: Applied on target variable\n"
    
    if smote_applied:
        report += f"- **SMOTE**: Applied to training set for class balancing\n"
    else:
        report += f"- **SMOTE**: Not applied (classes reasonably balanced)\n"
    
    ## Files Generated
    report += f"\n## Generated Files\n\n"
    report += f"### Datasets\n"
    report += f"- `preprocessed_data/train_data.csv`\n"
    report += f"- `preprocessed_data/val_data.csv`\n"
    report += f"- `preprocessed_data/test_data.csv`\n"
    report += f"- `preprocessed_data/feature_names.txt`\n"
    
    report += f"\n### Models and Transformers\n"
    report += f"- `models/encoders.pkl`\n"
    report += f"- `models/scalers.pkl`\n"
    if 'numerical_imputer.pkl' in os.listdir('models'):
        report += f"- `models/numerical_imputer.pkl`\n"
    if 'categorical_imputer.pkl' in os.listdir('models'):
        report += f"- `models/categorical_imputer.pkl`\n"
    if smote_applied:
        report += f"- `models/smote.pkl`\n"
    
    report += f"\n### Logs\n"
    report += f"- `{log_filename}`\n"
    
    ## Next Steps
    report += f"\n## Recommendations for Next Steps\n\n"
    report += f"1. **Model Training**: Use the preprocessed training data for model development\n"
    report += f"2. **Validation**: Use validation set for hyperparameter tuning\n"
    report += f"3. **Final Evaluation**: Use test set only for final model evaluation\n"
    report += f"4. **Feature Selection**: Consider feature importance analysis\n"
    report += f"5. **Cross-Validation**: Implement k-fold CV on training data\n"
    
    report += f"\n---\n*Generated by Alzheimer's Detection System Preprocessing Pipeline*\n"
    
    # Save the report
    with open('preprocessing_outputs/preprocessing_report.md', 'w') as f:
        f.write(report)
    
    logging.info("Preprocessing report saved to: preprocessing_outputs/preprocessing_report.md")

def main():
    """Main preprocessing pipeline"""
    print("Starting Alzheimer's Dataset Preprocessing Pipeline...")
    print("="*60)
    
    # Setup logging
    log_filename = setup_logging()
    
    # Create output directories
    create_output_dirs()
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return
    
    # Store original dataset for reporting
    original_df = df.copy()
    
    # 1. Data Cleaning
    df, outlier_info = data_cleaning(df)
    
    # 2. Feature Engineering
    df = feature_engineering(df)
    
    # 3. Feature Encoding
    df, encoders = encode_features(df)
    
    # 4. Feature Scaling
    df, scalers = scale_features(df)
    
    # 5. Data Splitting
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    if X_train is None:
        return
    
    # 6. Handle Class Imbalance
    X_train, y_train, smote_applied = apply_smote(X_train, y_train)
    
    # 7. Save Datasets
    save_datasets(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # 8. Generate Report
    generate_preprocessing_report(original_df, df, outlier_info, encoders, 
                                scalers, smote_applied, log_filename)
    
    print("\n" + "="*60)
    print("Preprocessing Pipeline Complete!")
    print("Check the following directories for outputs:")
    print("  - preprocessed_data/ : Train, validation, test datasets")
    print("  - models/ : Saved transformers and models")
    print("  - preprocessing_outputs/ : Reports and documentation")
    print(f"  - {log_filename} : Detailed processing log")

if __name__ == "__main__":
    main()