"""Lightweight preprocessing to produce pickles in outputs/preprocessed/
This mirrors the notebook's preprocessing so the model runner can load X_train/y_train etc.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'alzheimer_dataset.csv'
OUT_PRE = ROOT / 'outputs' / 'preprocessed'
OUT_MODELS = ROOT / 'outputs' / 'models'
OUT_PRE.mkdir(parents=True, exist_ok=True)
OUT_MODELS.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please add alzheimer_dataset.csv to the data folder.")

print('Loading raw dataset...')
df = pd.read_csv(DATA_PATH)
print('Rows, cols:', df.shape)

data = df.copy()
for c in ['PatientID','DoctorInCharge']:
    if c in data.columns:
        data.drop(columns=c, inplace=True)

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = data.select_dtypes(include=['object']).columns.tolist()

for col in numeric_cols:
    if data[col].isnull().any():
        med = data[col].median()
        data[col].fillna(med, inplace=True)

for col in cat_cols:
    if data[col].isnull().any():
        mode = data[col].mode()
        if not mode.empty:
            data[col].fillna(mode[0], inplace=True)
        else:
            data[col].fillna('Unknown', inplace=True)

# Feature engineering
if 'Age' in data.columns:
    bins = [0,50,70,200]
    labels = ['Young','Middle-aged','Elderly']
    data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels)

if 'BMI' in data.columns:
    def bmi_cat(bmi):
        try:
            bmi = float(bmi)
        except Exception:
            return 'Unknown'
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    data['BMICategory'] = data['BMI'].apply(bmi_cat)

if {'CholesterolLDL','CholesterolHDL','CholesterolTotal'}.issubset(set(data.columns)):
    data['CholesterolRiskScore'] = (data['CholesterolLDL'] - data['CholesterolHDL']) / (data['CholesterolTotal'].replace(0, np.nan))
    data['CholesterolRiskScore'].fillna(0, inplace=True)

symptom_cols = [c for c in data.columns if c in ['MemoryComplaints','BehavioralProblems','Confusion','Disorientation','PersonalityChanges','DifficultyCompletingTasks','Forgetfulness']]
if symptom_cols:
    data['TotalSymptomCount'] = data[symptom_cols].sum(axis=1)
else:
    data['TotalSymptomCount'] = 0

# Label encode some binary-like
label_encoders = {}
bin_candidates = ['Gender','Smoking','FamilyHistoryAlzheimers','CardiovascularDisease','Diabetes','Depression','HeadInjury','Hypertension']
for col in bin_candidates:
    if col in data.columns:
        le = LabelEncoder()
        try:
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
        except Exception:
            pass

# One-hot encode some categories
ohe_cols = [c for c in ['AgeGroup','BMICategory','Ethnicity','EducationLevel','BPCategory'] if c in data.columns]
if ohe_cols:
    data = pd.get_dummies(data, columns=ohe_cols, drop_first=True)

# Target
if 'Diagnosis' not in data.columns:
    raise KeyError('Diagnosis column not found in dataset')
le_target = LabelEncoder()
data['Diagnosis_encoded'] = le_target.fit_transform(data['Diagnosis'].astype(str))
y = data['Diagnosis_encoded']
X = data.drop(columns=['Diagnosis','Diagnosis_encoded'])

# Scaling
continuous = [c for c in X.select_dtypes(include=[np.number]).columns if c not in ['TotalSymptomCount']]
score_like = [c for c in ['MMSE','FunctionalAssessment','ADL'] if c in X.columns]

scaler = StandardScaler()
if continuous:
    X[continuous] = scaler.fit_transform(X[continuous])
    joblib.dump(scaler, OUT_MODELS / 'standard_scaler.pkl')

mm_scaler = MinMaxScaler()
if score_like:
    X[score_like] = mm_scaler.fit_transform(X[score_like])
    joblib.dump(mm_scaler, OUT_MODELS / 'minmax_scaler.pkl')

for k, enc in label_encoders.items():
    joblib.dump(enc, OUT_MODELS / f'encoder_{k}.pkl')

# split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print('Shapes —', X_train.shape, X_val.shape, X_test.shape)

joblib.dump(X_train, OUT_PRE / 'X_train.pkl')
joblib.dump(X_val, OUT_PRE / 'X_val.pkl')
joblib.dump(X_test, OUT_PRE / 'X_test.pkl')
joblib.dump(y_train, OUT_PRE / 'y_train.pkl')
joblib.dump(y_val, OUT_PRE / 'y_val.pkl')
joblib.dump(y_test, OUT_PRE / 'y_test.pkl')

print('Saved preprocessed pickles to', OUT_PRE)
