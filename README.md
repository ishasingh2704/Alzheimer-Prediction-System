# Alzheimer's Detection System

A comprehensive Next.js 14 application for Alzheimer's disease detection using machine learning, featuring a React frontend, Python Flask backend, and complete data science pipeline.

## 🏗️ Project Structure

```
alzheimer-prediction-system/
├── app/                    # Next.js 14 app directory (pages & routing)
├── components/             # Reusable React components
├── api/python/            # Python Flask backend for ML inference
├── models/                # ML model files and saved scalers
├── notebooks/             # Jupyter notebooks for analysis
├── data/                  # Dataset and processed data
├── outputs/               # Generated plots, reports, and results
├── utils/                 # Helper functions (image processing, API calls)
├── lib/                   # Database connections and utilities
├── public/                # Static assets
├── preprocessed_data/     # Train/validation/test splits
├── analysis_outputs/      # EDA visualizations and reports
└── preprocessing_outputs/ # Data preprocessing reports
```

## 🚀 Features

### Frontend (Next.js 14)
- Modern React interface for Alzheimer's detection
- Image upload and processing capabilities
- Real-time prediction results
- Responsive design with Tailwind CSS

### Backend (Python Flask)
- RESTful API for ML model inference
- Data preprocessing pipeline
- Model serving and prediction endpoints
- CORS-enabled for frontend integration

### Data Science Pipeline
- **Exploratory Data Analysis**: Comprehensive EDA with visualizations
- **Advanced Analysis**: Statistical tests and risk factor analysis
- **Data Preprocessing**: Feature engineering, encoding, and scaling
- **Model Training**: Support for various ML algorithms
- **Evaluation**: Performance metrics and validation

### Machine Learning
- Support for multiple model types (CNN, Random Forest, SVM, etc.)
- Feature engineering and selection
- Cross-validation and hyperparameter tuning
- Model persistence and versioning

## 🛠️ Setup Instructions

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+
- Git

### Frontend Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Run development server:**
   ```bash
   npm run dev
   ```

3. **Build for production:**
   ```bash
   npm run build
   npm start
   ```

### Backend Setup

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements_analysis.txt
   ```

3. **Start Flask server:**
   ```bash
   cd api/python
   python app.py
   ```

### Data Science Setup

1. **Place your dataset:**
   - Add `alzheimer_dataset.csv` to the `/data` folder
   - Ensure it contains all 35 required columns

2. **Launch Jupyter notebooks:**
   ```bash
   jupyter lab notebooks/
   ```

3. **Start with exploratory analysis:**
   - Open `notebooks/exploratory_data_analysis.ipynb`
   - Run all cells to generate comprehensive analysis
   - Review visualizations and insights

## 📊 Data Analysis Workflow

All data analysis is now conducted through Jupyter notebooks for better interactivity and visualization:

### `notebooks/exploratory_data_analysis.ipynb`
- Dataset overview and basic statistics
- Missing values analysis and visualization
- Target variable distribution analysis
- Feature correlations and heatmaps
- Demographic analysis and patterns
- Interactive visualizations with plotly
- Comprehensive EDA with statistical insights

### Planned Notebooks:
- `advanced_analysis.ipynb`: Statistical testing and risk factor analysis
- `data_preprocessing.ipynb`: Feature engineering and data preparation
- `model_training.ipynb`: ML model development and evaluation
- `model_evaluation.ipynb`: Performance analysis and validation

## 🔬 Jupyter Notebooks

### `notebooks/exploratory_data_analysis.ipynb`
- Interactive data exploration
- Statistical analysis
- Visualization with matplotlib, seaborn, and plotly
- Feature distribution analysis
- Correlation studies

## 📈 Model Development

### Supported Features (35 columns):
- **Demographics**: Age, Gender, Ethnicity, EducationLevel
- **Health Metrics**: BMI, SystolicBP, DiastolicBP, Cholesterol levels
- **Lifestyle**: Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality
- **Medical History**: FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes, Depression, etc.
- **Cognitive Assessment**: MMSE, MemoryComplaints, Confusion, Disorientation, etc.
- **Target**: Diagnosis (Alzheimer's vs Normal)

### Engineered Features:
- Age groups (Young/Middle/Elderly)
- BMI categories (Underweight/Normal/Overweight/Obese)
- Blood pressure categories (Normal/Elevated/Hypertension)
- Cholesterol risk score
- Cognitive decline score
- Lifestyle risk score
- Medical history risk score

## 🗃️ Database Integration

The project supports database integration through the `/lib` directory:
- Database connection utilities
- Data persistence for predictions
- User management (future feature)

## 📁 Output Files

### Analysis Outputs (`/outputs/`)
- Generated from Jupyter notebooks
- Interactive visualizations and reports
- Model evaluation results
- Statistical analysis summaries

### Model Files (`/models/`)
- `encoders.pkl`: Label and one-hot encoders
- `scalers.pkl`: StandardScaler and MinMaxScaler
- `*.pkl`: Trained model files
- `*.h5`: Deep learning models

### Processed Data (`/preprocessed_data/`)
- `train_data.csv`: Training dataset (70%)
- `val_data.csv`: Validation dataset (15%)
- `test_data.csv`: Test dataset (15%)
- `feature_names.txt`: List of all features

## 🔧 Environment Variables

Create a `.env.local` file for frontend:
```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

Create a `.env` file for backend:
```env
FLASK_ENV=development
MODEL_PATH=../models/
```

## 🧪 Testing

```bash
# Frontend tests
npm test

# Backend tests
python -m pytest api/python/tests/
```

## 📦 Deployment

### Frontend (Vercel)
```bash
npm run build
```

### Backend (Docker)
```dockerfile
# Dockerfile example in api/python/
FROM python:3.8
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation in `/notebooks/`
- Review analysis reports in `/outputs/`

## 🔬 Research Context

This system is designed for research purposes in Alzheimer's disease detection. The model incorporates:
- Demographic factors
- Lifestyle assessments
- Medical history
- Cognitive evaluations
- Biomarker data

**Note**: This tool is for research purposes only and should not be used as a substitute for professional medical diagnosis.

---

Built with ❤️ using Next.js 14, Python, and Machine Learning
