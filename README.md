# Alzheimer's Detection System

A Next.js 14 project for Alzheimer's disease detection using deep learning and a Python Flask backend.

## Project Structure

- `/app` — Next.js app directory for pages and routing
- `/components` — Reusable React components (UI, uploaders, etc.)
- `/api/python` — Python Flask backend for ML inference
- `/models` — Machine learning model files and documentation
- `/utils` — Helper functions (image processing, API calls)
- `/lib` — Database connection helpers

## Getting Started

### Frontend (Next.js)

1. Install dependencies:

   ```powershell
   npm install
   ```

2. Run the development server:

   ```powershell
   npm run dev
   ```

### Backend (Flask)

1. Install Python dependencies:

   ```powershell
   cd api/python
   pip install -r requirements.txt
   ```

2. Start the Flask server:

   ```powershell
   python app.py
   ```

## Model Files

Place your trained model files in the `/models` directory. See `/models/README.md` for details.

## Notes

- The Flask backend runs on port 5000 by default.
- Update CORS and API URLs as needed for deployment.
- This is a starter structure—extend as needed for your use case.
