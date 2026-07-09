# Kepler Exoplanet ML System — Full-Stack Flask & React Integration

Your codebase has been refactored so that your Python ML backend (`B8. Kepler Telescope`) and modern React frontend (`keplerml`) work together cleanly via Flask.

## Architecture
- **`ml_engine.py`**: Centralized ML pipeline class (`KeplerMLEngine`) managing dataset loading (`Dataset/cumulative.csv`), preprocessing, EDA, regression models (`CatBoost`, `Huber`, `KNN`, `Linear Regression`), and inference.
- **`app.py`**: Flask REST API server with CORS enabled on port `5000`. Exposes:
  - `GET /api/health`
  - `GET /api/overview`
  - `GET /api/dataset`
  - `POST /api/dataset/upload`
  - `POST /api/preprocess`
  - `GET /api/eda`
  - `GET /api/models`
  - `POST /api/models/train`
  - `POST /api/predict`
  - `POST /api/predict/upload`
- **`main.py`**: Your existing Tkinter desktop GUI, now refactored to share `KeplerMLEngine`.

---

## How to Run the Full-Stack Application

### 1. Start the Flask Backend Server
Open a terminal in the backend directory (`B8. Kepler Telescope`):
```powershell
# Install dependencies if not already installed
pip install -r requirements.txt

# Start the Flask REST API server
python app.py
```
The Flask server will listen on `http://127.0.0.1:5000`.

### 2. Start the React Frontend Application
Open a second terminal in the frontend directory (`keplerml`):
```powershell
# Install node dependencies
npm install

# Start Vite Dev Server
npm run dev
```
Open the UI in your browser (`http://localhost:5173`).
The Header will show **FLASK API CONNECTED** in green when the Flask server is running!
