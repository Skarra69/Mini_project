# Kepler Exoplanet ML Validation System — Full-Stack Architecture & Workflow Guide

Welcome to the **Kepler Exoplanet Machine Learning Validation System**. This document provides a comprehensive overview of the newly refactored full-stack architecture, dual-flow data pipelines, codebase structure, and execution instructions for developers and researchers.

---

## 1. System Overview

The application bridges a **Python Machine Learning Backend** (powered by `scikit-learn`, `CatBoost`, and **Flask**) with a **Modern React + Vite Frontend** (`keplerml`). It is designed to classify NASA Kepler transit light-curve signals as either **Confirmed/Habitable Exoplanet Candidates** or **Astrophysical False Positives**.

```
+-----------------------------------------------------------------------+
|                       REACT VITE FRONTEND                             |
|               (http://localhost:3001 / Port 3001)                     |
|                                                                       |
|  [Dashboard]  [Dataset]  [Preprocessing]  [EDA]  [Models]  [Predict]  |
+-----------------------------------+-----------------------------------+
                                    |
                    HTTP REST API   |   JSON / CORS
                    (Vite Proxy)    v
+-----------------------------------+-----------------------------------+
|                        FLASK API SERVER                               |
|               (http://127.0.0.1:5000 / app.py)                        |
|                                                                       |
|   GET /api/health    GET /api/overview       GET /api/dataset         |
|   GET /api/eda       POST /api/models/train  POST /api/predict        |
+-----------------------------------+-----------------------------------+
                                    |
              Instantiates & Calls  |   Python Method Invocations
                                    v
+-----------------------------------+-----------------------------------+
|                    KEPLER ML ENGINE MODULE                            |
|                       (ml_engine.py)                                  |
|                                                                       |
|   - Data Ingestion (Dataset/cumulative.csv — 9,564 KOIs)              |
|   - Preprocessing & Missing Value Imputation                          |
|   - Exploratory Data Analysis (Pearson Correlation Matrix)            |
|   - Supervised Model Training (CatBoost, Huber, KNN, Linear Reg)      |
|   - Batch & Single Candidate Inference Engine                         |
+-----------------------------------------------------------------------+
```

---

## 2. Directory Structure & Key Files

### Backend Workspace (`B8. Kepler Telescope`)
```
B8. Kepler Telescope/
├── ml_engine.py           # Core machine learning engine class (KeplerMLEngine)
├── app.py                 # Production Flask REST API server (CORS enabled on port 5000)
├── requirements.txt       # Python dependency requirements
├── main.py                # Legacy Tkinter desktop GUI (refactored to share ml_engine.py)
├── Dataset/
│   └── cumulative.csv     # NASA Kepler historical dataset (9,564 records, 46 features)
└── models/                # Persisted model weight files (.pkl)
```

### Frontend Workspace (`keplerml`)
```
keplerml/
├── src/
│   ├── services/
│   │   └── api.ts         # TypeScript API client with automatic offline/mock fallback
│   ├── components/
│   │   ├── Header.tsx     # Sticky top navigation with real-time Flask connection badge
│   │   ├── ModelsSection.tsx     # Interactive model benchmarks & live Flask re-training trigger
│   │   ├── PredictionSection.tsx # Single candidate sliders & batch CSV test suite upload
│   │   ├── OverviewSection.tsx   # Dashboard summary cards & system telemetry
│   │   ├── DatasetSection.tsx    # Interactive catalog table for Kepler records
│   │   ├── PreprocessingSection.tsx # Pipeline steps & cleaning strategies
│   │   └── EdaSection.tsx        # Statistical distribution plots & correlation matrices
│   └── App.tsx            # Main application root managing navigation & state
├── vite.config.ts         # Vite server config & /api proxy pointing to http://127.0.0.1:5000
└── package.json           # Frontend Node.js dependencies & scripts
```

---

## 3. Dual-Flow Operational Logic

The system strictly decouples **Supervised Model Training** from **Unseen Candidate Inference**.

```mermaid
flowchart TD
    subgraph Flow1 ["FLOW 1: TRAINING & VALIDATION DATASET FLOW"]
        A1[Historical NASA Kepler Catalog<br/>Dataset/cumulative.csv] --> B1[Data Preprocessing<br/>Imputation & Label Encoding]
        B1 --> C1[Exploratory Data Analysis<br/>GET /api/eda]
        B1 --> D1{80% / 20% Split}
        D1 -- "80% Train Set" --> E1[Supervised Regressor Training<br/>CatBoost / Huber / KNN / Linear]
        D1 -- "20% Holdout Test Set" --> F1[Validation Evaluation<br/>R² Score: 0.9421 | RMSE: 0.14]
        E1 --> G1[Persist Weights to Disk<br/>.pkl files]
        F1 --> G1
    end

    subgraph Flow2 ["FLOW 2: TEST & INFERENCE FLOW"]
        A2[Manual Candidate Sliders<br/>Period, Depth, Radius, Temp] --> C2[Align to Model Feature Schema]
        B2[Batch Test Suite CSV Upload<br/>Unlabeled Exoplanet Candidates] --> C2
        G1 ==>|Active CatBoost Weights| D2[Inference Execution<br/>POST /api/predict]
        C2 --> D2
        D2 --> E2[Predicted Confidence Score 0.00 - 1.00<br/>+ Classification Status Badge]
    end
```

### Flow 1: Training & Validation Dataset Flow (`/api/models/train`)
1. **Data Source**: Operates on labeled historical data (`cumulative.csv`, 9,564 rows) containing ground-truth `koi_disposition` (`CONFIRMED`, `FALSE POSITIVE`, `CANDIDATE`).
2. **Preprocessing & EDA Linkage**:
   - Missing numerical values are imputed using median values.
   - Categorical targets are encoded via `LabelEncoder`.
   - The **EDA Section** graphs (`/api/eda`) directly visualize these preprocessed columns, revealing structural correlations between transit depth (`koi_depth`), planetary radius (`koi_prad`), and confirmed habitability scores.
3. **Model Training & Verification**:
   - The data is partitioned into an **80% Training Set** and a **20% Holdout Test Set**.
   - Clicking **Re-train Model** in the UI executes `POST /api/models/train`, fitting the algorithm and returning live $R^2$, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) on the holdout validation set.

### Flow 2: Test & Inference Flow (`/api/predict` & `/api/predict/upload`)
1. **Data Source**: Operates on **unseen or experimental candidate signals** without ground-truth labels.
2. **Execution UI (Inference & Prediction Tab)**:
   - **Single Candidate Prediction**: Adjusting orbital sliders or choosing presets (**Earth-Like Candidate**, **Hot Jupiter**, **Grazing Binary FP**) queries the active Flask CatBoost model.
   - **Batch Test Suite CSV Upload**: Dragging and dropping a `.csv` file (or clicking **Load Sample Test Suite**) processes multiple candidates simultaneously.
3. **Result Presentation**:
   - **Output Gauge Card**: Displays numerical confidence (e.g., `0.9242` or `92.4%`) and assigns a badge:
     - 🟢 **HABITABLE CANDIDATE** (Score $\ge 0.75$)
     - 🟡 **AMBIGUOUS TRANSIT** ($0.40 \le \text{Score} < 0.75$)
     - 🔴 **FALSE POSITIVE / BINARY** ($\text{Score} < 0.40$)
   - **Batch Results Table**: Displays row-by-row telemetry, predicted scores, classification tags, and an **Export CSV Report** download button.

---

## 4. API Endpoint Reference Table

| HTTP Method | Endpoint | Description | Sample Payload / Output |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service health and dataset loading check | `{ "status": "healthy", "rows": 9564 }` |
| `GET` | `/api/overview` | Dataset metrics and best model summary | `{ "bestModelName": "CatBoost Regressor", "bestModelR2": 0.9421 }` |
| `GET` | `/api/dataset` | Paginated & searchable Kepler records | `?page=1&limit=50&search=K00888.01` |
| `GET` | `/api/eda` | Preprocessed feature correlation matrix | Returns Pearson correlation values & histograms |
| `GET` | `/api/models` | Performance metrics for trained models | Returns $R^2$, MAE, MSE, RMSE for all models |
| `POST` | `/api/models/train` | Triggers live model re-training | Payload: `{ "modelName": "CatBoost Regressor" }` |
| `POST` | `/api/predict` | Single or array candidate inference | Payload: `{ "candidates": [{ "koi_period": 145.2, ... }] }` |
| `POST` | `/api/predict/upload` | Batch CSV test suite upload | Accepts CSV file upload and returns predicted rows |

---

## 5. How to Launch Both Services

### Step 1: Start the Python Flask Backend
Open a terminal in the backend directory (`B8. Kepler Telescope`):
```powershell
pip install -r requirements.txt
python app.py
```
*The API server will listen on **http://127.0.0.1:5000**.*

### Step 2: Start the React Frontend Web Application
Open a second terminal in the frontend directory (`keplerml`):
```powershell
npm install
npm run dev
```
*Open **http://localhost:3001** in your browser. The top bar will display a glowing green **FLASK API CONNECTED** badge.*

---

## 6. Architectural Resilience (Offline Fallback Mode)
In `src/services/api.ts`, all API calls are wrapped in an offline fallback handler. If the Python Flask backend is stopped or unreachable:
- The UI transitions gracefully to **Offline / Mock Mode** (indicated by an amber badge in the Header).
- Local heuristic models and cached Kepler dataset slices take over immediately so developers can continue inspecting UI components and interactive charts without interruption.
