![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688) ![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C) ![Docker](https://img.shields.io/badge/Docker-Compose-2496ED) ![License](https://img.shields.io/badge/license-MIT-green)

# TimeGuard — Anomaly Detection System

Undetected anomalies in time series data can lead to significant financial losses, missed opportunities, or system failures. TimeGuard is a production-ready anomaly detection system that automatically identifies unusual patterns in any time series dataset — before they become costly problems.

## Why Three Models?

Most anomaly detection systems rely on a single model. The problem is that every model has blind spots:

- **Isolation Forest** excels at global outliers but can miss density-based anomalies
- **DBSCAN** catches local density anomalies but struggles with sudden spikes
- **LSTM Autoencoder** detects temporal pattern breaks but may overlook isolated outliers

TimeGuard runs all three simultaneously and marks a data point as anomalous only when **at least 2 out of 3 models agree**. This ensemble approach reduces false positives while ensuring real anomalies are never missed by a single model's blind spot.

---

## Features

- ✅ Auto-detect datetime and numeric columns from any CSV
- ✅ Train three unsupervised models with a single click
- ✅ Ensemble majority voting (2-of-3 agreement)
- ✅ SHAP explainability for every detected anomaly
- ✅ MLflow experiment tracking with full metrics
- ✅ PostgreSQL storage for runs and results
- ✅ FastAPI REST API with async endpoints
- ✅ Streamlit dashboard with interactive charts
- ✅ Docker Compose for one-command deployment
- ✅ Persistent model storage via joblib

---

## Tech Stack

| Layer         | Technology                              |
|---------------|-----------------------------------------|
| Language      | Python 3.11                             |
| ML Frameworks | scikit-learn, PyTorch, SHAP             |
| API           | FastAPI, Uvicorn                        |
| Database      | PostgreSQL 15 + SQLAlchemy              |
| Experiment    | MLflow                                  |
| Frontend      | Streamlit + Plotly                      |
| Containers    | Docker Compose                          |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Streamlit Dashboard                          │
│                     http://localhost:8501                            │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ HTTP
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          FastAPI Server                             │
│                     http://localhost:8000                            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    AnomalyDetectionPipeline                    │  │
│  │                                                                │  │
│  │   ┌──────────────┐  ┌─────────────┐  ┌──────────────────────┐ │  │
│  │   │Preprocessing │→ │  3 Models   │→ │ Ensemble + SHAP      │ │  │
│  │   │  (pandas)    │  │  (parallel) │  │ (majority voting)    │ │  │
│  │   └──────────────┘  └─────────────┘  └──────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────┘  │
└────────────┬────────────────────────────────┬───────────────────────┘
             │                                │
             ▼                                ▼
┌─────────────────────┐          ┌─────────────────────────┐
│   PostgreSQL:5432   │          │     MLflow:5000         │
│  (runs + results)   │          │  (metrics + artifacts)  │
└─────────────────────┘          └─────────────────────────┘
```

---

## Project Structure

```
anomaly-detection/
├── api.py                        # FastAPI server with 5 endpoints
├── app.py                        # Streamlit dashboard
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── docker-compose.yml            # Multi-container orchestration
├── Dockerfile.api                # API container image
├── Dockerfile.mlflow             # MLflow container image
├── .dockerignore                 # Docker build exclusions
├── models/                       # Persisted model files (joblib/torch)
│   └── <run_id>/                 # Per-run model artifacts
│       ├── isolation_forest.joblib
│       ├── dbscan.joblib
│       ├── dbscan_nn.joblib
│       ├── lstm_autoencoder.pt
│       ├── lstm_threshold.joblib
│       ├── lstm_config.joblib
│       └── metadata.json
└── src/
    ├── config.py                 # pydantic-settings configuration
    ├── preprocessing.py          # CSV parsing, scaling, windowing
    ├── explainability.py         # SHAP-based anomaly explanations
    ├── db.py                     # SQLAlchemy models and CRUD
    ├── utils.py                  # Model save/load utilities
    ├── pipeline.py               # Orchestrates train → predict → store
    └── models/
        ├── __init__.py           # Model class exports
        ├── isolation_forest.py   # Isolation Forest wrapper
        ├── dbscan.py             # DBSCAN wrapper + scoring
        └── lstm_autoencoder.py   # LSTM encoder-decoder + wrapper
```

---

## Quickstart

**1. Clone and set up the environment**

```bash
git clone <repository-url>
cd anomaly-detection
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Configure environment variables**

```bash
cp .env.example .env
# Edit .env if needed — defaults work for local development
```

**3a. Run locally (manual)**

```bash
# Start PostgreSQL and MLflow separately, then:
uvicorn api:app --host 0.0.0.0 --port 8000
streamlit run app.py
```

**3b. Run with Docker Compose**

```bash
docker-compose up --build
# API at http://localhost:8000
# MLflow at http://localhost:5000
# Dashboard at http://localhost:8501 (run locally)
```

**4. Train a model**

Upload a CSV through the Streamlit dashboard or call the API:

```bash
curl -X POST http://localhost:8000/train -F "file=@data.csv"
```

---

## API Endpoints

| Method | Endpoint          | Input                    | Description                                      |
|--------|-------------------|--------------------------|--------------------------------------------------|
| POST   | `/train`          | multipart CSV            | Train all 3 models, store results, save models   |
| POST   | `/detect`         | `?run_id=` (optional)    | Get ensemble predictions from DB                 |
| POST   | `/detect/file`    | multipart CSV + `?run_id=` | Predict on a new file using saved models       |
| GET    | `/detect/explain` | `?run_id=` (optional)    | Get SHAP explanations for all anomalies          |
| GET    | `/model/info`     | `?run_id=` (optional)    | Get run metadata and model metrics               |

> All endpoints accept optional `run_id`. If omitted, the latest run is used.

---

## How It Works

**1. Preprocessing** — The uploaded CSV is parsed: datetime and numeric columns are auto-detected, missing values are interpolated, features are scaled with StandardScaler, and sliding windows are generated for LSTM.

**2. Model Training** — Three unsupervised models are trained independently:
- Isolation Forest fits on scaled feature vectors
- DBSCAN clusters the data and marks low-density points
- LSTM Autoencoder learns to reconstruct normal sequences

**3. Prediction** — Each model produces binary predictions. DBSCAN uses NearestNeighbors distance to core points as its anomaly score.

**4. Ensemble Voting** — A point is flagged anomalous if at least 2 of the 3 models agree (configurable via `ENSEMBLE_MIN_VOTES`).

**5. SHAP Explanation** — For each anomaly where Isolation Forest also voted, SHAP TreeExplainer computes feature contributions. Anomalies flagged only by the ensemble (not IF) receive a descriptive message.

**6. Storage** — Run metadata and per-row results (including SHAP explanations) are saved to PostgreSQL. MLflow logs all model parameters and metrics.

**7. Model Persistence** — Trained models are saved to disk with joblib (IF, DBSCAN) and PyTorch (LSTM), allowing fast inference on new data without retraining.
