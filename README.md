# SCSE3040 MLOps Mini Project --- Group G075

## Project Details

- **Group ID:** G075
- **Problem ID:** MP153 (Warehouse Picking Time - Patna Fulfilment Centre)
- **Batch:** 64
- **Faculty:** Mr. Dhananjay Kumar

### Team Members

| Enrolment No. | Name | Role |
|---|---|---|
| **S24CSEU2638** | Kunal Sehrawat | Ops Lead |
| **S24CSEU2649** | Varnit Chourasiya | Model Lead |
| **S24CSEU2324** | Naman Kumar | Service Lead |
| **S24CSEU2355** | Varnit Agustya | Data Lead |

---

## Problem Statement

Patna Fulfilment Centre processes thousands of orders daily. Late picking leads to missed evening dispatch trucks. This project builds an end-to-end MLOps pipeline to predict `pick_seconds` for picking orders, prioritizing slow orders earlier in the shift.

- **Target Variable:** `pick_seconds` (regression, measured in MAE)
- **Evaluation Metric:** Mean Absolute Error (MAE)

---

## Repository Structure

```
├── .dvc/                           # DVC configuration
├── .github/workflows/ci.yml        # GitHub Actions CI workflow (pytest)
├── app/
│   └── main.py                     # FastAPI application (GET /health, POST /predict)
├── data/
│   └── raw/
│       ├── train.csv.dvc           # DVC pointer for training dataset
│       ├── later.csv.dvc           # DVC pointer for production arrival dataset
│       └── holdout_features.csv    # Test holdout features
├── k8s/
│   ├── deployment.yaml             # Kubernetes deployment (4 replicas)
│   └── service.yaml                # Kubernetes service (port 80 -> 8000)
├── models/
│   └── model.joblib                # Champion trained regression model
├── predictions/
│   └── holdout_predictions.csv     # Champion model predictions on holdout data
├── reports/
│   └── findings.yaml               # Project audit findings and metrics
├── src/
│   └── mp153/                      # Core package
│       ├── __init__.py
│       ├── data.py                 # Data cleaning and anomaly audits
│       ├── train.py                # MLflow tracked training routines
│       ├── evaluate.py             # Drift (PSI) and slice metrics
│       └── utils.py                # Helper utilities and SHA-256 verifier
├── tests/
│   ├── test_api.py                 # FastAPI endpoints & schema validation tests
│   └── test_data.py                # Data processing and model sanity tests
├── Dockerfile                      # Docker container specification (python:3.12-slim)
├── .dockerignore
├── .gitignore
├── findings.yaml                   # Root findings form
├── params.yaml                     # Pipeline parameters and hyperparameters
├── requirements.txt                # Pinned dependencies
└── README.md
```

---

## Setup & Execution

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Running Data Audit & Training

```bash
python -m src.mp153.train
```

### 3. Running Unit Tests

```bash
pytest tests/ -v
```

### 4. Running Prediction Service Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
