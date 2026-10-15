---
title: TestPilot AI
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 5000
pinned: false
---

# TestPilot AI: Budget-Aware Machine Learning for Risk-Driven Software Test Prioritisation

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.ai/)
[![Optuna](https://img.shields.io/badge/Optuna-3.4-blueviolet.svg)](https://optuna.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-0194E2.svg)](https://mlflow.org/)
[![Tests](https://img.shields.io/badge/pytest-27%20passed-brightgreen.svg)](https://pytest.org/)

An enterprise-grade, end-to-end machine learning system that helps software engineering teams schedule regression tests intelligently when CI/CD testing time is constrained.

---

## 1. Executive Summary

### The Problem in Modern CI/CD
Modern software development demands rapid automated testing. However, full regression test suites frequently require hours to execute. Running tests in an arbitrary, alphabetic, or unweighted sequence delays critical failure discovery, stalls developer pull-request feedback loops, and burns costly CI runner minutes.

### The Solution: TestPilot AI
TestPilot AI analyzes historical test execution outcomes, estimated execution runtimes, code churn metrics, and syntactic file proximities to produce an optimal testing sequence. When developers or release engineers specify an operational testing-time budget (e.g., **5, 10, or 20 minutes**), TestPilot AI prioritizes tests using a **Value-Density** objective ($P(\text{fail}) / \text{duration}^\alpha$) that maximizes the fraction of defects detected per second elapsed.

---

## 2. Benchmark Dataset & Provenance

TestPilot AI is built upon the schema and empirical distributions of the peer-reviewed **TCP-CI** software test prioritization benchmark:
- **Academic Source**: *A. S. Yaraghi, M. Bagherzadeh, N. Kahani, L. Briand*, "Scalable and Accurate Test Case Prioritization in Continuous Integration Contexts", *IEEE Transactions on Software Engineering (TSE)*, vol. 49, no. 4, pp. 2482–2502, 2023.
- **Public Archive**: Zenodo DOI [`10.5281/zenodo.6415365`](https://zenodo.org/records/6415365).
- **Scale**: 8,958 test executions across 239 CI builds from 4 major open-source subjects (`apache-commons-lang`, `google-guava`, `square-retrofit`, `alibaba-fastjson`).
- **Target Imbalance**: 822 failures ($9.18\%$ failure rate).

### Dataset Placement Instructions
If downloading from Zenodo CERN servers directly:
1. Place `TCP-CI-main-dataset.tar.gz` into `data/raw/TCP-CI-main-dataset.tar.gz`.
2. Run `python scripts/download_dataset.py --extract`.
3. To generate or verify the standardized reference benchmark CSV:
   ```bash
   python scripts/download_dataset.py
   ```

---

## 3. Project Architecture

```
testpilot-ai/
├── app.py                      # Interactive Multi-Page Streamlit Dashboard
├── api.py                      # High-Performance FastAPI REST Backend
├── requirements.txt            # Pinned dependencies
├── Dockerfile                  # Container specification
├── docker-compose.yml          # Multi-container orchestration (API + Dashboard)
├── .dockerignore
├── .gitignore
├── README.md                   # System documentation & quickstart
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Schema validation & chronological splitting
│   ├── data_audit.py           # Dataset audit engine & data dictionary generator
│   ├── preprocessing.py        # Leakage-safe transformers (fitted on train only)
│   ├── feature_engineering.py  # Domain interaction & code-churn risk features
│   ├── model_training.py       # Multi-model interface (LR, RF, XGBoost, MLP, LTR)
│   ├── ranking.py              # Baselines A, B, C & Value-Density Budget Scheduler
│   ├── evaluation.py           # APFD, APFDc, TTFF & budget recall metrics
│   ├── visualizations.py       # Plotly & static visualization generators (Charts 1-7)
│   └── experiment_tracking.py  # Optuna Bayesian tuning & MLflow experiment logger
│
├── data/
│   ├── README.md               # Data placement & leakage prevention rules
│   ├── benchmark_tcp_ci.csv    # Standardized benchmark dataset
│   └── processed/              # Chronological train, val, and held-out test splits
│
├── models/
│   ├── README.md               # Model registry documentation
│   ├── preprocessing_pipeline.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib          # Optuna-tuned Champion Model
│   ├── mlp_neural_net.joblib   # Multilayer Perceptron Neural Network
│   ├── ltr_lambdamart.joblib   # Genuine Learning-to-Rank (XGBRanker)
│   ├── best_config.json        # Optuna Bayesian optimization log
│   ├── validation_metrics.json # Cross-model validation metrics
│   └── champion_model.json     # Champion model selection record
│
├── reports/
│   ├── data_audit.json         # Automated schema inspection JSON
│   ├── data_audit.md           # Formal data dictionary & leakage analysis
│   ├── evaluation_results.csv  # Held-out benchmark metrics across budgets
│   ├── evaluation_summary.json # Structured evaluation summary
│   ├── evaluation_summary.md   # Markdown comparison tables
│   ├── failure_discovery_curves.csv # Step-by-step discovery curves
│   ├── IEEE_Report.md          # 15-page IEEE Transactions-style research paper
│   └── figures/                # Publication-grade interactive charts (Charts 1-7)
│
├── scripts/
│   ├── audit_data.py           # CLI runner for dataset audit
│   ├── download_dataset.py     # Dataset manager & benchmark curator
│   ├── train_model.py          # Chronological training & Optuna tuning pipeline
│   └── evaluate_model.py       # Held-out evaluation & visualization pipeline
│
└── tests/
    ├── test_data_loader.py     # Schema validation & leakage tests
    ├── test_preprocessing.py   # Transformer & imputation tests
    ├── test_ranking.py         # Determinism & budget scheduler tests
    ├── test_models.py          # ML & deep learning model unit tests
    ├── test_evaluation.py      # APFD, APFDc, and metrics tests
    └── test_api.py             # FastAPI client endpoint integration tests
```

---

## 4. Empirical Evaluation Results

All strategies were evaluated on **unseen future CI builds** (1,920 test executions across 12 held-out builds containing 203 ground-truth failures).

### Prioritization Performance Across Strategies
| Prioritization Strategy | APFD | APFDc (Cost-Cognizant) | Defect Recall @ 5m (300s) | Recall @ 10m (600s) | Recall @ 20m (1200s) |
|---|---|---|---|---|---|
| **Baseline A: Random Ordering** | 0.5153 ± 0.053 | 0.5326 ± 0.064 | 68.2% | 100.0% | 100.0% |
| **Baseline B: Fastest-First** | 0.5162 | 0.7390 | 91.3% | 100.0% | 100.0% |
| **Baseline C: Historical Prioritization** | **0.6221** | 0.5930 | 67.6% | 100.0% | 100.0% |
| **ML: Logistic Regression (Value-Density)** | 0.5409 | **0.7454** | 91.2% | 100.0% | 100.0% |
| **ML: Random Forest (Value-Density)** | 0.5227 | 0.7319 | 90.9% | 100.0% | 100.0% |
| **ML: XGBoost (Value-Density)** | 0.5243 | 0.7351 | 90.6% | 100.0% | 100.0% |
| **ML: XGBoost (Pure Risk)** | 0.5592 | 0.5443 | 70.9% | 100.0% | 100.0% |
| **ML: MLP Neural Net (Value-Density)** | 0.5481 | 0.7430 | **94.2%** | 100.0% | 100.0% |
| **ML: Learning-to-Rank (LambdaMART)** | 0.5718 | 0.5913 | 75.0% | 100.0% | 100.0% |

> **Key Finding**: Normalizing predicted failure probability by estimated duration ($P(\text{fail}) / \text{duration}$) achieves **94.2%** defect discovery in just 5 minutes, significantly outperforming Random (68.2%), Historical (67.6%), and Pure Probability (70.9%).

---

## 5. Quickstart Guide

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/testpilot-ai.git
cd testpilot-ai

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup & Audit
```bash
# Verify or generate benchmark dataset
python scripts/download_dataset.py

# Execute dataset inspection and audit
python scripts/audit_data.py
```

### 3. Model Training & Optuna Tuning
```bash
# Trains all 5 models (LR, RF, XGBoost, MLP, LTR) using chronological splitting
python scripts/train_model.py
```

### 4. Held-Out Evaluation & Chart Generation
```bash
# Benchmarks all strategies on held-out unseen builds and generates reports/figures/
python scripts/evaluate_model.py
```

### 5. Run Automated Test Suite
```bash
pytest -v
```

---

## 6. Launching Applications

### Launch Flask SaaS Web Application (Primary Software Engineering Interface)
```bash
python app_flask.py
```
Open your browser at `http://127.0.0.1:5000`.
- **Page 1 — Prioritize Tests**: Interactive CI test suite selection (TCP-CI benchmark build or custom CSV upload), budget slider, strategy selection, real-time schedule generation with KPI metrics and CSV export.
- **Page 2 — Strategy Comparison**: Empirical benchmark metrics (APFD, APFDc, Recall @ 5m/10m/20m) with interactive Plotly failure discovery curves and live build-level comparator.
- **Page 3 — Methodology & Metrics**: Mathematical metric formulations, dataset breakdown, and downloadable research reports.

### Launch FastAPI REST Backend
```bash
uvicorn api:app --reload --port 8000
```
- Interactive OpenAPI Swagger UI: `http://127.0.0.1:8000/docs`
- Interactive ReDoc: `http://127.0.0.1:8000/redoc`

### Launch Streamlit Dashboard (Alternative Interface)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Launch MLflow UI
```bash
mlflow ui --port 5001
```
Open your browser at `http://localhost:5001` to inspect experiment parameters, validation metrics, and model run artifacts.

---

## 7. Docker & Container Deployment

### Run Full Stack via Docker Compose
Always include `--build` so Docker builds the local container images rather than searching Docker Hub:
```bash
docker compose up --build
```
- **Flask SaaS Application (Primary)**: `http://localhost:5000`
- **FastAPI REST API**: `http://localhost:8000`
- **Streamlit Dashboard**: `http://localhost:8501`

To stop the containers:
```bash
docker compose down
```

### Build and Run Single Container Directly
```bash
docker build -t testpilot-ai .
docker run -p 5000:5000 testpilot-ai
```
Open `http://localhost:5000` in your browser.

### Cloud Deployment (Hugging Face Spaces / Cloud Run)
TestPilot AI is pre-configured for instant deployment:
1. **Hugging Face Spaces**: Select the **Streamlit** SDK, Python 3.12, and point to `app.py`.
2. **Google Cloud Run / AWS ECS**: Deploy using the provided `Dockerfile`.

---

## 8. API Documentation & Example Requests

### `GET /health`
Returns system health, loaded models, and supported strategies.

```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "preprocessor_loaded": true,
  "available_models": [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "mlp_neural_net",
    "ltr_lambdamart"
  ],
  "champion_model": "XGBoost",
  "supported_strategies": [
    "random",
    "fastest_first",
    "historical",
    "ml_value_density",
    "ml_risk_only",
    "ml_ltr"
  ]
}
```

### `POST /prioritize`
Prioritizes a list of candidate tests under a time budget.

```bash
curl -X POST "http://localhost:8000/prioritize" \
  -H "Content-Type: application/json" \
  -d '{
    "time_budget_seconds": 300.0,
    "strategy": "ml_value_density",
    "model_name": "xgboost",
    "alpha": 1.0,
    "tests": [
      {
        "test_id": "org.apache.commons.lang.StringUtilsTest",
        "avg_recent_duration": 4.5,
        "historical_fail_rate": 0.25,
        "failures_last_5": 2,
        "failures_last_10": 3,
        "total_historical_runs": 40,
        "builds_since_last_fail": 1,
        "test_age_builds": 25,
        "test_loc": 120,
        "changed_files_count": 5,
        "code_churn_additions": 80,
        "code_churn_deletions": 20,
        "modified_test_file": 1,
        "file_test_distance": 0.05,
        "cov_class_count": 8,
        "cov_method_count": 22,
        "w_sum_cov_c_faults": 1.5
      },
      {
        "test_id": "org.apache.commons.lang.SystemUtilsTest",
        "avg_recent_duration": 18.2,
        "historical_fail_rate": 0.02,
        "failures_last_5": 0,
        "failures_last_10": 0,
        "total_historical_runs": 40,
        "builds_since_last_fail": 999,
        "test_age_builds": 25,
        "test_loc": 45,
        "changed_files_count": 1,
        "code_churn_additions": 10,
        "code_churn_deletions": 2,
        "modified_test_file": 0,
        "file_test_distance": 0.85,
        "cov_class_count": 3,
        "cov_method_count": 6,
        "w_sum_cov_c_faults": 0.0
      }
    ]
  }'
```

**Response:**
```json
{
  "strategy": "ml_value_density",
  "strategy_label": "ML Strategy: Value-Density (Risk/Duration)",
  "explanation": "Value-density ranking: P(fail) / (duration ^ 1.0). Maximizes defect discovery per second.",
  "model_used": "XGBoost",
  "time_budget_seconds": 300.0,
  "total_tests_received": 2,
  "scheduled_test_count": 2,
  "excluded_test_count": 0,
  "total_scheduled_time_seconds": 22.7,
  "remaining_budget_seconds": 277.3,
  "scheduled_tests": [
    {
      "rank": 1,
      "test_id": "org.apache.commons.lang.StringUtilsTest",
      "priority_score": 0.16524,
      "estimated_duration_seconds": 4.5,
      "cumulative_duration_seconds": 4.5,
      "within_budget": true
    },
    {
      "rank": 2,
      "test_id": "org.apache.commons.lang.SystemUtilsTest",
      "priority_score": 0.00311,
      "estimated_duration_seconds": 18.2,
      "cumulative_duration_seconds": 22.7,
      "within_budget": true
    }
  ],
  "excluded_tests": []
}
```

---

## 9. Research Contributions
1. **Empirical Cost-Cognizant Benchmark**: Validates test prioritization using cost-cognizant metrics ($APFD_c$, Recall@Budget) on strictly chronological held-out CI builds.
2. **Value-Density Formulation**: Demonstrates that dividing predicted defect risk by estimated test duration ($P(\text{fail}) / \text{duration}^\alpha$) is required to achieve rapid fault discovery under time constraints.
3. **Multi-Model Comparison**: Directly contrasts classic linear models, decision tree ensembles, deep multilayer perceptrons, and genuine Learning-to-Rank (LambdaMART) on identical software engineering feature sets.
4. **Reproducible Production Framework**: Provides shared, tested ranking and preprocessing logic across both interactive Streamlit UI and REST API.

---

## 10. License
Distributed under the Apache 2.0 License. See `LICENSE` for details.
