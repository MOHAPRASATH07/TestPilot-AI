"""
Production FastAPI Backend for TestPilot AI.

Endpoints:
- GET  /health: Service status, preprocessor state, and loaded models.
- GET  /model-info: Feature schema, model hyperparameters, and training metadata.
- POST /prioritize: Budget-aware test case prioritization with feature evidence explanations.
- POST /evaluate: Evaluates prioritization efficacy on a user-submitted test suite.
- GET  /evaluation-results: Pre-computed held-out benchmark metrics across all strategies.
"""

import os
import sys
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import ALL_PRE_EXECUTION_FEATURES
from src.preprocessing import TestPilotPreprocessor
from src.ranking import rank_tests, STRATEGY_NAMES
from src.evaluation import compute_apfd, compute_apfdc
from src.model_training import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    MLPNeuralNetworkModel,
    LearningToRankModel
)

logger = logging.getLogger("TestPilotAI.API")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELS_DIR = "models"
REPORTS_DIR = "reports"

# Global model cache
LOADED_PREPROCESSOR: Optional[TestPilotPreprocessor] = None
LOADED_MODELS: Dict[str, Any] = {}
CHAMPION_INFO: Dict[str, Any] = {}


def load_all_resources():
    """Initializes and caches preprocessor and models."""
    global LOADED_PREPROCESSOR, LOADED_MODELS, CHAMPION_INFO
    
    prep_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
    if LOADED_PREPROCESSOR is None and os.path.exists(prep_path):
        LOADED_PREPROCESSOR = TestPilotPreprocessor.load(prep_path)
        
    champ_path = os.path.join(MODELS_DIR, "champion_model.json")
    if not CHAMPION_INFO and os.path.exists(champ_path):
        with open(champ_path, "r", encoding="utf-8") as f:
            CHAMPION_INFO = json.load(f)

    model_constructors = {
        "logistic_regression": LogisticRegressionModel,
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "mlp_neural_net": MLPNeuralNetworkModel,
        "ltr_lambdamart": LearningToRankModel
    }

    for key, ctor in model_constructors.items():
        if key not in LOADED_MODELS:
            m_path = os.path.join(MODELS_DIR, f"{key}.joblib")
            if os.path.exists(m_path):
                LOADED_MODELS[key] = ctor().load(m_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for modern FastAPI startup/shutdown."""
    load_all_resources()
    logger.info(f"TestPilot AI API ready. Loaded {len(LOADED_MODELS)} models.")
    yield


app = FastAPI(
    title="TestPilot AI - Prioritization Service",
    description="Budget-Aware Machine Learning for Risk-Driven Software Test Prioritization",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic Schemas
# ============================================================

class TestCaseInput(BaseModel):
    test_id: str = Field(..., json_schema_extra={"example": "org.apache.commons.lang.StringUtilsTest"})
    avg_recent_duration: float = Field(..., gt=0.0, description="Estimated duration in seconds", json_schema_extra={"example": 4.5})
    historical_fail_rate: Optional[float] = Field(0.0, ge=0.0, le=1.0, json_schema_extra={"example": 0.12})
    failures_last_5: Optional[int] = Field(0, ge=0, json_schema_extra={"example": 1})
    failures_last_10: Optional[int] = Field(0, ge=0, json_schema_extra={"example": 2})
    total_historical_runs: Optional[int] = Field(1, ge=0, json_schema_extra={"example": 35})
    builds_since_last_fail: Optional[int] = Field(999, ge=0, json_schema_extra={"example": 3})
    test_age_builds: Optional[int] = Field(1, ge=0, json_schema_extra={"example": 25})
    test_loc: Optional[int] = Field(50, ge=1, json_schema_extra={"example": 85})
    changed_files_count: Optional[int] = Field(1, ge=0, json_schema_extra={"example": 4})
    code_churn_additions: Optional[int] = Field(0, ge=0, json_schema_extra={"example": 120})
    code_churn_deletions: Optional[int] = Field(0, ge=0, json_schema_extra={"example": 30})
    modified_test_file: Optional[int] = Field(0, ge=0, le=1, json_schema_extra={"example": 1})
    file_test_distance: Optional[float] = Field(0.5, ge=0.0, le=1.0, json_schema_extra={"example": 0.15})
    cov_class_count: Optional[int] = Field(5, ge=0, json_schema_extra={"example": 8})
    cov_method_count: Optional[int] = Field(15, ge=0, json_schema_extra={"example": 24})
    w_sum_cov_c_faults: Optional[float] = Field(0.0, ge=0.0, json_schema_extra={"example": 1.2})
    verdict: Optional[int] = Field(None, ge=0, le=1, description="Optional ground-truth verdict (0=Pass, 1=Fail) for retrospective evaluation")


class PrioritizationRequest(BaseModel):
    tests: List[TestCaseInput]
    time_budget_seconds: Optional[float] = Field(300.0, gt=0.0, description="Testing time budget in seconds (e.g. 300s = 5m)", json_schema_extra={"example": 300.0})
    strategy: Optional[str] = Field("ml_value_density", description="Prioritization strategy", json_schema_extra={"example": "ml_value_density"})
    model_name: Optional[str] = Field("xgboost", description="ML model key", json_schema_extra={"example": "xgboost"})
    alpha: Optional[float] = Field(1.0, ge=0.1, le=3.0, description="Duration sensitivity exponent for value density", json_schema_extra={"example": 1.0})


class RankedItem(BaseModel):
    rank: int
    test_id: str
    priority_score: float
    estimated_duration_seconds: float
    cumulative_duration_seconds: float
    within_budget: bool
    supporting_evidence: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Feature evidence explaining why the test received this priority score"
    )


class PrioritizationResponse(BaseModel):
    strategy: str
    strategy_label: str
    explanation: str
    model_used: Optional[str]
    time_budget_seconds: float
    total_tests_received: int
    scheduled_test_count: int
    excluded_test_count: int
    total_scheduled_time_seconds: float
    remaining_budget_seconds: float
    failures_discovered: Optional[int] = None
    total_failures_in_suite: Optional[int] = None
    failure_recall: Optional[float] = None
    scheduled_tests: List[RankedItem]
    excluded_tests: List[RankedItem]


class EvaluationRequest(BaseModel):
    tests: List[TestCaseInput]
    strategy: Optional[str] = Field("ml_value_density", json_schema_extra={"example": "ml_value_density"})
    model_name: Optional[str] = Field("xgboost", json_schema_extra={"example": "xgboost"})
    budgets_seconds: Optional[List[float]] = Field(default=[300.0, 600.0, 1200.0])


class EvaluationResponse(BaseModel):
    strategy: str
    apfd: float
    apfdc: float
    total_tests: int
    total_failures: int
    budget_recall: Dict[str, float]


# ============================================================
# API Endpoints
# ============================================================

@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
def health_check():
    """Returns system status, preprocessor state, and loaded models."""
    load_all_resources()
    return {
        "status": "healthy",
        "preprocessor_loaded": LOADED_PREPROCESSOR is not None,
        "available_models": list(LOADED_MODELS.keys()),
        "champion_model": CHAMPION_INFO.get("champion_model_name", "None"),
        "supported_strategies": list(STRATEGY_NAMES.keys())
    }


@app.get("/model-info", status_code=status.HTTP_200_OK, tags=["Models"])
def get_model_info():
    """Returns training metadata, feature schema, and hyperparameter configuration."""
    load_all_resources()
    
    config_path = os.path.join(MODELS_DIR, "best_config.json")
    best_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            best_config = json.load(f)

    val_path = os.path.join(MODELS_DIR, "validation_metrics.json")
    val_metrics = {}
    if os.path.exists(val_path):
        with open(val_path, "r", encoding="utf-8") as f:
            val_metrics = json.load(f)

    return {
        "feature_schema": ALL_PRE_EXECUTION_FEATURES,
        "total_features": len(ALL_PRE_EXECUTION_FEATURES),
        "champion_model": CHAMPION_INFO,
        "optuna_best_config": best_config,
        "validation_metrics": val_metrics
    }


@app.post("/prioritize", response_model=PrioritizationResponse, status_code=status.HTTP_200_OK, tags=["Prioritization"])
def prioritize_tests(req: PrioritizationRequest):
    """
    Ranks tests according to specified strategy and testing time budget.
    Applies fitted preprocessor and model predictions deterministically.
    Explains the ranking using supporting feature evidence.
    """
    load_all_resources()
    
    if not req.tests:
        raise HTTPException(status_code=400, detail="Test list cannot be empty.")

    if req.strategy not in STRATEGY_NAMES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid strategy '{req.strategy}'. Supported: {list(STRATEGY_NAMES.keys())}"
        )

    # Convert request test items to DataFrame
    test_dicts = [t.model_dump() for t in req.tests]
    test_df = pd.DataFrame(test_dicts)

    probs = None
    ltr_scores = None
    model_used = None

    if req.strategy.startswith("ml_"):
        if LOADED_PREPROCESSOR is None:
            raise HTTPException(status_code=503, detail="Preprocessing pipeline not loaded on server.")
        
        m_key = req.model_name
        if req.strategy == "ml_ltr":
            m_key = "ltr_lambdamart"
            
        if m_key not in LOADED_MODELS:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{m_key}' not trained or loaded. Available: {list(LOADED_MODELS.keys())}"
            )
            
        model = LOADED_MODELS[m_key]
        model_used = model.name
        
        X_scaled = LOADED_PREPROCESSOR.transform(test_df, scale=True)
        if req.strategy == "ml_ltr":
            ltr_scores = model.predict_proba(X_scaled)
        else:
            probs = model.predict_proba(X_scaled)

    # Execute ranking using shared ranking module
    res = rank_tests(
        test_df=test_df,
        strategy=req.strategy,
        predicted_probs=probs,
        ltr_scores=ltr_scores,
        time_budget_seconds=req.time_budget_seconds,
        alpha=req.alpha,
        seed=42
    )

    def extract_evidence(row: pd.Series) -> Dict[str, Any]:
        """Constructs an explanatory feature evidence dictionary for each test."""
        return {
            "historical_fail_rate": float(row.get("historical_fail_rate", 0.0)),
            "failures_last_5": int(row.get("failures_last_5", 0)),
            "modified_test_file": bool(row.get("modified_test_file", 0)),
            "file_test_distance": float(row.get("file_test_distance", 0.5)),
            "changed_files_count": int(row.get("changed_files_count", 0)),
            "estimated_duration": float(row.get("est_duration", 1.0))
        }

    def df_to_ranked_items(sub_df: pd.DataFrame) -> List[RankedItem]:
        items = []
        for _, row in sub_df.iterrows():
            items.append(RankedItem(
                rank=int(row["rank"]),
                test_id=str(row["test_id"]),
                priority_score=round(float(row["priority_score"]), 5),
                estimated_duration_seconds=round(float(row["est_duration"]), 2),
                cumulative_duration_seconds=round(float(row["cum_duration"]), 2),
                within_budget=bool(row["within_budget"]),
                supporting_evidence=extract_evidence(row)
            ))
        return items

    scheduled_items = df_to_ranked_items(res["scheduled_tests"])
    excluded_items = df_to_ranked_items(res["excluded_tests"])

    return PrioritizationResponse(
        strategy=res["strategy"],
        strategy_label=res["strategy_label"],
        explanation=res["explanation"],
        model_used=model_used,
        time_budget_seconds=req.time_budget_seconds,
        total_tests_received=res["total_tests_available"],
        scheduled_test_count=res["scheduled_test_count"],
        excluded_test_count=res["excluded_test_count"],
        total_scheduled_time_seconds=res["total_scheduled_time_seconds"],
        remaining_budget_seconds=res["remaining_budget_seconds"],
        failures_discovered=res.get("failures_discovered"),
        total_failures_in_suite=res.get("total_failures_in_build"),
        failure_recall=res.get("failure_recall"),
        scheduled_tests=scheduled_items,
        excluded_tests=excluded_items
    )


@app.post("/evaluate", response_model=EvaluationResponse, status_code=status.HTTP_200_OK, tags=["Evaluation"])
def evaluate_prioritization(req: EvaluationRequest):
    """
    Retrospectively evaluates a prioritization strategy on submitted test cases with ground-truth verdicts.
    Calculates APFD, cost-cognizant APFD (APFDc), and Defect Recall @ Budgets.
    """
    load_all_resources()
    
    if not req.tests:
        raise HTTPException(status_code=400, detail="Test list cannot be empty.")

    test_dicts = [t.model_dump() for t in req.tests]
    test_df = pd.DataFrame(test_dicts)

    if "verdict" not in test_df.columns or test_df["verdict"].isnull().all():
        raise HTTPException(
            status_code=400, 
            detail="Evaluation requires 'verdict' (0=Pass, 1=Fail) in the test case records."
        )

    probs = None
    ltr_scores = None
    if req.strategy.startswith("ml_"):
        if LOADED_PREPROCESSOR is None:
            raise HTTPException(status_code=503, detail="Preprocessing pipeline not loaded.")
        m_key = req.model_name
        if req.strategy == "ml_ltr":
            m_key = "ltr_lambdamart"
        if m_key not in LOADED_MODELS:
            raise HTTPException(status_code=404, detail=f"Model '{m_key}' not loaded.")
        model = LOADED_MODELS[m_key]
        X_s = LOADED_PREPROCESSOR.transform(test_df, scale=True)
        if req.strategy == "ml_ltr":
            ltr_scores = model.predict_proba(X_s)
        else:
            probs = model.predict_proba(X_s)

    res = rank_tests(
        test_df=test_df,
        strategy=req.strategy,
        predicted_probs=probs,
        ltr_scores=ltr_scores,
        time_budget_seconds=None
    )

    ranked_df = res["ranked_tests"]
    verdicts = ranked_df["verdict"].fillna(0).astype(int).tolist()
    durations = ranked_df["est_duration"].tolist()

    apfd = compute_apfd(verdicts)
    apfdc = compute_apfdc(verdicts, durations)
    total_fails = sum(verdicts)

    budget_recalls = {}
    for b in req.budgets_seconds:
        sub = ranked_df[ranked_df["cum_duration"] <= b]
        f_found = int(sub["verdict"].sum())
        rec = float(f_found / max(1, total_fails)) if total_fails > 0 else 1.0
        budget_recalls[f"recall_at_{int(b)}s"] = round(rec, 4)

    return EvaluationResponse(
        strategy=req.strategy,
        apfd=apfd,
        apfdc=apfdc,
        total_tests=len(ranked_df),
        total_failures=total_fails,
        budget_recall=budget_recalls
    )


@app.get("/evaluation-results", status_code=status.HTTP_200_OK, tags=["Evaluation"])
def get_evaluation_results():
    """Returns actual held-out empirical evaluation benchmark results."""
    summary_path = os.path.join(REPORTS_DIR, "evaluation_summary.json")
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Evaluation results not yet generated. Run scripts/evaluate_model.py")
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
