"""
Experiment Tracking and Hyperparameter Optimization Module for TestPilot AI.

Integrates:
1. Optuna: Automated Bayesian hyperparameter optimization evaluated strictly on validation set.
2. MLflow: Experiment registry tracking parameters, metrics, artifacts, and runs.
3. Fallback JSON metadata persistence in models/best_config.json.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import optuna

# Suppress Optuna verbose logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("TestPilotAI.ExperimentTracking")
BEST_CONFIG_PATH = os.path.join("models", "best_config.json")


def optimize_xgboost_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Tunes XGBoost hyperparameters using Optuna strictly on validation split.
    Optimizes Average Precision (PR-AUC) to address severe class imbalance.
    """
    from sklearn.metrics import average_precision_score
    import xgboost as xgb

    logger.info(f"Starting Optuna hyperparameter optimization ({n_trials} trials)...")

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = float(num_neg / max(1, num_pos))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 40, 150, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "random_state": seed,
            "eval_metric": "logloss"
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, y_val_prob)
        return float(score)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_score = float(study.best_value)

    logger.info(f"Optuna completed. Best validation PR-AUC: {best_score:.4f} with params: {best_params}")

    config_record = {
        "model_name": "XGBoost",
        "primary_metric": "validation_pr_auc",
        "best_validation_score": round(best_score, 4),
        "hyperparameters": best_params,
        "n_trials": n_trials,
        "random_seed": seed,
        "optimization_timestamp": datetime.now().isoformat(),
        "evaluation_split_rule": "Fitted on train_split, evaluated strictly on validation_split"
    }

    os.makedirs(os.path.dirname(BEST_CONFIG_PATH), exist_ok=True)
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    return config_record


def log_experiment_to_mlflow(
    experiment_name: str,
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Optional[Dict[str, str]] = None
) -> bool:
    """
    Logs model run, parameters, metrics, and artifact paths to MLflow.
    Gracefully handles environment restrictions without raising exceptions.
    """
    try:
        import mlflow
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            if artifacts:
                for art_name, art_path in artifacts.items():
                    if os.path.exists(art_path):
                        mlflow.log_artifact(art_path)
        logger.info(f"Successfully logged experiment '{experiment_name}' run '{run_name}' to MLflow.")
        return True
    except Exception as e:
        logger.warning(f"MLflow tracking skipped or unavailable: {e}. Experiment metadata saved locally to JSON.")
        return False
