"""
Model Training and Benchmark Optimization Script for TestPilot AI.

Executes:
1. Leakage-safe chronological splitting (Train / Val / Held-out Test)
2. Preprocessing pipeline fitting on Train split only
3. Optuna Bayesian hyperparameter optimization on Validation split
4. Multi-model training:
   - Logistic Regression
   - Random Forest
   - Tuned XGBoost
   - Multilayer Perceptron (MLP)
   - Learning-to-Rank (LambdaMART)
5. Model persistence and validation performance registration.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_dataset, chronological_split
from src.preprocessing import TestPilotPreprocessor
from src.feature_engineering import add_domain_features
from src.model_training import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    MLPNeuralNetworkModel,
    LearningToRankModel
)
from src.experiment_tracking import optimize_xgboost_hyperparameters, log_experiment_to_mlflow
from src.evaluation import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestPilotAI.TrainModel")

MODELS_DIR = "models"
REPORTS_DIR = "reports"


def main():
    print("=" * 70)
    print("TestPilot AI: Starting Leakage-Safe Multi-Model Training Pipeline")
    print("=" * 70)

    # 1. Load Data
    df = load_dataset()
    logger.info(f"Loaded dataset with {len(df)} records across {df['project_id'].nunique()} projects.")

    # 2. Chronological Split (Train 60%, Val 20%, Test 20%)
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.60, val_ratio=0.20)
    
    # Save splits for exact reproducibility and held-out evaluation
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    train_df.to_csv(os.path.join("data", "processed", "train.csv"), index=False)
    val_df.to_csv(os.path.join("data", "processed", "val.csv"), index=False)
    test_df.to_csv(os.path.join("data", "processed", "test.csv"), index=False)
    logger.info("Saved train.csv, val.csv, and test.csv under data/processed/")

    # 3. Fit Preprocessing Pipeline strictly on Train split
    preprocessor = TestPilotPreprocessor()
    preprocessor.fit(train_df)
    preprocessor.save(os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib"))

    # Transform splits
    X_train_scaled = preprocessor.transform(train_df, scale=True)
    y_train = train_df["verdict"].values

    X_val_scaled = preprocessor.transform(val_df, scale=True)
    y_val = val_df["verdict"].values

    # Query groups for Learning-to-Rank (number of tests per build in train and val)
    train_groups = train_df.groupby(["project_id", "build_id"], sort=False).size().values
    val_groups = val_df.groupby(["project_id", "build_id"], sort=False).size().values

    # 4. Optuna Hyperparameter Tuning for XGBoost on Validation set
    best_config = optimize_xgboost_hyperparameters(
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        n_trials=12,
        seed=42
    )

    # 5. Initialize and Train Models
    models = {
        "logistic_regression": LogisticRegressionModel(),
        "random_forest": RandomForestModel(),
        "xgboost": XGBoostModel(params=best_config["hyperparameters"]),
        "mlp_neural_net": MLPNeuralNetworkModel(),
        "ltr_lambdamart": LearningToRankModel()
    }

    validation_results = {}

    for key, model in models.items():
        logger.info(f"--- Training {model.name} ---")
        if key == "ltr_lambdamart":
            model.fit(X_train_scaled, y_train, groups=train_groups)
        else:
            model.fit(X_train_scaled, y_train)

        # Save model artifact
        model_path = os.path.join(MODELS_DIR, f"{key}.joblib")
        model.save(model_path)

        # Validation Metrics
        probs = model.predict_proba(X_val_scaled)
        prob_positive = probs[:, 1] if probs.ndim == 2 else probs
        metrics = compute_classification_metrics(y_val, prob_positive)
        validation_results[key] = {
            "model_name": model.name,
            "metrics": metrics
        }
        logger.info(f"  {model.name} Validation Metrics: PR-AUC={metrics['pr_auc']}, F1={metrics['f1']}, Recall={metrics['recall']}")

        # Track with MLflow
        log_experiment_to_mlflow(
            experiment_name="TestPilotAI_Benchmarking",
            run_name=f"train_{key}",
            params=model.params,
            metrics=metrics,
            artifacts={f"{key}_model": model_path}
        )

    # Save validation metrics summary
    val_summary_path = os.path.join(MODELS_DIR, "validation_metrics.json")
    with open(val_summary_path, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=2)

    # Select champion model strictly based on validation PR-AUC
    best_model_key = max(validation_results.keys(), key=lambda k: validation_results[k]["metrics"]["pr_auc"])
    champion_info = {
        "champion_model_key": best_model_key,
        "champion_model_name": validation_results[best_model_key]["model_name"],
        "selection_metric": "validation_pr_auc",
        "validation_pr_auc": validation_results[best_model_key]["metrics"]["pr_auc"],
        "selection_rule": "Selected strictly on validation split without inspecting held-out test data."
    }
    with open(os.path.join(MODELS_DIR, "champion_model.json"), "w", encoding="utf-8") as f:
        json.dump(champion_info, f, indent=2)

    print("\n" + "=" * 70)
    print("[SUCCESS] All 5 Models Trained and Persisted Successfully!")
    print(f"  Champion Model Selected: {champion_info['champion_model_name']} (Val PR-AUC: {champion_info['validation_pr_auc']})")
    print(f"  Model files saved to: {MODELS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
