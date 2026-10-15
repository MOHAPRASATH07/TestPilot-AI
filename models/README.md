# TestPilot AI: Model Registry and Training Artifacts

## Overview
This directory stores fitted model artifacts, preprocessors, and hyperparameter optimization metadata for the TestPilot AI benchmark suite.

## Artifacts Inventory

| File | Description | Source / Architecture |
|---|---|---|
| `preprocessing_pipeline.joblib` | Fitted `TestPilotPreprocessor` | Fitted strictly on `data/processed/train.csv` |
| `logistic_regression.joblib` | Baseline Linear Model | L2-penalized Logistic Regression with class balancing |
| `random_forest.joblib` | Ensemble Tree Model | 100 Estimators, max_depth=10, balanced class weights |
| `xgboost.joblib` | Gradient Boosted Decision Tree | Champion Model, Optuna-tuned hyperparameters with `scale_pos_weight` |
| `mlp_neural_net.joblib` | Multilayer Perceptron Neural Network | Dense(64, ReLU) -> Dense(32, ReLU) -> Sigmoid with early stopping |
| `ltr_lambdamart.joblib` | Genuine Learning-to-Rank | `XGBRanker(objective='rank:pairwise')` grouped by build query IDs |
| `best_config.json` | Optuna Hyperparameter Optimization Log | 12 Bayesian TPE trials evaluated on validation split |
| `validation_metrics.json` | Cross-Model Validation Metrics | PR-AUC, F1, Recall, ROC-AUC, Brier score on validation split |
| `champion_model.json` | Champion Model Selection Record | Declared champion model selected strictly on validation split PR-AUC |

## Model Selection Protocol
The champion model is selected strictly on the **Validation Split** using Average Precision (**PR-AUC**), avoiding data snooping on the held-out test split.
Held-out test evaluations are executed only after model selection and hyperparameter tuning are finalized.
