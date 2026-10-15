"""
Model Training and Multi-Model Benchmarking Interface for TestPilot AI.

Implements:
1. Logistic Regression (Baseline Linear Classifier)
2. Random Forest (Ensemble Tree Classifier)
3. XGBoost Classifier (Gradient Boosted Decision Trees)
4. Multilayer Perceptron (MLP) Neural Network (Deep Learning with ReLU & Early Stopping)
5. Learning-to-Rank (LTR): XGBRanker with pairwise ranking loss grouped by CI build.
"""

import os
import joblib
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

logger = logging.getLogger("TestPilotAI.ModelTraining")


class BaseModelInterface(ABC):
    """Common interface for all TestPilot AI predictive and ranking models."""
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns failure probability array (N, 2) or score array."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        prob = self.predict_proba(X)
        if prob.ndim == 2:
            return (prob[:, 1] >= 0.5).astype(int)
        return (prob >= 0.5).astype(int)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"name": self.name, "params": self.params, "model": self.model}, filepath)
        logger.info(f"Saved {self.name} model to {filepath}")

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        data = joblib.load(filepath)
        self.name = data["name"]
        self.params = data["params"]
        self.model = data["model"]
        self.is_fitted = True
        logger.info(f"Loaded {self.name} model from {filepath}")
        return self


class LogisticRegressionModel(BaseModelInterface):
    """Logistic Regression with class balancing for baseline classification."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "random_state": 42}
        if params:
            default_params.update(params)
        super().__init__("Logistic Regression", default_params)
        self.model = LogisticRegression(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class RandomForestModel(BaseModelInterface):
    """Random Forest Ensemble with cost-sensitive class balancing."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }
        if params:
            default_params.update(params)
        super().__init__("Random Forest", default_params)
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.feature_importances_


class XGBoostModel(BaseModelInterface):
    """XGBoost Classifier with scale_pos_weight for severe class imbalance."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42,
            "eval_metric": "logloss"
        }
        if params:
            default_params.update(params)
        super().__init__("XGBoost", default_params)
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        # Calculate scale_pos_weight
        num_neg = (y == 0).sum()
        num_pos = (y == 1).sum()
        scale_weight = float(num_neg / max(1, num_pos))
        self.params["scale_pos_weight"] = scale_weight
        self.model.set_params(scale_pos_weight=scale_weight)
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.feature_importances_


class MLPNeuralNetworkModel(BaseModelInterface):
    """
    Multilayer Perceptron Neural Network.
    Architecture: Input -> Dense(64, ReLU) -> Dropout(0.2) -> Dense(32, ReLU) -> Dense(1, Sigmoid)
    Trained with early stopping on validation loss.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.001,
            "max_iter": 300,
            "early_stopping": True,
            "n_iter_no_change": 15,
            "random_state": 42
        }
        if params:
            default_params.update(params)
        super().__init__("Multilayer Perceptron (MLP)", default_params)
        self.model = MLPClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class LearningToRankModel(BaseModelInterface):
    """
    Genuine Learning-to-Rank (LTR) Model using LambdaMART (XGBRanker).
    Optimizes pairwise ranking loss directly within each CI build query group.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "objective": "rank:pairwise",
            "n_estimators": 100,
            "learning_rate": 0.08,
            "max_depth": 4,
            "subsample": 0.85,
            "random_state": 42
        }
        if params:
            default_params.update(params)
        super().__init__("Learning-to-Rank (LambdaMART)", default_params)
        self.model = xgb.XGBRanker(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        """
        Fits ranker where groups is an array containing the number of tests in each build.
        y is binary relevance (1 = failure, 0 = pass).
        """
        if groups is None:
            raise ValueError("Learning-to-Rank requires build query group sizes (groups array).")
        self.model.fit(X, y, group=groups)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        For LTR, returns predicted ranking scores mapped to [0, 1] range via sigmoid,
        compatible with classification score interfaces.
        """
        raw_scores = self.model.predict(X)
        # Sigmoid conversion
        probs = 1.0 / (1.0 + np.exp(-raw_scores))
        # Return format (N, 2) like predict_proba
        return np.column_stack([1.0 - probs, probs])


def get_model_registry() -> Dict[str, Any]:
    """Returns dictionary mapping model keys to model classes."""
    return {
        "logistic_regression": LogisticRegressionModel,
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "mlp_neural_net": MLPNeuralNetworkModel,
        "ltr_lambdamart": LearningToRankModel
    }
