"""
Leakage-Safe Preprocessing Pipeline for TestPilot AI.

Fits scalers, imputers, and transformers strictly on training data splits,
persisting artifacts to models/preprocessing_pipeline.joblib for reproducible inference.
"""

import os
import joblib
import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.data_loader import ALL_PRE_EXECUTION_FEATURES

logger = logging.getLogger("TestPilotAI.Preprocessing")
PIPELINE_PATH = os.path.join("models", "preprocessing_pipeline.joblib")


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selects valid pre-execution feature columns and enforces float32 typing."""
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            # Fill missing expected features with 0.0 if not present
            data = {}
            for col in self.feature_names:
                if col in X.columns:
                    data[col] = X[col].values
                else:
                    data[col] = np.zeros(len(X), dtype=np.float32)
            return pd.DataFrame(data, columns=self.feature_names)
        return X


class TestPilotPreprocessor:
    """
    Manages end-to-end preprocessing pipeline.
    Produces scaled features for linear/neural models and unscaled for tree ensembles.
    """
    __test__ = False

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or ALL_PRE_EXECUTION_FEATURES
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        """Fits imputer and scaler ONLY on training data split."""
        selector = FeatureSelector(self.feature_names)
        X_selected = selector.transform(X)
        imputed = self.imputer.fit_transform(X_selected)
        self.scaler.fit(imputed)
        self.is_fitted = True
        logger.info(f"Fitted preprocessor on {len(X)} training samples across {len(self.feature_names)} features.")
        return self

    def transform(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """Transforms input feature dataframe into sanitized numpy array."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        selector = FeatureSelector(self.feature_names)
        X_selected = selector.transform(X)
        imputed = self.imputer.transform(X_selected)
        if scale:
            return self.scaler.transform(imputed)
        return imputed

    def fit_transform(self, X: pd.DataFrame, y=None, scale: bool = True) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X, scale=scale)

    def save(self, filepath: str = PIPELINE_PATH):
        """Saves fitted preprocessor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved fitted preprocessor to {filepath}")

    @classmethod
    def load(cls, filepath: str = PIPELINE_PATH) -> "TestPilotPreprocessor":
        """Loads fitted preprocessor from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessing pipeline file not found at {filepath}")
        preprocessor = joblib.load(filepath)
        logger.info(f"Loaded fitted preprocessor from {filepath}")
        return preprocessor
