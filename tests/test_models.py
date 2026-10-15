"""
Tests for ML and Deep Learning Models.
"""

import os
import pytest
import numpy as np
from src.model_training import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    MLPNeuralNetworkModel,
    LearningToRankModel
)


@pytest.fixture
def dummy_train_data():
    X = np.random.randn(80, 10)
    y = np.random.choice([0, 1], size=80, p=[0.85, 0.15])
    groups = np.array([20, 20, 20, 20])
    return X, y, groups


def test_logistic_regression(dummy_train_data):
    X, y, _ = dummy_train_data
    model = LogisticRegressionModel()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (80, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_random_forest(dummy_train_data):
    X, y, _ = dummy_train_data
    model = RandomForestModel()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (80, 2)
    assert len(model.get_feature_importances()) == 10


def test_xgboost(dummy_train_data):
    X, y, _ = dummy_train_data
    model = XGBoostModel()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (80, 2)
    assert len(model.get_feature_importances()) == 10


def test_mlp_neural_network(dummy_train_data):
    X, y, _ = dummy_train_data
    model = MLPNeuralNetworkModel()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (80, 2)


def test_learning_to_rank(dummy_train_data):
    X, y, groups = dummy_train_data
    model = LearningToRankModel()
    model.fit(X, y, groups=groups)
    probs = model.predict_proba(X)
    assert probs.shape == (80, 2)
