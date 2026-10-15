"""
Tests for APFD, APFDc, and Evaluation Metrics.
"""

import pytest
import numpy as np
from src.evaluation import compute_apfd, compute_apfdc, compute_classification_metrics


def test_apfd_perfect_ordering():
    # If all failing tests are at the beginning:
    # 5 tests, 2 failures, ranks at 1, 2
    # APFD = 1 - (1+2)/(5*2) + 1/(2*5) = 1 - 3/10 + 1/10 = 0.8
    verdicts = [1, 1, 0, 0, 0]
    apfd = compute_apfd(verdicts)
    assert apfd == 0.8


def test_apfd_worst_ordering():
    # 5 tests, 2 failures, ranks at 4, 5
    # APFD = 1 - (4+5)/(5*2) + 1/(2*5) = 1 - 9/10 + 1/10 = 0.2
    verdicts = [0, 0, 0, 1, 1]
    apfd = compute_apfd(verdicts)
    assert apfd == 0.2


def test_apfd_no_failures():
    verdicts = [0, 0, 0]
    assert compute_apfd(verdicts) == 1.0


def test_apfdc_cost_cognizant():
    verdicts = [1, 0, 1]
    durations = [1.0, 10.0, 2.0]
    apfdc = compute_apfdc(verdicts, durations)
    assert 0.0 <= apfdc <= 1.0


def test_classification_metrics():
    y_true = np.array([0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    m = compute_classification_metrics(y_true, y_prob)
    assert "precision" in m
    assert "recall" in m
    assert "f1" in m
    assert "pr_auc" in m
    assert "roc_auc" in m
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
