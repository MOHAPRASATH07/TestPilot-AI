"""
Tests for Prioritization and Budget Scheduler.
"""

import pytest
import numpy as np
import pandas as pd
from src.ranking import rank_tests


@pytest.fixture
def sample_test_df():
    return pd.DataFrame({
        "test_id": [f"Test_{i}" for i in range(1, 11)],
        "avg_recent_duration": [10.0, 50.0, 20.0, 100.0, 5.0, 15.0, 30.0, 45.0, 60.0, 25.0],
        "historical_fail_rate": [0.0, 0.5, 0.2, 0.8, 0.0, 0.1, 0.3, 0.4, 0.1, 0.6],
        "failures_last_5": [0, 2, 1, 3, 0, 0, 1, 2, 0, 2],
        "verdict": [0, 1, 0, 1, 0, 0, 1, 0, 0, 1]
    })


def test_ranking_determinism(sample_test_df):
    """Deterministic inputs must yield identical rankings."""
    res1 = rank_tests(sample_test_df, strategy="fastest_first")
    res2 = rank_tests(sample_test_df, strategy="fastest_first")
    
    assert list(res1["ranked_tests"]["test_id"]) == list(res2["ranked_tests"]["test_id"])
    assert list(res1["ranked_tests"]["priority_score"]) == list(res2["ranked_tests"]["priority_score"])


def test_fastest_first_orders_ascending_duration(sample_test_df):
    res = rank_tests(sample_test_df, strategy="fastest_first")
    durations = list(res["ranked_tests"]["est_duration"])
    assert durations == sorted(durations)


def test_budget_constraint_enforcement(sample_test_df):
    budget = 60.0  # 1 minute budget
    res = rank_tests(sample_test_df, strategy="fastest_first", time_budget_seconds=budget)
    
    scheduled = res["scheduled_tests"]
    excluded = res["excluded_tests"]
    
    assert len(scheduled) + len(excluded) == len(sample_test_df)
    assert res["total_scheduled_time_seconds"] <= budget
    assert scheduled["cum_duration"].max() <= budget
    # Disjoint test ids
    assert set(scheduled["test_id"]).isdisjoint(set(excluded["test_id"]))


def test_no_duplicate_tests_in_output(sample_test_df):
    res = rank_tests(sample_test_df, strategy="historical")
    test_ids = list(res["ranked_tests"]["test_id"])
    assert len(test_ids) == len(set(test_ids))


def test_ml_value_density_scoring(sample_test_df):
    probs = np.array([0.1, 0.9, 0.2, 0.8, 0.05, 0.1, 0.3, 0.4, 0.2, 0.7])
    res = rank_tests(sample_test_df, strategy="ml_value_density", predicted_probs=probs, alpha=1.0)
    
    scores = res["ranked_tests"]["priority_score"].values
    # Ensure scores are strictly monotonically non-increasing
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1]
