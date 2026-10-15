"""
Tests for FastAPI Backend Endpoints, Validation, and Evaluation.
"""

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "available_models" in data
    assert "supported_strategies" in data
    assert len(data["available_models"]) == 5


def test_model_info_endpoint():
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "feature_schema" in data
    assert len(data["feature_schema"]) == 16


def test_prioritize_endpoint_valid_request():
    payload = {
        "tests": [
            {
                "test_id": "TestA",
                "avg_recent_duration": 12.0,
                "historical_fail_rate": 0.2,
                "failures_last_5": 1,
                "failures_last_10": 2,
                "total_historical_runs": 20,
                "builds_since_last_fail": 2,
                "test_age_builds": 15,
                "test_loc": 60,
                "changed_files_count": 3,
                "code_churn_additions": 40,
                "code_churn_deletions": 10,
                "modified_test_file": 1,
                "file_test_distance": 0.1,
                "cov_class_count": 5,
                "cov_method_count": 15,
                "w_sum_cov_c_faults": 0.8
            },
            {
                "test_id": "TestB",
                "avg_recent_duration": 2.5,
                "historical_fail_rate": 0.05,
                "failures_last_5": 0,
                "failures_last_10": 0,
                "total_historical_runs": 20,
                "builds_since_last_fail": 999,
                "test_age_builds": 15,
                "test_loc": 25,
                "changed_files_count": 3,
                "code_churn_additions": 40,
                "code_churn_deletions": 10,
                "modified_test_file": 0,
                "file_test_distance": 0.8,
                "cov_class_count": 2,
                "cov_method_count": 5,
                "w_sum_cov_c_faults": 0.0
            }
        ],
        "time_budget_seconds": 10.0,
        "strategy": "fastest_first"
    }
    resp = client.post("/prioritize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["strategy"] == "fastest_first"
    assert data["total_tests_received"] == 2
    assert len(data["scheduled_tests"]) >= 1
    assert data["scheduled_tests"][0]["test_id"] == "TestB"  # Shorter test ranks first
    assert "supporting_evidence" in data["scheduled_tests"][0]
    assert "historical_fail_rate" in data["scheduled_tests"][0]["supporting_evidence"]


def test_prioritize_endpoint_validation_error():
    # Invalid duration <= 0
    payload = {
        "tests": [
            {
                "test_id": "TestInvalid",
                "avg_recent_duration": -5.0
            }
        ]
    }
    resp = client.post("/prioritize", json=payload)
    assert resp.status_code == 422  # Pydantic validation error


def test_prioritize_endpoint_empty_tests():
    payload = {
        "tests": []
    }
    resp = client.post("/prioritize", json=payload)
    assert resp.status_code == 400


def test_evaluation_results_endpoint():
    resp = client.get("/evaluation-results")
    assert resp.status_code == 200
    data = resp.json()
    assert "classification_metrics" in data
    assert "prioritization_benchmarks" in data


def test_post_evaluate_endpoint():
    payload = {
        "tests": [
            {
                "test_id": "Test1",
                "avg_recent_duration": 5.0,
                "historical_fail_rate": 0.5,
                "verdict": 1
            },
            {
                "test_id": "Test2",
                "avg_recent_duration": 15.0,
                "historical_fail_rate": 0.1,
                "verdict": 0
            }
        ],
        "strategy": "historical",
        "budgets_seconds": [10.0, 30.0]
    }
    resp = client.post("/evaluate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "apfd" in data
    assert "apfdc" in data
    assert data["total_tests"] == 2
    assert data["total_failures"] == 1
