"""
Comprehensive Integration Tests for the TestPilot AI Flask Web Application.
"""

import io
import json
import pytest
from app_flask import app, get_benchmark_df


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_redirect(client):
    """GET / should redirect to /prioritize."""
    res = client.get("/")
    assert res.status_code == 302
    assert "/prioritize" in res.headers["Location"]


def test_prioritize_view(client):
    """GET /prioritize should render HTML with expected sections."""
    res = client.get("/prioritize")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Prioritize Test Suite" in html
    assert "Ranking Strategy" in html
    assert "Benchmark Build" in html


def test_comparison_view(client):
    """GET /comparison should render HTML with benchmark evaluation table and curves."""
    res = client.get("/comparison")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Strategy Comparison" in html
    assert "APFDc" in html
    assert "Held-Out Test Suite Benchmark" in html


def test_about_view(client):
    """GET /about should render formal formulas and empirical findings."""
    res = client.get("/about")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Methodology" in html
    assert "Average Percentage of Faults Detected" in html
    assert "Cost-Cognizant APFD" in html


def test_api_benchmark_builds(client):
    """GET /api/benchmark-builds/<project_id> returns list of builds."""
    df = get_benchmark_df()
    if df is not None and not df.empty:
        proj = df["project_id"].iloc[0]
        res = client.get(f"/api/benchmark-builds/{proj}")
        assert res.status_code == 200
        data = res.get_json()
        assert "builds" in data
        assert len(data["builds"]) > 0


def test_api_benchmark_build_info(client):
    """GET /api/benchmark-build-info/<project_id>/<build_id> returns test counts and durations."""
    df = get_benchmark_df()
    if df is not None and not df.empty:
        row = df.iloc[0]
        proj = row["project_id"]
        b_id = row["build_id"]
        res = client.get(f"/api/benchmark-build-info/{proj}/{b_id}")
        assert res.status_code == 200
        data = res.get_json()
        assert data["test_count"] > 0
        assert data["total_duration"] > 0.0


def test_api_upload_validate_valid(client):
    """POST /api/upload-validate with valid CSV returns valid status."""
    csv_content = "test_id,avg_recent_duration,historical_fail_rate\ntest_one,4.2,0.1\ntest_two,1.5,0.0"
    data = {
        "csv_file": (io.BytesIO(csv_content.encode("utf-8")), "tests.csv")
    }
    res = client.post("/api/upload-validate", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    info = res.get_json()
    assert info["status"] == "valid"
    assert info["total_tests"] == 2
    assert info["total_duration"] == pytest.approx(5.7, 0.01)


def test_api_upload_validate_missing_required(client):
    """POST /api/upload-validate with missing required column returns 400 error."""
    csv_content = "test_name,duration\ntest_one,4.2"
    data = {
        "csv_file": (io.BytesIO(csv_content.encode("utf-8")), "bad.csv")
    }
    res = client.post("/api/upload-validate", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    info = res.get_json()
    assert info["status"] == "invalid"
    assert "Missing required column" in info["message"]


def test_api_prioritize_benchmark(client):
    """POST /api/prioritize on benchmark build returns scheduled tests and KPIs."""
    df = get_benchmark_df()
    if df is not None and not df.empty:
        row = df.iloc[0]
        proj = str(row["project_id"])
        b_id = int(row["build_id"])

        payload = {
            "source_type": "benchmark",
            "project_id": proj,
            "build_id": b_id,
            "strategy": "ml_value_density",
            "model_name": "xgboost",
            "time_budget_seconds": 300.0,
            "alpha": 1.0
        }
        res = client.post("/api/prioritize", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["total_tests_received"] > 0
        assert data["scheduled_test_count"] > 0
        assert len(data["all_ranked_tests"]) == data["total_tests_received"]
        assert "priority_score" in data["all_ranked_tests"][0]
        assert "within_budget" in data["all_ranked_tests"][0]


def test_api_prioritize_heuristics(client):
    """POST /api/prioritize with Fastest-First and Random baselines."""
    df = get_benchmark_df()
    if df is not None and not df.empty:
        row = df.iloc[0]
        proj = str(row["project_id"])
        b_id = int(row["build_id"])

        for strat in ["fastest_first", "random", "historical"]:
            payload = {
                "source_type": "benchmark",
                "project_id": proj,
                "build_id": b_id,
                "strategy": strat,
                "time_budget_seconds": 600.0
            }
            res = client.post("/api/prioritize", json=payload)
            assert res.status_code == 200
            data = res.get_json()
            assert data["strategy"] == strat
            assert data["scheduled_test_count"] > 0


def test_api_compare_build(client):
    """POST /api/compare-build runs all strategies side-by-side."""
    df = get_benchmark_df()
    if df is not None and not df.empty:
        row = df.iloc[0]
        proj = str(row["project_id"])
        b_id = int(row["build_id"])

        payload = {
            "project_id": proj,
            "build_id": int(b_id),
            "time_budget_seconds": 300.0
        }
        res = client.post("/api/compare-build", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert "results" in data
        assert len(data["results"]) >= 3
        # Check that strategies have APFD and APFDc computed
        for r in data["results"]:
            assert "strategy" in r
            assert "apfd" in r
            assert "apfdc" in r


def test_download_report(client):
    """GET /download-report/IEEE_Report.md downloads report file."""
    res = client.get("/download-report/IEEE_Report.md")
    assert res.status_code == 200
    assert "TestPilot AI" in res.get_data(as_text=True)
