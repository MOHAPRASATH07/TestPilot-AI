"""
TestPilot AI - Flask Web Application.

Production SaaS developer tool interface for budget-aware software test prioritisation.
Serves interactive workspace, strategy comparison benchmarks, and REST API.
"""

import os
import sys
import io
import time
import json
import logging
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    send_from_directory,
    Response
)
from flask.json.provider import DefaultJSONProvider

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import (
    ALL_PRE_EXECUTION_FEATURES,
    IDENTIFIER_COLUMNS,
    POST_EXECUTION_COLUMNS
)
from src.preprocessing import TestPilotPreprocessor
from src.ranking import rank_tests, STRATEGY_NAMES
from src.evaluation import compute_apfd, compute_apfdc
from src.model_training import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    MLPNeuralNetworkModel,
    LearningToRankModel
)

logger = logging.getLogger("TestPilotAI.Flask")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data")
BENCHMARK_PATH = os.path.join(DATA_DIR, "benchmark_tcp_ci.csv")

class CustomJSONProvider(DefaultJSONProvider):
    """Custom JSON provider that seamlessly handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.json = CustomJSONProvider(app)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

# Global Singletons
LOADED_PREPROCESSOR: Optional[TestPilotPreprocessor] = None
LOADED_MODELS: Dict[str, Any] = {}
BENCHMARK_DF: Optional[pd.DataFrame] = None


def get_preprocessor() -> Optional[TestPilotPreprocessor]:
    """Loads and caches the fitted training preprocessor pipeline."""
    global LOADED_PREPROCESSOR
    if LOADED_PREPROCESSOR is None:
        p_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
        if os.path.exists(p_path):
            LOADED_PREPROCESSOR = TestPilotPreprocessor.load(p_path)
            logger.info("Loaded fitted preprocessor pipeline.")
    return LOADED_PREPROCESSOR


def get_models() -> Dict[str, Any]:
    """Loads and caches all trained ML model artifacts."""
    global LOADED_MODELS
    ctors = {
        "xgboost": XGBoostModel,
        "random_forest": RandomForestModel,
        "logistic_regression": LogisticRegressionModel,
        "mlp_neural_net": MLPNeuralNetworkModel,
        "ltr_lambdamart": LearningToRankModel
    }
    for key, ctor in ctors.items():
        if key not in LOADED_MODELS:
            m_path = os.path.join(MODELS_DIR, f"{key}.joblib")
            if os.path.exists(m_path):
                try:
                    LOADED_MODELS[key] = ctor().load(m_path)
                except Exception as e:
                    logger.warning(f"Could not load model {key}: {e}")
    return LOADED_MODELS


def get_benchmark_df() -> Optional[pd.DataFrame]:
    """Loads and caches the benchmark TCP-CI dataset."""
    global BENCHMARK_DF
    if BENCHMARK_DF is None and os.path.exists(BENCHMARK_PATH):
        BENCHMARK_DF = pd.read_csv(BENCHMARK_PATH)
        logger.info(f"Loaded benchmark dataset with {len(BENCHMARK_DF)} records.")
    return BENCHMARK_DF


def extract_evidence(row: pd.Series) -> Dict[str, Any]:
    """Formats feature drivers explaining priority decisions."""
    return {
        "historical_fail_rate": float(row.get("historical_fail_rate", 0.0)),
        "failures_last_5": int(row.get("failures_last_5", 0)),
        "modified_test_file": bool(row.get("modified_test_file", 0)),
        "changed_files_count": int(row.get("changed_files_count", 0)),
        "file_test_distance": float(row.get("file_test_distance", 0.5)),
        "estimated_duration": float(row.get("est_duration", 1.0))
    }


# ============================================================
# Page Routes
# ============================================================

@app.route("/")
def index():
    """Default landing route redirects to prioritization workspace."""
    return redirect(url_for("prioritize_view"))


@app.route("/prioritize")
def prioritize_view():
    """Renders the main interactive test prioritization workspace."""
    df = get_benchmark_df()
    projects = []
    if df is not None and "project_id" in df.columns:
        projects = sorted(df["project_id"].unique().tolist())

    return render_template(
        "prioritize.html",
        active_page="prioritize",
        projects=projects
    )


@app.route("/comparison")
def comparison_view():
    """Renders the strategy comparison page with held-out metrics & curves."""
    df = get_benchmark_df()
    projects = []
    if df is not None and "project_id" in df.columns:
        projects = sorted(df["project_id"].unique().tolist())

    # Load pre-computed evaluation results
    eval_csv_path = os.path.join(REPORTS_DIR, "evaluation_results.csv")
    eval_records = []
    if os.path.exists(eval_csv_path):
        eval_df = pd.read_csv(eval_csv_path)
        eval_records = eval_df.to_dict(orient="records")

    # Load failure discovery curves
    curves_csv_path = os.path.join(REPORTS_DIR, "failure_discovery_curves.csv")
    curves_data = []
    if os.path.exists(curves_csv_path):
        curves_df = pd.read_csv(curves_csv_path)
        curves_data = curves_df.to_dict(orient="records")

    return render_template(
        "comparison.html",
        active_page="comparison",
        projects=projects,
        eval_records=eval_records,
        curves_json=json.dumps(curves_data)
    )


@app.route("/about")
def about_view():
    """Renders the methodology, mathematical formulas, and dataset documentation."""
    return render_template("about.html", active_page="about")


# ============================================================
# API Endpoints
# ============================================================

@app.route("/api/benchmark-builds/<project_id>")
def api_benchmark_builds(project_id: str):
    """Returns sorted build sequence IDs for a selected project."""
    df = get_benchmark_df()
    if df is None:
        return jsonify({"builds": []})

    builds = sorted(df[df["project_id"] == project_id]["build_id"].unique().tolist())
    return jsonify({"project_id": project_id, "builds": builds})


@app.route("/api/benchmark-build-info/<project_id>/<int:build_id>")
def api_benchmark_build_info(project_id: str, build_id: int):
    """Returns metadata summary for a specific benchmark build."""
    df = get_benchmark_df()
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 404

    sub_df = df[(df["project_id"] == project_id) & (df["build_id"] == build_id)]
    if sub_df.empty:
        return jsonify({"error": "Build not found"}), 404

    total_dur = float(sub_df["avg_recent_duration"].sum()) if "avg_recent_duration" in sub_df.columns else 0.0
    fails = int(sub_df["verdict"].sum()) if "verdict" in sub_df.columns else None

    return jsonify({
        "project_id": project_id,
        "build_id": build_id,
        "test_count": len(sub_df),
        "total_duration": total_dur,
        "failures_count": fails
    })


@app.route("/api/upload-validate", methods=["POST"])
def api_upload_validate():
    """Validates an uploaded custom test suite CSV."""
    if "csv_file" not in request.files:
        return jsonify({"status": "invalid", "message": "No file uploaded."}), 400

    file = request.files["csv_file"]
    if file.filename == "":
        return jsonify({"status": "invalid", "message": "No selected file."}), 400

    try:
        content = file.read().decode("utf-8", errors="replace")
        csv_df = pd.read_csv(io.StringIO(content))

        req_cols = ["test_id", "avg_recent_duration"]
        missing_req = [c for c in req_cols if c not in csv_df.columns]
        if missing_req:
            return jsonify({
                "status": "invalid",
                "message": f"Missing required column(s): {', '.join(missing_req)}"
            }), 400

        feats_present = sum(1 for c in ALL_PRE_EXECUTION_FEATURES if c in csv_df.columns)
        total_dur = float(csv_df["avg_recent_duration"].sum())

        return jsonify({
            "status": "valid",
            "total_tests": len(csv_df),
            "total_duration": total_dur,
            "features_present": feats_present,
            "total_possible_features": len(ALL_PRE_EXECUTION_FEATURES)
        })

    except Exception as e:
        return jsonify({"status": "invalid", "message": f"Parsing error: {str(e)}"}), 400


@app.route("/api/prioritize", methods=["POST"])
def api_prioritize():
    """
    Executes real test prioritization pipeline.
    Accepts either multipart form or JSON payloads.
    """
    t_start = time.time()
    
    # Extract parameters
    if request.is_json:
        data = request.get_json()
        source_type = data.get("source_type", "benchmark")
        strategy = data.get("strategy", "ml_value_density")
        model_name = data.get("model_name", "xgboost")
        time_budget = float(data.get("time_budget_seconds", 300.0))
        alpha = float(data.get("alpha", 1.0))
        project_id = data.get("project_id")
        build_id = data.get("build_id")
        raw_tests = data.get("tests")
    else:
        source_type = request.form.get("source_type", "benchmark")
        strategy = request.form.get("strategy", "ml_value_density")
        model_name = request.form.get("model_name", "xgboost")
        time_budget = float(request.form.get("time_budget_seconds", 300.0))
        alpha = float(request.form.get("alpha", 1.0))
        project_id = request.form.get("project_id")
        build_id = request.form.get("build_id")
        raw_tests = None

    # Load target test cases
    if source_type == "upload" or "csv_file" in request.files:
        if "csv_file" not in request.files:
            return jsonify({"message": "No CSV file uploaded."}), 400
        file = request.files["csv_file"]
        content = file.read().decode("utf-8", errors="replace")
        test_df = pd.read_csv(io.StringIO(content))
    elif raw_tests:
        test_df = pd.DataFrame(raw_tests)
    else:
        # Benchmark mode
        df = get_benchmark_df()
        if df is None:
            return jsonify({"message": "Benchmark dataset not found."}), 404
        if not project_id or build_id is None:
            return jsonify({"message": "project_id and build_id are required for benchmark source."}), 400
        
        try:
            b_id = int(build_id)
        except ValueError:
            return jsonify({"message": f"Invalid build_id: {build_id}"}), 400

        sub_df = df[(df["project_id"] == project_id) & (df["build_id"] == b_id)].copy()
        if sub_df.empty:
            return jsonify({"message": f"No test records for {project_id} build #{b_id}."}), 404
        test_df = sub_df.reset_index(drop=True)

    if test_df.empty:
        return jsonify({"message": "Empty test suite provided."}), 400

    # Execute ML Inference if needed
    probs = None
    ltr_scores = None
    model_used = None

    if strategy.startswith("ml_"):
        prep = get_preprocessor()
        models = get_models()

        if prep is None:
            return jsonify({"message": "Preprocessing pipeline not found. Run train_models.py first."}), 500

        target_model_key = model_name
        if strategy == "ml_ltr":
            target_model_key = "ltr_lambdamart"

        if target_model_key not in models:
            return jsonify({"message": f"Model '{target_model_key}' not loaded. Available: {list(models.keys())}"}), 404

        model = models[target_model_key]
        model_used = model.name

        try:
            X_scaled = prep.transform(test_df, scale=True)
            if strategy == "ml_ltr":
                ltr_scores = model.predict_proba(X_scaled)
            else:
                probs = model.predict_proba(X_scaled)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return jsonify({"message": f"Inference pipeline error: {str(e)}"}), 500

    # Run budget-aware ranking
    try:
        res = rank_tests(
            test_df=test_df,
            strategy=strategy,
            predicted_probs=probs,
            ltr_scores=ltr_scores,
            time_budget_seconds=time_budget,
            alpha=alpha,
            seed=42
        )
    except Exception as e:
        logger.error(f"Ranking execution failed: {e}")
        return jsonify({"message": f"Ranking execution failed: {str(e)}"}), 500

    elapsed_ms = (time.time() - t_start) * 1000.0

    # Build response records
    ranked_df = res["ranked_tests"]
    has_verdicts = "verdict" in ranked_df.columns and not ranked_df["verdict"].isnull().all()

    all_ranked_tests = []
    for _, row in ranked_df.iterrows():
        all_ranked_tests.append({
            "rank": int(row["rank"]),
            "test_id": str(row["test_id"]),
            "priority_score": round(float(row["priority_score"]), 4),
            "predicted_risk": round(float(row["predicted_risk"]), 4) if pd.notnull(row.get("predicted_risk")) else None,
            "estimated_duration_seconds": round(float(row["est_duration"]), 2),
            "cumulative_duration_seconds": round(float(row["cum_duration"]), 2),
            "within_budget": bool(row["within_budget"]),
            "verdict": int(row["verdict"]) if has_verdicts and pd.notnull(row.get("verdict")) else None,
            "supporting_evidence": extract_evidence(row)
        })

    return jsonify({
        "strategy": res["strategy"],
        "strategy_label": res["strategy_label"],
        "explanation": res["explanation"],
        "model_used": model_used,
        "time_budget_seconds": time_budget,
        "total_tests_received": res["total_tests_available"],
        "scheduled_test_count": res["scheduled_test_count"],
        "excluded_test_count": res["excluded_test_count"],
        "total_scheduled_time_seconds": res["total_scheduled_time_seconds"],
        "remaining_budget_seconds": res["remaining_budget_seconds"],
        "failures_discovered": res.get("failures_discovered"),
        "total_failures_in_suite": res.get("total_failures_in_build"),
        "failure_recall": res.get("failure_recall"),
        "execution_time_ms": elapsed_ms,
        "all_ranked_tests": all_ranked_tests
    })


@app.route("/api/compare-build", methods=["POST"])
def api_compare_build():
    """Runs all 6 strategies side-by-side on a selected benchmark build."""
    data = request.get_json() or {}
    project_id = data.get("project_id")
    build_id = data.get("build_id")
    budget = float(data.get("time_budget_seconds", 300.0))

    df = get_benchmark_df()
    if df is None or not project_id or build_id is None:
        return jsonify({"error": "Invalid request"}), 400

    sub_df = df[(df["project_id"] == project_id) & (df["build_id"] == int(build_id))].copy().reset_index(drop=True)
    if sub_df.empty:
        return jsonify({"error": "Build not found"}), 404

    prep = get_preprocessor()
    models = get_models()
    has_v = "verdict" in sub_df.columns and not sub_df["verdict"].isnull().all()

    strategies_to_run = [
        ("Baseline A: Random Ordering", "random", None, None),
        ("Baseline B: Fastest-First", "fastest_first", None, None),
        ("Baseline C: Historical Prioritization", "historical", None, None),
    ]

    if prep is not None:
        X_s = prep.transform(sub_df, scale=True)
        if "xgboost" in models:
            p_xgb = models["xgboost"].predict_proba(X_s)
            strategies_to_run.append(("ML: XGBoost (Value-Density)", "ml_value_density", p_xgb, None))
            strategies_to_run.append(("ML: XGBoost (Pure Risk)", "ml_risk_only", p_xgb, None))
        if "ltr_lambdamart" in models:
            p_ltr = models["ltr_lambdamart"].predict_proba(X_s)
            strategies_to_run.append(("ML: LambdaMART (Rank)", "ml_ltr", None, p_ltr))

    results = []
    plot_series = []

    for label, strat_key, p_prob, l_prob in strategies_to_run:
        res = rank_tests(
            test_df=sub_df,
            strategy=strat_key,
            predicted_probs=p_prob,
            ltr_scores=l_prob,
            time_budget_seconds=budget,
            seed=42
        )
        r_tests = res["ranked_tests"]
        v_list = r_tests["verdict"].tolist() if has_v else []
        d_list = r_tests["est_duration"].tolist()

        apfd = compute_apfd(v_list) if has_v else 0.0
        apfdc = compute_apfdc(v_list, d_list) if has_v else 0.0

        results.append({
            "strategy": label,
            "scheduled_count": res["scheduled_test_count"],
            "total_tests": res["total_tests_available"],
            "runtime_seconds": res["total_scheduled_time_seconds"],
            "failures_found": res["failures_discovered"] if has_v else 0,
            "total_failures": res["total_failures_in_build"] if has_v else 0,
            "recall": res["failure_recall"] if has_v else 0.0,
            "apfd": round(apfd, 4),
            "apfdc": round(apfdc, 4)
        })

        if has_v:
            cum_f = 0
            # Pre-pend zero point
            plot_series.append({"strategy": label, "time_s": 0.0, "cum_failures": 0})
            for _, r_row in r_tests.iterrows():
                cum_f += int(r_row["verdict"])
                plot_series.append({
                    "strategy": label,
                    "time_s": round(float(r_row["cum_duration"]), 1),
                    "cum_failures": cum_f
                })

    return jsonify({
        "project_id": project_id,
        "build_id": build_id,
        "budget_seconds": budget,
        "results": results,
        "plot_series": plot_series
    })


@app.route("/download-report/<filename>")
def download_report(filename: str):
    """Safely serves pre-computed report artifacts from the reports directory."""
    allowed = {
        "IEEE_Report.md",
        "evaluation_summary.json",
        "evaluation_results.csv",
        "failure_discovery_curves.csv",
        "data_audit.json"
    }
    if filename not in allowed:
        return jsonify({"error": "File not found or access restricted"}), 404
    return send_from_directory(REPORTS_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    # Preload resources
    get_benchmark_df()
    get_preprocessor()
    get_models()
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting TestPilot AI Flask Server on http://{host}:{port}")
    app.run(host=host, port=port, debug=True)
