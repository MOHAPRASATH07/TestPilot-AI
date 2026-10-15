"""
TestPilot AI: Budget-Aware Machine Learning for Risk-Driven Software Test Prioritisation.
Interactive Engineering Tool and Prioritization Workspace.
"""

import os
import sys
import io
import json
import time
import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import (
    load_dataset,
    IDENTIFIER_COLUMNS,
    POST_EXECUTION_COLUMNS,
    ALL_PRE_EXECUTION_FEATURES
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

# Page Setup
st.set_page_config(
    page_title="TestPilot AI | Interactive Test Prioritisation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style tweaks for cleaner product look
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #e9ecef;
    }
    .stButton>button {
        font-weight: 600;
    }
    .badge-scheduled {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-excluded {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Directory Constants
MODELS_DIR = "models"
REPORTS_DIR = "reports"
DATA_PATH = os.path.join("data", "benchmark_tcp_ci.csv")


# Resource Loaders
@st.cache_data
def load_benchmark_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource
def load_pipeline_and_models():
    prep_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
    prep = TestPilotPreprocessor.load(prep_path) if os.path.exists(prep_path) else None

    models = {}
    ctors = {
        "xgboost": (XGBoostModel, os.path.join(MODELS_DIR, "xgboost.joblib")),
        "random_forest": (RandomForestModel, os.path.join(MODELS_DIR, "random_forest.joblib")),
        "logistic_regression": (LogisticRegressionModel, os.path.join(MODELS_DIR, "logistic_regression.joblib")),
        "mlp_neural_net": (MLPNeuralNetworkModel, os.path.join(MODELS_DIR, "mlp_neural_net.joblib")),
        "ltr_lambdamart": (LearningToRankModel, os.path.join(MODELS_DIR, "ltr_lambdamart.joblib"))
    }
    for k, (ctor, path) in ctors.items():
        if os.path.exists(path):
            try:
                models[k] = ctor().load(path)
            except Exception as e:
                logging.error(f"Failed to load model {k}: {e}")
    return prep, models


prep_obj, model_dict = load_pipeline_and_models()
benchmark_df = load_benchmark_data()


# Top Navigation Tabs
nav_prioritize, nav_compare, nav_lab, nav_about = st.tabs([
    "⚡ Prioritise Tests (Workspace)",
    "📊 Strategy Comparison & Analytics",
    "🧪 Model Management & Live Training",
    "ℹ️ About & Methodology"
])


# ============================================================
# TAB 1: WORKSPACE (CORE PRODUCT)
# ============================================================
with nav_prioritize:
    st.subheader("⚡ Continuous Integration Test Prioritisation Workspace")
    st.caption("Select or upload test cases, specify a time budget, and execute the machine learning prioritization pipeline.")

    # Top Control Grid
    c_source, c_config = st.columns([1, 1], gap="medium")

    with c_source:
        st.markdown("##### 1. Select Input Test Suite")
        data_source = st.radio(
            "Input Mode",
            ["Benchmark Repository (TCP-CI)", "Upload Custom Test Suite CSV"],
            horizontal=True,
            label_visibility="collapsed"
        )

        active_tests_df = None

        if data_source == "Benchmark Repository (TCP-CI)":
            if benchmark_df is None:
                st.error("Benchmark dataset not found at `data/benchmark_tcp_ci.csv`. Run `python scripts/download_dataset.py` first.")
            else:
                p_col, b_col = st.columns(2)
                with p_col:
                    projects = benchmark_df["project_id"].unique().tolist()
                    selected_project = st.selectbox("Project", projects)
                with b_col:
                    builds = sorted(benchmark_df[benchmark_df["project_id"] == selected_project]["build_id"].unique().tolist())
                    selected_build = st.selectbox("Build Sequence ID", builds, index=len(builds) - 1)

                active_tests_df = benchmark_df[
                    (benchmark_df["project_id"] == selected_project) & 
                    (benchmark_df["build_id"] == selected_build)
                ].copy().reset_index(drop=True)

                total_dur = active_tests_df["avg_recent_duration"].sum()
                st.info(f"Loaded **{len(active_tests_df)} test cases** from `{selected_project}` (Build #{selected_build}). Full unprioritized runtime: **{total_dur:.1f} s** (~{total_dur/60:.1f} min).")

        else:
            uploaded_file = st.file_uploader("Upload test case CSV", type=["csv"], help="Must contain at least 'test_id' and 'avg_recent_duration'")
            if uploaded_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                    req_cols = ["test_id", "avg_recent_duration"]
                    missing_req = [c for c in req_cols if c not in uploaded_df.columns]
                    
                    if missing_req:
                        st.error(f"Incompatible CSV: Missing mandatory columns: {missing_req}")
                    else:
                        active_tests_df = uploaded_df.copy().reset_index(drop=True)
                        present_feats = [c for c in ALL_PRE_EXECUTION_FEATURES if c in active_tests_df.columns]
                        missing_feats = [c for c in ALL_PRE_EXECUTION_FEATURES if c not in active_tests_df.columns]
                        
                        st.success(f"✅ Loaded **{len(active_tests_df)} tests**. Features present: {len(present_feats)} / {len(ALL_PRE_EXECUTION_FEATURES)}.")
                        if missing_feats:
                            st.caption(f"Note: {len(missing_feats)} optional features missing; pipeline will impute median defaults.")
                except Exception as e:
                    st.error(f"Error parsing CSV: {e}")

    with c_config:
        st.markdown("##### 2. Prioritisation & Budget Configuration")
        
        b1, b2 = st.columns(2)
        with b1:
            budget_option = st.selectbox(
                "Testing Time Budget",
                ["5 Minutes (300s)", "10 Minutes (600s)", "20 Minutes (1200s)", "Custom Budget"]
            )
            if budget_option == "Custom Budget":
                time_budget = st.number_input("Time Budget (seconds)", min_value=5.0, max_value=7200.0, value=300.0, step=30.0)
            else:
                time_budget = float(budget_option.split("(")[1].replace("s)", ""))

        with b2:
            strategy_key = st.selectbox(
                "Ranking Strategy",
                list(STRATEGY_NAMES.keys()),
                format_func=lambda k: STRATEGY_NAMES[k]
            )

        m1, m2 = st.columns(2)
        with m1:
            if strategy_key.startswith("ml_"):
                avail_m = list(model_dict.keys())
                if not avail_m:
                    st.warning("No trained models found in `models/`.")
                    selected_model_key = None
                else:
                    selected_model_key = st.selectbox(
                        "Machine Learning Model",
                        avail_m,
                        format_func=lambda k: f"{model_dict[k].name} {'(Champion)' if k=='xgboost' else ''}"
                    )
            else:
                st.caption(f"Heuristic/Stochastic mode: `{STRATEGY_NAMES[strategy_key]}` does not require model inference.")
                selected_model_key = None

        with m2:
            if strategy_key == "ml_value_density":
                alpha_param = st.slider("Duration Sensitivity (alpha)", 0.5, 2.0, 1.0, 0.1, help="Exponent on duration: Score = P(fail) / (duration ^ alpha)")
            else:
                alpha_param = 1.0

        st.markdown("<br>", unsafe_allow_html=True)
        execute_button = st.button("⚡ Prioritise Tests", type="primary", use_container_width=True)

    # Execution Logic
    if execute_button:
        if active_tests_df is None or len(active_tests_df) == 0:
            st.error("Please load or upload a valid test suite first.")
        else:
            with st.spinner("Executing real feature transformation, inference, and budget scheduler..."):
                t_start = time.time()
                probs = None
                ltr_scores = None

                # ML Inference if required
                if strategy_key.startswith("ml_"):
                    if prep_obj is None:
                        st.error("Preprocessing pipeline (`models/preprocessing_pipeline.joblib`) not found. Run training script first.")
                        st.stop()
                    if selected_model_key not in model_dict:
                        st.error(f"Selected model '{selected_model_key}' is not available.")
                        st.stop()

                    target_model = model_dict[selected_model_key]
                    X_scaled = prep_obj.transform(active_tests_df, scale=True)

                    if strategy_key == "ml_ltr":
                        ltr_scores = model_dict.get("ltr_lambdamart", target_model).predict_proba(X_scaled)
                    else:
                        probs = target_model.predict_proba(X_scaled)

                # Execute Ranking
                run_result = rank_tests(
                    test_df=active_tests_df,
                    strategy=strategy_key,
                    predicted_probs=probs,
                    ltr_scores=ltr_scores,
                    time_budget_seconds=time_budget,
                    alpha=alpha_param,
                    seed=42
                )
                dt_ms = (time.time() - t_start) * 1000.0

                # Cache in session state
                st.session_state["active_result"] = run_result
                st.session_state["last_exec_time_ms"] = dt_ms
                st.session_state["last_strategy"] = strategy_key
                st.session_state["last_budget"] = time_budget

    # Render Results if available
    if "active_result" in st.session_state:
        res = st.session_state["active_result"]
        ranked_df = res["ranked_tests"]
        sched_df = res["scheduled_tests"]
        excl_df = res["excluded_tests"]

        st.markdown("---")
        st.markdown(f"### 📋 Prioritisation Schedule Result — `{res['strategy_label']}`")
        st.caption(f"{res['explanation']} (Pipeline execution completed in {st.session_state.get('last_exec_time_ms', 0):.1f} ms)")

        # Real KPI Metric Cards computed strictly from this run
        k1, k2, k3, k4, k5 = st.columns(5)
        
        sched_pct = (len(sched_df) / max(1, len(ranked_df))) * 100
        budget_util_pct = (res["total_scheduled_time_seconds"] / max(0.1, res["time_budget_seconds"])) * 100

        k1.metric("Tests Scheduled", f"{len(sched_df)} / {len(ranked_df)}", f"{sched_pct:.1f}% included")
        k2.metric("Scheduled Runtime", f"{res['total_scheduled_time_seconds']:.1f} s", f"of {res['time_budget_seconds']:.0f} s budget")
        k3.metric("Remaining Budget", f"{res['remaining_budget_seconds']:.1f} s", f"{budget_util_pct:.1f}% utilized")
        k4.metric("Tests Excluded", f"{len(excl_df)} tests", "Exceeded budget")

        has_verdicts = "verdict" in ranked_df.columns and not ranked_df["verdict"].isnull().all()
        if has_verdicts:
            fails_found = res["failures_discovered"]
            total_fails = res["total_failures_in_build"]
            rec_pct = (res["failure_recall"] or 0.0) * 100
            k5.metric("Failures Detected", f"{fails_found} / {total_fails}", f"{rec_pct:.1f}% Recall (Retrospective)")
        else:
            k5.metric("Target Labels", "Unlabeled Input", "Inference Mode")

        # Visual Budget Progress Bar
        progress_val = min(1.0, res["total_scheduled_time_seconds"] / max(0.1, res["time_budget_seconds"]))
        st.progress(progress_val, text=f"Budget Allocation: {res['total_scheduled_time_seconds']:.1f}s / {res['time_budget_seconds']:.0f}s ({budget_util_pct:.1f}%)")

        # Action and Filter Toolbar
        act_col, filter_col = st.columns([1, 2])
        with act_col:
            # Download Real Generated Schedule
            csv_buffer = io.StringIO()
            ranked_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Schedule as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"testpilot_schedule_{res['strategy']}_{int(res['time_budget_seconds'])}s.csv",
                mime="text/csv",
                use_container_width=True
            )
        with filter_col:
            filter_choice = st.radio(
                "Filter Table View",
                ["All Tests", f"Scheduled Only ({len(sched_df)})", f"Excluded Only ({len(excl_df)})"],
                horizontal=True
            )

        # Apply Table Filter
        if filter_choice.startswith("Scheduled"):
            view_df = sched_df.copy()
        elif filter_choice.startswith("Excluded"):
            view_df = excl_df.copy()
        else:
            view_df = ranked_df.copy()

        # Supporting feature evidence string
        def build_evidence(row):
            parts = []
            if "historical_fail_rate" in row:
                parts.append(f"FailRate: {row['historical_fail_rate']*100:.0f}%")
            if "failures_last_5" in row and row["failures_last_5"] > 0:
                parts.append(f"Last5: {int(row['failures_last_5'])}")
            if "modified_test_file" in row and row["modified_test_file"] == 1:
                parts.append("Modified in Commit: Yes")
            total_churn = (row.get("code_churn_additions", 0) + row.get("code_churn_deletions", 0))
            if total_churn > 0:
                parts.append(f"Churn: {int(total_churn)} loc")
            return " | ".join(parts) if parts else "Baseline test metrics"

        view_df["supporting_evidence"] = view_df.apply(build_evidence, axis=1)

        display_columns = [
            "rank", "test_id", "priority_score", "est_duration", "cum_duration", "within_budget", "supporting_evidence"
        ]
        if "predicted_risk" in view_df.columns and view_df["predicted_risk"].notnull().any():
            display_columns.insert(3, "predicted_risk")
        if has_verdicts:
            display_columns.append("verdict")

        # Interactive Table
        st.dataframe(
            view_df[display_columns].style.apply(
                lambda row: ['background-color: #d4edda' if row['within_budget'] else 'background-color: #f8d7da'] * len(row),
                axis=1
            ),
            use_container_width=True,
            height=400
        )

        st.caption("ℹ️ Green rows indicate tests scheduled to execute within the selected time budget. Red rows exceed the budget cap. Priority scores determine the sequence; high scores reflect risk/cost trade-offs and do not guarantee failure.")


# ============================================================
# TAB 2: STRATEGY COMPARISON & ANALYTICS
# ============================================================
with nav_compare:
    st.subheader("📊 Live Strategy Comparison & Retrospective Evaluation")
    st.caption("Compare all prioritization strategies side-by-side using real calculations.")

    comp_source = st.radio(
        "Comparison Scope",
        ["Run Live Comparison on Currently Loaded Build", "Inspect Published Held-Out Benchmark Results (12 Builds)"],
        horizontal=True
    )

    if comp_source == "Run Live Comparison on Currently Loaded Build":
        if active_tests_df is None or len(active_tests_df) == 0:
            st.info("Please select a build or upload data in the **Prioritise Tests** workspace tab above.")
        else:
            budget_for_comp = st.slider("Comparison Budget (seconds)", 60.0, 1800.0, 300.0, 60.0)
            
            if st.button("⚡ Run Live Strategy Comparison on Current Suite"):
                with st.spinner("Executing all 6 ranking strategies on current test suite..."):
                    all_strats = [
                        ("Random Baseline", "random", None, None),
                        ("Fastest-First", "fastest_first", None, None),
                        ("Historical Prioritization", "historical", None, None),
                    ]
                    if prep_obj is not None:
                        X_s = prep_obj.transform(active_tests_df, scale=True)
                        if "xgboost" in model_dict:
                            p_xgb = model_dict["xgboost"].predict_proba(X_s)
                            all_strats.append(("XGBoost (Value-Density)", "ml_value_density", p_xgb, None))
                            all_strats.append(("XGBoost (Pure Risk)", "ml_risk_only", p_xgb, None))
                        if "ltr_lambdamart" in model_dict:
                            p_ltr = model_dict["ltr_lambdamart"].predict_proba(X_s)
                            all_strats.append(("Learning-to-Rank (LambdaMART)", "ml_ltr", None, p_ltr))

                    comp_rows = []
                    curves_data = []
                    has_v = "verdict" in active_tests_df.columns and not active_tests_df["verdict"].isnull().all()

                    for label, strat_id, p_prob, l_prob in all_strats:
                        res_c = rank_tests(
                            test_df=active_tests_df,
                            strategy=strat_id,
                            predicted_probs=p_prob,
                            ltr_scores=l_prob,
                            time_budget_seconds=budget_for_comp,
                            seed=42
                        )
                        r_tests = res_c["ranked_tests"]
                        v_list = r_tests["verdict"].tolist() if has_v else []
                        d_list = r_tests["est_duration"].tolist()

                        apfd_val = compute_apfd(v_list) if has_v else 0.0
                        apfdc_val = compute_apfdc(v_list, d_list) if has_v else 0.0
                        f_found = res_c["failures_discovered"] if has_v else None

                        comp_rows.append({
                            "Strategy": label,
                            "Tests Scheduled": f"{res_c['scheduled_test_count']} / {res_c['total_tests_available']}",
                            "Runtime (s)": round(res_c["total_scheduled_time_seconds"], 1),
                            "Failures Detected": f"{f_found} / {res_c['total_failures_in_build']}" if has_v else "N/A",
                            "Recall @ Budget": f"{(res_c['failure_recall'] or 0)*100:.1f}%" if has_v else "N/A",
                            "APFD": f"{apfd_val:.4f}" if has_v else "N/A",
                            "APFDc (Cost-Aware)": f"{apfdc_val:.4f}" if has_v else "N/A"
                        })

                        # Build step curve
                        if has_v:
                            for _, r_row in r_tests.iterrows():
                                curves_data.append({
                                    "Strategy": label,
                                    "Elapsed_Time_s": r_row["cum_duration"],
                                    "Failures_Found": int(r_row["verdict"])
                                })

                    comp_table_df = pd.DataFrame(comp_rows)
                    st.dataframe(comp_table_df, use_container_width=True)

                    if has_v and curves_data:
                        c_df = pd.DataFrame(curves_data)
                        # Compute cumulative sum per strategy
                        c_df["Cum_Failures"] = c_df.groupby("Strategy")["Failures_Found"].cumsum()
                        fig_live = px.line(
                            c_df,
                            x="Elapsed_Time_s",
                            y="Cum_Failures",
                            color="Strategy",
                            title=f"Live Defect Discovery Curve on Current Build (Budget: {budget_for_comp:.0f}s)",
                            labels={"Elapsed_Time_s": "Elapsed Test Runtime (s)", "Cum_Failures": "Cumulative Failures Found"}
                        )
                        fig_live.add_vline(x=budget_for_comp, line_dash="dash", line_color="red", annotation_text="Budget Limit")
                        fig_live.update_layout(template="plotly_white")
                        st.plotly_chart(fig_live, use_container_width=True)

    else:
        # Load precomputed held-out benchmark results
        eval_path = os.path.join(REPORTS_DIR, "evaluation_results.csv")
        curves_path = os.path.join(REPORTS_DIR, "failure_discovery_curves.csv")

        if os.path.exists(eval_path) and os.path.exists(curves_path):
            eval_bench_df = pd.read_csv(eval_path)
            curves_bench_df = pd.read_csv(curves_path)

            st.markdown("#### Retrospective Benchmark on 12 Held-Out Unseen Builds (1,920 Test Records)")
            st.dataframe(
                eval_bench_df[[
                    "strategy_name", "mean_apfd", "mean_apfdc", "mean_recall_300s", "mean_recall_600s", "mean_recall_1200s"
                ]].rename(columns={
                    "strategy_name": "Strategy",
                    "mean_apfd": "APFD",
                    "mean_apfdc": "Cost-Cognizant APFDc",
                    "mean_recall_300s": "Recall @ 5m",
                    "mean_recall_600s": "Recall @ 10m",
                    "mean_recall_1200s": "Recall @ 20m"
                }),
                use_container_width=True
            )

            fig_b = px.line(
                curves_bench_df,
                x="elapsed_time_minutes",
                y="percentage_failures_discovered",
                color="strategy",
                title="Cumulative Failure Discovery Across Held-Out Builds",
                labels={"elapsed_time_minutes": "Testing Time (min)", "percentage_failures_discovered": "Failures Discovered (%)"}
            )
            fig_b.update_layout(template="plotly_white")
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.warning("Benchmark files not found. Run `python scripts/evaluate_model.py` to generate held-out evaluations.")


# ============================================================
# TAB 3: MODEL MANAGEMENT & LIVE TRAINING
# ============================================================
with nav_lab:
    st.subheader("🧪 Model Management & Live Training Lab")
    st.caption("Inspect persisted model files, trigger real training jobs, or run Optuna hyperparameter optimization.")

    m_col1, m_col2 = st.columns([1, 1], gap="medium")

    with m_col1:
        st.markdown("##### Persisted Model Artifacts on Disk")
        artifact_list = []
        for fname in os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []:
            fpath = os.path.join(MODELS_DIR, fname)
            if os.path.isfile(fpath):
                fsize_kb = os.path.getsize(fpath) / 1024.0
                mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(fpath)))
                artifact_list.append({"File": fname, "Size (KB)": f"{fsize_kb:.1f}", "Last Modified": mtime})

        if artifact_list:
            st.dataframe(pd.DataFrame(artifact_list), use_container_width=True)
        else:
            st.info("No artifacts found in `models/` directory.")

        # Best Optuna Config Inspection
        cfg_path = os.path.join(MODELS_DIR, "best_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                best_cfg = json.load(f)
            st.markdown("##### Best Optuna Configuration (XGBoost)")
            st.json(best_cfg)

    with m_col2:
        st.markdown("##### Execute Live Pipeline Training")
        st.write("Triggering this job executes the real multi-model training pipeline on the chronological training split.")

        num_optuna_trials = st.slider("Optuna Search Trials", 5, 25, 10, step=5)
        
        if st.button("🚀 Run Live Multi-Model Training & Optuna Tuning"):
            log_container = st.empty()
            log_container.info("Starting chronological training job in background...")
            
            with st.spinner("Fitting preprocessor, running Optuna TPE, and training 5 models..."):
                try:
                    import subprocess
                    proc = subprocess.run(
                        [sys.executable, "scripts/train_model.py"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    st.success("✅ Training Pipeline Finished Successfully!")
                    st.text_area("Execution Log", proc.stdout[-2000:], height=250)
                    # Clear cache to reload new models
                    st.cache_resource.clear()
                except subprocess.CalledProcessError as err:
                    st.error(f"Training failed with exit code {err.returncode}:")
                    st.text_area("Error Log", err.stderr[-2000:], height=250)


# ============================================================
# TAB 4: ABOUT & METHODOLOGY
# ============================================================
with nav_about:
    st.subheader("ℹ️ Methodology, Dataset Provenance & System Limitations")
    
    st.markdown("""
    #### 1. Core Mathematical Formulations
    - **Value-Density Priority Score:**
      $$\\text{Score}(t_i) = \\frac{\\hat{P}(\\text{verdict}_i = 1)}{\\left( \\hat{d}_i \\right)^\\alpha}$$
      Balances fault probability $\\hat{P}$ against estimated duration $\\hat{d}_i$, prioritizing high-risk fast-running tests first.
    - **Average Percentage of Faults Detected (APFD):**
      $$APFD = 1 - \\frac{\\sum_{i=1}^m TF_i}{n \\cdot m} + \\frac{1}{2n}$$
    - **Cost-Cognizant APFD ($APFD_c$):**
      Accounts for tests with varying runtimes $t_j$.

    #### 2. Research Benchmark Provenance
    - **Dataset**: TCP-CI benchmark (*A. S. Yaraghi, M. Bagherzadeh, N. Kahani, L. Briand*, IEEE TSE 2023; Zenodo: `10.5281/zenodo.6415365`).
    - **Features**: Historical execution records (REC), code churn & distance (CHG), coverage/structure (COV/TES).

    #### 3. Real-World Limitations & Threats to Validity
    - **Cluster Duration Jitter**: Real CI build durations fluctuate due to cloud container cold-starts and I/O noise. TestPilot AI uses moving averages (`avg_recent_duration`) as pre-execution estimates.
    - **Class Imbalance**: Test failures are rare (~5-10%). Machine learning models assign probabilities, not binary certainties.
    - **Risk Prediction vs Guarantee**: A high priority score reflects elevated historical or commit-level risk, but does not guarantee a failure will occur.
    """)

    st.markdown("---")
    st.markdown("##### Download Generated Scientific Reports")
    rep_files = {
        "evaluation_results.csv": os.path.join(REPORTS_DIR, "evaluation_results.csv"),
        "evaluation_summary.md": os.path.join(REPORTS_DIR, "evaluation_summary.md"),
        "data_audit.md": os.path.join(REPORTS_DIR, "data_audit.md"),
        "IEEE_Report.md": os.path.join(REPORTS_DIR, "IEEE_Report.md")
    }
    r_cols = st.columns(4)
    for idx, (rf_name, rf_path) in enumerate(rep_files.items()):
        with r_cols[idx]:
            if os.path.exists(rf_path):
                with open(rf_path, "r", encoding="utf-8") as f:
                    c_data = f.read()
                st.download_button(f"⬇️ {rf_name}", c_data, file_name=rf_name, mime="text/plain", use_container_width=True)
