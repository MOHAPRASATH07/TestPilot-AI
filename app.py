"""
TestPilot AI — Gradio Interface for Hugging Face Spaces.
Budget-Aware ML for Risk-Driven CI Test Prioritisation.
"""

import os
import sys
import io
import json
import logging
import pandas as pd
import numpy as np
import gradio as gr
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing import TestPilotPreprocessor
from src.ranking import rank_tests, STRATEGY_NAMES
from src.evaluation import compute_apfd, compute_apfdc
from src.model_training import (
    LogisticRegressionModel, RandomForestModel, XGBoostModel,
    MLPNeuralNetworkModel, LearningToRankModel
)

logging.basicConfig(level=logging.INFO)

MODELS_DIR = "models"
DATA_PATH  = os.path.join("data", "benchmark_tcp_ci.csv")
REPORTS_DIR = "reports"

# ── Singletons ────────────────────────────────────────────────────────────────

def _load_resources():
    prep, models, bench = None, {}, None

    prep_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
    if os.path.exists(prep_path):
        try:
            prep = TestPilotPreprocessor.load(prep_path)
        except Exception as e:
            logging.warning(f"Preprocessor load failed: {e}")

    ctors = {
        "xgboost":             XGBoostModel,
        "random_forest":       RandomForestModel,
        "logistic_regression": LogisticRegressionModel,
        "mlp_neural_net":      MLPNeuralNetworkModel,
        "ltr_lambdamart":      LearningToRankModel,
    }
    for key, ctor in ctors.items():
        path = os.path.join(MODELS_DIR, f"{key}.joblib")
        if os.path.exists(path):
            try:
                models[key] = ctor().load(path)
            except Exception as e:
                logging.warning(f"Model {key} load failed: {e}")

    if os.path.exists(DATA_PATH):
        bench = pd.read_csv(DATA_PATH)

    return prep, models, bench


PREP, MODELS, BENCH_DF = _load_resources()

MODEL_LABELS = {
    "xgboost":             "XGBoost (Champion)",
    "random_forest":       "Random Forest",
    "logistic_regression": "Logistic Regression",
    "mlp_neural_net":      "MLP Neural Net",
    "ltr_lambdamart":      "LambdaMART (LTR)",
}

BUDGET_OPTIONS = {
    "5 Minutes  (300 s)":  300.0,
    "10 Minutes (600 s)":  600.0,
    "20 Minutes (1200 s)": 1200.0,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_projects():
    if BENCH_DF is None:
        return []
    return sorted(BENCH_DF["project_id"].unique().tolist())


def _get_builds(project):
    if BENCH_DF is None or not project:
        return gr.update(choices=[], value=None)
    builds = sorted(BENCH_DF[BENCH_DF["project_id"] == project]["build_id"].unique().tolist())
    return gr.update(choices=[str(b) for b in builds], value=str(builds[-1]) if builds else None)


def _run_prioritize(source, project, build_id, csv_file,
                    budget_label, custom_budget,
                    strategy_key, model_key, alpha):

    # ── Resolve test dataframe ────────────────────────────────────────────────
    if source == "Benchmark (TCP-CI)":
        if BENCH_DF is None:
            return "❌ Benchmark dataset not found.", None, None
        if not project or not build_id:
            return "❌ Select a project and build.", None, None
        test_df = BENCH_DF[
            (BENCH_DF["project_id"] == project) &
            (BENCH_DF["build_id"] == int(build_id))
        ].copy().reset_index(drop=True)
        if test_df.empty:
            return "❌ No tests found for that build.", None, None
    else:
        if csv_file is None:
            return "❌ Please upload a CSV file.", None, None
        try:
            test_df = pd.read_csv(csv_file.name)
        except Exception as e:
            return f"❌ CSV parse error: {e}", None, None
        for col in ["test_id", "avg_recent_duration"]:
            if col not in test_df.columns:
                return f"❌ Missing required column: `{col}`", None, None

    # ── Resolve budget ────────────────────────────────────────────────────────
    budget = BUDGET_OPTIONS.get(budget_label, float(custom_budget or 300))

    # ── ML inference ─────────────────────────────────────────────────────────
    probs, ltr_scores = None, None
    if strategy_key.startswith("ml_"):
        if PREP is None:
            return "❌ Preprocessing pipeline not found. Run `scripts/train_model.py` first.", None, None
        target_key = "ltr_lambdamart" if strategy_key == "ml_ltr" else model_key
        if target_key not in MODELS:
            return f"❌ Model `{target_key}` not loaded.", None, None
        try:
            X = PREP.transform(test_df, scale=True)
            if strategy_key == "ml_ltr":
                ltr_scores = MODELS[target_key].predict_proba(X)
            else:
                probs = MODELS[target_key].predict_proba(X)
        except Exception as e:
            return f"❌ Inference error: {e}", None, None

    # ── Rank ──────────────────────────────────────────────────────────────────
    try:
        res = rank_tests(
            test_df=test_df,
            strategy=strategy_key,
            predicted_probs=probs,
            ltr_scores=ltr_scores,
            time_budget_seconds=budget,
            alpha=float(alpha),
            seed=42,
        )
    except Exception as e:
        return f"❌ Ranking error: {e}", None, None

    ranked = res["ranked_tests"]
    has_v  = "verdict" in ranked.columns and not ranked["verdict"].isnull().all()

    # ── Summary text ──────────────────────────────────────────────────────────
    recall_str = ""
    if has_v:
        rec = (res["failure_recall"] or 0) * 100
        recall_str = f"  |  Failures detected: **{res['failures_discovered']} / {res['total_failures_in_build']}** ({rec:.1f}% recall)"

    summary = (
        f"**Strategy:** {res['strategy_label']}  \n"
        f"**Budget:** {budget:.0f} s  |  "
        f"**Scheduled:** {res['scheduled_test_count']} / {res['total_tests_available']} tests  |  "
        f"**Runtime:** {res['total_scheduled_time_seconds']:.1f} s"
        f"{recall_str}"
    )

    # ── Table ─────────────────────────────────────────────────────────────────
    cols = ["rank", "test_id", "priority_score", "est_duration", "cum_duration", "within_budget"]
    if "predicted_risk" in ranked.columns and ranked["predicted_risk"].notnull().any():
        cols.insert(3, "predicted_risk")
    if has_v:
        cols.append("verdict")
    table = ranked[cols].copy()
    table["priority_score"] = table["priority_score"].round(5)
    table["est_duration"]   = table["est_duration"].round(2)
    table["cum_duration"]   = table["cum_duration"].round(2)

    # ── Discovery curve (if verdicts available) ───────────────────────────────
    fig = None
    if has_v:
        cum_fails = ranked["verdict"].cumsum().tolist()
        cum_time  = ranked["cum_duration"].tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum_time, y=cum_fails,
            mode="lines", name=res["strategy_label"],
            line=dict(color="#38bdf8", width=2)
        ))
        fig.add_vline(x=budget, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Budget ({budget:.0f}s)")
        fig.update_layout(
            title="Failure Discovery Curve",
            xaxis_title="Elapsed Test Runtime (s)",
            yaxis_title="Cumulative Failures Found",
            template="plotly_dark",
            paper_bgcolor="#1e293b",
            plot_bgcolor="#0f172a",
            font=dict(color="#f8fafc"),
            margin=dict(l=40, r=20, t=40, b=40),
        )

    return summary, table, fig


def _run_comparison(project, build_id, budget_label, custom_budget):
    if BENCH_DF is None:
        return "❌ Benchmark dataset not found.", None, None
    if not project or not build_id:
        return "❌ Select a project and build.", None, None

    test_df = BENCH_DF[
        (BENCH_DF["project_id"] == project) &
        (BENCH_DF["build_id"] == int(build_id))
    ].copy().reset_index(drop=True)

    if test_df.empty:
        return "❌ No tests found for that build.", None, None

    budget = BUDGET_OPTIONS.get(budget_label, float(custom_budget or 300))
    has_v  = "verdict" in test_df.columns and not test_df["verdict"].isnull().all()

    strategies = [
        ("random",          None,  None),
        ("fastest_first",   None,  None),
        ("historical",      None,  None),
    ]
    if PREP is not None:
        try:
            X = PREP.transform(test_df, scale=True)
            if "xgboost" in MODELS:
                p = MODELS["xgboost"].predict_proba(X)
                strategies += [
                    ("ml_value_density", p,    None),
                    ("ml_risk_only",     p,    None),
                ]
            if "ltr_lambdamart" in MODELS:
                l = MODELS["ltr_lambdamart"].predict_proba(X)
                strategies.append(("ml_ltr", None, l))
        except Exception as e:
            logging.warning(f"Comparison inference error: {e}")

    rows, fig = [], go.Figure()
    colors = ["#38bdf8","#34d399","#f59e0b","#a78bfa","#f472b6","#fb923c"]

    for i, (strat, p, l) in enumerate(strategies):
        res = rank_tests(test_df=test_df, strategy=strat, predicted_probs=p,
                         ltr_scores=l, time_budget_seconds=budget, seed=42)
        ranked = res["ranked_tests"]
        v_list = ranked["verdict"].tolist() if has_v else []
        d_list = ranked["est_duration"].tolist()

        apfd  = compute_apfd(v_list)  if has_v else None
        apfdc = compute_apfdc(v_list, d_list) if has_v else None

        rows.append({
            "Strategy":        STRATEGY_NAMES[strat],
            "Scheduled":       f"{res['scheduled_test_count']} / {res['total_tests_available']}",
            "Runtime (s)":     round(res["total_scheduled_time_seconds"], 1),
            "Failures Found":  f"{res['failures_discovered']} / {res['total_failures_in_build']}" if has_v else "N/A",
            "Recall @ Budget": f"{(res['failure_recall'] or 0)*100:.1f}%" if has_v else "N/A",
            "APFD":            f"{apfd:.4f}" if apfd is not None else "N/A",
            "APFDc":           f"{apfdc:.4f}" if apfdc is not None else "N/A",
        })

        if has_v:
            cum_f = ranked["verdict"].cumsum().tolist()
            fig.add_trace(go.Scatter(
                x=ranked["cum_duration"].tolist(), y=cum_f,
                mode="lines", name=STRATEGY_NAMES[strat],
                line=dict(color=colors[i % len(colors)], width=2)
            ))

    if has_v:
        fig.add_vline(x=budget, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Budget ({budget:.0f}s)")
        fig.update_layout(
            title="Failure Discovery Curves — All Strategies",
            xaxis_title="Elapsed Test Runtime (s)",
            yaxis_title="Cumulative Failures Found",
            template="plotly_dark",
            paper_bgcolor="#1e293b",
            plot_bgcolor="#0f172a",
            font=dict(color="#f8fafc"),
            margin=dict(l=40, r=20, t=40, b=40),
        )
    else:
        fig = None

    return f"Compared {len(strategies)} strategies on {len(test_df)} tests.", pd.DataFrame(rows), fig


# ── UI ────────────────────────────────────────────────────────────────────────

projects = _get_projects()
default_project = projects[0] if projects else None
default_builds  = (
    sorted(BENCH_DF[BENCH_DF["project_id"] == default_project]["build_id"].unique().tolist())
    if default_project and BENCH_DF is not None else []
)
default_build = str(default_builds[-1]) if default_builds else None

CSS = """
body, .gradio-container { background: #0f172a !important; color: #f8fafc !important; }
.gr-button-primary { background: #0284c7 !important; border-color: #38bdf8 !important; }
"""

with gr.Blocks(title="TestPilot AI") as demo:

    gr.Markdown("""
# ⚡ TestPilot AI
**Budget-Aware Machine Learning for Risk-Driven CI Test Prioritisation**

Prioritize your regression test suite intelligently under time constraints using ML-powered value-density scoring.
""")

    # ── Shared build selector (reused across tabs) ────────────────────────────
    with gr.Tab("⚡ Prioritise Tests"):

        with gr.Row():
            source_radio = gr.Radio(
                ["Benchmark (TCP-CI)", "Upload CSV"],
                value="Benchmark (TCP-CI)", label="Input Source"
            )

        with gr.Row():
            proj_dd   = gr.Dropdown(projects, value=default_project, label="Project")
            build_dd  = gr.Dropdown(
                [str(b) for b in default_builds], value=default_build, label="Build ID"
            )
            csv_upload = gr.File(label="Upload CSV", file_types=[".csv"], visible=False)

        source_radio.change(
            fn=lambda s: (
                gr.update(visible=s == "Benchmark (TCP-CI)"),
                gr.update(visible=s == "Benchmark (TCP-CI)"),
                gr.update(visible=s == "Upload CSV"),
            ),
            inputs=source_radio,
            outputs=[proj_dd, build_dd, csv_upload]
        )
        proj_dd.change(fn=_get_builds, inputs=proj_dd, outputs=build_dd)

        with gr.Row():
            budget_dd = gr.Dropdown(
                list(BUDGET_OPTIONS.keys()), value="5 Minutes  (300 s)", label="Time Budget"
            )
            custom_budget = gr.Number(value=300, label="Custom Budget (s)", visible=False)

        budget_dd.change(
            fn=lambda v: gr.update(visible=(v not in BUDGET_OPTIONS)),
            inputs=budget_dd, outputs=custom_budget
        )

        with gr.Row():
            strategy_dd = gr.Dropdown(
                list(STRATEGY_NAMES.keys()),
                value="ml_value_density",
                label="Ranking Strategy",
                info="Value-Density = P(fail) / duration^α"
            )
            model_dd = gr.Dropdown(
                list(MODEL_LABELS.keys()),
                value="xgboost",
                label="ML Model",
            )
            alpha_sl = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Alpha (duration exponent)")

        run_btn = gr.Button("⚡ Prioritise Tests", variant="primary")

        summary_out = gr.Markdown()
        table_out   = gr.Dataframe(label="Prioritised Schedule", wrap=True)
        curve_out   = gr.Plot(label="Failure Discovery Curve")

        run_btn.click(
            fn=_run_prioritize,
            inputs=[source_radio, proj_dd, build_dd, csv_upload,
                    budget_dd, custom_budget, strategy_dd, model_dd, alpha_sl],
            outputs=[summary_out, table_out, curve_out]
        )

    # ── Tab 2: Strategy Comparison ────────────────────────────────────────────
    with gr.Tab("📊 Strategy Comparison"):

        gr.Markdown("Compare all 6 strategies side-by-side on a single benchmark build.")

        with gr.Row():
            c_proj  = gr.Dropdown(projects, value=default_project, label="Project")
            c_build = gr.Dropdown(
                [str(b) for b in default_builds], value=default_build, label="Build ID"
            )

        c_proj.change(fn=_get_builds, inputs=c_proj, outputs=c_build)

        with gr.Row():
            c_budget = gr.Dropdown(
                list(BUDGET_OPTIONS.keys()), value="5 Minutes  (300 s)", label="Time Budget"
            )
            c_custom = gr.Number(value=300, label="Custom Budget (s)", visible=False)

        c_budget.change(
            fn=lambda v: gr.update(visible=(v not in BUDGET_OPTIONS)),
            inputs=c_budget, outputs=c_custom
        )

        cmp_btn = gr.Button("⚡ Run Comparison", variant="primary")

        cmp_status = gr.Markdown()
        cmp_table  = gr.Dataframe(label="Strategy Comparison Results", wrap=True)
        cmp_curve  = gr.Plot(label="Failure Discovery Curves")

        cmp_btn.click(
            fn=_run_comparison,
            inputs=[c_proj, c_build, c_budget, c_custom],
            outputs=[cmp_status, cmp_table, cmp_curve]
        )

    # ── Tab 3: About ──────────────────────────────────────────────────────────
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
## Methodology

**Value-Density Score:**
$$\\text{Score}(t_i) = \\frac{\\hat{P}(\\text{fail}_i)}{\\hat{d}_i^{\\alpha}}$$

Prioritizes high-risk, fast-running tests first — maximizing defect discovery per second of CI budget.

**Metrics:**
- **APFD** — Average Percentage of Faults Detected (order quality)
- **APFDc** — Cost-Cognizant APFD (accounts for test duration)
- **Recall @ Budget** — Fraction of failures found within the time limit

## Benchmark Results (Held-Out, 12 Builds)

| Strategy | APFD | APFDc | Recall @ 5m |
|---|---|---|---|
| Random | 0.5153 | 0.5326 | 68.2% |
| Fastest-First | 0.5162 | 0.7390 | 91.3% |
| Historical | 0.6221 | 0.5930 | 67.6% |
| XGBoost Value-Density | 0.5243 | 0.7351 | 90.6% |
| **MLP Value-Density** | 0.5481 | 0.7430 | **94.2%** |
| LambdaMART LTR | 0.5718 | 0.5913 | 75.0% |

## Dataset
TCP-CI benchmark — *Yaraghi et al., IEEE TSE 2023* — 8,958 test executions across 239 CI builds.
Zenodo DOI: [10.5281/zenodo.6415365](https://zenodo.org/records/6415365)
""")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(), css=CSS)
