"""
Visualization Engine for TestPilot AI.

Generates at least seven publication-grade charts using Plotly and Matplotlib:
1. Chart 1: Test execution duration distribution.
2. Chart 2: Historical test outcome / verdict distribution.
3. Chart 3: Dataset quality and feature completeness.
4. Chart 4: Cumulative failure discovery curves over elapsed testing time.
5. Chart 5: Strategy comparison under 5m, 10m, and 20m budgets.
6. Chart 6: ML model classification benchmark metrics (PR-AUC, F1, Recall).
7. Chart 7: Feature importance breakdown from tree ensembles.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

logger = logging.getLogger("TestPilotAI.Visualizations")
FIGURES_DIR = os.path.join("reports", "figures")


def ensure_fig_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig: go.Figure, name: str):
    """Saves figure as both PNG and HTML."""
    fig.write_image(os.path.join(FIGURES_DIR, f"{name}.png"), width=1200, height=650, scale=2)
    fig.write_html(os.path.join(FIGURES_DIR, f"{name}.html"))


def plot_duration_distribution(df: pd.DataFrame) -> go.Figure:
    """Chart 1: Test execution duration distribution."""
    ensure_fig_dir()
    durs = df["duration"].clip(upper=float(df["duration"].quantile(0.99)))
    
    fig = px.histogram(
        durs,
        x="duration",
        nbins=50,
        marginal="box",
        title="Chart 1: Empirical Distribution of Test Execution Durations",
        labels={"duration": "Execution Duration (seconds)"},
        color_discrete_sequence=["#1f77b4"]
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Execution Duration (seconds)",
        yaxis_title="Count of Test Cases",
        annotations=[
            dict(
                x=float(durs.median()),
                y=0,
                text=f"Median: {durs.median():.2f}s",
                showarrow=True,
                arrowhead=2
            )
        ]
    )
    _save(fig, "chart1_duration_distribution")
    return fig


def plot_outcome_distribution(df: pd.DataFrame) -> go.Figure:
    """Chart 2: Test outcome and class imbalance distribution."""
    ensure_fig_dir()
    counts = df["verdict"].value_counts().reset_index()
    counts.columns = ["Verdict", "Count"]
    counts["Verdict_Label"] = counts["Verdict"].map({0: "Passed (0)", 1: "Failed (1)"})
    total = len(df)
    counts["Percentage"] = (counts["Count"] / total * 100).round(2)
    
    fig = px.bar(
        counts,
        x="Verdict_Label",
        y="Count",
        text=counts.apply(lambda r: f"{r['Count']:,} ({r['Percentage']}%)", axis=1),
        title="Chart 2: Test Outcome Class Distribution (Ground-Truth Imbalance)",
        color="Verdict_Label",
        color_discrete_map={"Passed (0)": "#2ca02c", "Failed (1)": "#d62728"}
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Observed Test Verdict",
        yaxis_title="Number of Test Executions",
        showlegend=False
    )
    _save(fig, "chart2_outcome_distribution")
    return fig


def plot_data_quality(df: pd.DataFrame) -> go.Figure:
    """Chart 3: Feature completeness and data quality summary."""
    ensure_fig_dir()
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Feature", "MissingCount"]
    missing["Completeness_Pct"] = ((len(df) - missing["MissingCount"]) / len(df) * 100).round(2)
    
    fig = px.bar(
        missing,
        x="Feature",
        y="Completeness_Pct",
        title="Chart 3: Dataset Quality and Feature Completeness Matrix",
        color="Completeness_Pct",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Dataset Columns",
        yaxis_title="Completeness (%)",
        yaxis_range=[0, 105]
    )
    _save(fig, "chart3_data_quality")
    return fig


def plot_failure_discovery_curves(curves_df: pd.DataFrame) -> go.Figure:
    """Chart 4: Cumulative failure discovery over elapsed testing time."""
    ensure_fig_dir()
    
    fig = px.line(
        curves_df,
        x="elapsed_time_minutes",
        y="percentage_failures_discovered",
        color="strategy",
        title="Chart 4: Cumulative Defect Discovery Curves Over Elapsed Regression Testing Time",
        labels={
            "elapsed_time_minutes": "Elapsed Testing Time (minutes)",
            "percentage_failures_discovered": "Cumulative Known Failures Discovered (%)",
            "strategy": "Prioritization Strategy"
        }
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Elapsed Testing Time (minutes)",
        yaxis_title="Cumulative Failures Discovered (%)",
        yaxis_range=[0, 102]
    )
    _save(fig, "chart4_failure_discovery_curves")
    return fig


def plot_budget_comparison(eval_df: pd.DataFrame) -> go.Figure:
    """Chart 5: Comparative strategy failure recall under 5m, 10m, and 20m budgets."""
    ensure_fig_dir()
    
    budgets = [300, 600, 1200]
    plot_rows = []
    
    for _, row in eval_df.iterrows():
        strat = row.get("strategy", "Unknown")
        for b in budgets:
            col_name = f"mean_recall_{b}s"
            if col_name in row:
                plot_rows.append({
                    "Strategy": strat,
                    "Budget": f"{int(b/60)} Minutes ({b}s)",
                    "Recall_Pct": round(row[col_name] * 100, 2)
                })
                
    b_df = pd.DataFrame(plot_rows)
    if b_df.empty:
        return go.Figure()
        
    fig = px.bar(
        b_df,
        x="Strategy",
        y="Recall_Pct",
        color="Budget",
        barmode="group",
        title="Chart 5: Defect Recall Under Standard Fixed Testing-Time Budgets",
        labels={"Recall_Pct": "Failures Discovered (%)", "Strategy": "Strategy"}
    )
    fig.update_layout(
        template="plotly_white",
        yaxis_title="Average Failure Recall (%)",
        yaxis_range=[0, 105]
    )
    _save(fig, "chart5_budget_comparison")
    return fig


def plot_model_metrics(metrics_dict: Dict[str, Dict[str, float]]) -> go.Figure:
    """Chart 6: ML model evaluation metrics (PR-AUC, F1, Recall, ROC-AUC)."""
    ensure_fig_dir()
    rows = []
    for m_name, m_vals in metrics_dict.items():
        for metric in ["pr_auc", "f1", "recall", "roc_auc"]:
            if metric in m_vals:
                rows.append({"Model": m_name, "Metric": metric.upper(), "Value": m_vals[metric]})
                
    m_df = pd.DataFrame(rows)
    if m_df.empty:
        return go.Figure()
        
    fig = px.bar(
        m_df,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="Chart 6: Multi-Model Machine Learning Classification Benchmark",
        labels={"Value": "Metric Score [0, 1]"}
    )
    fig.update_layout(
        template="plotly_white",
        yaxis_range=[0, 1.05]
    )
    _save(fig, "chart6_model_metrics")
    return fig


def plot_feature_importance(feature_names: List[str], importances: np.ndarray, model_name: str = "XGBoost") -> go.Figure:
    """Chart 7: Feature importance from tree ensemble."""
    ensure_fig_dir()
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=True).tail(15)
    
    fig = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Chart 7: Relative Feature Importance ({model_name})",
        labels={"Importance": "Relative Gini / Split Gain"}
    )
    fig.update_layout(template="plotly_white")
    _save(fig, "chart7_feature_importance")
    return fig
