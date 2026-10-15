"""
Evaluation and Benchmarking Script for TestPilot AI.

Evaluates all baselines and ML strategies strictly on held-out test builds.
Generates:
- reports/evaluation_results.csv
- reports/evaluation_summary.json
- reports/evaluation_summary.md
- reports/failure_discovery_curves.csv
- reports/figures/ (Charts 1 to 7)
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_dataset
from src.preprocessing import TestPilotPreprocessor
from src.model_training import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    MLPNeuralNetworkModel,
    LearningToRankModel
)
from src.evaluation import (
    evaluate_test_suite_across_builds,
    generate_failure_discovery_curve,
    compute_classification_metrics,
    STANDARD_BUDGETS_SECONDS
)
from src.visualizations import (
    plot_duration_distribution,
    plot_outcome_distribution,
    plot_data_quality,
    plot_failure_discovery_curves,
    plot_budget_comparison,
    plot_model_metrics,
    plot_feature_importance
)
from src.ranking import STRATEGY_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestPilotAI.Evaluate")

REPORTS_DIR = "reports"
MODELS_DIR = "models"


def main():
    print("=" * 70)
    print("TestPilot AI: Starting Held-Out Evaluation on Unseen Future Builds")
    print("=" * 70)

    # 1. Load Data and Splits
    test_csv_path = os.path.join("data", "processed", "test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError("data/processed/test.csv not found. Run scripts/train_model.py first.")
    test_df = pd.read_csv(test_csv_path)

    full_df = load_dataset()
    preprocessor = TestPilotPreprocessor.load(os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib"))

    # 2. Load Models
    models = {
        "logistic_regression": LogisticRegressionModel().load(os.path.join(MODELS_DIR, "logistic_regression.joblib")),
        "random_forest": RandomForestModel().load(os.path.join(MODELS_DIR, "random_forest.joblib")),
        "xgboost": XGBoostModel().load(os.path.join(MODELS_DIR, "xgboost.joblib")),
        "mlp_neural_net": MLPNeuralNetworkModel().load(os.path.join(MODELS_DIR, "mlp_neural_net.joblib")),
        "ltr_lambdamart": LearningToRankModel().load(os.path.join(MODELS_DIR, "ltr_lambdamart.joblib"))
    }

    # 3. Held-out Classification Metrics
    X_test_scaled = preprocessor.transform(test_df, scale=True)
    y_test = test_df["verdict"].values

    classification_results = {}
    for key, model in models.items():
        probs = model.predict_proba(X_test_scaled)
        p_pos = probs[:, 1] if probs.ndim == 2 else probs
        m = compute_classification_metrics(y_test, p_pos)
        classification_results[model.name] = m
        logger.info(f"Classification Held-Out [{model.name}]: PR-AUC={m['pr_auc']}, ROC-AUC={m['roc_auc']}, F1={m['f1']}")

    # 4. Evaluate Ranking Strategies on Held-Out Builds
    strategy_evaluations = []
    discovery_curves_list = []

    # Strategy 1: Baseline A - Random
    logger.info("Evaluating Baseline A: Random Ordering (5 seeds)...")
    res_rand = evaluate_test_suite_across_builds(test_df, strategy="random", num_random_seeds=5)
    res_rand["strategy_name"] = "Baseline A: Random Ordering"
    strategy_evaluations.append(res_rand)
    curve_rand = generate_failure_discovery_curve(test_df, strategy="random")
    curve_rand["strategy"] = "Random Baseline"
    discovery_curves_list.append(curve_rand)

    # Strategy 2: Baseline B - Fastest-First
    logger.info("Evaluating Baseline B: Fastest-First...")
    res_ff = evaluate_test_suite_across_builds(test_df, strategy="fastest_first")
    res_ff["strategy_name"] = "Baseline B: Fastest-First"
    strategy_evaluations.append(res_ff)
    curve_ff = generate_failure_discovery_curve(test_df, strategy="fastest_first")
    curve_ff["strategy"] = "Fastest-First"
    discovery_curves_list.append(curve_ff)

    # Strategy 3: Baseline C - Historical
    logger.info("Evaluating Baseline C: Historical Failure Prioritization...")
    res_hist = evaluate_test_suite_across_builds(test_df, strategy="historical")
    res_hist["strategy_name"] = "Baseline C: Historical Prioritization"
    strategy_evaluations.append(res_hist)
    curve_hist = generate_failure_discovery_curve(test_df, strategy="historical")
    curve_hist["strategy"] = "Historical Prioritization"
    discovery_curves_list.append(curve_hist)

    # Strategy 4: ML - Logistic Regression (Value-Density)
    logger.info("Evaluating ML: Logistic Regression (Value-Density)...")
    res_lr = evaluate_test_suite_across_builds(test_df, strategy="ml_value_density", model_obj=models["logistic_regression"], preprocessor=preprocessor)
    res_lr["strategy_name"] = "ML: Logistic Regression (Value-Density)"
    strategy_evaluations.append(res_lr)

    # Strategy 5: ML - Random Forest (Value-Density)
    logger.info("Evaluating ML: Random Forest (Value-Density)...")
    res_rf = evaluate_test_suite_across_builds(test_df, strategy="ml_value_density", model_obj=models["random_forest"], preprocessor=preprocessor)
    res_rf["strategy_name"] = "ML: Random Forest (Value-Density)"
    strategy_evaluations.append(res_rf)

    # Strategy 6: ML - XGBoost (Value-Density)
    logger.info("Evaluating ML: XGBoost (Value-Density)...")
    res_xgb = evaluate_test_suite_across_builds(test_df, strategy="ml_value_density", model_obj=models["xgboost"], preprocessor=preprocessor)
    res_xgb["strategy_name"] = "ML: XGBoost (Value-Density)"
    strategy_evaluations.append(res_xgb)
    curve_xgb = generate_failure_discovery_curve(test_df, strategy="ml_value_density", model_obj=models["xgboost"], preprocessor=preprocessor)
    curve_xgb["strategy"] = "XGBoost (Value-Density)"
    discovery_curves_list.append(curve_xgb)

    # Strategy 7: ML - XGBoost (Pure Risk)
    logger.info("Evaluating ML: XGBoost (Pure Risk)...")
    res_xgb_risk = evaluate_test_suite_across_builds(test_df, strategy="ml_risk_only", model_obj=models["xgboost"], preprocessor=preprocessor)
    res_xgb_risk["strategy_name"] = "ML: XGBoost (Pure Risk)"
    strategy_evaluations.append(res_xgb_risk)

    # Strategy 8: ML - MLP Neural Network (Value-Density)
    logger.info("Evaluating ML: MLP Neural Net (Value-Density)...")
    res_mlp = evaluate_test_suite_across_builds(test_df, strategy="ml_value_density", model_obj=models["mlp_neural_net"], preprocessor=preprocessor)
    res_mlp["strategy_name"] = "ML: MLP Neural Net (Value-Density)"
    strategy_evaluations.append(res_mlp)

    # Strategy 9: ML - Learning-to-Rank (LambdaMART)
    logger.info("Evaluating ML: Learning-to-Rank (LambdaMART)...")
    res_ltr = evaluate_test_suite_across_builds(test_df, strategy="ml_ltr", model_obj=models["ltr_lambdamart"], preprocessor=preprocessor)
    res_ltr["strategy_name"] = "ML: Learning-to-Rank (LambdaMART)"
    strategy_evaluations.append(res_ltr)
    curve_ltr = generate_failure_discovery_curve(test_df, strategy="ml_ltr", model_obj=models["ltr_lambdamart"], preprocessor=preprocessor)
    curve_ltr["strategy"] = "Learning-to-Rank (LambdaMART)"
    discovery_curves_list.append(curve_ltr)

    # 5. Compile DataFrames
    eval_df = pd.DataFrame(strategy_evaluations)
    curves_df = pd.concat(discovery_curves_list, ignore_index=True)

    # 6. Save Evaluation Output Files
    eval_csv_path = os.path.join(REPORTS_DIR, "evaluation_results.csv")
    eval_df.to_csv(eval_csv_path, index=False)
    logger.info(f"Saved evaluation results to {eval_csv_path}")

    curves_csv_path = os.path.join(REPORTS_DIR, "failure_discovery_curves.csv")
    curves_df.to_csv(curves_csv_path, index=False)
    logger.info(f"Saved failure discovery curves to {curves_csv_path}")

    summary_json_path = os.path.join(REPORTS_DIR, "evaluation_summary.json")
    summary_dict = {
        "classification_metrics": classification_results,
        "prioritization_benchmarks": strategy_evaluations,
        "evaluation_dataset": {
            "source": "Held-Out Test Builds (Chronological Split)",
            "num_test_records": len(test_df),
            "num_test_builds": test_df["build_id"].nunique(),
            "num_test_failures": int(test_df["verdict"].sum())
        }
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    # Generate Markdown Summary
    summary_md_path = os.path.join(REPORTS_DIR, "evaluation_summary.md")
    md_lines = [
        "# TestPilot AI: Rigorous Empirical Evaluation Summary",
        "",
        "## 1. Experimental Overview",
        "",
        "- **Evaluation Split**: Chronological Held-Out Test Builds (strictly unseen future builds).",
        f"- **Evaluated Test Records**: {len(test_df):,}",
        f"- **Evaluated CI Builds**: {test_df['build_id'].nunique()}",
        f"- **Ground-Truth Test Failures in Split**: {int(test_df['verdict'].sum()):,}",
        "",
        "## 2. Test Prioritization Performance Across Strategies",
        "",
        "| Prioritization Strategy | APFD | APFDc | Recall @ 5m (300s) | Recall @ 10m (600s) | Recall @ 20m (1200s) |",
        "|---|---|---|---|---|---|"
    ]
    for _, r in eval_df.iterrows():
        s_name = r["strategy_name"]
        apfd_str = f"{r['mean_apfd']:.4f} ± {r['std_apfd']:.3f}" if r["std_apfd"] > 0 else f"{r['mean_apfd']:.4f}"
        apfdc_str = f"{r['mean_apfdc']:.4f} ± {r['std_apfdc']:.3f}" if r["std_apfdc"] > 0 else f"{r['mean_apfdc']:.4f}"
        rec_5 = f"{r.get('mean_recall_300s', 0.0)*100:.1f}%"
        rec_10 = f"{r.get('mean_recall_600s', 0.0)*100:.1f}%"
        rec_20 = f"{r.get('mean_recall_1200s', 0.0)*100:.1f}%"
        md_lines.append(f"| **{s_name}** | {apfd_str} | {apfdc_str} | {rec_5} | {rec_10} | {rec_20} |")

    md_lines.extend([
        "",
        "## 3. Multi-Model Classification Metrics (Held-Out Test Set)",
        "",
        "| Model Name | Precision | Recall | F1 Score | PR-AUC | ROC-AUC | Brier Score |",
        "|---|---|---|---|---|---|---|"
    ])
    for m_name, m in classification_results.items():
        md_lines.append(f"| **{m_name}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | {m['brier_score']:.4f} |")

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved evaluation markdown summary to {summary_md_path}")

    # 7. Generate All 7 Figures
    logger.info("Generating publication-grade charts...")
    plot_duration_distribution(full_df)
    plot_outcome_distribution(full_df)
    plot_data_quality(full_df)
    plot_failure_discovery_curves(curves_df)
    plot_budget_comparison(eval_df)
    plot_model_metrics(classification_results)

    # Feature Importance from champion XGBoost model
    xgb_feat_importances = models["xgboost"].get_feature_importances()
    plot_feature_importance(preprocessor.feature_names, xgb_feat_importances, model_name="XGBoost")

    print("\n" + "=" * 70)
    print("[SUCCESS] Comprehensive Evaluation and Figure Generation Complete!")
    print(f"  Summary Markdown: {summary_md_path}")
    print(f"  Summary JSON: {summary_json_path}")
    print(f"  Figures Directory: reports/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
