"""
Evaluation and Benchmarking Engine for TestPilot AI.

Implements rigorous, leakage-safe empirical software engineering metrics:
1. Classification Metrics: Precision, Recall, F1, ROC-AUC, Average Precision (PR-AUC), Brier Score.
2. Ranking Metrics:
   - APFD (Average Percentage of Faults Detected)
   - APFDc (Cost-Cognizant Average Percentage of Faults Detected)
   - Time-to-First-Failure (TTFF)
   - Failures Discovered @ Budget (5 min, 10 min, 20 min)
   - Recall @ Budget
3. Multi-seed Variance for Random Baselines.
4. Export to CSV, JSON, and Markdown summaries.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)
from src.ranking import rank_tests

logger = logging.getLogger("TestPilotAI.Evaluation")

STANDARD_BUDGETS_SECONDS = [300.0, 600.0, 1200.0]  # 5 min, 10 min, 20 min


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes standard binary classification performance metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    # Check if both classes are present in y_true
    has_pos = (y_true == 1).any()
    has_neg = (y_true == 0).any()
    
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    
    roc_auc = float(roc_auc_score(y_true, y_prob)) if (has_pos and has_neg) else 0.5
    avg_prec = float(average_precision_score(y_true, y_prob)) if has_pos else 0.0
    
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(avg_prec, 4),
        "brier_score": round(brier, 4)
    }


def compute_apfd(ranked_verdicts: List[int]) -> float:
    """
    Computes Average Percentage of Faults Detected (APFD).
    APFD = 1 - (sum(TF_i) / (n * m)) + (1 / (2 * n))
    where:
      n = total number of tests
      m = total number of failing tests
      TF_i = rank of the test that detects fault i
    """
    n = len(ranked_verdicts)
    if n == 0:
        return 0.0
    fault_indices = [idx + 1 for idx, v in enumerate(ranked_verdicts) if v == 1]
    m = len(fault_indices)
    if m == 0:
        return 1.0  # No faults to detect
    apfd = 1.0 - (sum(fault_indices) / (n * m)) + (1.0 / (2.0 * n))
    return round(float(np.clip(apfd, 0.0, 1.0)), 4)


def compute_apfdc(ranked_verdicts: List[int], ranked_durations: List[float]) -> float:
    """
    Computes Cost-Cognizant APFD (APFDc) where tests have varying execution durations.
    APFDc = sum_{i=1}^m ( sum_{j=TF_i}^n t_j - 0.5 * t_{TF_i} ) / ( m * sum_{j=1}^n t_j )
    """
    n = len(ranked_verdicts)
    if n == 0:
        return 0.0
    m = sum(ranked_verdicts)
    if m == 0:
        return 1.0
        
    total_cost = sum(ranked_durations)
    if total_cost <= 0:
        return compute_apfd(ranked_verdicts)
        
    cum_cost_from_end = np.cumsum(ranked_durations[::-1])[::-1]
    
    numerator = 0.0
    for idx, v in enumerate(ranked_verdicts):
        if v == 1:
            t_curr = ranked_durations[idx]
            cost_remaining = cum_cost_from_end[idx]
            numerator += (cost_remaining - 0.5 * t_curr)
            
    apfdc = numerator / (m * total_cost)
    return round(float(np.clip(apfdc, 0.0, 1.0)), 4)


def evaluate_build(
    build_df: pd.DataFrame,
    strategy: str,
    predicted_probs: Optional[np.ndarray] = None,
    ltr_scores: Optional[np.ndarray] = None,
    budgets: List[float] = STANDARD_BUDGETS_SECONDS,
    seed: int = 42
) -> Dict[str, Any]:
    """Evaluates a single build under a given test prioritization strategy."""
    ranked_res = rank_tests(
        test_df=build_df,
        strategy=strategy,
        predicted_probs=predicted_probs,
        ltr_scores=ltr_scores,
        time_budget_seconds=None,  # Full sequence to measure APFD
        seed=seed
    )
    
    df_ranked = ranked_res["ranked_tests"]
    verdicts = df_ranked["verdict"].tolist()
    durations = df_ranked["est_duration"].tolist()
    
    total_tests = len(df_ranked)
    total_failures = sum(verdicts)
    
    apfd = compute_apfd(verdicts)
    apfdc = compute_apfdc(verdicts, durations)
    
    # Time to first failure
    ttff = None
    cum_time = 0.0
    for idx, v in enumerate(verdicts):
        cum_time += durations[idx]
        if v == 1:
            ttff = cum_time
            break

    # Budget-specific evaluations
    budget_metrics = {}
    for b in budgets:
        sub_df = df_ranked[df_ranked["cum_duration"] <= b]
        sub_fails = int(sub_df["verdict"].sum())
        sub_tests = len(sub_df)
        sub_time = float(sub_df["est_duration"].sum())
        recall_at_b = float(sub_fails / max(1, total_failures)) if total_failures > 0 else 1.0
        
        budget_metrics[f"fails_at_{int(b)}s"] = sub_fails
        budget_metrics[f"recall_at_{int(b)}s"] = round(recall_at_b, 4)
        budget_metrics[f"tests_at_{int(b)}s"] = sub_tests
        budget_metrics[f"time_at_{int(b)}s"] = round(sub_time, 2)
        
    return {
        "apfd": apfd,
        "apfdc": apfdc,
        "ttff_seconds": round(ttff, 2) if ttff is not None else None,
        "total_failures": total_failures,
        "total_tests": total_tests,
        "budget_metrics": budget_metrics,
        "ranked_df": df_ranked
    }


def evaluate_test_suite_across_builds(
    test_df: pd.DataFrame,
    strategy: str,
    model_obj=None,
    preprocessor=None,
    budgets: List[float] = STANDARD_BUDGETS_SECONDS,
    num_random_seeds: int = 5
) -> Dict[str, Any]:
    """
    Evaluates prioritization across all held-out builds independently.
    Averages APFD, APFDc, and budget metrics across builds with failures.
    """
    build_groups = list(test_df.groupby(["project_id", "build_id"]))
    build_results = []
    
    # Stochastic Baseline A handles multiple seeds
    seeds_to_run = list(range(42, 42 + num_random_seeds)) if strategy == "random" else [42]
    
    seed_apfds = []
    seed_apfdcs = []
    seed_budget_recalls = {b: [] for b in budgets}
    
    for seed in seeds_to_run:
        build_apfds = []
        build_apfdcs = []
        b_recalls = {b: [] for b in budgets}
        b_fails = {b: [] for b in budgets}
        
        for (proj, build_id), b_df in build_groups:
            # Skip builds with zero failures when evaluating prioritization discovery
            n_fails = b_df["verdict"].sum()
            if n_fails == 0:
                continue
                
            probs = None
            ltr_scores = None
            if model_obj is not None and preprocessor is not None:
                X_scaled = preprocessor.transform(b_df, scale=True)
                if strategy == "ml_ltr":
                    ltr_scores = model_obj.predict_proba(X_scaled)
                else:
                    probs = model_obj.predict_proba(X_scaled)
                    
            res = evaluate_build(
                build_df=b_df,
                strategy=strategy,
                predicted_probs=probs,
                ltr_scores=ltr_scores,
                budgets=budgets,
                seed=seed
            )
            
            build_apfds.append(res["apfd"])
            build_apfdcs.append(res["apfdc"])
            for b in budgets:
                b_recalls[b].append(res["budget_metrics"][f"recall_at_{int(b)}s"])
                b_fails[b].append(res["budget_metrics"][f"fails_at_{int(b)}s"])
                
        seed_apfds.append(np.mean(build_apfds))
        seed_apfdcs.append(np.mean(build_apfdcs))
        for b in budgets:
            seed_budget_recalls[b].append(np.mean(b_recalls[b]))

    summary = {
        "strategy": strategy,
        "mean_apfd": round(float(np.mean(seed_apfds)), 4),
        "std_apfd": round(float(np.std(seed_apfds)), 4) if len(seed_apfds) > 1 else 0.0,
        "mean_apfdc": round(float(np.mean(seed_apfdcs)), 4),
        "std_apfdc": round(float(np.std(seed_apfdcs)), 4) if len(seed_apfdcs) > 1 else 0.0,
    }
    
    for b in budgets:
        summary[f"mean_recall_{int(b)}s"] = round(float(np.mean(seed_budget_recalls[b])), 4)
        summary[f"std_recall_{int(b)}s"] = round(float(np.std(seed_budget_recalls[b])), 4) if len(seeds_to_run) > 1 else 0.0
        
    return summary


def generate_failure_discovery_curve(
    test_df: pd.DataFrame,
    strategy: str,
    model_obj=None,
    preprocessor=None,
    seed: int = 42,
    max_time_seconds: float = 1200.0,
    step_seconds: float = 30.0
) -> pd.DataFrame:
    """Generates cumulative failure discovery curve over elapsed testing time."""
    build_groups = list(test_df.groupby(["project_id", "build_id"]))
    time_points = np.arange(step_seconds, max_time_seconds + step_seconds, step_seconds)
    
    total_failures_in_eval = 0
    failures_at_time = {t: 0 for t in time_points}
    
    for (proj, build_id), b_df in build_groups:
        n_fails = b_df["verdict"].sum()
        if n_fails == 0:
            continue
        total_failures_in_eval += n_fails
        
        probs = None
        ltr_scores = None
        if model_obj is not None and preprocessor is not None:
            X_scaled = preprocessor.transform(b_df, scale=True)
            if strategy == "ml_ltr":
                ltr_scores = model_obj.predict_proba(X_scaled)
            else:
                probs = model_obj.predict_proba(X_scaled)
                
        res = rank_tests(
            test_df=b_df,
            strategy=strategy,
            predicted_probs=probs,
            ltr_scores=ltr_scores,
            time_budget_seconds=None,
            seed=seed
        )
        ranked = res["ranked_tests"]
        
        for t in time_points:
            sub = ranked[ranked["cum_duration"] <= t]
            failures_at_time[t] += int(sub["verdict"].sum())
            
    records = []
    for t in time_points:
        count = failures_at_time[t]
        pct = (count / max(1, total_failures_in_eval)) * 100.0
        records.append({
            "strategy": strategy,
            "elapsed_time_seconds": t,
            "elapsed_time_minutes": round(t / 60.0, 2),
            "cumulative_failures_discovered": count,
            "percentage_failures_discovered": round(pct, 2)
        })
        
    return pd.DataFrame(records)
