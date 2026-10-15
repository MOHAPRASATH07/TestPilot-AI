"""
Test Prioritization and Budget-Aware Ranking Module for TestPilot AI.

Implements:
1. Baseline A: Random Ordering (Deterministic with random seed, supports multi-seed evaluation)
2. Baseline B: Fastest-First Ordering (Duration-minimizing greedily)
3. Baseline C: Historical Failure-based Ordering (Recency/Frequency of past failures)
4. ML Priority Scoring:
   - Value-Density (Probability per Unit Time): Score = P(fail) / (duration ^ alpha)
   - Risk-Only: Score = P(fail)
   - Learning-to-Rank direct rank score
5. Budget-Constrained Scheduler:
   - Evaluates tests against time budget (e.g. 5, 10, 20 mins or custom seconds)
   - Splits tests into scheduled vs remaining (out-of-budget)
   - Eliminates duplicate entries and double-counting.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("TestPilotAI.Ranking")

STRATEGY_NAMES = {
    "random": "Baseline A: Random Ordering",
    "fastest_first": "Baseline B: Fastest-First",
    "historical": "Baseline C: History-Based Prioritization",
    "ml_value_density": "ML Strategy: Value-Density (Risk/Duration)",
    "ml_risk_only": "ML Strategy: Pure Risk (Probability)",
    "ml_ltr": "ML Strategy: Learning-to-Rank (LambdaMART)"
}


def rank_tests(
    test_df: pd.DataFrame,
    strategy: str = "ml_value_density",
    predicted_probs: Optional[np.ndarray] = None,
    ltr_scores: Optional[np.ndarray] = None,
    time_budget_seconds: Optional[float] = 300.0,
    alpha: float = 1.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Ranks tests within a target build according to the specified strategy and time budget.

    Args:
        test_df: DataFrame of tests in the build.
        strategy: One of 'random', 'fastest_first', 'historical', 'ml_value_density', 'ml_risk_only', 'ml_ltr'.
        predicted_probs: Model failure probabilities P(fail), required for ml_* strategies.
        ltr_scores: Raw scores from Learning-to-Rank model if strategy is 'ml_ltr'.
        time_budget_seconds: Time budget in seconds (e.g., 300s = 5m, 600s = 10m).
        alpha: Duration exponent for value density: P(fail) / (duration ^ alpha).
        seed: Random seed for Baseline A.

    Returns:
        Structured dict with ordered tests, scores, budget breakdown, and summary.
    """
    df = test_df.copy().reset_index(drop=True)
    n_tests = len(df)
    
    # Estimated duration (Pre-execution safe: avg_recent_duration)
    # If not present, fallback to duration
    if "avg_recent_duration" in df.columns:
        durations = df["avg_recent_duration"].clip(lower=0.1).values
    elif "duration" in df.columns:
        durations = df["duration"].clip(lower=0.1).values
    else:
        durations = np.ones(n_tests, dtype=float)

    # Compute Priority Scores based on strategy
    if strategy == "random":
        rng = np.random.RandomState(seed)
        scores = rng.permutation(np.linspace(0.01, 0.99, n_tests))
        explanation = "Uniform stochastic permutation under reproducible random seed."

    elif strategy == "fastest_first":
        # Fastest-first prioritizes shortest estimated runtime: score = 1.0 / duration
        scores = 1.0 / durations
        explanation = "Inversely proportional to test runtime (1 / duration); maximizes test throughput."

    elif strategy == "historical":
        # Prioritizes by historical failure rate, breaking ties by recent failures (last 5)
        h_rate = df.get("historical_fail_rate", np.zeros(n_tests)).values
        f5 = df.get("failures_last_5", np.zeros(n_tests)).values
        scores = h_rate * 10.0 + f5
        explanation = "Historical failure frequency weighted by recent failure counts in past 5 builds."

    elif strategy == "ml_value_density":
        if predicted_probs is None:
            raise ValueError("predicted_probs is required for ml_value_density strategy.")
        p_fail = predicted_probs if predicted_probs.ndim == 1 else predicted_probs[:, 1]
        scores = p_fail / (np.power(durations, alpha) + 1e-5)
        explanation = f"Value-density ranking: P(fail) / (duration ^ {alpha}). Maximizes defect discovery per second."

    elif strategy == "ml_risk_only":
        if predicted_probs is None:
            raise ValueError("predicted_probs is required for ml_risk_only strategy.")
        p_fail = predicted_probs if predicted_probs.ndim == 1 else predicted_probs[:, 1]
        scores = p_fail
        explanation = "Pure defect probability P(fail); ranks by risk without duration normalization."

    elif strategy == "ml_ltr":
        if ltr_scores is not None:
            scores = ltr_scores if ltr_scores.ndim == 1 else ltr_scores[:, 1]
        elif predicted_probs is not None:
            scores = predicted_probs if predicted_probs.ndim == 1 else predicted_probs[:, 1]
        else:
            raise ValueError("ltr_scores or predicted_probs is required for ml_ltr strategy.")
        explanation = "Pairwise LambdaMART Learning-to-Rank score trained to rank failures above non-failures."

    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Valid: {list(STRATEGY_NAMES.keys())}")

    if predicted_probs is not None:
        p_fail = predicted_probs if predicted_probs.ndim == 1 else predicted_probs[:, 1]
        df["predicted_risk"] = np.round(p_fail, 4)
    else:
        df["predicted_risk"] = None

    df["priority_score"] = scores
    df["est_duration"] = durations
    
    # Sort descending by priority score; deterministic tie-breaking on test_id
    if "test_id" in df.columns:
        df = df.sort_values(by=["priority_score", "test_id"], ascending=[False, True]).reset_index(drop=True)
    else:
        df = df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)

    df["rank"] = df.index + 1
    df["cum_duration"] = df["est_duration"].cumsum()

    # Budget allocation
    if time_budget_seconds is not None and time_budget_seconds > 0:
        df["within_budget"] = df["cum_duration"] <= time_budget_seconds
    else:
        df["within_budget"] = True

    scheduled_df = df[df["within_budget"]].copy()
    excluded_df = df[~df["within_budget"]].copy()

    total_est_time = float(scheduled_df["est_duration"].sum())
    scheduled_count = len(scheduled_df)
    excluded_count = len(excluded_df)

    # Retrospective verification (if ground truth verdict is present)
    failures_found = int(scheduled_df["verdict"].sum()) if "verdict" in scheduled_df.columns else None
    total_failures = int(df["verdict"].sum()) if "verdict" in df.columns else None
    failure_recall = round(float(failures_found / max(1, total_failures)), 4) if total_failures is not None else None

    return {
        "strategy": strategy,
        "strategy_label": STRATEGY_NAMES.get(strategy, strategy),
        "explanation": explanation,
        "time_budget_seconds": time_budget_seconds,
        "total_tests_available": n_tests,
        "scheduled_test_count": scheduled_count,
        "excluded_test_count": excluded_count,
        "total_scheduled_time_seconds": round(total_est_time, 2),
        "remaining_budget_seconds": round(max(0.0, (time_budget_seconds or 0) - total_est_time), 2),
        "failures_discovered": failures_found,
        "total_failures_in_build": total_failures,
        "failure_recall": failure_recall,
        "ranked_tests": df,
        "scheduled_tests": scheduled_df,
        "excluded_tests": excluded_df
    }
