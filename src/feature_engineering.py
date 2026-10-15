"""
Feature Engineering Module for TestPilot AI.

Generates pre-execution composite interaction and domain features
strictly without accessing post-execution targets or future data.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("TestPilotAI.FeatureEngineering")


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes pre-execution interaction and risk-density features.
    
    Safe transformations based on historical testing and code churn theory:
    - Code churn magnitude (Additions + Deletions)
    - Modified test interaction with churn
    - Proximity-weighted failure probability
    - Methods per class coverage density
    - Historical failure trend (short-term vs long-term)
    - Historical failure per duration unit
    """
    df = df.copy()
    
    # 1. Total Code Churn
    additions = df.get("code_churn_additions", 0)
    deletions = df.get("code_churn_deletions", 0)
    df["total_code_churn"] = additions + deletions
    
    # 2. Test Churn Interaction
    modified_test = df.get("modified_test_file", 0)
    df["churn_test_modified"] = df["total_code_churn"] * modified_test
    
    # 3. Proximity-Weighted Risk
    dist = df.get("file_test_distance", 0.5)
    hist_fail = df.get("historical_fail_rate", 0.0)
    df["proximity_risk_score"] = (1.0 - dist) * (hist_fail + 0.05)
    
    # 4. Coverage Density
    cov_c = df.get("cov_class_count", 1)
    cov_m = df.get("cov_method_count", 1)
    df["methods_per_class"] = cov_m / (cov_c + 1.0)
    
    # 5. Failure Recency Trend (Last 5 builds vs Last 10 builds)
    f5 = df.get("failures_last_5", 0)
    f10 = df.get("failures_last_10", 0)
    df["failure_recency_trend"] = f5 / ((f10 / 2.0) + 1.0)
    
    # 6. Cost-Normalized Historical Risk
    dur = df.get("avg_recent_duration", 1.0)
    df["fail_rate_per_duration"] = hist_fail / (dur + 0.1)
    
    return df
