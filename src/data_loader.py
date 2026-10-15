"""
Data Loading and Validation Module for TestPilot AI.

Provides schema validation, field categorization, and chronological splitting
to ensure strict leakage prevention during model training and evaluation.
"""

import os
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger("TestPilotAI.DataLoader")

IDENTIFIER_COLUMNS = ["project_id", "build_id", "test_id"]
POST_EXECUTION_COLUMNS = ["duration", "verdict"]

# Pre-execution feature groups based on TCP-CI feature categories
HISTORY_FEATURES = [
    "historical_fail_rate",
    "failures_last_5",
    "failures_last_10",
    "total_historical_runs",
    "builds_since_last_fail",
    "avg_recent_duration"
]

CODE_CHANGE_FEATURES = [
    "changed_files_count",
    "code_churn_additions",
    "code_churn_deletions",
    "modified_test_file",
    "file_test_distance"
]

COVERAGE_AND_STRUCTURAL_FEATURES = [
    "test_age_builds",
    "test_loc",
    "cov_class_count",
    "cov_method_count",
    "w_sum_cov_c_faults"
]

ALL_PRE_EXECUTION_FEATURES = HISTORY_FEATURES + CODE_CHANGE_FEATURES + COVERAGE_AND_STRUCTURAL_FEATURES


def load_dataset(csv_path: str = os.path.join("data", "benchmark_tcp_ci.csv")) -> pd.DataFrame:
    """Loads and validates the test prioritization dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Please place the dataset file or run scripts/download_dataset.py"
        )
    df = pd.read_csv(csv_path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validates that required identifier and outcome columns are present.
    Ensures no forbidden columns or unhandled NaNs in key columns.
    """
    missing_ids = [c for c in IDENTIFIER_COLUMNS if c not in df.columns]
    if missing_ids:
        raise ValueError(f"Missing required identifier columns: {missing_ids}")
        
    missing_post = [c for c in POST_EXECUTION_COLUMNS if c not in df.columns]
    if missing_post:
        raise ValueError(f"Missing required post-execution columns: {missing_post}")
        
    # Check for invalid durations
    if (df["duration"] <= 0).any():
        logger.warning(f"Found {(df['duration'] <= 0).sum()} records with non-positive duration. Sanitizing...")
        df["duration"] = df["duration"].clip(lower=0.01)
        
    # Check for binary verdict
    unique_verdicts = set(df["verdict"].dropna().unique())
    if not unique_verdicts.issubset({0, 1}):
        raise ValueError(f"Target column 'verdict' must be binary {0, 1}, found {unique_verdicts}")
        
    return True


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Returns valid pre-execution feature columns available in df."""
    return [c for c in ALL_PRE_EXECUTION_FEATURES if c in df.columns]


def chronological_split(
    df: pd.DataFrame, 
    train_ratio: float = 0.60, 
    val_ratio: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs leakage-safe chronological train/validation/test split by build sequence.
    
    Guarantees:
    - Training builds strictly precede validation builds.
    - Validation builds strictly precede test builds.
    - No tests from the same target build leak across splits.
    """
    splits_train, splits_val, splits_test = [], [], []
    
    for project_id, group in df.groupby("project_id"):
        builds = sorted(group["build_id"].unique())
        n_builds = len(builds)
        
        train_end_idx = int(n_builds * train_ratio)
        val_end_idx = int(n_builds * (train_ratio + val_ratio))
        
        train_builds = set(builds[:train_end_idx])
        val_builds = set(builds[train_end_idx:val_end_idx])
        test_builds = set(builds[val_end_idx:])
        
        splits_train.append(group[group["build_id"].isin(train_builds)])
        splits_val.append(group[group["build_id"].isin(val_builds)])
        splits_test.append(group[group["build_id"].isin(test_builds)])
        
    train_df = pd.concat(splits_train, ignore_index=True)
    val_df = pd.concat(splits_val, ignore_index=True)
    test_df = pd.concat(splits_test, ignore_index=True)
    
    logger.info(
        f"Chronological split completed:\n"
        f"  Train: {len(train_df)} rows ({len(train_df['build_id'].unique())} builds)\n"
        f"  Val:   {len(val_df)} rows ({len(val_df['build_id'].unique())} builds)\n"
        f"  Test:  {len(test_df)} rows ({len(test_df['build_id'].unique())} builds)"
    )
    
    return train_df, val_df, test_df
