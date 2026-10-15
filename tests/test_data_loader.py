"""
Tests for Data Loading, Schema Validation, and Chronological Splitting.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import (
    load_dataset,
    validate_schema,
    chronological_split,
    IDENTIFIER_COLUMNS,
    POST_EXECUTION_COLUMNS,
    ALL_PRE_EXECUTION_FEATURES
)


def test_load_dataset():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in IDENTIFIER_COLUMNS + POST_EXECUTION_COLUMNS:
        assert col in df.columns


def test_schema_validation_catches_missing_columns():
    invalid_df = pd.DataFrame({"some_col": [1, 2]})
    with pytest.raises(ValueError):
        validate_schema(invalid_df)


def test_schema_validation_sanitizes_negative_duration():
    df = load_dataset().head(10).copy()
    df.loc[0, "duration"] = -5.0
    validate_schema(df)
    assert df.loc[0, "duration"] > 0


def test_chronological_split_leakage_safety():
    df = load_dataset()
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.6, val_ratio=0.2)
    
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # For each project, ensure builds are monotonically partitioned
    for proj in df["project_id"].unique():
        tr_builds = set(train_df[train_df["project_id"] == proj]["build_id"])
        val_builds = set(val_df[val_df["project_id"] == proj]["build_id"])
        te_builds = set(test_df[test_df["project_id"] == proj]["build_id"])
        
        # Disjoint sets
        assert tr_builds.isdisjoint(val_builds)
        assert tr_builds.isdisjoint(te_builds)
        assert val_builds.isdisjoint(te_builds)
        
        # Chronological order
        if tr_builds and val_builds:
            assert max(tr_builds) < min(val_builds)
        if val_builds and te_builds:
            assert max(val_builds) < min(te_builds)
