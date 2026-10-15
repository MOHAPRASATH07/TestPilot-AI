"""
Tests for Preprocessing Pipeline.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_dataset
from src.preprocessing import TestPilotPreprocessor


def test_preprocessor_fit_transform():
    df = load_dataset().head(100)
    prep = TestPilotPreprocessor()
    
    # Must raise if transforming before fit
    with pytest.raises(RuntimeError):
        prep.transform(df)
        
    X_scaled = prep.fit_transform(df, scale=True)
    assert isinstance(X_scaled, np.ndarray)
    assert X_scaled.shape[0] == 100
    assert X_scaled.shape[1] == len(prep.feature_names)
    assert not np.isnan(X_scaled).any()


def test_preprocessor_handles_missing_columns():
    df = pd.DataFrame({
        "avg_recent_duration": [1.5, 2.0],
        "historical_fail_rate": [0.1, 0.0]
    })
    prep = TestPilotPreprocessor(feature_names=["avg_recent_duration", "historical_fail_rate", "non_existent_feature"])
    X = prep.fit_transform(df, scale=False)
    assert X.shape == (2, 3)
    # The non-existent feature should be filled with 0.0
    assert X[0, 2] == 0.0
