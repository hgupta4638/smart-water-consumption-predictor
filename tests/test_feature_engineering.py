"""
test_feature_engineering.py - Unit tests for feature transforms and data leakage prevention
"""

import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_dataset
from src.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    compute_time_features,
    compute_lag_and_rolling_features,
    build_feature_matrix,
    prepare_next_day_features
)


def test_compute_time_features():
    df = load_dataset().head(50)
    df_feat = compute_time_features(df)

    assert "is_weekend" in df_feat.columns
    assert "sin_day_of_week" in df_feat.columns
    assert "cos_day_of_week" in df_feat.columns
    assert "sin_month" in df_feat.columns
    assert "cos_month" in df_feat.columns

    # Weekend values must be 0 or 1
    assert set(df_feat["is_weekend"].unique()).issubset({0, 1})


def test_leakage_prevention_lags_and_rolling():
    df = load_dataset().head(50)
    df_feat = compute_lag_and_rolling_features(df)

    # Critical check: lag_1 at index k MUST equal target at index k-1
    for k in range(1, 20):
        actual_prev = df.loc[k - 1, TARGET_COLUMN]
        lag_val = df_feat.loc[k, "lag_1"]
        assert np.isclose(lag_val, actual_prev), f"Leakage violation: lag_1 at {k} does not match target at {k-1}"

    # Critical check: rolling_mean_7 at index 7 must strictly equal average of indices 0 through 6
    expected_roll = df.loc[0:6, TARGET_COLUMN].mean()
    actual_roll = df_feat.loc[7, "rolling_mean_7"]
    assert np.isclose(actual_roll, expected_roll), "Leakage violation: rolling_mean_7 does not match shifted window"


def test_build_feature_matrix():
    df = load_dataset()
    X, y = build_feature_matrix(df, drop_warmup=True)

    assert len(X) == len(y)
    assert len(X) > 0
    # No NaNs in feature matrix after drop_warmup
    assert not X.isna().any().any()
    assert list(X.columns) == FEATURE_COLUMNS


def test_prepare_next_day_features():
    df = load_dataset()
    X_next = prepare_next_day_features(
        df,
        forecast_date="2026-01-01",
        household_size=4,
        temperature_c=30.0,
        rainfall_mm=0.0
    )

    assert len(X_next) == 1
    assert list(X_next.columns) == FEATURE_COLUMNS
    assert X_next.loc[0, "household_size"] == 4
    assert X_next.loc[0, "temperature_c"] == 30.0
    assert not X_next.isna().any().any()
