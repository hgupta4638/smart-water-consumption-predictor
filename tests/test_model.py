"""
test_model.py - Unit tests for model loading, evaluation metrics, prediction, and 7-day forecasting
"""

import pytest
import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.model import (
    load_trained_model,
    evaluate_predictions,
    predict_next_day,
    forecast_next_n_days
)


def test_evaluate_predictions_metrics():
    y_true = np.array([500.0, 520.0, 480.0, 550.0])
    y_pred = np.array([510.0, 515.0, 490.0, 540.0])

    metrics = evaluate_predictions(y_true, y_pred)

    assert "MAE_Litres" in metrics
    assert "RMSE_Litres" in metrics
    assert "MAPE_Percent" in metrics
    assert "R2_Score" in metrics

    # MAE should be (10 + 5 + 10 + 10) / 4 = 8.75
    assert np.isclose(metrics["MAE_Litres"], 8.75)
    assert metrics["R2_Score"] > 0.8


def test_load_trained_model_and_artifacts():
    model, metadata, metrics_df = load_trained_model()

    assert model is not None
    assert "best_model_name" in metadata
    assert not metrics_df.empty
    assert "Model" in metrics_df.columns
    assert "MAE_Litres" in metrics_df.columns


def test_predict_next_day():
    df = load_dataset()
    model, _, _ = load_trained_model()

    pred, lower, upper = predict_next_day(
        model=model,
        historical_df=df,
        household_size=4,
        temperature_c=28.0,
        rainfall_mm=0.0
    )

    assert pred > 0.0
    assert lower <= pred <= upper
    # Typical 4-person house consumption range (300 to 800 Litres)
    assert 250.0 <= pred <= 850.0


def test_forecast_next_7_days():
    df = load_dataset()
    model, _, _ = load_trained_model()

    forecast_df = forecast_next_n_days(
        model=model,
        historical_df=df,
        n_days=7,
        household_size=4
    )

    assert len(forecast_df) == 7
    assert "predicted_consumption_l" in forecast_df.columns
    assert "lower_bound_l" in forecast_df.columns
    assert "upper_bound_l" in forecast_df.columns
    assert "day_name" in forecast_df.columns

    # Verify bounds validity
    for _, row in forecast_df.iterrows():
        assert row["lower_bound_l"] <= row["predicted_consumption_l"] <= row["upper_bound_l"]
