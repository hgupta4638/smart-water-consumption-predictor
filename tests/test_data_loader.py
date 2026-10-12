"""
test_data_loader.py - Unit tests for data loading, cleaning, validation, and user input
"""

import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    load_dataset,
    calculate_consumption_summaries,
    validate_user_input,
    append_user_reading,
    aggregate_by_period
)


def test_load_dataset_valid():
    df = load_dataset()
    assert not df.empty
    assert len(df) >= 700
    assert "date" in df.columns
    assert "total_consumption_l" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    # Non-negative check
    assert (df["total_consumption_l"] >= 0).all()


def test_calculate_consumption_summaries():
    df = load_dataset()
    summaries = calculate_consumption_summaries(df)
    assert summaries["total_recorded_days"] >= 700
    assert summaries["average_daily_l"] > 100.0
    assert summaries["per_capita_daily_l"] > 25.0
    assert summaries["recent_7d_avg_l"] > 0


def test_validate_user_input_valid():
    # Standard 4-person family, 500 Litres total
    is_valid, msg = validate_user_input(500.0, 4)
    assert is_valid is True
    assert msg is None


def test_validate_user_input_invalid_values():
    # Negative consumption
    valid, msg = validate_user_input(-50.0, 4)
    assert valid is False
    assert "greater than 0" in msg

    # Unrealistic extreme consumption
    valid, msg = validate_user_input(15000.0, 4)
    assert valid is False
    assert "extraordinarily high" in msg

    # Household size 0
    valid, msg = validate_user_input(400.0, 0)
    assert valid is False
    assert "Household size" in msg

    # Below survival minimum
    valid, msg = validate_user_input(20.0, 4)  # 5 L/person
    assert valid is False
    assert "below the WHO survival baseline" in msg


def test_validate_user_input_category_mismatch():
    categories = {
        "usage_bathroom_l": 200.0,
        "usage_toilet_l": 100.0,
        "usage_kitchen_l": 50.0
    }
    # Total entered is 500, but categories sum to 350
    valid, msg = validate_user_input(500.0, 4, categories=categories)
    assert valid is False
    assert "Sum of usage breakdown categories" in msg


def test_append_user_reading():
    df = load_dataset()
    initial_len = len(df)
    future_date = "2026-05-15"

    updated_df = append_user_reading(
        df=df,
        date_str=future_date,
        total_l=520.0,
        household_size=4,
        temperature_c=31.5,
        rainfall_mm=0.0
    )

    assert len(updated_df) == initial_len + 1
    last_row = updated_df.iloc[-1]
    assert last_row["date"].strftime("%Y-%m-%d") == future_date
    assert last_row["total_consumption_l"] == 520.0
    assert last_row["data_source_flag"] == "USER_LOGGED_ENTRY"


def test_aggregate_by_period():
    df = load_dataset()
    weekly = aggregate_by_period(df, freq="W")
    assert not weekly.empty
    assert "total_consumption_l_sum" in weekly.columns
    assert len(weekly) < len(df)
