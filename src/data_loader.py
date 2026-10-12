"""
data_loader.py - Data ingestion, cleaning, validation, and aggregation
Project: Smart Water Consumption Predictor Using Machine Learning
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "household_size",
    "total_consumption_l"
]

CATEGORY_COLUMNS = [
    "usage_bathroom_l",
    "usage_toilet_l",
    "usage_kitchen_l",
    "usage_laundry_l",
    "usage_outdoor_l"
]


def get_default_data_path() -> str:
    """Returns the default absolute path to the processed water consumption dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "processed", "water_consumption_daily.csv")


def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and cleans the daily water consumption dataset.
    
    Steps performed:
    1. Reads CSV file from specified or default location.
    2. Validates existence of required columns.
    3. Converts 'date' to datetime and sets as ordered time-series.
    4. Handles missing values via forward/backward fill and sensible defaults.
    5. Validates non-negative water values.
    
    Returns:
        pd.DataFrame: Cleaned, sorted daily water consumption dataframe.
    """
    if file_path is None:
        file_path = get_default_data_path()

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found at '{file_path}'. "
            "Please run 'python data/generate_dataset.py' to generate the demonstration dataset."
        )

    df = pd.read_csv(file_path)

    # 1. Column validation
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    # 2. Date parsing and sorting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Check for duplicate dates
    if df["date"].duplicated().any():
        # Keep latest record if duplicated
        df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    # 3. Numeric conversions & Missing Value Handling
    numeric_cols = [
        "household_size", "total_consumption_l", "temperature_c", "rainfall_mm",
        "guests_present", "effective_occupants"
    ] + [c for c in CATEGORY_COLUMNS if c in df.columns]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing value handling: forward-fill then backward-fill for time-series continuity
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Fill any remaining NaNs with logical domain defaults
    if "household_size" in df.columns:
        df["household_size"] = df["household_size"].fillna(4).clip(lower=1)
    if "total_consumption_l" in df.columns:
        df["total_consumption_l"] = df["total_consumption_l"].fillna(135.0 * df.get("household_size", 4)).clip(lower=10.0)

    # Validate non-negative consumption
    for c in ["total_consumption_l"] + [col for col in CATEGORY_COLUMNS if col in df.columns]:
        df[c] = df[c].clip(lower=0.0)

    # Fill category columns if missing or partial
    for cat in CATEGORY_COLUMNS:
        if cat not in df.columns:
            # Default ratio allocation if individual categories are missing
            if cat == "usage_bathroom_l":
                df[cat] = df["total_consumption_l"] * 0.35
            elif cat == "usage_toilet_l":
                df[cat] = df["total_consumption_l"] * 0.22
            elif cat == "usage_kitchen_l":
                df[cat] = df["total_consumption_l"] * 0.15
            elif cat == "usage_laundry_l":
                df[cat] = df["total_consumption_l"] * 0.18
            elif cat == "usage_outdoor_l":
                df[cat] = df["total_consumption_l"] * 0.10

    # Ensure day_of_week and is_weekend exist
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["date"].dt.dayofweek
    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def calculate_consumption_summaries(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes primary descriptive statistics and KPI metrics from the dataset.
    """
    if df.empty:
        return {}

    total_days = len(df)
    avg_daily = float(df["total_consumption_l"].mean())
    median_daily = float(df["total_consumption_l"].median())
    std_daily = float(df["total_consumption_l"].std())
    min_daily = float(df["total_consumption_l"].min())
    max_daily = float(df["total_consumption_l"].max())

    latest_row = df.iloc[-1]
    latest_date = latest_row["date"].strftime("%Y-%m-%d")
    latest_val = float(latest_row["total_consumption_l"])
    household_size = int(latest_row.get("household_size", 4))

    per_capita_daily = avg_daily / max(1, household_size)

    # Recent 7-day average
    last_7_days = df.tail(7)
    recent_7d_avg = float(last_7_days["total_consumption_l"].mean())

    # Recent 30-day average
    last_30_days = df.tail(30)
    recent_30d_avg = float(last_30_days["total_consumption_l"].mean())

    # Weekend vs Weekday average
    weekend_avg = float(df[df["is_weekend"] == 1]["total_consumption_l"].mean()) if (df["is_weekend"] == 1).any() else avg_daily
    weekday_avg = float(df[df["is_weekend"] == 0]["total_consumption_l"].mean()) if (df["is_weekend"] == 0).any() else avg_daily

    return {
        "total_recorded_days": total_days,
        "latest_date": latest_date,
        "latest_consumption_l": round(latest_val, 1),
        "household_size": household_size,
        "average_daily_l": round(avg_daily, 1),
        "median_daily_l": round(median_daily, 1),
        "std_daily_l": round(std_daily, 1),
        "min_daily_l": round(min_daily, 1),
        "max_daily_l": round(max_daily, 1),
        "per_capita_daily_l": round(per_capita_daily, 1),
        "recent_7d_avg_l": round(recent_7d_avg, 1),
        "recent_30d_avg_l": round(recent_30d_avg, 1),
        "weekday_avg_l": round(weekday_avg, 1),
        "weekend_avg_l": round(weekend_avg, 1)
    }


def aggregate_by_period(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Aggregates consumption by time frequency ('W' for weekly, 'ME' for monthly).
    """
    df_copy = df.copy()
    df_copy = df_copy.set_index("date")
    
    agg_dict = {"total_consumption_l": ["sum", "mean", "count"]}
    for cat in CATEGORY_COLUMNS:
        if cat in df_copy.columns:
            agg_dict[cat] = "sum"

    resampled = df_copy.resample(freq).agg(agg_dict)
    resampled.columns = [
        f"{c[0]}_{c[1]}" if c[1] != "" else c[0]
        for c in resampled.columns
    ]
    resampled = resampled.reset_index()
    return resampled


def validate_user_input(
    total_l: float,
    household_size: int,
    categories: Optional[Dict[str, float]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validates user consumption entry with logical physical and statistical constraints.
    
    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    if household_size < 1 or household_size > 25:
        return False, f"Household size ({household_size}) must be between 1 and 25 persons."

    if total_l <= 0:
        return False, "Total daily water consumption must be greater than 0 litres."

    if total_l > 10000:
        return False, (
            f"Consumption value of {total_l} L is extraordinarily high for a single household. "
            "Please verify reading or check for a catastrophic water burst."
        )

    # Per capita sanity check (< 15 L/day is dangerously below survival basic minimum)
    lpcd = total_l / household_size
    if lpcd < 15.0:
        return False, (
            f"Calculated per-capita usage ({lpcd:.1f} L/person/day) is below the WHO survival baseline (20 L/day). "
            "Please check if the entry is accurate."
        )

    # Category sum consistency check if categories are provided
    if categories:
        cat_sum = sum(categories.values())
        if abs(cat_sum - total_l) > 1.5:
            return False, (
                f"Sum of usage breakdown categories ({cat_sum:.1f} L) does not match total entered ({total_l:.1f} L). "
                "Please reconcile category inputs."
            )

    return True, None


def append_user_reading(
    df: pd.DataFrame,
    date_str: str,
    total_l: float,
    household_size: int,
    temperature_c: float = 28.0,
    rainfall_mm: float = 0.0,
    categories: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Appends or updates a daily reading into the existing dataframe.
    """
    valid, err = validate_user_input(total_l, household_size, categories)
    if not valid:
        raise ValueError(err)

    date = pd.to_datetime(date_str)
    day_of_week = date.dayofweek
    is_weekend = 1 if day_of_week in [5, 6] else 0

    cats = categories or {}
    bath = cats.get("usage_bathroom_l", round(total_l * 0.35, 1))
    toilet = cats.get("usage_toilet_l", round(total_l * 0.22, 1))
    kitchen = cats.get("usage_kitchen_l", round(total_l * 0.15, 1))
    laundry = cats.get("usage_laundry_l", round(total_l * 0.18, 1))
    outdoor = cats.get("usage_outdoor_l", round(total_l * 0.10, 1))

    new_row = {
        "date": date,
        "household_size": household_size,
        "guests_present": 0,
        "effective_occupants": household_size,
        "temperature_c": temperature_c,
        "rainfall_mm": rainfall_mm,
        "is_weekend": is_weekend,
        "day_of_week": day_of_week,
        "season": "Custom",
        "usage_bathroom_l": bath,
        "usage_toilet_l": toilet,
        "usage_kitchen_l": kitchen,
        "usage_laundry_l": laundry,
        "usage_outdoor_l": outdoor,
        "total_consumption_l": total_l,
        "is_anomaly_simulated": 0,
        "data_source_flag": "USER_LOGGED_ENTRY"
    }

    # If date already exists, update row; otherwise append and sort
    existing_idx = df[df["date"] == date].index
    if not existing_idx.empty:
        df.loc[existing_idx[0]] = new_row
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df = df.sort_values("date").reset_index(drop=True)
    return df
