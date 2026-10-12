"""
feature_engineering.py - Leak-free time-series feature engineering
Project: Smart Water Consumption Predictor Using Machine Learning

CRITICAL TIME-SERIES PRINCIPLE:
To prevent data leakage, all historical lag and rolling window features are computed
strictly on past observations using .shift(1). The target observation at time t
is NEVER incorporated into rolling or lag aggregates for time t.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "household_size",
    "temperature_c",
    "rainfall_mm",
    "is_weekend",
    "is_rainy",
    "sin_day_of_week",
    "cos_day_of_week",
    "sin_month",
    "cos_month",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_min_7",
    "rolling_max_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "lag1_to_rolling7_ratio"
]

TARGET_COLUMN = "total_consumption_l"


def get_feature_names() -> List[str]:
    """Returns the ordered list of predictive feature column names."""
    return list(FEATURE_COLUMNS)


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes calendar and cyclical time features from the 'date' column.
    """
    df_feat = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_feat["date"]):
        df_feat["date"] = pd.to_datetime(df_feat["date"])

    day_of_week = df_feat["date"].dt.dayofweek
    month = df_feat["date"].dt.month

    df_feat["day_of_week"] = day_of_week
    df_feat["month"] = month
    df_feat["day_of_month"] = df_feat["date"].dt.day
    df_feat["is_weekend"] = day_of_week.isin([5, 6]).astype(int)

    # Cyclical trigonometric transforms
    # Captures continuous continuity (e.g. Sunday -> Monday, Dec -> Jan)
    df_feat["sin_day_of_week"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df_feat["cos_day_of_week"] = np.cos(2 * np.pi * day_of_week / 7.0)
    df_feat["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    df_feat["cos_month"] = np.cos(2 * np.pi * month / 12.0)

    # Rainfall indicator
    if "rainfall_mm" in df_feat.columns:
        df_feat["is_rainy"] = (df_feat["rainfall_mm"] > 1.0).astype(int)
    else:
        df_feat["rainfall_mm"] = 0.0
        df_feat["is_rainy"] = 0

    if "temperature_c" not in df_feat.columns:
        df_feat["temperature_c"] = 28.0

    return df_feat


def compute_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes lag features and rolling window aggregates.
    
    SAFETY CHECK:
    Notice .shift(1) is applied BEFORE rolling.
    This guarantees that rolling_mean_7 at index t uses records [t-7 ... t-1],
    completely excluding record t.
    """
    df_feat = df.copy()
    target = df_feat[TARGET_COLUMN]

    # Explicit Lags
    df_feat["lag_1"] = target.shift(1)
    df_feat["lag_2"] = target.shift(2)
    df_feat["lag_3"] = target.shift(3)
    df_feat["lag_7"] = target.shift(7)

    # Shifted target series for rolling windows
    shifted_target = target.shift(1)

    # Rolling window statistics over shifted history
    df_feat["rolling_mean_7"] = shifted_target.rolling(window=7, min_periods=7).mean()
    df_feat["rolling_std_7"] = shifted_target.rolling(window=7, min_periods=7).std().fillna(0.0)
    df_feat["rolling_min_7"] = shifted_target.rolling(window=7, min_periods=7).min()
    df_feat["rolling_max_7"] = shifted_target.rolling(window=7, min_periods=7).max()

    df_feat["rolling_mean_14"] = shifted_target.rolling(window=14, min_periods=14).mean()
    df_feat["rolling_mean_30"] = shifted_target.rolling(window=30, min_periods=30).mean()

    # Interaction / Ratio feature: how yesterday compares to recent 7-day average
    df_feat["lag1_to_rolling7_ratio"] = df_feat["lag_1"] / (df_feat["rolling_mean_7"] + 1e-5)

    return df_feat


def build_feature_matrix(
    df: pd.DataFrame,
    drop_warmup: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs the final design matrix X and target vector y.
    
    Args:
        df: Input daily water consumption dataframe with 'date' and 'total_consumption_l'.
        drop_warmup: If True, drops the initial 30 days needed to warm up rolling window features.
        
    Returns:
        Tuple of (X: pd.DataFrame, y: pd.Series)
    """
    df_proc = compute_time_features(df)
    df_proc = compute_lag_and_rolling_features(df_proc)

    if drop_warmup:
        # Drop rows where rolling_mean_30 or lag_7 is NaN (first 30 records)
        df_proc = df_proc.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    X = df_proc[FEATURE_COLUMNS].copy()
    y = df_proc[TARGET_COLUMN].copy()

    return X, y


def prepare_next_day_features(
    historical_df: pd.DataFrame,
    forecast_date: Optional[str] = None,
    household_size: Optional[int] = None,
    temperature_c: Optional[float] = None,
    rainfall_mm: Optional[float] = None
) -> pd.DataFrame:
    """
    Prepares a single-row feature vector to predict next-day consumption
    based on the latest historical observations.
    """
    if len(historical_df) < 30:
        raise ValueError("At least 30 historical days are required to construct lag and rolling features.")

    latest_row = historical_df.iloc[-1]
    latest_date = pd.to_datetime(latest_row["date"])

    if forecast_date is None:
        target_dt = latest_date + pd.Timedelta(days=1)
    else:
        target_dt = pd.to_datetime(forecast_date)

    if household_size is None:
        household_size = int(latest_row.get("household_size", 4))
    if temperature_c is None:
        temperature_c = float(latest_row.get("temperature_c", 28.0))
    if rainfall_mm is None:
        rainfall_mm = 0.0

    # Extract historical target series
    history_target = historical_df[TARGET_COLUMN].values

    # Compute lags directly from recent points
    lag_1 = float(history_target[-1])
    lag_2 = float(history_target[-2])
    lag_3 = float(history_target[-3])
    lag_7 = float(history_target[-7])

    # Compute rolling aggregates from latest available days
    last_7 = history_target[-7:]
    last_14 = history_target[-14:]
    last_30 = history_target[-30:]

    rolling_mean_7 = float(np.mean(last_7))
    rolling_std_7 = float(np.std(last_7, ddof=1)) if len(last_7) > 1 else 0.0
    rolling_min_7 = float(np.min(last_7))
    rolling_max_7 = float(np.max(last_7))
    rolling_mean_14 = float(np.mean(last_14))
    rolling_mean_30 = float(np.mean(last_30))
    lag1_ratio = lag_1 / (rolling_mean_7 + 1e-5)

    day_of_week = target_dt.dayofweek
    month = target_dt.month
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_rainy = 1 if rainfall_mm > 1.0 else 0

    feature_dict = {
        "household_size": household_size,
        "temperature_c": temperature_c,
        "rainfall_mm": rainfall_mm,
        "is_weekend": is_weekend,
        "is_rainy": is_rainy,
        "sin_day_of_week": np.sin(2 * np.pi * day_of_week / 7.0),
        "cos_day_of_week": np.cos(2 * np.pi * day_of_week / 7.0),
        "sin_month": np.sin(2 * np.pi * month / 12.0),
        "cos_month": np.cos(2 * np.pi * month / 12.0),
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "lag_7": lag_7,
        "rolling_mean_7": rolling_mean_7,
        "rolling_std_7": rolling_std_7,
        "rolling_min_7": rolling_min_7,
        "rolling_max_7": rolling_max_7,
        "rolling_mean_14": rolling_mean_14,
        "rolling_mean_30": rolling_mean_30,
        "lag1_to_rolling7_ratio": lag1_ratio
    }

    return pd.DataFrame([feature_dict])[FEATURE_COLUMNS]
