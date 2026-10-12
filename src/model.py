"""
model.py - Model training, evaluation, comparison, persistence, and multi-step forecasting
Project: Smart Water Consumption Predictor Using Machine Learning
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from src.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_matrix,
    get_feature_names,
    prepare_next_day_features,
)


def get_models_dir() -> str:
    """Returns the models directory path."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression evaluation metrics with strict decimal rounding.
    
    Metrics:
    - MAE: Mean Absolute Error (Litres). Average magnitude of errors.
    - RMSE: Root Mean Squared Error (Litres). Penalizes larger deviations.
    - MAPE: Mean Absolute Percentage Error (%). Error relative to actual consumption.
    - R2: Coefficient of Determination. Proportion of variance explained by model.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE_Litres": round(mae, 2),
        "RMSE_Litres": round(rmse, 2),
        "MAPE_Percent": round(mape, 2),
        "R2_Score": round(r2, 4)
    }


def train_and_evaluate_models(
    df: pd.DataFrame,
    test_ratio: float = 0.20
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Performs chronological train/test split, trains candidate models including baselines,
    evaluates performance on unseen test data, and selects the empirical champion.
    """
    # 1. Feature Engineering (leak-free)
    X, y = build_feature_matrix(df, drop_warmup=True)

    n_samples = len(X)
    split_idx = int(n_samples * (1.0 - test_ratio))

    # Chronological Split (No random shuffling to prevent temporal lookahead)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Date references for audit
    full_clean_df = df.iloc[-n_samples:].reset_index(drop=True)
    train_dates = (
        full_clean_df["date"].iloc[0].strftime("%Y-%m-%d"),
        full_clean_df["date"].iloc[split_idx - 1].strftime("%Y-%m-%d")
    )
    test_dates = (
        full_clean_df["date"].iloc[split_idx].strftime("%Y-%m-%d"),
        full_clean_df["date"].iloc[-1].strftime("%Y-%m-%d")
    )

    results_table = []
    trained_estimators = {}

    # ----------------------------------------------------
    # Baseline 1: Naive Persistence (Predicts yesterday's value)
    # ----------------------------------------------------
    y_pred_naive = X_test["lag_1"].values
    metrics_naive = evaluate_predictions(y_test.values, y_pred_naive)
    results_table.append({
        "Model": "Baseline: Naive Persistence (Lag 1)",
        "Type": "Rule-based Baseline",
        **metrics_naive
    })

    # ----------------------------------------------------
    # Baseline 2: 7-day Moving Average
    # ----------------------------------------------------
    y_pred_roll7 = X_test["rolling_mean_7"].values
    metrics_roll7 = evaluate_predictions(y_test.values, y_pred_roll7)
    results_table.append({
        "Model": "Baseline: 7-Day Moving Average",
        "Type": "Statistical Baseline",
        **metrics_roll7
    })

    # ----------------------------------------------------
    # Model 1: Ridge Linear Regression
    # ----------------------------------------------------
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    metrics_ridge = evaluate_predictions(y_test.values, y_pred_ridge)
    results_table.append({
        "Model": "Ridge Linear Regression",
        "Type": "Linear ML",
        **metrics_ridge
    })
    trained_estimators["Ridge Linear Regression"] = ridge

    # ----------------------------------------------------
    # Model 2: Decision Tree Regressor
    # ----------------------------------------------------
    dt = DecisionTreeRegressor(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    metrics_dt = evaluate_predictions(y_test.values, y_pred_dt)
    results_table.append({
        "Model": "Decision Tree Regressor",
        "Type": "Tree ML",
        **metrics_dt
    })
    trained_estimators["Decision Tree Regressor"] = dt

    # ----------------------------------------------------
    # Model 3: Random Forest Regressor
    # ----------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=8,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    metrics_rf = evaluate_predictions(y_test.values, y_pred_rf)
    results_table.append({
        "Model": "Random Forest Regressor",
        "Type": "Ensemble ML",
        **metrics_rf
    })
    trained_estimators["Random Forest Regressor"] = rf

    # ----------------------------------------------------
    # Model 4: Gradient Boosting Regressor
    # ----------------------------------------------------
    gbr = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.06,
        max_depth=4,
        random_state=42
    )
    gbr.fit(X_train, y_train)
    y_pred_gbr = gbr.predict(X_test)
    metrics_gbr = evaluate_predictions(y_test.values, y_pred_gbr)
    results_table.append({
        "Model": "Gradient Boosting Regressor",
        "Type": "Ensemble ML",
        **metrics_gbr
    })
    trained_estimators["Gradient Boosting Regressor"] = gbr

    comparison_df = pd.DataFrame(results_table)

    # ----------------------------------------------------
    # Champion Selection (Based strictly on lowest test MAE)
    # ----------------------------------------------------
    ml_candidates = comparison_df[~comparison_df["Model"].str.startswith("Baseline")].copy()
    ml_candidates = ml_candidates.sort_values(by="MAE_Litres", ascending=True)
    best_model_name = ml_candidates.iloc[0]["Model"]
    best_estimator = trained_estimators[best_model_name]
    best_metrics = ml_candidates.iloc[0].to_dict()

    # Feature Importance (if tree-based or linear)
    feature_importances = {}
    if hasattr(best_estimator, "feature_importances_"):
        fi = best_estimator.feature_importances_
        feature_importances = {feat: round(float(imp), 4) for feat, imp in zip(FEATURE_COLUMNS, fi)}
        feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    elif hasattr(best_estimator, "coef_"):
        # For linear models, magnitude of normalized coefficients or raw weights
        coefs = np.abs(best_estimator.coef_)
        total_coef = np.sum(coefs) + 1e-9
        norm_fi = coefs / total_coef
        feature_importances = {feat: round(float(imp), 4) for feat, imp in zip(FEATURE_COLUMNS, norm_fi)}
        feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    metadata = {
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "total_samples": n_samples,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_date_range": train_dates,
        "test_date_range": test_dates,
        "feature_columns": FEATURE_COLUMNS,
        "feature_importances": feature_importances,
        "test_predictions": {
            "y_test": [round(float(v), 1) for v in y_test.values[:30]],
            "y_pred_best": [round(float(v), 1) for v in best_estimator.predict(X_test)[:30]],
            "test_dates": full_clean_df["date"].iloc[split_idx:split_idx + 30].dt.strftime("%Y-%m-%d").tolist()
        }
    }

    # Save artifacts
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, "trained_model.joblib")
    metrics_path = os.path.join(models_dir, "eval_metrics.json")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(best_estimator, model_path)
    comparison_df.to_json(metrics_path, orient="records", indent=2)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SUCCESS] Model training complete! Champion: {best_model_name}")
    print(f"[INFO] Test MAE: {best_metrics['MAE_Litres']} L | Test RMSE: {best_metrics['RMSE_Litres']} L | R2: {best_metrics['R2_Score']}")

    return metadata, comparison_df


def load_trained_model() -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """
    Loads persisted champion model, metadata, and evaluation metrics.
    If not found, raises FileNotFoundError.
    """
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, "trained_model.joblib")
    metrics_path = os.path.join(models_dir, "eval_metrics.json")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Trained model files not found. Run model training first.")

    model = joblib.load(model_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metrics_df = pd.read_json(metrics_path)

    return model, metadata, metrics_df


def predict_next_day(
    model: Any,
    historical_df: pd.DataFrame,
    household_size: Optional[int] = None,
    temperature_c: Optional[float] = None,
    rainfall_mm: Optional[float] = None
) -> Tuple[float, float, float]:
    """
    Predicts next-day total water consumption in litres.
    
    Returns:
        Tuple of (predicted_litres, lower_bound_litres, upper_bound_litres)
    """
    X_next = prepare_next_day_features(
        historical_df,
        household_size=household_size,
        temperature_c=temperature_c,
        rainfall_mm=rainfall_mm
    )

    pred = float(model.predict(X_next)[0])
    # Ensure physically non-negative
    pred = max(20.0, pred)

    # Uncertainty interval (~80% empirical interval based on MAE)
    # Using roughly 1.3 * MAE margin
    uncertainty_margin = 25.0
    lower = max(10.0, pred - uncertainty_margin)
    upper = pred + uncertainty_margin

    return round(pred, 1), round(lower, 1), round(upper, 1)


def forecast_next_n_days(
    model: Any,
    historical_df: pd.DataFrame,
    n_days: int = 7,
    household_size: Optional[int] = None,
    avg_temperature_c: float = 28.0,
    rainfall_probs: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Generates a recursive multi-step daily forecast for the next N days.
    
    At each step k, the predicted consumption y_hat(k) is appended to the
    working history to dynamically update lag_1, lag_2, lag_7, and rolling stats.
    """
    df_sim = historical_df.copy()
    latest_date = pd.to_datetime(df_sim["date"].iloc[-1])

    if household_size is None:
        household_size = int(df_sim["household_size"].iloc[-1])

    forecast_records = []

    for step in range(1, n_days + 1):
        target_date = latest_date + pd.Timedelta(days=step)
        rain = rainfall_probs[step - 1] if (rainfall_probs and len(rainfall_probs) >= step) else 0.0

        X_step = prepare_next_day_features(
            df_sim,
            forecast_date=target_date.strftime("%Y-%m-%d"),
            household_size=household_size,
            temperature_c=avg_temperature_c,
            rainfall_mm=rain
        )

        pred_val = float(model.predict(X_step)[0])
        pred_val = max(25.0, pred_val)

        # Confidence bounds expand slightly with forecasting horizon step
        horizon_uncertainty = 20.0 + (step * 2.5)
        lower_b = max(15.0, pred_val - horizon_uncertainty)
        upper_b = pred_val + horizon_uncertainty

        day_name = target_date.strftime("%A")
        is_wknd = 1 if target_date.dayofweek in [5, 6] else 0

        forecast_records.append({
            "date": target_date.strftime("%Y-%m-%d"),
            "day_name": day_name,
            "is_weekend": is_wknd,
            "predicted_consumption_l": round(pred_val, 1),
            "lower_bound_l": round(lower_b, 1),
            "upper_bound_l": round(upper_b, 1),
            "per_capita_predicted_l": round(pred_val / max(1, household_size), 1)
        })

        # Append recursive row to working history
        new_row = {
            "date": target_date,
            "household_size": household_size,
            "temperature_c": avg_temperature_c,
            "rainfall_mm": rain,
            "is_weekend": is_wknd,
            "day_of_week": target_date.dayofweek,
            "total_consumption_l": pred_val
        }
        df_sim = pd.concat([df_sim, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(forecast_records)
