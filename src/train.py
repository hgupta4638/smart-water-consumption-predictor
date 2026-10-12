"""
train.py - Pipeline execution script for model training and comparison
Project: Smart Water Consumption Predictor Using Machine Learning
"""

import sys
import os

# Add project root to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_dataset
from src.model import train_and_evaluate_models


def main():
    print("=" * 70)
    print(" Smart Water Consumption Predictor - Model Training & Evaluation ")
    print("=" * 70)

    try:
        df = load_dataset()
        print(f"[DATA] Successfully loaded {len(df)} records ({df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}).")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    print("\n[TRAINING] Performing chronological 80/20 train-test split and evaluating models...")
    metadata, comparison_df = train_and_evaluate_models(df, test_ratio=0.20)

    print("\n" + "=" * 70)
    print(" Model Comparison Results on Chronological Holdout Test Set:")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    print("\n[METRICS EXPLANATION]")
    print("- MAE (Mean Absolute Error): Average absolute difference in Litres between actual and predicted.")
    print("- RMSE (Root Mean Squared Error): Penalizes larger errors more heavily (useful for peak detection).")
    print("- MAPE (%): Average percentage error relative to actual consumption.")
    print("- R² Score: Variance explained (closer to 1.0 indicates superior predictive capability).")

    print("\n[SELECTED CHAMPION]")
    print(f"Model: {metadata['best_model_name']}")
    print(f"MAE: {metadata['best_metrics']['MAE_Litres']} L | RMSE: {metadata['best_metrics']['RMSE_Litres']} L | R²: {metadata['best_metrics']['R2_Score']}")

    if metadata["feature_importances"]:
        print("\n[TOP 5 PREDICTIVE FEATURES]")
        for i, (k, v) in enumerate(list(metadata["feature_importances"].items())[:5], 1):
            print(f" {i}. {k}: {v * 100:.2f}%")

    print("\n[DONE] Model artifacts successfully saved to 'models/' directory.")


if __name__ == "__main__":
    main()
