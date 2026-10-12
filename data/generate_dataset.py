"""
generate_dataset.py - Calibrated Synthetic Demonstration Dataset Generator
Project: Smart Water Consumption Predictor Using Machine Learning

NOTICE:
This script generates a clearly labelled synthetic demonstration dataset
calibrated against real-world residential water engineering benchmarks:
- Bureau of Indian Standards (BIS IS 1172: 1993: 135-150 Litres/capita/day)
- World Health Organization (WHO) Domestic Water Quantity Guidelines
- American Water Works Association (AWWA) Residential End Uses of Water

This dataset is created for educational, modeling, and demonstration purposes.
It is NEVER represented as real municipal smart meter telemetry.
"""

import os
import numpy as np
import pandas as pd


def generate_water_consumption_data(
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    household_size: int = 4,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a daily water consumption time-series dataframe with realistic
    hydrological, seasonal, and behavioral patterns.
    """
    np.random.seed(random_seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(date_range)

    # Base per-capita daily consumption (Litres/capita/day)
    # IS 1172 standard: ~135 L/person/day
    base_lpcd = 135.0

    records = []

    for i, date in enumerate(date_range):
        day_of_year = date.dayofyear
        month = date.month
        day_of_week = date.dayofweek  # 0=Monday, 6=Sunday
        is_weekend = 1 if day_of_week in [5, 6] else 0

        # 1. Seasonality & Temperature simulation (°C)
        # Assuming representative subtropical / tropical seasonal curve (min 15°C in Jan, peak 38°C in May/June)
        temp_base = 27.0 + 9.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        temp_noise = np.random.normal(0, 1.8)
        temperature_c = round(float(temp_base + temp_noise), 1)

        # 2. Rainfall simulation (mm) - Monsoon peak July-September
        if month in [6, 7, 8, 9]:
            rain_prob = 0.42
            rainfall_mm = round(float(np.random.exponential(12.0)) if np.random.rand() < rain_prob else 0.0, 1)
        else:
            rain_prob = 0.08
            rainfall_mm = round(float(np.random.exponential(4.0)) if np.random.rand() < rain_prob else 0.0, 1)

        # 3. Occasional Guests (more likely on weekends)
        guest_prob = 0.22 if is_weekend else 0.04
        guests_present = int(np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])) if np.random.rand() < guest_prob else 0
        effective_occupants = household_size + guests_present

        # 4. Temperature multiplier: hot days require more showers, drinking, and garden hydration
        temp_factor = 1.0 + max(0.0, (temperature_c - 28.0) * 0.015) - max(0.0, (20.0 - temperature_c) * 0.01)

        # 5. End-use categories (in Litres)
        # Bathing & Personal Hygiene (standard ~35% of daily usage)
        bath_base = effective_occupants * 48.0 * temp_factor
        usage_bathroom = bath_base + np.random.normal(0, 6.0 * effective_occupants)
        usage_bathroom = max(usage_bathroom, effective_occupants * 25.0)

        # Toilet Flushing (standard ~22% of daily usage)
        toilet_base = effective_occupants * 30.0
        usage_toilet = toilet_base + np.random.normal(0, 3.5 * effective_occupants)
        usage_toilet = max(usage_toilet, effective_occupants * 15.0)

        # Kitchen & Cooking (standard ~15% of daily usage)
        kitchen_base = 20.0 * effective_occupants + (18.0 if is_weekend else 0.0)
        usage_kitchen = kitchen_base + np.random.normal(0, 4.0 * effective_occupants)
        usage_kitchen = max(usage_kitchen, 15.0 * effective_occupants)

        # Laundry (standard ~18% of daily usage, heavily concentrated on weekends)
        if is_weekend:
            laundry_base = household_size * 32.0 + np.random.uniform(10, 30)
        else:
            laundry_base = household_size * 14.0 if (i % 2 == 0) else household_size * 6.0
        usage_laundry = max(10.0, laundry_base + np.random.normal(0, 5.0))

        # Outdoor / Cleaning / Gardening (standard ~10% of daily usage)
        # Reduced to near 0 if raining heavily (> 5mm)
        if rainfall_mm > 5.0:
            usage_outdoor = np.random.uniform(0.0, 5.0)
        else:
            outdoor_base = 35.0 * temp_factor + (25.0 if is_weekend else 0.0)
            usage_outdoor = max(0.0, outdoor_base + np.random.normal(0, 6.0))

        # 6. Simulated anomaly / leak event (e.g. minor pipe drip or valve leak during days 280-284)
        is_leak = False
        if 280 <= i <= 284:
            is_leak = True
            leak_loss = 65.0  # continuous slow toilet flapper or bibcock leak
            usage_toilet += leak_loss

        # Calculate Total Daily Consumption
        total_l = round(usage_bathroom + usage_toilet + usage_kitchen + usage_laundry + usage_outdoor, 1)

        season = (
            "Winter" if month in [12, 1, 2] else
            "Pre-Summer" if month in [3, 4] else
            "Summer" if month in [5, 6] else
            "Monsoon" if month in [7, 8, 9] else
            "Autumn"
        )

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "household_size": household_size,
            "guests_present": guests_present,
            "effective_occupants": effective_occupants,
            "temperature_c": temperature_c,
            "rainfall_mm": rainfall_mm,
            "is_weekend": is_weekend,
            "day_of_week": day_of_week,
            "season": season,
            "usage_bathroom_l": round(usage_bathroom, 1),
            "usage_toilet_l": round(usage_toilet, 1),
            "usage_kitchen_l": round(usage_kitchen, 1),
            "usage_laundry_l": round(usage_laundry, 1),
            "usage_outdoor_l": round(usage_outdoor, 1),
            "total_consumption_l": total_l,
            "is_anomaly_simulated": 1 if is_leak else 0,
            "data_source_flag": "SYNTHETIC_CALIBRATED_BENCHMARK"
        })

    df = pd.DataFrame(records)
    return df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "water_consumption_daily.csv")

    print("[INFO] Generating calibrated synthetic water consumption dataset...")
    df = generate_water_consumption_data()
    df.to_csv(output_file, index=False)

    print(f"[SUCCESS] Dataset generated at: {output_file}")
    print(f"[INFO] Records: {len(df)} days ({df['date'].min()} to {df['date'].max()})")
    print(f"[INFO] Average Total Daily Consumption: {df['total_consumption_l'].mean():.2f} L")
    print(f"[INFO] Average Per-Capita Daily: {df['total_consumption_l'].mean() / 4:.2f} L (Benchmark ~135 L/capita/day)")
    print(f"[INFO] Min: {df['total_consumption_l'].min():.2f} L | Max: {df['total_consumption_l'].max():.2f} L")


if __name__ == "__main__":
    main()
