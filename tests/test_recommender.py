"""
test_recommender.py - Unit tests for conservation recommender, leak alerts, and target tracking
"""

import pytest

from src.recommender import (
    analyze_consumption_profile,
    calculate_target_tracking,
    detect_leak_or_anomaly,
    generate_conservation_recommendations,
    BIS_BENCHMARK_LPCD
)


def test_analyze_consumption_profile():
    # 4 persons consuming 400 L/day -> 100 LPCD (Eco-Conscious)
    profile_eco = analyze_consumption_profile(daily_consumption_l=400.0, household_size=4)
    assert profile_eco["lpcd"] == 100.0
    assert profile_eco["tier"] == "Eco-Conscious & Sustainable"

    # 4 persons consuming 900 L/day -> 225 LPCD (Excessive)
    profile_high = analyze_consumption_profile(daily_consumption_l=900.0, household_size=4)
    assert profile_high["lpcd"] == 225.0
    assert "High Consumption" in profile_high["tier"]


def test_calculate_target_tracking():
    # Target 500 L, actual 450 L (within budget)
    res_under = calculate_target_tracking(consumption_l=450.0, target_daily_l=500.0)
    assert res_under["is_within_target"] is True
    assert res_under["delta_l"] == -50.0
    assert res_under["pct_used"] == 90.0
    assert res_under["monthly_projected_delta_l"] == -1500.0

    # Target 500 L, actual 600 L (over budget)
    res_over = calculate_target_tracking(consumption_l=600.0, target_daily_l=500.0)
    assert res_over["is_within_target"] is False
    assert res_over["delta_l"] == 100.0
    assert res_over["pct_used"] == 120.0
    assert res_over["monthly_projected_delta_l"] == 3000.0


def test_detect_leak_or_anomaly():
    # Normal usage
    is_spike, msg = detect_leak_or_anomaly(daily_consumption_l=520.0, rolling_mean_7=510.0, rolling_std_7=20.0)
    assert is_spike is False

    # Massive spike (e.g. 510 + 3.5 * 20 = 580+)
    is_spike, msg = detect_leak_or_anomaly(daily_consumption_l=650.0, rolling_mean_7=500.0, rolling_std_7=20.0)
    assert is_spike is True
    assert "Unusual spike detected" in msg


def test_generate_conservation_recommendations():
    tips = generate_conservation_recommendations(
        daily_consumption_l=700.0,
        household_size=4,
        categories={
            "usage_bathroom_l": 280.0,
            "usage_toilet_l": 160.0,
            "usage_kitchen_l": 100.0,
            "usage_laundry_l": 90.0,
            "usage_outdoor_l": 70.0
        }
    )

    assert len(tips) >= 3
    # Check each tip has structured fields
    for tip in tips:
        assert "category" in tip
        assert "title" in tip
        assert "action" in tip
        assert tip["estimated_daily_saving_l"] > 0
