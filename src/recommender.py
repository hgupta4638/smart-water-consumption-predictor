"""
recommender.py - Rule-based water conservation recommendation and target tracking engine
Project: Smart Water Consumption Predictor Using Machine Learning

Standards Reference:
- Bureau of Indian Standards (BIS IS 1172: 1993): 135 L/capita/day residential benchmark
- World Health Organization (WHO): Minimum 50-100 LPCD for health & hygiene
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


BIS_BENCHMARK_LPCD = 135.0  # Litres per capita per day


def analyze_consumption_profile(
    daily_consumption_l: float,
    household_size: int,
    target_daily_l: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyzes household consumption against national/global hydrological benchmarks.
    """
    household_size = max(1, household_size)
    lpcd = daily_consumption_l / household_size
    standard_expected_l = household_size * BIS_BENCHMARK_LPCD

    # Benchmark efficiency ratio
    ratio_to_benchmark = lpcd / BIS_BENCHMARK_LPCD

    if lpcd < 90.0:
        tier = "Ultra-Conserving"
        status_color = "#10b981"  # Emerald
        assessment = "Your water usage is remarkably low, well below standard urban baselines."
    elif lpcd <= 135.0:
        tier = "Eco-Conscious & Sustainable"
        status_color = "#059669"  # Green
        assessment = "Your consumption satisfies BIS IS 1172 sustainable benchmarks comfortably."
    elif lpcd <= 175.0:
        tier = "Moderate / Above Average"
        status_color = "#f59e0b"  # Amber
        assessment = "Your consumption is moderately elevated compared to standard residential benchmarks."
    else:
        tier = "High Consumption / Water Waste Risk"
        status_color = "#ef4444"  # Red
        assessment = "Usage significantly exceeds optimal per-capita benchmarks. Immediate conservation actions recommended."

    profile = {
        "daily_consumption_l": round(daily_consumption_l, 1),
        "household_size": household_size,
        "lpcd": round(lpcd, 1),
        "benchmark_lpcd": BIS_BENCHMARK_LPCD,
        "standard_expected_l": round(standard_expected_l, 1),
        "variance_from_benchmark_l": round(daily_consumption_l - standard_expected_l, 1),
        "ratio_to_benchmark": round(ratio_to_benchmark, 2),
        "tier": tier,
        "status_color": status_color,
        "assessment": assessment
    }

    if target_daily_l is not None and target_daily_l > 0:
        profile.update(calculate_target_tracking(daily_consumption_l, target_daily_l))

    return profile


def calculate_target_tracking(
    consumption_l: float,
    target_daily_l: float
) -> Dict[str, Any]:
    """
    Computes budget progress and variance metrics against user-defined consumption target.
    """
    target_daily_l = max(1.0, float(target_daily_l))
    pct_used = (consumption_l / target_daily_l) * 100.0
    delta_l = consumption_l - target_daily_l
    is_within_target = delta_l <= 0.0

    monthly_projected_delta_l = delta_l * 30.0

    if is_within_target:
        budget_status = "Within Target Budget"
        budget_color = "#10b981"
        summary_msg = f"On track! Consuming {abs(delta_l):.1f} L less than your daily target (Projected monthly saving: {abs(monthly_projected_delta_l):.0f} L)."
    else:
        budget_status = "Exceeding Target Budget"
        budget_color = "#ef4444"
        summary_msg = f"Over target! Exceeding daily budget by {delta_l:.1f} L (Projected monthly excess: {monthly_projected_delta_l:.0f} L)."

    return {
        "target_daily_l": round(target_daily_l, 1),
        "consumption_l": round(consumption_l, 1),
        "delta_l": round(delta_l, 1),
        "pct_used": round(pct_used, 1),
        "is_within_target": is_within_target,
        "budget_status": budget_status,
        "budget_color": budget_color,
        "monthly_projected_delta_l": round(monthly_projected_delta_l, 1),
        "summary_msg": summary_msg
    }


def detect_leak_or_anomaly(
    daily_consumption_l: float,
    rolling_mean_7: float,
    rolling_std_7: float
) -> Tuple[bool, str]:
    """
    Detects potential unmetered leaks or abnormal consumption spikes using statistical z-scores.
    """
    if rolling_std_7 <= 1.0:
        rolling_std_7 = 15.0

    z_score = (daily_consumption_l - rolling_mean_7) / rolling_std_7

    if z_score >= 2.5:
        return (
            True,
            f"Unusual spike detected (Z-score: +{z_score:.2f}). Consumption is {daily_consumption_l - rolling_mean_7:.1f} L "
            "above your 7-day average. Check for toilet flapper leaks, continuously open garden valves, or overflow tanks."
        )
    elif z_score <= -2.5:
        return (
            False,
            f"Unusually low consumption recorded ({daily_consumption_l:.1f} L vs {rolling_mean_7:.1f} L avg). "
            "Likely due to occupants traveling or temporary supply restriction."
        )
    return False, "Normal pattern detected within expected statistical variation."


def generate_conservation_recommendations(
    daily_consumption_l: float,
    household_size: int,
    categories: Optional[Dict[str, float]] = None,
    target_daily_l: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Generates actionable, prioritized, quantified conservation recommendations.
    """
    household_size = max(1, household_size)
    lpcd = daily_consumption_l / household_size
    tips: List[Dict[str, Any]] = []

    # Category proportions
    cats = categories or {
        "usage_bathroom_l": daily_consumption_l * 0.35,
        "usage_toilet_l": daily_consumption_l * 0.22,
        "usage_kitchen_l": daily_consumption_l * 0.15,
        "usage_laundry_l": daily_consumption_l * 0.18,
        "usage_outdoor_l": daily_consumption_l * 0.10,
    }

    bath_lpcd = cats.get("usage_bathroom_l", 0) / household_size
    toilet_lpcd = cats.get("usage_toilet_l", 0) / household_size
    kitchen_lpcd = cats.get("usage_kitchen_l", 0) / household_size
    laundry_total = cats.get("usage_laundry_l", 0)
    outdoor_total = cats.get("usage_outdoor_l", 0)

    # 1. Bathroom / Shower Recommendation
    if bath_lpcd > 45.0 or lpcd > 140.0:
        tips.append({
            "category": "Bathroom & Bathing",
            "priority": "High" if bath_lpcd > 55.0 else "Medium",
            "title": "Fit Low-Flow Tap Aerators & Restrict Shower Duration",
            "action": "Retrofit faucets with 3-star aerators and limit showers to 4-5 minutes (or use bucket baths). Standard shower heads flow at 15-20 L/min, while aerated heads use 6-8 L/min.",
            "estimated_daily_saving_l": round(household_size * 25.0, 1),
            "estimated_monthly_saving_l": round(household_size * 25.0 * 30.0, 0),
            "difficulty": "Easy (Under $5 DIY fix)"
        })

    # 2. Toilet Flushing Recommendation
    if toilet_lpcd > 30.0 or lpcd > 135.0:
        tips.append({
            "category": "Toilet Flushing",
            "priority": "High",
            "title": "Upgrade to Dual-Flush Cistern or Install Cistern Displacement Bag",
            "action": "Traditional single-flush cisterns release 10-12 L per flush. Dual-flush mechanisms reduce short-flush volume to 3 L, saving significant water throughout the day.",
            "estimated_daily_saving_l": round(household_size * 20.0, 1),
            "estimated_monthly_saving_l": round(household_size * 20.0 * 30.0, 0),
            "difficulty": "Medium (Plumbing retrofit)"
        })

    # 3. Laundry & Washing Recommendation
    if laundry_total > 80.0 or (laundry_total / household_size) > 25.0:
        tips.append({
            "category": "Laundry & Washing",
            "priority": "Medium",
            "title": "Consolidate Laundry to Full Loads & Select Eco-Cycles",
            "action": "Washing machines consume roughly 45-70 L per cycle regardless of load size. Only run full loads and skip redundant extra rinses.",
            "estimated_daily_saving_l": 35.0,
            "estimated_monthly_saving_l": 35.0 * 30.0,
            "difficulty": "Immediate (Behavioral)"
        })

    # 4. Kitchen & Dishwashing Recommendation
    if kitchen_lpcd > 22.0 or lpcd > 150.0:
        tips.append({
            "category": "Kitchen & Cooking",
            "priority": "Medium",
            "title": "Scrape Plates & Wash Dishes in a Basin Rather Than Running Taps",
            "action": "Running a tap continuously while rinsing vegetables or dishes consumes 10-12 L/minute. Using a soak basin saves over 60% of kitchen flow.",
            "estimated_daily_saving_l": round(household_size * 12.0, 1),
            "estimated_monthly_saving_l": round(household_size * 12.0 * 30.0, 0),
            "difficulty": "Immediate (Behavioral)"
        })

    # 5. Outdoor / Gardening / Vehicle Washing
    if outdoor_total > 30.0:
        tips.append({
            "category": "Outdoors & Garden",
            "priority": "Medium",
            "title": "Shift Garden Watering to Dawn/Dusk & Avoid Hosepipes for Cars",
            "action": "Watering during mid-day heat loses 30-40% to evaporation. Use a bucket and microfiber sponge for vehicle cleaning instead of an open running hose.",
            "estimated_daily_saving_l": round(outdoor_total * 0.45, 1),
            "estimated_monthly_saving_l": round(outdoor_total * 0.45 * 30.0, 0),
            "difficulty": "Immediate (Behavioral)"
        })

    # 6. Universal Leak Auditing
    tips.append({
        "category": "Preventive Maintenance",
        "priority": "High" if lpcd > 175.0 else "Low",
        "title": "Perform a Night-time Water Meter / Tank Auditing",
        "action": "Record your water meter or overhead tank level before sleeping and recheck upon waking before any water is used. Any drop indicates an unmetered concealed leak.",
        "estimated_daily_saving_l": 40.0,
        "estimated_monthly_saving_l": 1200.0,
        "difficulty": "Easy (Inspection)"
    })

    return tips


def get_campus_extension_recommendations() -> List[Dict[str, str]]:
    """
    Provides architectural and operational recommendations for extending
    the predictor to College Campuses, Hostels, and Academic Facilities.
    """
    return [
        {
            "sector": "Hostel Residential Blocks",
            "strategy": "Wing-level & Floor-level Smart Sub-Metering",
            "description": "Deploy pulse-output or ultrasonic flow meters per hostel floor. Predict per-block consumption based on student occupancy calendars and exam schedules to flag abnormal common-washroom leakage."
        },
        {
            "sector": "Campus Dining Mess & Canteen",
            "strategy": "Push-Button Foot Valves & Dishwasher Heat-Exchange Recycling",
            "description": "Kitchen prep accounts for high continuous flow during rush hours. Install foot-pedal valves for kitchen staff and graywater recycling for grease traps."
        },
        {
            "sector": "Academic & Admin Buildings",
            "strategy": "Occupancy-Driven Automated Solenoid Shutoff",
            "description": "Automatically shut off secondary restroom header valves after 8:00 PM and on public holidays via scheduled IoT solenoid controllers to prevent overnight phantom leaks."
        },
        {
            "sector": "Sports Grounds & Landscaping",
            "strategy": "Weather-Aware Drip Irrigation & Rainwater Catchment",
            "description": "Integrate soil moisture sensors with next-day rainfall predictions to completely bypass automated campus lawn sprinklers when precipitation is imminent."
        }
    ]
