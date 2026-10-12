"""
app.py - Streamlit Web Application
Smart Water Consumption Predictor Using Machine Learning
"""

import datetime
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Configure wide layout and page metadata
st.set_page_config(
    page_title="Smart Water Consumption Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project root path setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    load_dataset,
    calculate_consumption_summaries,
    validate_user_input,
    append_user_reading,
    aggregate_by_period,
    CATEGORY_COLUMNS
)
from src.model import (
    load_trained_model,
    predict_next_day,
    forecast_next_n_days
)
from src.recommender import (
    analyze_consumption_profile,
    calculate_target_tracking,
    detect_leak_or_anomaly,
    generate_conservation_recommendations,
    get_campus_extension_recommendations,
    BIS_BENCHMARK_LPCD
)
from src.visualization import (
    plot_historical_and_forecast,
    plot_category_breakdown,
    plot_weekly_monthly_aggregation,
    plot_target_progress_gauge,
    plot_model_residuals
)

# ----------------------------------------------------
# Custom CSS Styling for Modern & Polished Aesthetics
# ----------------------------------------------------
st.markdown("""
<style>
    /* Global styles */
    .main {
        background-color: #f8fafc;
    }
    
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #075985 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.18);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 15px;
        opacity: 0.92;
        margin-bottom: 0px;
        line-height: 1.5;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(2, 132, 199, 0.08);
    }
    .kpi-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .kpi-subtext {
        font-size: 12px;
        color: #64748b;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-blue { background-color: #e0f2fe; color: #0284c7; }
    .badge-green { background-color: #d1fae5; color: #059669; }
    .badge-amber { background-color: #fef3c7; color: #d97706; }
    .badge-red { background-color: #fee2e2; color: #dc2626; }

    /* Recommendation Card */
    .tip-card {
        background: #ffffff;
        border-left: 5px solid #0284c7;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .tip-title {
        font-size: 16px;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .tip-action {
        font-size: 14px;
        color: #334155;
        line-height: 1.5;
        margin-bottom: 8px;
    }
    .tip-metric {
        font-size: 13px;
        font-weight: 600;
        color: #059669;
    }

    /* Disclaimer box */
    .disclaimer-card {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 14px 18px;
        color: #92400e;
        font-size: 13px;
        line-height: 1.5;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# Session State & Cache Initialization
# ----------------------------------------------------
@st.cache_data
def get_base_data() -> pd.DataFrame:
    """Loads baseline dataset from disk with cache."""
    return load_dataset()


@st.cache_resource
def get_cached_model():
    """Loads trained ML champion model, metadata, and evaluation metrics."""
    return load_trained_model()


if "working_df" not in st.session_state:
    st.session_state.working_df = get_base_data().copy()

# Load model artifacts
try:
    model, metadata, metrics_df = get_cached_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ----------------------------------------------------
# Sidebar Controls & Parameters
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Household Parameters")
    
    current_df = st.session_state.working_df
    default_h_size = int(current_df["household_size"].iloc[-1]) if "household_size" in current_df.columns else 4
    
    household_size = st.number_input(
        "Household Size (Persons)",
        min_value=1,
        max_value=20,
        value=default_h_size,
        step=1,
        help="Number of regular household residents (used for per-capita calculations)."
    )

    # Standard BIS target baseline: 135 L/person/day
    suggested_target = float(household_size * BIS_BENCHMARK_LPCD)
    
    daily_target_l = st.number_input(
        "Daily Target Budget (Litres)",
        min_value=50.0,
        max_value=4000.0,
        value=suggested_target,
        step=25.0,
        help="Your conservation threshold. BIS IS 1172 recommended benchmark is 135 L/capita/day."
    )

    st.markdown("---")
    st.markdown("### 🌤️ Weather Forecast Inputs")
    forecast_temp = st.slider(
        "Expected Temperature (°C)",
        min_value=12.0,
        max_value=46.0,
        value=29.0,
        step=0.5,
        help="Ambient temperature influences drinking, showers, and garden evaporation."
    )
    
    forecast_rain = st.slider(
        "Expected Rainfall (mm)",
        min_value=0.0,
        max_value=60.0,
        value=0.0,
        step=1.0,
        help="Rainfall reduces outdoor gardening and vehicle cleaning usage."
    )

    st.markdown("---")
    st.markdown("### 🔄 Dataset Management")
    if st.button("Reset to Clean Dataset", use_container_width=True):
        st.session_state.working_df = get_base_data().copy()
        st.success("Dataset reset to original 731-day baseline!")
        st.rerun()

    # st.markdown("---")
    # st.markdown(
    #     "<div style='font-size: 11px; color: #64748b;'>"
    #     "<b>Project:</b> 6-Week AI/ML Internship<br>"
    #     "<b>Model:</b> " + (metadata.get('best_model_name', 'Ridge Regression') if model_loaded else 'Not loaded') + "<br>"
    #     "<b>Standards:</b> BIS IS 1172:1993 / WHO"
    #     "</div>",
    #     unsafe_allow_html=True
    # )


# ----------------------------------------------------
# Main Header Banner
# ----------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">
        <span>💧 Smart Water Consumption Predictor</span>
        <span class="badge badge-green">v1.0 Live</span>
    </div>
    <div class="hero-subtitle">
        AI-powered residential water demand analytics, next-day forecasting, target budget tracking, 
        and evidence-based conservation recommendations. Extensible to university campuses and hostels.
    </div>
</div>
""", unsafe_allow_html=True)


# Compute current summaries
summaries = calculate_consumption_summaries(st.session_state.working_df)
latest_consumption = summaries["latest_consumption_l"]
latest_date_str = summaries["latest_date"]
recent_7d_avg = summaries["recent_7d_avg_l"]
per_capita_daily = round(latest_consumption / max(1, household_size), 1)

# Generate Prediction for Next Day
if model_loaded:
    pred_val, lower_bound, upper_bound = predict_next_day(
        model=model,
        historical_df=st.session_state.working_df,
        household_size=household_size,
        temperature_c=forecast_temp,
        rainfall_mm=forecast_rain
    )
    forecast_7d_df = forecast_next_n_days(
        model=model,
        historical_df=st.session_state.working_df,
        n_days=7,
        household_size=household_size,
        avg_temperature_c=forecast_temp,
        rainfall_probs=[forecast_rain] * 7
    )
    avg_7d_forecast = round(float(forecast_7d_df["predicted_consumption_l"].mean()), 1)
else:
    pred_val, lower_bound, upper_bound = recent_7d_avg, recent_7d_avg - 30, recent_7d_avg + 30
    forecast_7d_df = pd.DataFrame()
    avg_7d_forecast = recent_7d_avg

# Budget Tracking against Latest Reading
target_info = calculate_target_tracking(latest_consumption, daily_target_l)


# ----------------------------------------------------
# Navigation Tabs
# ----------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Predictions & Forecast",
    "📈 Analytics & Trends",
    "🎯 Target & Conservation Tips",
    "🤖 ML Model Evaluation",
    "📖 Dataset & Methodology"
])


# ====================================================
# TAB 1: PREDICTIONS & FORECAST
# ====================================================
with tab1:
    # 4 Quick KPI Summary Cards
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Latest Recorded Usage</div>
            <div class="kpi-value">{latest_consumption:.1f} <span style="font-size:16px;">L</span></div>
            <div class="kpi-subtext">📅 Recorded on {latest_date_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #0284c7;">
            <div class="kpi-label">Predicted Next-Day</div>
            <div class="kpi-value" style="color: #0284c7;">{pred_val:.1f} <span style="font-size:16px;">L</span></div>
            <div class="kpi-subtext">Interval: {lower_bound:.0f} – {upper_bound:.0f} L (±MAE)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">7-Day Forecast Avg</div>
            <div class="kpi-value">{avg_7d_forecast:.1f} <span style="font-size:16px;">L</span></div>
            <div class="kpi-subtext">Projected next week daily average</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi4:
        budget_color = target_info["budget_color"]
        budget_status = target_info["budget_status"]
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid {budget_color};">
            <div class="kpi-label">Target Budget Progress</div>
            <div class="kpi-value" style="color: {budget_color};">{target_info['pct_used']:.0f}%</div>
            <div class="kpi-subtext">{budget_status} ({target_info['delta_l']:+.1f} L)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Historical vs Forecast Trend Chart
    st.subheader("Historical vs. Predicted Consumption Trend")
    col_hist_ctrl, _ = st.columns([2, 4])
    with col_hist_ctrl:
        history_window = st.select_slider(
            "Historical Display Window (Days)",
            options=[14, 30, 45, 60, 90, 180, 365],
            value=45
        )

    fig_trend = plot_historical_and_forecast(
        historical_df=st.session_state.working_df,
        forecast_df=forecast_7d_df,
        recent_days=history_window,
        target_daily_l=daily_target_l
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # 7-Day Multi-Step Forecast Breakdown
    st.markdown("---")
    st.subheader("🗓️ 7-Day Forecast Schedule & Daily Breakdown")
    if not forecast_7d_df.empty:
        col_table, col_fchart = st.columns([3, 3])
        with col_table:
            display_f_df = forecast_7d_df[[
                "date", "day_name", "predicted_consumption_l", "lower_bound_l", "upper_bound_l", "per_capita_predicted_l"
            ]].copy()
            display_f_df.columns = [
                "Forecast Date", "Day", "Predicted (L)", "Lower (L)", "Upper (L)", "Per Capita (L/day)"
            ]
            st.dataframe(display_f_df, hide_index=True, use_container_width=True)
            
            st.caption(
                "💡 **Forecast Note:** Multi-step forecasting is performed recursively, dynamically propagating "
                "prior day predictions into lag and 7-day rolling aggregates."
            )

        with col_fchart:
            # Bar chart of 7-day predictions
            fig_fc_bar = plot_historical_and_forecast(
                historical_df=st.session_state.working_df.tail(7),
                forecast_df=forecast_7d_df,
                recent_days=7,
                target_daily_l=daily_target_l
            )
            fig_fc_bar.update_layout(title="7-Day Horizon Detailed Forecast Curve", height=320)
            st.plotly_chart(fig_fc_bar, use_container_width=True)

    # Form to Log New Daily Reading
    st.markdown("---")
    st.subheader("📝 Log Today's Water Consumption")
    st.markdown(
        "Record a new meter reading to append to the active time-series and instantly recalculate forecasts. "
        "Inputs are strictly validated against physical boundaries."
    )

    with st.form("log_reading_form"):
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            log_date = st.date_input(
                "Reading Date",
                value=datetime.date.today()
            )
        with col_in2:
            log_total_l = st.number_input(
                "Total Water Consumed (Litres)",
                min_value=1.0,
                max_value=8000.0,
                value=float(latest_consumption),
                step=5.0
            )
        with col_in3:
            log_household_size = st.number_input(
                "Household Size for this date",
                min_value=1,
                max_value=20,
                value=household_size,
                step=1
            )

        with st.expander("Optional: Detailed Category Breakdown (Litres)"):
            st.caption("Leave blank to automatically apportion based on standard residential ratios.")
            col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
            with col_c1:
                cat_bath = st.number_input("Bathing / Shower", min_value=0.0, value=round(log_total_l * 0.35, 1))
            with col_c2:
                cat_toilet = st.number_input("Toilet Flushing", min_value=0.0, value=round(log_total_l * 0.22, 1))
            with col_c3:
                cat_kitchen = st.number_input("Kitchen & Cooking", min_value=0.0, value=round(log_total_l * 0.15, 1))
            with col_c4:
                cat_laundry = st.number_input("Laundry Washing", min_value=0.0, value=round(log_total_l * 0.18, 1))
            with col_c5:
                cat_outdoor = st.number_input("Outdoors / Cleaning", min_value=0.0, value=round(log_total_l * 0.10, 1))

        submit_reading = st.form_submit_button("Log Reading & Update Predictions", use_container_width=True)

        if submit_reading:
            cats = {
                "usage_bathroom_l": cat_bath,
                "usage_toilet_l": cat_toilet,
                "usage_kitchen_l": cat_kitchen,
                "usage_laundry_l": cat_laundry,
                "usage_outdoor_l": cat_outdoor
            }
            # Validation
            is_valid, err_msg = validate_user_input(
                total_l=log_total_l,
                household_size=log_household_size,
                categories=cats
            )

            if not is_valid:
                st.error(f"❌ Input Validation Error: {err_msg}")
            else:
                updated_df = append_user_reading(
                    df=st.session_state.working_df,
                    date_str=log_date.strftime("%Y-%m-%d"),
                    total_l=log_total_l,
                    household_size=log_household_size,
                    temperature_c=forecast_temp,
                    rainfall_mm=forecast_rain,
                    categories=cats
                )
                st.session_state.working_df = updated_df
                st.success(f"✅ Successfully logged {log_total_l:.1f} Litres for {log_date.strftime('%Y-%m-%d')}!")
                st.rerun()

    # Disclaimer Note
    st.markdown("""
    <div class="disclaimer-card">
        <b>⚠️ Scientific Disclaimer:</b> Predictions generated by this application are empirical statistical estimates 
        derived from historical consumption trends, calendar cycles, and environmental factors. They are intended for 
        conservation guidance and demand planning, not certified physical meter readings.
    </div>
    """, unsafe_allow_html=True)


# ====================================================
# TAB 2: ANALYTICS & TRENDS
# ====================================================
with tab2:
    st.subheader("Consumption Trends & Longitudinal Patterns")
    
    col_an_kpi1, col_an_kpi2, col_an_kpi3, col_an_kpi4 = st.columns(4)
    with col_an_kpi1:
        st.metric("All-Time Daily Average", f"{summaries['average_daily_l']} L")
    with col_an_kpi2:
        st.metric("Per-Capita Average (LPCD)", f"{summaries['per_capita_daily_l']} L/day", help="Litres Per Capita per Day")
    with col_an_kpi3:
        st.metric("Weekday Average", f"{summaries['weekday_avg_l']} L")
    with col_an_kpi4:
        diff_wknd = summaries['weekend_avg_l'] - summaries['weekday_avg_l']
        st.metric("Weekend Average", f"{summaries['weekend_avg_l']} L", delta=f"{diff_wknd:+.1f} L vs Weekday")

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart_left, col_chart_right = st.columns([3, 2])
    
    # Weekly & Monthly Aggregates
    weekly_agg = aggregate_by_period(st.session_state.working_df, freq="W")
    monthly_agg = aggregate_by_period(st.session_state.working_df, freq="ME")
    fig_weekly, fig_monthly = plot_weekly_monthly_aggregation(weekly_agg.tail(16), monthly_agg.tail(12))

    with col_chart_left:
        agg_choice = st.radio("Aggregation Level", ["Weekly Aggregation", "Monthly Aggregation"], horizontal=True)
        if agg_choice == "Weekly Aggregation":
            st.plotly_chart(fig_weekly, use_container_width=True)
        else:
            st.plotly_chart(fig_monthly, use_container_width=True)

    with col_chart_right:
        # Category Breakdown of latest row
        latest_row = st.session_state.working_df.iloc[-1]
        active_cats = {
            "usage_bathroom_l": float(latest_row.get("usage_bathroom_l", latest_consumption * 0.35)),
            "usage_toilet_l": float(latest_row.get("usage_toilet_l", latest_consumption * 0.22)),
            "usage_kitchen_l": float(latest_row.get("usage_kitchen_l", latest_consumption * 0.15)),
            "usage_laundry_l": float(latest_row.get("usage_laundry_l", latest_consumption * 0.18)),
            "usage_outdoor_l": float(latest_row.get("usage_outdoor_l", latest_consumption * 0.10)),
        }
        fig_donut = plot_category_breakdown(active_cats)
        st.plotly_chart(fig_donut, use_container_width=True)

    # Anomaly & Spike Detection Table
    st.markdown("---")
    st.subheader("🔍 Automated Anomaly & Outlier Detection")
    st.markdown("Statistical outlier scanning (records exceeding $2.5\\sigma$ over the 7-day rolling window):")
    
    # Compute z-scores for all records
    df_scan = st.session_state.working_df.copy()
    df_scan["rolling_7"] = df_scan["total_consumption_l"].shift(1).rolling(7).mean()
    df_scan["rolling_std"] = df_scan["total_consumption_l"].shift(1).rolling(7).std().fillna(15.0)
    df_scan["z_score"] = (df_scan["total_consumption_l"] - df_scan["rolling_7"]) / (df_scan["rolling_std"] + 1e-5)
    
    anomalies = df_scan[df_scan["z_score"] >= 2.5].copy()
    if not anomalies.empty:
        anom_display = anomalies[["date", "total_consumption_l", "rolling_7", "z_score", "guests_present"]].copy()
        anom_display["date"] = anom_display["date"].dt.strftime("%Y-%m-%d")
        anom_display.columns = ["Date", "Actual (L)", "7-Day Avg (L)", "Z-Score", "Guests Present"]
        st.dataframe(anom_display.tail(10), hide_index=True, use_container_width=True)
    else:
        st.info("No statistically significant consumption spikes identified in the recent period.")


# ====================================================
# TAB 3: TARGET TRACKER & CONSERVATION TIPS
# ====================================================
with tab3:
    st.subheader("🎯 Water Budget Target Tracking")
    
    col_gauge, col_profile = st.columns([2, 3])
    
    with col_gauge:
        fig_gauge = plot_target_progress_gauge(latest_consumption, daily_target_l)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_profile:
        profile = analyze_consumption_profile(
            daily_consumption_l=latest_consumption,
            household_size=household_size,
            target_daily_l=daily_target_l
        )
        
        st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:18px;">
            <div style="font-size:16px; font-weight:700; color:{profile['status_color']}; margin-bottom:8px;">
                Status: {profile['tier']}
            </div>
            <div style="font-size:14px; color:#334155; margin-bottom:12px;">
                {profile['assessment']}
            </div>
            <div style="font-size:13px; color:#64748b; line-height:1.7;">
                • <b>Per-Capita Usage:</b> {profile['lpcd']} Litres/person/day<br>
                • <b>National Standard (BIS IS 1172):</b> {profile['benchmark_lpcd']:.0f} Litres/person/day<br>
                • <b>Variance from National Standard:</b> {profile['variance_from_benchmark_l']:+.1f} Litres/day<br>
                • <b>Target Budget Delta:</b> {target_info['delta_l']:+.1f} Litres/day ({target_info['budget_status']})<br>
                • <b>30-Day Projected Budget Impact:</b> {target_info['monthly_projected_delta_l']:+.0f} Litres
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Leak Detection Diagnostic
    st.markdown("---")
    st.subheader("🩺 Real-Time Leak Diagnostic")
    is_leak, leak_msg = detect_leak_or_anomaly(
        daily_consumption_l=latest_consumption,
        rolling_mean_7=recent_7d_avg,
        rolling_std_7=float(st.session_state.working_df["total_consumption_l"].tail(14).std())
    )
    if is_leak:
        st.warning(f"⚠️ **Alert:** {leak_msg}")
    else:
        st.success(f"✅ **Diagnostic Result:** {leak_msg}")

    # Personalized Conservation Tips
    st.markdown("---")
    st.subheader("💡 Tailored Conservation Recommendations")
    st.markdown("Prioritized, quantified recommendations calculated based on your household size and usage categories:")

    tips = generate_conservation_recommendations(
        daily_consumption_l=latest_consumption,
        household_size=household_size,
        categories=active_cats,
        target_daily_l=daily_target_l
    )

    total_potential_daily_savings = sum(t["estimated_daily_saving_l"] for t in tips)
    total_potential_monthly_savings = sum(t["estimated_monthly_saving_l"] for t in tips)

    st.markdown(
        f"**Potential Household Water Savings:** Up to **{total_potential_daily_savings:.0f} Litres/day** "
        f"(~**{total_potential_monthly_savings:,.0f} Litres/month**) by adopting the measures below:"
    )

    for tip in tips:
        priority_badge = (
            '<span class="badge badge-red">High Priority</span>' if tip['priority'] == 'High'
            else '<span class="badge badge-amber">Medium Priority</span>' if tip['priority'] == 'Medium'
            else '<span class="badge badge-blue">Preventive</span>'
        )
        st.markdown(f"""
        <div class="tip-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="tip-title">{tip['title']}</span>
                {priority_badge}
            </div>
            <div class="tip-action">{tip['action']}</div>
            <div style="display:flex; gap:16px;">
                <span class="tip-metric">💧 Est. Savings: {tip['estimated_daily_saving_l']:.0f} L/day ({tip['estimated_monthly_saving_l']:.0f} L/mo)</span>
                <span style="font-size:12px; color:#64748b;">🛠️ Feasibility: {tip['difficulty']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Campus & Hostel Extension Blueprint
    st.markdown("---")
    st.subheader("🏫 Architecture Extension: College Campuses & Hostels")
    st.markdown(
        "While this model is demonstrated on single-household telemetry, its predictive architecture "
        "is directly extensible to institutional student hostels and academic departments:"
    )
    campus_recs = get_campus_extension_recommendations()
    col_c1, col_c2 = st.columns(2)
    for i, rec in enumerate(campus_recs):
        target_col = col_c1 if (i % 2 == 0) else col_c2
        with target_col:
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:14px; margin-bottom:12px;">
                <div style="font-weight:600; color:#0284c7; font-size:14px;">{rec['sector']}: {rec['strategy']}</div>
                <div style="font-size:13px; color:#334155; margin-top:4px; line-height:1.4;">{rec['description']}</div>
            </div>
            """, unsafe_allow_html=True)


# ====================================================
# TAB 4: ML MODEL EVALUATION & TRANSPARENCY
# ====================================================
with tab4:
    st.subheader("Model Evaluation & Empirical Comparison")
    st.markdown(
        "Candidate models were evaluated against an un-shuffled chronological test set (last 20% of records). "
        "The winning model was selected purely on empirical generalization accuracy, avoiding subjective bias."
    )

    if model_loaded:
        # Display Metrics Comparison Table
        st.markdown("### 🏆 Chronological Holdout Evaluation Results")
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        # Champion Banner
        best_name = metadata.get("best_model_name", "Ridge Linear Regression")
        best_mae = metadata.get("best_metrics", {}).get("MAE_Litres", 43.16)
        best_rmse = metadata.get("best_metrics", {}).get("RMSE_Litres", 56.67)
        best_r2 = metadata.get("best_metrics", {}).get("R2_Score", 0.6239)

        st.success(
            f"🏅 **Selected Champion Model:** **{best_name}** | "
            f"**Test MAE:** {best_mae} L | **Test RMSE:** {best_rmse} L | **R²:** {best_r2}"
        )

        # Metrics Plain-English Explanation
        st.markdown("### 📚 Understanding the Evaluation Metrics")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            - **MAE (Mean Absolute Error):** Measures average absolute deviation between actual and predicted consumption in litres. An MAE of ~43 L means predictions are on average within ±43 Litres (~8% of total daily volume).
            - **RMSE (Root Mean Squared Error):** Squaring deviations penalizes large outlier spikes much more severely. A lower RMSE indicates the model handles seasonal surges smoothly.
            """)
        with col_m2:
            st.markdown("""
            - **MAPE (Mean Absolute Percentage Error):** Normalized percentage error relative to actual volume. A MAPE under 10% is classified as high forecasting accuracy in industrial hydrological forecasting.
            - **R² Score (Coefficient of Determination):** Measures the proportion of variance explained by the model over a simple mean baseline.
            """)

        # Feature Importance / Weights
        st.markdown("---")
        col_fi, col_res = st.columns(2)
        
        with col_fi:
            st.subheader("Top Predictive Features")
            fi_dict = metadata.get("feature_importances", {})
            if fi_dict:
                fi_df = pd.DataFrame(list(fi_dict.items()), columns=["Feature", "Relative Weight / Importance"]).head(8)
                fi_df["Relative Weight (%)"] = (fi_df["Relative Weight / Importance"] * 100).round(2)
                st.dataframe(fi_df[["Feature", "Relative Weight (%)"]], hide_index=True, use_container_width=True)
            else:
                st.info("Feature weights available in model_metadata.json.")

        with col_res:
            st.subheader("Holdout Residual Fit")
            test_preds = metadata.get("test_predictions", {})
            if "y_test" in test_preds and "y_pred_best" in test_preds:
                fig_res = plot_model_residuals(test_preds["y_test"], test_preds["y_pred_best"])
                st.plotly_chart(fig_res, use_container_width=True)

    else:
        st.error(f"Could not load model: {model_error}. Please run `python src/train.py`.")

    # Explicit Model Limitations Section
    st.markdown("---")
    st.subheader("⚠️ Model Scope & Limitations")
    st.markdown("""
    1. **Synthetic Demonstration Data:** In accordance with rigorous scientific ethics, this demonstration dataset was generated using BIS IS 1172 standards. It simulates realistic behavioral fluctuations, but cannot replace high-frequency pulse smart water meter telemetry.
    2. **Lag Warmup Requirement:** Lag ($t-7$) and 30-day rolling statistics require at least 30 consecutive days of prior historical records before initial inference is possible.
    3. **Unforeseen Extreme Events:** Sudden anomalies (e.g. major burst pipe or 10 unexpected festival guests) cannot be predicted in advance by seasonal calendar models without sensor telemetry.
    """)


# ====================================================
# TAB 5: DATASET & METHODOLOGY
# ====================================================
with tab5:
    st.subheader("Dataset Provenance & Research Methodology")
    
    st.markdown("""
    ### 1. Dataset Origin & Benchmark Standards
    Public municipal water datasets typically offer coarse annual or quarterly postal-code aggregates (e.g. Barcelona Open Data) 
    or require proprietary credentials. To provide a completely reproducible, transparent foundation for this 6-week internship project:
    - **Benchmark Base:** Calibrated against **Bureau of Indian Standards (BIS IS 1172: 1993)** recommended domestic supply benchmark of **135 to 150 Litres/capita/day (LPCD)**.
    - **Category Ratios:** Bathing (35%), Toilet Flushing (22%), Laundry (18%), Kitchen (15%), Outdoors (10%).
    - **Duration:** 731 continuous daily observations (2024-01-01 to 2025-12-31).
    - **Features:** Date, household size, guests, temperature (°C), rainfall (mm), day-of-week, weekend flag, category breakdowns.

    ### 2. Time-Series Modeling Methodology & Leakage Prevention
    - **Chronological Split:** The dataset is partitioned chronologically (First 80% Train, Final 20% Test). Shuffled cross-validation is strictly forbidden in time-series to prevent future data from leaking into the past.
    - **Strict Shifted Aggregation:** All rolling features (7-day, 14-day, 30-day means) and lag variables ($y_{t-1}, y_{t-7}$) are calculated using `.shift(1)`. The observation at time $t$ is NEVER included in features for time $t$.

    ### 3. Internship Project Roadmap (6-Week Timeline)
    - **Week 1:** Literature review, municipal water benchmark analysis, dataset synthesis & EDA.
    - **Week 2:** Data cleaning, missing value strategies, and strict time-series feature engineering.
    - **Week 3:** Baseline model implementation (Naive persistence, rolling average) and candidate ML training.
    - **Week 4:** Model evaluation, hyperparameter tuning, model persistence with Joblib, and leak prevention audit.
    - **Week 5:** Interactive Streamlit dashboard creation, Plotly visualizations, and conservation recommender rules.
    - **Week 6:** Unit testing with Pytest, documentation, Lean Canvas, Concept Note, and final presentation synthesis.
    """)

    st.markdown("---")
    st.markdown("### 📥 Raw Dataset Sample Inspection")
    st.dataframe(st.session_state.working_df.tail(10), hide_index=True, use_container_width=True)
