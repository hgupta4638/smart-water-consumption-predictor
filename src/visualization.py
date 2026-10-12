"""
visualization.py - Plotly interactive charts and data visualizations
Project: Smart Water Consumption Predictor Using Machine Learning
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# Consistent professional color palette
PRIMARY_BLUE = "#0284c7"
TEAL_ACCENT = "#0d9488"
ALERT_RED = "#ef4444"
WARNING_AMBER = "#f59e0b"
SUCCESS_GREEN = "#10b981"
DARK_NAVY = "#0f172a"
LIGHT_BG = "#f8fafc"
GRID_COLOR = "#e2e8f0"


def plot_historical_and_forecast(
    historical_df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
    recent_days: int = 45,
    target_daily_l: Optional[float] = None
) -> go.Figure:
    """
    Renders an interactive time-series chart showing historical consumption
    alongside predicted future forecast with confidence interval.
    """
    df_recent = historical_df.tail(recent_days).copy()

    fig = go.Figure()

    # 1. Historical Actual Consumption
    fig.add_trace(
        go.Scatter(
            x=df_recent["date"],
            y=df_recent["total_consumption_l"],
            mode="lines+markers",
            name="Actual Consumption (L)",
            line=dict(color=PRIMARY_BLUE, width=2.5),
            marker=dict(size=5, color=PRIMARY_BLUE),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> %{y:.1f} Litres<extra></extra>"
        )
    )

    # 2. Target Reference Line (if set)
    if target_daily_l and target_daily_l > 0:
        fig.add_hline(
            y=target_daily_l,
            line_dash="dot",
            line_color=WARNING_AMBER,
            annotation_text=f"Target Budget ({target_daily_l:.0f} L)",
            annotation_position="top right"
        )

    # 3. Forecast Traces (if provided)
    if forecast_df is not None and not forecast_df.empty:
        # Forecast connector: connect last historical point to first forecast point
        last_hist_date = df_recent["date"].iloc[-1]
        last_hist_val = df_recent["total_consumption_l"].iloc[-1]

        conn_dates = [last_hist_date] + list(forecast_df["date"])
        conn_preds = [last_hist_val] + list(forecast_df["predicted_consumption_l"])
        conn_uppers = [last_hist_val] + list(forecast_df["upper_bound_l"])
        conn_lowers = [last_hist_val] + list(forecast_df["lower_bound_l"])

        # Upper bound (invisible line for fill)
        fig.add_trace(
            go.Scatter(
                x=conn_dates,
                y=conn_uppers,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            )
        )

        # Lower bound with shaded uncertainty band
        fig.add_trace(
            go.Scatter(
                x=conn_dates,
                y=conn_lowers,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(13, 148, 136, 0.18)",
                name="Forecast Uncertainty (±MAE)",
                hoverinfo="skip"
            )
        )

        # Forecast Line
        fig.add_trace(
            go.Scatter(
                x=conn_dates,
                y=conn_preds,
                mode="lines+markers",
                name="Predicted Forecast (L)",
                line=dict(color=TEAL_ACCENT, width=3, dash="dash"),
                marker=dict(size=7, symbol="diamond", color=TEAL_ACCENT),
                hovertemplate="<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:.1f} Litres<extra></extra>"
            )
        )

    fig.update_layout(
        title=dict(text=f"Water Consumption Trend & Forecast (Recent {recent_days} Days)", font=dict(size=16, color=DARK_NAVY)),
        xaxis=dict(title="Date", showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(title="Consumption (Litres / Day)", showgrid=True, gridcolor=GRID_COLOR),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )

    return fig


def plot_category_breakdown(
    categories: Dict[str, float]
) -> go.Figure:
    """
    Renders a donut chart displaying usage breakdown by category.
    """
    label_map = {
        "usage_bathroom_l": "Bathing & Shower",
        "usage_toilet_l": "Toilet Flushing",
        "usage_kitchen_l": "Kitchen & Cooking",
        "usage_laundry_l": "Laundry Washing",
        "usage_outdoor_l": "Outdoor & Cleaning"
    }

    labels = [label_map.get(k, k) for k in categories.keys()]
    values = [round(v, 1) for v in categories.values()]

    colors = ["#0284c7", "#0d9488", "#38bdf8", "#818cf8", "#fbbf24"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Volume: %{value:.1f} L<br>Share: %{percent}<extra></extra>"
            )
        ]
    )

    total_litres = sum(values)

    fig.update_layout(
        title=dict(text="Consumption Breakdown by End-Use Category", font=dict(size=15, color=DARK_NAVY)),
        annotations=[
            dict(
                text=f"<b>{total_litres:.0f} L</b><br><span style='font-size:11px;color:#64748b'>Total</span>",
                x=0.5,
                y=0.5,
                font_size=16,
                showarrow=False
            )
        ],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=50),
        paper_bgcolor="#ffffff"
    )

    return fig


def plot_weekly_monthly_aggregation(
    weekly_df: pd.DataFrame,
    monthly_df: pd.DataFrame
) -> Tuple[go.Figure, go.Figure]:
    """
    Renders weekly and monthly aggregated consumption bar charts.
    """
    # Weekly Chart
    fig_weekly = go.Figure(
        data=[
            go.Bar(
                x=weekly_df["date"],
                y=weekly_df["total_consumption_l_sum"],
                marker_color=PRIMARY_BLUE,
                hovertemplate="<b>Week Ending:</b> %{x|%Y-%m-%d}<br><b>Weekly Total:</b> %{y:.0f} L<extra></extra>"
            )
        ]
    )
    fig_weekly.update_layout(
        title=dict(text="Weekly Aggregate Water Consumption (Litres)", font=dict(size=14, color=DARK_NAVY)),
        xaxis=dict(title="Week Ending", showgrid=False),
        yaxis=dict(title="Total Litres", showgrid=True, gridcolor=GRID_COLOR),
        margin=dict(l=30, r=30, t=40, b=30),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )

    # Monthly Chart
    fig_monthly = go.Figure(
        data=[
            go.Bar(
                x=monthly_df["date"].dt.strftime("%b %Y"),
                y=monthly_df["total_consumption_l_sum"],
                marker_color=TEAL_ACCENT,
                hovertemplate="<b>Month:</b> %{x}<br><b>Monthly Total:</b> %{y:.0f} L<extra></extra>"
            )
        ]
    )
    fig_monthly.update_layout(
        title=dict(text="Monthly Aggregate Water Consumption (Litres)", font=dict(size=14, color=DARK_NAVY)),
        xaxis=dict(title="Month", showgrid=False),
        yaxis=dict(title="Total Litres", showgrid=True, gridcolor=GRID_COLOR),
        margin=dict(l=30, r=30, t=40, b=30),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )

    return fig_weekly, fig_monthly


def plot_target_progress_gauge(
    consumption_l: float,
    target_daily_l: float
) -> go.Figure:
    """
    Renders an indicator gauge displaying budget status against daily target.
    """
    pct = min(150.0, (consumption_l / max(1.0, target_daily_l)) * 100.0)
    bar_color = SUCCESS_GREEN if consumption_l <= target_daily_l else ALERT_RED

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(consumption_l, 1),
            delta=dict(reference=target_daily_l, increasing=dict(color=ALERT_RED), decreasing=dict(color=SUCCESS_GREEN), suffix=" L"),
            title=dict(text=f"Daily Target Budget: {target_daily_l:.0f} L", font=dict(size=15)),
            number=dict(suffix=" L", font=dict(size=24)),
            gauge=dict(
                axis=dict(range=[0, max(target_daily_l * 1.5, consumption_l * 1.2)], ticksuffix=" L"),
                bar=dict(color=bar_color),
                steps=[
                    dict(range=[0, target_daily_l], color="rgba(16, 185, 129, 0.15)"),
                    dict(range=[target_daily_l, max(target_daily_l * 1.5, consumption_l * 1.2)], color="rgba(239, 68, 68, 0.15)")
                ],
                threshold=dict(line=dict(color=ALERT_RED, width=4), thickness=0.8, value=target_daily_l)
            )
        )
    )

    fig.update_layout(
        margin=dict(l=25, r=25, t=40, b=25),
        height=240,
        paper_bgcolor="#ffffff"
    )

    return fig


def plot_model_residuals(
    y_test: List[float],
    y_pred: List[float],
    dates: Optional[List[str]] = None
) -> go.Figure:
    """
    Renders actual vs predicted scatter with ideal diagonal for model evaluation transparency.
    """
    min_val = min(min(y_test), min(y_pred)) * 0.95
    max_val = max(max(y_test), max(y_pred)) * 1.05

    fig = go.Figure()

    # Actual vs Predicted points
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            name="Test Predictions",
            marker=dict(color=PRIMARY_BLUE, size=7, opacity=0.8),
            hovertemplate="<b>Actual:</b> %{x:.1f} L<br><b>Predicted:</b> %{y:.1f} L<extra></extra>"
        )
    )

    # 45-degree reference line (Perfect prediction)
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal 1:1 Line",
            line=dict(color="#94a3b8", dash="dash", width=2)
        )
    )

    fig.update_layout(
        title=dict(text="Evaluation: Actual vs Predicted (Holdout Test Set)", font=dict(size=15, color=DARK_NAVY)),
        xaxis=dict(title="Actual Consumption (Litres)", showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(title="Predicted Consumption (Litres)", showgrid=True, gridcolor=GRID_COLOR),
        margin=dict(l=35, r=35, t=45, b=35),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )

    return fig
