"""
Reusable Plotly visualization functions for the Streamlit dashboard.

All functions return plotly.graph_objects.Figure objects with consistent
theming, interactive tooltips, and professional styling.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COLOR_PALETTE, PLOTLY_TEMPLATE, MONTHS


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent dark theme to any figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=title,
            font=dict(size=20, color=COLOR_PALETTE["text"]),
            x=0.5,
        ),
        paper_bgcolor=COLOR_PALETTE["bg_dark"],
        plot_bgcolor=COLOR_PALETTE["bg_dark"],
        font=dict(color=COLOR_PALETTE["text"], family="Inter, sans-serif"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# Temperature Charts
# ═══════════════════════════════════════════════════════════════════

def plot_temperature_heatmap(
    temp_df: pd.DataFrame,
    value_col: str = "Temperature",
    title: str = "Monthly Temperature Heatmap (°C)",
) -> go.Figure:
    """Create a year × month heatmap of temperatures."""
    # Pivot to matrix
    if "Month_Name" in temp_df.columns:
        month_col = "Month_Name"
    elif "Month" in temp_df.columns:
        month_col = "Month"
    else:
        month_col = temp_df.columns[1]

    pivot = temp_df.pivot_table(
        values=value_col, index="Year", columns=month_col, aggfunc="mean"
    )

    # Reorder months
    ordered_months = [m for m in MONTHS if m in pivot.columns]
    if ordered_months:
        pivot = pivot[ordered_months]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="YlOrRd",
        text=np.round(pivot.values, 1),
        texttemplate="%{text}°",
        textfont=dict(size=11),
        hovertemplate="Month: %{x}<br>Year: %{y}<br>Temp: %{z:.1f}°C<extra></extra>",
        colorbar=dict(title="°C"),
    ))

    return _apply_theme(fig, title)


def plot_temperature_trend(
    temp_df: pd.DataFrame,
    value_col: str = "Temperature",
    title: str = "Temperature Trend Over Time",
) -> go.Figure:
    """Line chart of temperature over time."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(temp_df))),
        y=temp_df[value_col],
        mode="lines",
        line=dict(color=COLOR_PALETTE["primary"], width=2),
        fill="tozeroy",
        fillcolor="rgba(255,107,107,0.1)",
        name="Temperature",
        hovertemplate="Index: %{x}<br>Temp: %{y:.1f}°C<extra></extra>",
    ))

    return _apply_theme(fig, title)


def plot_temperature_comparison(
    tn_temps: pd.Series,
    ap_temps: pd.Series,
    title: str = "Tamil Nadu vs Andhra Pradesh — Temperature",
) -> go.Figure:
    """Compare temperature trends of two states."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=tn_temps.values,
        mode="lines",
        name="Tamil Nadu",
        line=dict(color=COLOR_PALETTE["primary"], width=2.5),
    ))

    fig.add_trace(go.Scatter(
        y=ap_temps.values,
        mode="lines",
        name="Andhra Pradesh",
        line=dict(color=COLOR_PALETTE["secondary"], width=2.5),
    ))

    fig.update_xaxes(title_text="Month Index")
    fig.update_yaxes(title_text="Temperature (°C)")

    return _apply_theme(fig, title)


# ═══════════════════════════════════════════════════════════════════
# Electricity Charts
# ═══════════════════════════════════════════════════════════════════

def plot_demand_heatmap(
    demand_df: pd.DataFrame,
    title: str = "Monthly Electricity Peak Demand (MW)",
) -> go.Figure:
    """Heatmap of electricity demand by year and month."""
    demand_col = "Peak Demand (in MW)"
    if demand_col not in demand_df.columns:
        demand_col = demand_df.columns[-1]

    # Parse year from "2024-25" format
    df = demand_df.copy()
    if df["Year"].dtype == object and "-" in str(df["Year"].iloc[0]):
        df["Year_Num"] = df["Year"].str[:4].astype(int)
    else:
        df["Year_Num"] = df["Year"]

    # Create matrix (assumes 12 months per year, data ordered by year desc)
    years = sorted(df["Year_Num"].unique())
    matrix = []
    year_labels = []

    for yr in years:
        yr_data = df[df["Year_Num"] == yr][demand_col].values
        if len(yr_data) >= 12:
            matrix.append(yr_data[:12])
            year_labels.append(str(yr))

    if not matrix:
        return go.Figure()

    matrix = np.array(matrix)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=MONTHS,
        y=year_labels,
        colorscale="Blues",
        text=np.round(matrix, 0).astype(int),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="Month: %{x}<br>Year: %{y}<br>Demand: %{z:,.0f} MW<extra></extra>",
        colorbar=dict(title="MW"),
    ))

    return _apply_theme(fig, title)


def plot_demand_trend(
    demand_df: pd.DataFrame,
    title: str = "Electricity Peak Demand Trend",
) -> go.Figure:
    """Line chart of electricity demand over time."""
    demand_col = "Peak Demand (in MW)"
    if demand_col not in demand_df.columns:
        demand_col = demand_df.columns[-1]

    values = demand_df[demand_col].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values,
        mode="lines",
        line=dict(
            color=COLOR_PALETTE["secondary"],
            width=2,
        ),
        fill="tozeroy",
        fillcolor="rgba(78,205,196,0.1)",
        name="Peak Demand",
        hovertemplate="Demand: %{y:,.0f} MW<extra></extra>",
    ))

    fig.update_yaxes(title_text="Peak Demand (MW)")
    return _apply_theme(fig, title)


def plot_demand_comparison(
    old_values: list,
    new_values: list,
    old_label: str = "2015",
    new_label: str = "2024",
    title: str = "Electricity Demand Comparison",
) -> go.Figure:
    """Grouped bar chart comparing two years."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=MONTHS,
        y=new_values,
        name=new_label,
        marker_color=COLOR_PALETTE["primary"],
    ))
    fig.add_trace(go.Bar(
        x=MONTHS,
        y=old_values,
        name=old_label,
        marker_color=COLOR_PALETTE["secondary"],
    ))

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Peak Demand (MW)")

    return _apply_theme(fig, title)


def plot_production_pie(
    gen_df: pd.DataFrame,
    year: str,
    title: str = "",
) -> go.Figure:
    """Pie chart of power production by source for a given year."""
    year_data = gen_df[gen_df["Year"] == year]

    if year_data.empty:
        return go.Figure()

    gen_col = "Generation (in MU)"
    if gen_col not in year_data.columns:
        gen_col = year_data.columns[-1]

    fig = go.Figure(data=go.Pie(
        labels=year_data["Source"].values,
        values=year_data[gen_col].values,
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set2),
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="%{label}: %{value:,.0f} MU (%{percent})<extra></extra>",
    ))

    t = title or f"Power Production Mix — {year}"
    return _apply_theme(fig, t)


def plot_consumption_pie(
    cons_df: pd.DataFrame,
    year: str,
    title: str = "",
) -> go.Figure:
    """Pie chart of sector-wise consumption for a given year."""
    year_data = cons_df[cons_df["Year"] == year]

    if year_data.empty:
        return go.Figure()

    fig = go.Figure(data=go.Pie(
        labels=year_data["Sector"].values,
        values=year_data["Consumption (%)"].values,
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Pastel),
        textinfo="label+percent",
        textposition="outside",
    ))

    t = title or f"Sector-wise Consumption — {year}"
    return _apply_theme(fig, t)


# ═══════════════════════════════════════════════════════════════════
# Correlation & Comparison Charts
# ═══════════════════════════════════════════════════════════════════

def plot_standardized_overlay(
    series_a: np.ndarray,
    series_b: np.ndarray,
    label_a: str = "Temperature",
    label_b: str = "Electricity Demand",
    x_labels: list = None,
    title: str = "Standardized Comparison",
) -> go.Figure:
    """Overlay two standardized series on the same axis."""
    fig = go.Figure()

    x = x_labels or list(range(len(series_a)))

    fig.add_trace(go.Scatter(
        x=x, y=series_a,
        mode="lines",
        name=label_a,
        line=dict(color=COLOR_PALETTE["primary"], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=series_b,
        mode="lines",
        name=label_b,
        line=dict(color=COLOR_PALETTE["secondary"], width=2.5),
    ))

    fig.update_yaxes(title_text="Standardized Value (Z-score)")
    return _apply_theme(fig, title)


def plot_scatter_regression(
    x_vals: pd.Series,
    y_vals: pd.Series,
    y_pred: np.ndarray = None,
    x_label: str = "Temperature (°C)",
    y_label: str = "Peak Demand (MW)",
    title: str = "Temperature vs Electricity Demand",
) -> go.Figure:
    """Scatter plot with optional regression line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals.values,
        y=y_vals.values,
        mode="markers",
        marker=dict(
            color=COLOR_PALETTE["accent"],
            size=8,
            opacity=0.7,
            line=dict(width=1, color="white"),
        ),
        name="Actual",
        hovertemplate=f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:,.0f}}<extra></extra>",
    ))

    if y_pred is not None:
        # Sort for clean line
        sort_idx = np.argsort(x_vals.values)
        fig.add_trace(go.Scatter(
            x=x_vals.values[sort_idx],
            y=y_pred[sort_idx],
            mode="lines",
            line=dict(color=COLOR_PALETTE["primary"], width=3, dash="dash"),
            name="Regression Line",
        ))

    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return _apply_theme(fig, title)


# ═══════════════════════════════════════════════════════════════════
# Economic Charts
# ═══════════════════════════════════════════════════════════════════

def plot_gdp_trend(
    gdp_df: pd.DataFrame,
    value_col: str = "Price (in Rs.Lakh Crore)",
    title: str = "Tamil Nadu GDP Trend",
) -> go.Figure:
    """Line chart of GDP over time."""
    if value_col not in gdp_df.columns:
        value_col = gdp_df.select_dtypes(include=[np.number]).columns[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gdp_df["Year"] if "Year" in gdp_df.columns else list(range(len(gdp_df))),
        y=gdp_df[value_col],
        mode="lines+markers",
        line=dict(color=COLOR_PALETTE["accent"], width=3),
        marker=dict(size=8),
        name="GDP",
        fill="tozeroy",
        fillcolor="rgba(255,230,109,0.1)",
    ))

    fig.update_yaxes(title_text="Rs. Lakh Crore")
    return _apply_theme(fig, title)


def plot_profit_loss(
    pl_df: pd.DataFrame,
    title: str = "TANGEDCO Profit & Loss (Rs. Crores)",
) -> go.Figure:
    """Bar chart of TANGEDCO profit/loss by year."""
    pl_col = "Profit and Loss (in Rs Crores)"
    values = pl_df[pl_col].values
    years = pl_df["Year"].values

    colors = [COLOR_PALETTE["secondary"] if v >= 0 else COLOR_PALETTE["primary"]
              for v in values]

    fig = go.Figure(data=go.Bar(
        x=years,
        y=values,
        marker_color=colors,
        text=[f"₹{v:,.0f}" for v in values],
        textposition="outside",
        hovertemplate="Year: %{x}<br>P&L: ₹%{y:,.0f} Cr<extra></extra>",
    ))

    fig.update_yaxes(title_text="Rs. Crores")
    return _apply_theme(fig, title)


def plot_gdp_state_comparison(
    tn_values: list,
    ka_values: list,
    mh_values: list,
    years: list,
    title: str = "GDP Comparison — TN vs Karnataka vs Maharashtra",
) -> go.Figure:
    """Grouped bar chart comparing GDP of 3 states."""
    fig = go.Figure()

    fig.add_trace(go.Bar(x=years, y=tn_values, name="Tamil Nadu",
                         marker_color=COLOR_PALETTE["primary"]))
    fig.add_trace(go.Bar(x=years, y=ka_values, name="Karnataka",
                         marker_color=COLOR_PALETTE["secondary"]))
    fig.add_trace(go.Bar(x=years, y=mh_values, name="Maharashtra",
                         marker_color=COLOR_PALETTE["accent"]))

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Rs. Lakh Crore")

    return _apply_theme(fig, title)


# ═══════════════════════════════════════════════════════════════════
# Forecasting Charts
# ═══════════════════════════════════════════════════════════════════

def plot_forecast(
    actual: pd.Series,
    forecast: pd.Series,
    forecast_ci: pd.DataFrame = None,
    test_actual: pd.Series = None,
    test_predicted: pd.Series = None,
    title: str = "SARIMAX Forecast",
    y_label: str = "Value",
) -> go.Figure:
    """Plot actual data, test predictions, and future forecast."""
    fig = go.Figure()

    # Actual data
    fig.add_trace(go.Scatter(
        y=actual.values,
        mode="lines",
        name="Historical",
        line=dict(color=COLOR_PALETTE["secondary"], width=2),
    ))

    offset = len(actual)

    # Test predictions
    if test_actual is not None and len(test_actual) > 0:
        test_start = offset - len(test_actual)
        fig.add_trace(go.Scatter(
            x=list(range(test_start, offset)),
            y=test_predicted.values if test_predicted is not None else [],
            mode="lines",
            name="Test Prediction",
            line=dict(color=COLOR_PALETTE["accent"], width=2, dash="dash"),
        ))

    # Future forecast
    if len(forecast) > 0:
        forecast_x = list(range(offset, offset + len(forecast)))
        fig.add_trace(go.Scatter(
            x=forecast_x,
            y=forecast.values,
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLOR_PALETTE["primary"], width=3),
            marker=dict(size=6),
        ))

        # Confidence interval
        if forecast_ci is not None and not forecast_ci.empty:
            ci_cols = forecast_ci.columns
            fig.add_trace(go.Scatter(
                x=forecast_x + forecast_x[::-1],
                y=list(forecast_ci[ci_cols[1]].values) +
                  list(forecast_ci[ci_cols[0]].values[::-1]),
                fill="toself",
                fillcolor="rgba(255,107,107,0.15)",
                line=dict(color="rgba(255,107,107,0)"),
                name="95% Confidence",
                hoverinfo="skip",
            ))

    fig.update_yaxes(title_text=y_label)
    fig.update_xaxes(title_text="Time Period (Monthly)")
    return _apply_theme(fig, title)


def plot_population_pie(
    tn_percent: float = 5.96,
    title: str = "Tamil Nadu Population Share in India",
) -> go.Figure:
    """Pie chart of TN's population share."""
    fig = go.Figure(data=go.Pie(
        labels=["Rest of India", "Tamil Nadu"],
        values=[100 - tn_percent, tn_percent],
        hole=0.5,
        marker=dict(colors=[COLOR_PALETTE["secondary"], COLOR_PALETTE["primary"]]),
        textinfo="label+percent",
        textposition="outside",
    ))
    return _apply_theme(fig, title)


def plot_urban_rural(
    urban_pct: float,
    rural_pct: float,
    title: str = "Urban vs Rural Population",
) -> go.Figure:
    """Donut chart of urban vs rural split."""
    fig = go.Figure(data=go.Pie(
        labels=["Urban", "Rural"],
        values=[urban_pct, rural_pct],
        hole=0.5,
        marker=dict(colors=[COLOR_PALETTE["accent"], COLOR_PALETTE["secondary"]]),
        textinfo="label+percent",
        textposition="outside",
    ))
    return _apply_theme(fig, title)


# ═══════════════════════════════════════════════════════════════════
# Advanced Analytics Charts
# ═══════════════════════════════════════════════════════════════════

def plot_seasonal_decomposition(
    observed: np.ndarray,
    trend: np.ndarray,
    seasonal: np.ndarray,
    residual: np.ndarray,
    title: str = "Seasonal Decomposition",
) -> go.Figure:
    """4-panel decomposition: Observed / Trend / Seasonal / Residual."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
        vertical_spacing=0.06,
    )

    x = list(range(len(observed)))

    fig.add_trace(go.Scatter(
        x=x, y=observed, mode="lines",
        line=dict(color=COLOR_PALETTE["secondary"], width=1.5), name="Observed",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=trend, mode="lines",
        line=dict(color=COLOR_PALETTE["accent"], width=2.5), name="Trend",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=seasonal, mode="lines",
        line=dict(color=COLOR_PALETTE["primary"], width=1.5), name="Seasonal",
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=residual, mode="markers",
        marker=dict(color=COLOR_PALETTE["text"], size=3, opacity=0.5), name="Residual",
    ), row=4, col=1)

    fig.update_layout(height=700, showlegend=False)
    return _apply_theme(fig, title)


def plot_rolling_correlation(
    rolling_corr: pd.Series,
    window: int = 12,
    title: str = "Rolling Correlation (12-month window)",
) -> go.Figure:
    """Line chart of rolling correlation over time."""
    fig = go.Figure()

    values = rolling_corr.values
    x = list(range(len(values)))

    # Color-code positive vs negative correlation
    fig.add_trace(go.Scatter(
        x=x, y=values,
        mode="lines",
        line=dict(color=COLOR_PALETTE["accent"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(255,230,109,0.15)",
        name=f"Correlation (w={window})",
        hovertemplate="Period: %{x}<br>r = %{y:.3f}<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    fig.add_hline(y=0.5, line_dash="dot", line_color=COLOR_PALETTE["secondary"], opacity=0.4,
                  annotation_text="r=0.5")
    fig.add_hline(y=-0.5, line_dash="dot", line_color=COLOR_PALETTE["primary"], opacity=0.4,
                  annotation_text="r=-0.5")

    fig.update_yaxes(title_text="Pearson r", range=[-1, 1])
    fig.update_xaxes(title_text="Time Period")
    return _apply_theme(fig, title)


def plot_anomaly_detection(
    values: pd.Series,
    z_scores: np.ndarray,
    anomaly_indices: list,
    title: str = "Anomaly Detection",
    y_label: str = "Value",
) -> go.Figure:
    """Scatter plot with anomalies highlighted in red."""
    fig = go.Figure()

    x = list(range(len(values)))
    vals = values.values if hasattr(values, 'values') else values

    # Normal points
    normal_mask = np.ones(len(vals), dtype=bool)
    normal_mask[anomaly_indices] = False

    fig.add_trace(go.Scatter(
        x=[x[i] for i in range(len(x)) if normal_mask[i]],
        y=[vals[i] for i in range(len(vals)) if normal_mask[i]],
        mode="markers",
        marker=dict(color=COLOR_PALETTE["secondary"], size=6, opacity=0.6),
        name="Normal",
    ))

    # Anomaly points
    if len(anomaly_indices) > 0:
        fig.add_trace(go.Scatter(
            x=[x[i] for i in anomaly_indices],
            y=[vals[i] for i in anomaly_indices],
            mode="markers",
            marker=dict(
                color=COLOR_PALETTE["primary"], size=12, symbol="x",
                line=dict(width=2, color="white"),
            ),
            name=f"Anomaly ({len(anomaly_indices)})",
            hovertemplate=f"Index: %{{x}}<br>{y_label}: %{{y:.1f}}<br>⚠️ Anomaly<extra></extra>",
        ))

    # Trend line
    fig.add_trace(go.Scatter(
        x=x, y=vals,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.2)", width=1),
        showlegend=False,
    ))

    fig.update_yaxes(title_text=y_label)
    return _apply_theme(fig, title)


def plot_distribution(
    values: pd.Series,
    name: str = "Value",
    title: str = "Distribution Analysis",
) -> go.Figure:
    """Histogram + KDE + box plot combination."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    vals = values.dropna()

    # Histogram
    fig.add_trace(go.Histogram(
        x=vals,
        nbinsx=30,
        marker_color=COLOR_PALETTE["secondary"],
        opacity=0.7,
        name="Distribution",
        hovertemplate=f"{name}: %{{x:.1f}}<br>Count: %{{y}}<extra></extra>",
    ), row=1, col=1)

    # KDE overlay
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(vals)
        x_range = np.linspace(vals.min(), vals.max(), 200)
        kde_vals = kde(x_range) * len(vals) * (vals.max() - vals.min()) / 30

        fig.add_trace(go.Scatter(
            x=x_range, y=kde_vals,
            mode="lines",
            line=dict(color=COLOR_PALETTE["accent"], width=2.5),
            name="KDE",
        ), row=1, col=1)
    except Exception:
        pass

    # Box plot
    fig.add_trace(go.Box(
        x=vals,
        marker_color=COLOR_PALETTE["primary"],
        name="Box Plot",
        boxpoints="outliers",
    ), row=2, col=1)

    fig.update_layout(height=500, showlegend=True)
    fig.update_xaxes(title_text=name, row=2, col=1)
    return _apply_theme(fig, title)


def plot_model_comparison(
    leaderboard: list[dict],
    title: str = "Model Performance Leaderboard",
) -> go.Figure:
    """Grouped bar chart comparing model metrics."""
    names = [r["name"] for r in leaderboard]
    r2_scores = [r["metrics"].get("R²", 0) for r in leaderboard]
    mae_scores = [r["metrics"].get("MAE", 0) for r in leaderboard]
    rmse_scores = [r["metrics"].get("RMSE", 0) for r in leaderboard]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["R² Score (↑ better)", "MAE (↓ better)", "RMSE (↓ better)"],
        horizontal_spacing=0.08,
    )

    colors = [COLOR_PALETTE["secondary"], COLOR_PALETTE["accent"],
              COLOR_PALETTE["primary"], "#A78BFA"]

    # R²
    fig.add_trace(go.Bar(
        x=names, y=r2_scores,
        marker_color=colors[:len(names)],
        text=[f"{v:.4f}" for v in r2_scores],
        textposition="outside",
        showlegend=False,
    ), row=1, col=1)

    # MAE
    fig.add_trace(go.Bar(
        x=names, y=mae_scores,
        marker_color=colors[:len(names)],
        text=[f"{v:.1f}" for v in mae_scores],
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)

    # RMSE
    fig.add_trace(go.Bar(
        x=names, y=rmse_scores,
        marker_color=colors[:len(names)],
        text=[f"{v:.1f}" for v in rmse_scores],
        textposition="outside",
        showlegend=False,
    ), row=1, col=3)

    fig.update_layout(height=400)
    return _apply_theme(fig, title)


def plot_feature_importance(
    importance: dict,
    title: str = "Random Forest — Feature Importance",
) -> go.Figure:
    """Horizontal bar chart for feature importance."""
    sorted_feat = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_feat]
    values = [f[1] for f in sorted_feat]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(
            color=values,
            colorscale=[[0, COLOR_PALETTE["secondary"]], [1, COLOR_PALETTE["primary"]]],
        ),
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))

    fig.update_xaxes(title_text="Importance Score")
    fig.update_layout(height=max(250, len(names) * 50))
    return _apply_theme(fig, title)


def plot_residual_analysis(
    residuals: np.ndarray,
    predictions: np.ndarray,
    title: str = "Residual Analysis",
) -> go.Figure:
    """3-panel residual analysis: residuals vs fitted, histogram, Q-Q plot."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Residuals vs Fitted",
            "Residual Distribution",
            "Q-Q Plot",
        ],
        horizontal_spacing=0.08,
    )

    # 1. Residuals vs Fitted
    fig.add_trace(go.Scatter(
        x=predictions, y=residuals,
        mode="markers",
        marker=dict(color=COLOR_PALETTE["secondary"], size=5, opacity=0.6),
        showlegend=False,
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=1, col=1)

    # 2. Residual histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=20,
        marker_color=COLOR_PALETTE["accent"],
        opacity=0.7,
        showlegend=False,
    ), row=1, col=2)

    # 3. Q-Q plot
    sorted_resid = np.sort(residuals)
    theoretical = sp_stats.norm.ppf(
        np.linspace(0.01, 0.99, len(sorted_resid))
    )
    fig.add_trace(go.Scatter(
        x=theoretical, y=sorted_resid,
        mode="markers",
        marker=dict(color=COLOR_PALETTE["primary"], size=5),
        showlegend=False,
    ), row=1, col=3)
    # 45-degree reference line
    line_range = [min(theoretical), max(theoretical)]
    fig.add_trace(go.Scatter(
        x=line_range, y=line_range,
        mode="lines",
        line=dict(color="white", dash="dash", width=1),
        showlegend=False,
    ), row=1, col=3)

    fig.update_layout(height=350)
    fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Residual Value", row=1, col=2)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=3)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=3)
    return _apply_theme(fig, title)


def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
) -> go.Figure:
    """Enhanced heatmap for all numeric variables."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr.values, 3),
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="r"),
    ))

    fig.update_layout(height=500)
    return _apply_theme(fig, title)


def plot_granger_results(
    granger_results: dict,
    title: str = "Granger Causality Test Results",
) -> go.Figure:
    """Bar chart of p-values by lag with significance markers."""
    if "results" not in granger_results or not granger_results["results"]:
        fig = go.Figure()
        fig.add_annotation(text="Granger test could not be completed", showarrow=False)
        return _apply_theme(fig, title)

    results = granger_results["results"]
    lags = list(results.keys())
    p_values = [results[lag]["p_value"] for lag in lags]
    significant = [results[lag]["significant"] for lag in lags]

    colors = [COLOR_PALETTE["secondary"] if s else COLOR_PALETTE["primary"] for s in significant]

    fig = go.Figure(go.Bar(
        x=[f"Lag {lag}" for lag in lags],
        y=p_values,
        marker_color=colors,
        text=[f"p={p:.4f}" for p in p_values],
        textposition="outside",
        hovertemplate="Lag %{x}<br>p-value: %{y:.4f}<extra></extra>",
    ))

    # Significance threshold line
    fig.add_hline(y=0.05, line_dash="dash", line_color=COLOR_PALETTE["accent"],
                  annotation_text="α = 0.05 (significance)", opacity=0.8)

    fig.update_yaxes(title_text="P-value", range=[0, max(p_values) * 1.3 if p_values else 1])
    fig.update_layout(height=400)
    return _apply_theme(fig, title)

