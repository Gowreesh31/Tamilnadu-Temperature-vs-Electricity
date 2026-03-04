"""
Statistical analysis module for Tamil Nadu Temperature vs Electricity.

Provides functions for correlation analysis, seasonal decomposition,
standardization, and comparative analytics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats


def compute_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    method: str = "pearson",
) -> dict:
    """
    Compute correlation between two series.

    Parameters
    ----------
    series_a, series_b : pd.Series
        The two series to correlate. Must be same length.
    method : str
        'pearson' or 'spearman'.

    Returns
    -------
    dict with keys: coefficient, p_value, method, strength
    """
    a = series_a.dropna()
    b = series_b.dropna()

    # Align lengths
    min_len = min(len(a), len(b))
    a, b = a.iloc[:min_len], b.iloc[:min_len]

    if method == "spearman":
        coeff, p_val = stats.spearmanr(a, b)
    else:
        coeff, p_val = stats.pearsonr(a, b)

    # Classify strength
    abs_coeff = abs(coeff)
    if abs_coeff >= 0.7:
        strength = "Strong"
    elif abs_coeff >= 0.4:
        strength = "Moderate"
    elif abs_coeff >= 0.2:
        strength = "Weak"
    else:
        strength = "Very Weak"

    direction = "Positive" if coeff > 0 else "Negative"

    return {
        "coefficient": round(coeff, 4),
        "p_value": round(p_val, 6),
        "method": method.title(),
        "strength": f"{strength} {direction}",
        "significant": p_val < 0.05,
    }


def standardize_series(*series_list: pd.Series) -> list[np.ndarray]:
    """
    Standardize multiple series using StandardScaler for comparison.

    Returns list of standardized numpy arrays in same order as input.
    """
    scaler = StandardScaler()
    results = []

    for s in series_list:
        arr = s.values.reshape(-1, 1)
        scaled = scaler.fit_transform(arr).flatten()
        results.append(scaled)

    return results


def compute_yoy_growth(series: pd.Series) -> pd.Series:
    """Compute year-over-year percentage growth."""
    return series.pct_change() * 100


def compute_monthly_stats(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Compute monthly statistics (mean, min, max, std) for a value column.

    Expects df to have 'Month' or 'Month_Name' column.
    """
    group_col = "Month_Name" if "Month_Name" in df.columns else "Month"
    stats_df = df.groupby(group_col)[value_col].agg(
        ["mean", "min", "max", "std"]
    ).round(2)
    return stats_df


def compute_seasonal_pattern(
    values: pd.Series,
    period: int = 12,
) -> dict:
    """
    Decompose a series into seasonal components.

    Returns dict with: trend, seasonal, residual (as numpy arrays).
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Need at least 2 full periods
    if len(values) < period * 2:
        return {"trend": None, "seasonal": None, "residual": None}

    result = seasonal_decompose(values, model="additive", period=period)
    return {
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
        "observed": result.observed,
    }


def build_temp_demand_dataset(
    temp_df: pd.DataFrame,
    demand_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge temperature and electricity demand into a single DataFrame
    for correlation and regression analysis.

    Both DataFrames should have Year and Month columns.
    """
    # Ensure common format
    if "Peak Demand (in MW)" in demand_df.columns:
        demand_col = "Peak Demand (in MW)"
    else:
        demand_col = demand_df.columns[-1]

    temp_col = "Temperature" if "Temperature" in temp_df.columns else temp_df.columns[-1]

    # Create year-month keys for merging
    temp_clean = temp_df[["Year", "Month", temp_col]].copy()
    temp_clean = temp_clean.rename(columns={temp_col: "Temperature"})

    demand_clean = demand_df.copy()
    if "Year" not in demand_clean.columns and demand_col in demand_clean.columns:
        # Parse year from Year column like "2024-25"
        if demand_clean.columns[0] == "Year":
            demand_clean["Year_Num"] = demand_clean["Year"].str[:4].astype(int)
        demand_clean = demand_clean.rename(columns={demand_col: "Demand_MW"})

    merged = pd.merge(temp_clean, demand_clean, on=["Year", "Month"], how="inner")
    return merged
