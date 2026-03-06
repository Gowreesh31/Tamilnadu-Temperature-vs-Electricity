"""
Advanced statistical analysis toolkit for Tamil Nadu Temperature × Electricity.

This module provides professional-grade analytics functions:
  - Granger Causality testing (does temperature cause demand?)
  - Augmented Dickey-Fuller stationarity test
  - Mann-Kendall monotonic trend test
  - Rolling correlation analysis
  - Statistical anomaly detection (modified Z-score)
  - Automated data profiling
  - Auto-insight generator
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Granger Causality Test
# ═══════════════════════════════════════════════════════════════════

def granger_causality_test(
    cause: pd.Series,
    effect: pd.Series,
    max_lag: int = 6,
) -> dict:
    """
    Test if 'cause' Granger-causes 'effect'.

    Uses statsmodels grangercausalitytests. Null hypothesis is that
    cause does NOT Granger-cause effect.

    Parameters
    ----------
    cause : pd.Series
        Potential causal variable (e.g., temperature).
    effect : pd.Series
        Potential effect variable (e.g., electricity demand).
    max_lag : int
        Maximum number of lags to test.

    Returns
    -------
    dict with keys per lag: {lag: {f_stat, p_value, significant}}
    Plus 'best_lag' and 'conclusion'.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    # Align and clean
    min_len = min(len(cause), len(effect))
    df = pd.DataFrame({
        "effect": effect.values[:min_len],
        "cause": cause.values[:min_len],
    }).dropna()

    if len(df) < max_lag * 3:
        return {"error": "Insufficient data for Granger test", "results": {}}

    try:
        results = grangercausalitytests(df[["effect", "cause"]], maxlag=max_lag, verbose=False)

        lag_results = {}
        best_p = 1.0
        best_lag = 1

        for lag in range(1, max_lag + 1):
            f_test = results[lag][0]["ssr_ftest"]
            f_stat = round(f_test[0], 4)
            p_value = round(f_test[1], 6)

            lag_results[lag] = {
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }

            if p_value < best_p:
                best_p = p_value
                best_lag = lag

        # Generate conclusion
        if best_p < 0.01:
            conclusion = f"Strong evidence (p={best_p:.4f}) that temperature Granger-causes electricity demand at lag {best_lag}."
        elif best_p < 0.05:
            conclusion = f"Significant evidence (p={best_p:.4f}) that temperature Granger-causes demand at lag {best_lag}."
        elif best_p < 0.10:
            conclusion = f"Weak evidence (p={best_p:.4f}) of Granger causality at lag {best_lag}."
        else:
            conclusion = f"No significant Granger causality detected (best p={best_p:.4f})."

        return {
            "results": lag_results,
            "best_lag": best_lag,
            "best_p_value": best_p,
            "conclusion": conclusion,
        }

    except Exception as e:
        logger.error("Granger causality test failed: %s", e)
        return {"error": str(e), "results": {}}


# ═══════════════════════════════════════════════════════════════════
# Augmented Dickey-Fuller Stationarity Test
# ═══════════════════════════════════════════════════════════════════

def adf_stationarity_test(series: pd.Series, name: str = "Series") -> dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    H0: Series has a unit root (non-stationary).
    H1: Series is stationary.

    Returns
    -------
    dict with: test_statistic, p_value, critical_values, is_stationary,
               interpretation
    """
    from statsmodels.tsa.stattools import adfuller

    clean = series.dropna()
    if len(clean) < 10:
        return {"error": "Insufficient data for ADF test"}

    result = adfuller(clean, autolag="AIC")

    is_stationary = result[1] < 0.05

    if result[1] < 0.01:
        interpretation = f"✅ {name} is **stationary** (p={result[1]:.4f} < 0.01). No differencing needed."
    elif result[1] < 0.05:
        interpretation = f"✅ {name} is **stationary** (p={result[1]:.4f} < 0.05)."
    else:
        interpretation = f"❌ {name} is **non-stationary** (p={result[1]:.4f}). Differencing recommended for time series modeling."

    return {
        "test_statistic": round(result[0], 4),
        "p_value": round(result[1], 6),
        "lags_used": result[2],
        "n_observations": result[3],
        "critical_values": {k: round(v, 4) for k, v in result[4].items()},
        "is_stationary": is_stationary,
        "interpretation": interpretation,
    }


# ═══════════════════════════════════════════════════════════════════
# Mann-Kendall Trend Test
# ═══════════════════════════════════════════════════════════════════

def mann_kendall_trend_test(series: pd.Series, name: str = "Series") -> dict:
    """
    Perform Mann-Kendall test for monotonic trend.

    Non-parametric test — doesn't assume normal distribution.
    More robust than linear regression for trend detection.

    Returns
    -------
    dict with: tau, p_value, trend_direction, interpretation
    """
    clean = series.dropna().values
    n = len(clean)

    if n < 8:
        return {"error": "Need at least 8 observations for Mann-Kendall test"}

    # Calculate S statistic
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            diff = clean[j] - clean[k]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Variance of S
    var_s = (n * (n - 1) * (2 * n + 5)) / 18

    # Kendall's tau
    tau = s / (n * (n - 1) / 2)

    # Z-test
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))

    # Direction
    if p_value < 0.05:
        if tau > 0:
            direction = "Increasing"
            interpretation = f"📈 **{name}** shows a statistically significant **increasing trend** (τ={tau:.4f}, p={p_value:.4f})."
        else:
            direction = "Decreasing"
            interpretation = f"📉 **{name}** shows a statistically significant **decreasing trend** (τ={tau:.4f}, p={p_value:.4f})."
    else:
        direction = "No significant trend"
        interpretation = f"➡️ **{name}** shows no statistically significant trend (τ={tau:.4f}, p={p_value:.4f})."

    return {
        "tau": round(tau, 4),
        "s_statistic": s,
        "z_statistic": round(z, 4),
        "p_value": round(p_value, 6),
        "trend_direction": direction,
        "significant": p_value < 0.05,
        "interpretation": interpretation,
    }


# ═══════════════════════════════════════════════════════════════════
# Rolling Correlation
# ═══════════════════════════════════════════════════════════════════

def rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 12,
) -> pd.Series:
    """
    Compute rolling Pearson correlation between two series.

    Use this to show how the temperature-demand relationship
    changes over time (e.g., stronger in summer months).

    Parameters
    ----------
    series_a, series_b : pd.Series
        Two aligned series of the same length.
    window : int
        Rolling window size (default 12 = 1 year for monthly data).

    Returns
    -------
    pd.Series of rolling correlation values.
    """
    min_len = min(len(series_a), len(series_b))
    a = pd.Series(series_a.values[:min_len])
    b = pd.Series(series_b.values[:min_len])

    return a.rolling(window=window, min_periods=window).corr(b)


# ═══════════════════════════════════════════════════════════════════
# Anomaly Detection (Modified Z-Score)
# ═══════════════════════════════════════════════════════════════════

def detect_anomalies(
    series: pd.Series,
    threshold: float = 2.5,
    method: str = "modified_zscore",
) -> dict:
    """
    Detect statistical anomalies in a time series.

    Parameters
    ----------
    series : pd.Series
        The series to analyze.
    threshold : float
        Z-score threshold for anomaly classification.
    method : str
        'zscore' or 'modified_zscore' (MAD-based, more robust).

    Returns
    -------
    dict with: anomaly_indices, anomaly_values, z_scores, stats
    """
    clean = series.dropna()

    if method == "modified_zscore":
        median = np.median(clean)
        mad = np.median(np.abs(clean - median))
        if mad == 0:
            mad = np.std(clean) * 0.6745  # fallback
        z_scores = 0.6745 * (clean - median) / mad if mad > 0 else np.zeros(len(clean))
    else:
        mean, std = clean.mean(), clean.std()
        z_scores = (clean - mean) / std if std > 0 else np.zeros(len(clean))

    z_scores = np.array(z_scores)
    anomaly_mask = np.abs(z_scores) > threshold

    anomaly_indices = np.where(anomaly_mask)[0]
    anomaly_values = clean.iloc[anomaly_indices] if len(anomaly_indices) > 0 else pd.Series()

    # Classify anomalies
    high_anomalies = np.sum(z_scores[anomaly_mask] > 0)
    low_anomalies = np.sum(z_scores[anomaly_mask] < 0)

    return {
        "anomaly_indices": anomaly_indices.tolist(),
        "anomaly_values": anomaly_values,
        "z_scores": z_scores,
        "total_anomalies": len(anomaly_indices),
        "high_anomalies": int(high_anomalies),
        "low_anomalies": int(low_anomalies),
        "pct_anomalies": round(len(anomaly_indices) / len(clean) * 100, 2),
        "threshold": threshold,
        "method": method,
    }


# ═══════════════════════════════════════════════════════════════════
# Data Profiling
# ═══════════════════════════════════════════════════════════════════

def profile_dataframe(df: pd.DataFrame, name: str = "Dataset") -> dict:
    """
    Generate comprehensive data quality profile for a DataFrame.

    Returns
    -------
    dict with: overview, column_profiles, quality_score
    """
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    missing_cells = df.isnull().sum().sum()
    completeness = round((1 - missing_cells / total_cells) * 100, 1) if total_cells > 0 else 0

    # Column-level profiles
    col_profiles = []
    for col in df.columns:
        series = df[col]
        profile = {
            "column": col,
            "dtype": str(series.dtype),
            "non_null": int(series.count()),
            "null_count": int(series.isnull().sum()),
            "null_pct": round(series.isnull().mean() * 100, 1),
            "unique": int(series.nunique()),
            "unique_pct": round(series.nunique() / len(series) * 100, 1) if len(series) > 0 else 0,
        }

        # Numeric stats
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            profile.update({
                "mean": round(desc.get("mean", 0), 2),
                "std": round(desc.get("std", 0), 2),
                "min": round(desc.get("min", 0), 2),
                "max": round(desc.get("max", 0), 2),
                "skewness": round(series.skew(), 3) if len(series.dropna()) > 2 else None,
                "kurtosis": round(series.kurtosis(), 3) if len(series.dropna()) > 3 else None,
            })

        col_profiles.append(profile)

    # Quality score (0-100)
    quality_score = completeness

    return {
        "name": name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_cells": total_cells,
        "missing_cells": int(missing_cells),
        "completeness": completeness,
        "quality_score": round(quality_score, 1),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "column_profiles": col_profiles,
    }


# ═══════════════════════════════════════════════════════════════════
# Auto-Insight Generator
# ═══════════════════════════════════════════════════════════════════

def generate_insights(
    temp_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    pl_df: pd.DataFrame = None,
) -> list[dict]:
    """
    Automatically discover and articulate key findings from the data.

    Returns a list of insight dicts: {icon, title, body, category, priority}
    """
    insights = []

    # ── Temperature insights ──
    if not temp_df.empty:
        temp_col = "Temperature" if "Temperature" in temp_df.columns else temp_df.columns[-1]
        temps = temp_df[temp_col].dropna()

        if len(temps) > 0:
            # Hottest/coldest
            max_temp = temps.max()
            min_temp = temps.min()
            avg_temp = temps.mean()
            insights.append({
                "icon": "🌡️",
                "title": "Temperature Range",
                "body": f"Tamil Nadu temperatures range from **{min_temp:.1f}°C** to **{max_temp:.1f}°C**, with an average of **{avg_temp:.1f}°C**.",
                "category": "Temperature",
                "priority": 1,
            })

            # Check for warming trend
            if "Year" in temp_df.columns:
                yearly_avg = temp_df.groupby("Year")[temp_col].mean()
                if len(yearly_avg) >= 3:
                    first_avg = yearly_avg.iloc[:3].mean()
                    last_avg = yearly_avg.iloc[-3:].mean()
                    change = last_avg - first_avg
                    if abs(change) > 0.3:
                        direction = "warming" if change > 0 else "cooling"
                        insights.append({
                            "icon": "📈" if change > 0 else "📉",
                            "title": f"Temperature {direction.title()} Detected",
                            "body": f"Average temperature has shifted by **{change:+.2f}°C** (3-year rolling avg: {first_avg:.1f}°C → {last_avg:.1f}°C).",
                            "category": "Temperature",
                            "priority": 2,
                        })

            # Hottest month
            if "Month" in temp_df.columns:
                monthly_avg = temp_df.groupby("Month")[temp_col].mean()
                hottest_month = monthly_avg.idxmax()
                month_names = {1: "January", 2: "February", 3: "March", 4: "April",
                              5: "May", 6: "June", 7: "July", 8: "August",
                              9: "September", 10: "October", 11: "November", 12: "December"}
                month_name = month_names.get(hottest_month, str(hottest_month))
                insights.append({
                    "icon": "🔥",
                    "title": "Peak Heat Month",
                    "body": f"**{month_name}** is historically the hottest month with an average of **{monthly_avg.max():.1f}°C**.",
                    "category": "Temperature",
                    "priority": 3,
                })

    # ── Demand insights ──
    if not demand_df.empty:
        demand_col = "Peak Demand (in MW)" if "Peak Demand (in MW)" in demand_df.columns else demand_df.columns[-1]
        demands = demand_df[demand_col].dropna()

        if len(demands) > 0:
            max_demand = demands.max()
            min_demand = demands.min()
            volatility = (demands.std() / demands.mean()) * 100

            insights.append({
                "icon": "⚡",
                "title": "Demand Volatility",
                "body": f"Peak demand ranges from **{min_demand:,.0f}** to **{max_demand:,.0f} MW** — a coefficient of variation of **{volatility:.1f}%**.",
                "category": "Electricity",
                "priority": 1,
            })

            # YoY growth
            if len(demands) >= 24:
                recent_12 = demands.iloc[:12].mean()
                older_12 = demands.iloc[-12:].mean()
                growth = ((recent_12 - older_12) / older_12) * 100
                insights.append({
                    "icon": "📊",
                    "title": "Demand Growth Trajectory",
                    "body": f"Average monthly demand has changed by **{growth:+.1f}%** comparing the first and last 12 months of data.",
                    "category": "Electricity",
                    "priority": 2,
                })

    # ── Correlation insight ──
    if not temp_df.empty and not demand_df.empty:
        temp_col = "Temperature" if "Temperature" in temp_df.columns else temp_df.columns[-1]
        demand_col = "Peak Demand (in MW)" if "Peak Demand (in MW)" in demand_df.columns else demand_df.columns[-1]

        t = temp_df[temp_col].dropna().values
        d = demand_df[demand_col].dropna().values
        min_len = min(len(t), len(d))

        if min_len > 5:
            corr, p_val = sp_stats.pearsonr(t[:min_len], d[:min_len])
            strength = "strong" if abs(corr) > 0.6 else "moderate" if abs(corr) > 0.3 else "weak"
            sig = "statistically significant" if p_val < 0.05 else "not statistically significant"

            insights.append({
                "icon": "🔗",
                "title": "Temperature-Demand Link",
                "body": f"A **{strength}** correlation (r={corr:.3f}, p={p_val:.4f}) exists between temperature and electricity demand — **{sig}**.",
                "category": "Correlation",
                "priority": 1,
            })

    # ── Financial insights ──
    if pl_df is not None and not pl_df.empty and "Profit and Loss (in Rs Crores)" in pl_df.columns:
        pl_values = pl_df["Profit and Loss (in Rs Crores)"].dropna()
        n_loss_years = (pl_values < 0).sum()
        total_years = len(pl_values)
        total_loss = pl_values[pl_values < 0].sum()

        if n_loss_years > 0:
            insights.append({
                "icon": "💸",
                "title": "TANGEDCO Financial Stress",
                "body": f"TANGEDCO reported losses in **{n_loss_years}/{total_years}** years, with cumulative losses of **₹{abs(total_loss):,.0f} Crores**.",
                "category": "Economics",
                "priority": 2,
            })

    # Sort by priority
    insights.sort(key=lambda x: x["priority"])
    return insights
