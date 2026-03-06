"""
Machine learning models for Tamil Nadu Temperature vs Electricity.

Implements:
  - SARIMAX time series forecasting with proper train/test evaluation
  - Linear regression for temperature → electricity demand relationship
  - Polynomial regression (degree 2, 3) for non-linear modelling
  - Random Forest regressor with feature importance
  - Multi-model comparison leaderboard
  - Model evaluation metrics (MAE, RMSE, R², MAPE)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute comprehensive regression metrics.

    Returns
    -------
    dict with: MAE, RMSE, R2, MAPE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R²": round(r2, 4),
        "MAPE (%)": round(mape, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# SARIMAX Forecasting
# ═══════════════════════════════════════════════════════════════════

def forecast_sarimax(
    series: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 0, 12),
    forecast_steps: int = 24,
    test_size: int = 12,
) -> dict:
    """
    Fit SARIMAX model with train/test evaluation and forecast.

    Parameters
    ----------
    series : pd.Series
        Time series to model.
    order : tuple
        ARIMA (p, d, q) order.
    seasonal_order : tuple
        Seasonal (P, D, Q, s) order.
    forecast_steps : int
        Number of future periods to forecast.
    test_size : int
        Number of periods to hold out for evaluation.

    Returns
    -------
    dict with keys:
        - train_metrics: evaluation metrics on test set
        - forecast: predicted future values
        - forecast_ci: confidence intervals
        - test_actual: actual test values
        - test_predicted: predicted test values
        - model_summary: string summary of the model
    """
    # Train/test split
    if len(series) <= test_size:
        test_size = max(1, len(series) // 5)

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    try:
        # Fit on training data
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False, maxiter=200)

        # Evaluate on test set
        test_pred = results.get_forecast(steps=test_size)
        test_predicted = test_pred.predicted_mean.values

        metrics = evaluate_model(test.values, test_predicted)

        # Refit on full data for final forecast
        full_model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        full_results = full_model.fit(disp=False, maxiter=200)

        # Forecast future periods
        forecast = full_results.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int(alpha=0.05)

        return {
            "train_metrics": metrics,
            "forecast": forecast_mean,
            "forecast_ci": forecast_ci,
            "test_actual": test,
            "test_predicted": pd.Series(test_predicted, index=test.index),
            "model_summary": str(results.summary()),
            "aic": round(results.aic, 2),
            "bic": round(results.bic, 2),
        }

    except Exception as e:
        logger.error("SARIMAX failed: %s", e)
        return {
            "train_metrics": {},
            "forecast": pd.Series(),
            "forecast_ci": pd.DataFrame(),
            "test_actual": test,
            "test_predicted": pd.Series(),
            "model_summary": f"Model fitting failed: {e}",
            "aic": None,
            "bic": None,
        }


# ═══════════════════════════════════════════════════════════════════
# Linear Regression: Temperature → Demand
# ═══════════════════════════════════════════════════════════════════

def regression_temp_demand(
    temperature: pd.Series,
    demand: pd.Series,
) -> dict:
    """
    Fit linear regression: Temperature → Electricity Demand.

    Parameters
    ----------
    temperature : pd.Series
        Monthly average temperature values.
    demand : pd.Series
        Monthly peak electricity demand values.

    Returns
    -------
    dict with: model, metrics, coefficient, intercept, predictions
    """
    # Align lengths
    min_len = min(len(temperature), len(demand))
    X = temperature.values[:min_len].reshape(-1, 1)
    y = demand.values[:min_len]

    # Remove NaN
    mask = ~(np.isnan(X.flatten()) | np.isnan(y))
    X, y = X[mask], y[mask]

    if len(X) < 2:
        return {"model": None, "metrics": {}, "coefficient": 0, "intercept": 0}

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    metrics = evaluate_model(y, y_pred)

    return {
        "model": model,
        "metrics": metrics,
        "coefficient": round(model.coef_[0], 4),
        "intercept": round(model.intercept_, 4),
        "predictions": y_pred,
        "equation": f"Demand = {model.coef_[0]:.2f} × Temperature + {model.intercept_:.2f}",
    }


# ═══════════════════════════════════════════════════════════════════
# Polynomial Regression
# ═══════════════════════════════════════════════════════════════════

def polynomial_regression(
    temperature: pd.Series,
    demand: pd.Series,
    degree: int = 2,
) -> dict:
    """
    Fit polynomial regression: Temperature → Electricity Demand.

    Captures non-linear relationships (e.g., exponential demand increase
    at very high temperatures due to AC usage).

    Parameters
    ----------
    temperature : pd.Series
    demand : pd.Series
    degree : int
        Polynomial degree (2 = quadratic, 3 = cubic).

    Returns
    -------
    dict with: model, metrics, predictions, degree, equation
    """
    min_len = min(len(temperature), len(demand))
    X = temperature.values[:min_len].reshape(-1, 1)
    y = demand.values[:min_len]

    mask = ~(np.isnan(X.flatten()) | np.isnan(y))
    X, y = X[mask], y[mask]

    if len(X) < degree + 1:
        return {"model": None, "metrics": {}, "predictions": np.array([])}

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)

    metrics = evaluate_model(y, y_pred)

    # Build equation string
    lr = model.named_steps["linearregression"]
    coeffs = lr.coef_
    intercept = lr.intercept_
    terms = [f"{intercept:.2f}"]
    for i in range(1, len(coeffs)):
        if abs(coeffs[i]) > 0.001:
            terms.append(f"{coeffs[i]:+.4f}·T^{i}")
    equation = "Demand = " + " ".join(terms)

    return {
        "model": model,
        "metrics": metrics,
        "predictions": y_pred,
        "degree": degree,
        "equation": equation,
        "residuals": y - y_pred,
    }


# ═══════════════════════════════════════════════════════════════════
# Random Forest Regressor
# ═══════════════════════════════════════════════════════════════════

def random_forest_regression(
    temperature: pd.Series,
    demand: pd.Series,
    month: pd.Series = None,
    n_estimators: int = 100,
) -> dict:
    """
    Fit Random Forest: Temperature (+ Month) → Electricity Demand.

    Random Forest can capture complex non-linear interactions and
    provides feature importance scores.

    Parameters
    ----------
    temperature : pd.Series
    demand : pd.Series
    month : pd.Series, optional
        Month numbers (1-12) to add seasonality as a feature.
    n_estimators : int
        Number of trees in the forest.

    Returns
    -------
    dict with: model, metrics, predictions, feature_importance, cv_scores
    """
    min_len = min(len(temperature), len(demand))
    temp_vals = temperature.values[:min_len]
    demand_vals = demand.values[:min_len]

    # Build feature matrix
    feature_names = ["Temperature"]
    if month is not None and len(month) >= min_len:
        month_vals = month.values[:min_len]
        X = np.column_stack([temp_vals, month_vals])
        feature_names.append("Month")
    else:
        X = temp_vals.reshape(-1, 1)

    y = demand_vals

    # Remove NaN
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[valid], y[valid]

    if len(X) < 10:
        return {"model": None, "metrics": {}, "predictions": np.array([])}

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    metrics = evaluate_model(y, y_pred)

    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_.round(4)))

    # Cross-validation (5-fold)
    try:
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(X) // 3), scoring="r2")
        cv_mean = round(cv_scores.mean(), 4)
        cv_std = round(cv_scores.std(), 4)
    except Exception:
        cv_mean, cv_std = None, None
        cv_scores = np.array([])

    return {
        "model": model,
        "metrics": metrics,
        "predictions": y_pred,
        "feature_importance": importance,
        "cv_r2_mean": cv_mean,
        "cv_r2_std": cv_std,
        "cv_scores": cv_scores,
        "residuals": y - y_pred,
    }


# ═══════════════════════════════════════════════════════════════════
# Model Comparison Leaderboard
# ═══════════════════════════════════════════════════════════════════

def compare_models(
    temperature: pd.Series,
    demand: pd.Series,
    month: pd.Series = None,
) -> list[dict]:
    """
    Run all regression models and return a ranked leaderboard.

    Compares: Linear, Polynomial (deg 2), Polynomial (deg 3), Random Forest.

    Returns
    -------
    list of dicts sorted by R² (best first):
        [{name, metrics, predictions, model_details}, ...]
    """
    results = []

    # 1. Linear Regression
    lr = regression_temp_demand(temperature, demand)
    if lr["model"] is not None:
        results.append({
            "name": "Linear Regression",
            "type": "linear",
            "metrics": lr["metrics"],
            "predictions": lr["predictions"],
            "equation": lr.get("equation", ""),
        })

    # 2. Polynomial Regression (degree 2)
    poly2 = polynomial_regression(temperature, demand, degree=2)
    if poly2["model"] is not None:
        results.append({
            "name": "Polynomial (Degree 2)",
            "type": "polynomial",
            "metrics": poly2["metrics"],
            "predictions": poly2["predictions"],
            "equation": poly2.get("equation", ""),
            "residuals": poly2.get("residuals"),
        })

    # 3. Polynomial Regression (degree 3)
    poly3 = polynomial_regression(temperature, demand, degree=3)
    if poly3["model"] is not None:
        results.append({
            "name": "Polynomial (Degree 3)",
            "type": "polynomial",
            "metrics": poly3["metrics"],
            "predictions": poly3["predictions"],
            "equation": poly3.get("equation", ""),
            "residuals": poly3.get("residuals"),
        })

    # 4. Random Forest
    rf = random_forest_regression(temperature, demand, month)
    if rf["model"] is not None:
        results.append({
            "name": "Random Forest",
            "type": "random_forest",
            "metrics": rf["metrics"],
            "predictions": rf["predictions"],
            "feature_importance": rf.get("feature_importance", {}),
            "cv_r2_mean": rf.get("cv_r2_mean"),
            "cv_r2_std": rf.get("cv_r2_std"),
            "residuals": rf.get("residuals"),
        })

    # Sort by R² (descending)
    results.sort(key=lambda x: x["metrics"].get("R²", -999), reverse=True)

    # Add rank
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results

