"""
Machine learning models for Tamil Nadu Temperature vs Electricity.

Implements:
  - SARIMAX time series forecasting with proper train/test evaluation
  - Linear regression for temperature → electricity demand relationship
  - Model evaluation metrics (MAE, RMSE, R², MAPE)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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
