# -*- coding: utf-8 -*-
"""
Common evaluation metrics for time series forecasting models.

Optimized with NumPy for vectorized calculations and improved numerical stability.
"""

from typing import List, Union

import numpy as np

# Epsilon for numerical stability in MAPE calculation
EPS = 1e-12


def _validate_inputs(y_true: List[float], y_pred: List[float]) -> None:
    """Validate inputs for metric functions."""
    if not y_true:
        return  # Handled in functions
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value, or NaN if input is empty
    """
    _validate_inputs(y_true, y_pred)
    if not y_true:
        return float("nan")
    y_true_arr, y_pred_arr = (
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_pred, dtype=np.float64),
    )
    return np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))


def mae(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value, or NaN if input is empty
    """
    _validate_inputs(y_true, y_pred)
    if not y_true:
        return float("nan")
    y_true_arr, y_pred_arr = (
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_pred, dtype=np.float64),
    )
    return np.mean(np.abs(y_true_arr - y_pred_arr))


def mape(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE value in percentage, or NaN if input is empty
    """
    _validate_inputs(y_true, y_pred)
    if not y_true:
        return float("nan")
    y_true_arr, y_pred_arr = (
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_pred, dtype=np.float64),
    )
    diff = np.abs(y_true_arr - y_pred_arr)
    denom = np.where(np.abs(y_true_arr) > EPS, y_true_arr, EPS)
    return 100.0 * np.mean(diff / denom)


def r2(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate R-squared (coefficient of determination).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² value, or NaN if input is empty
    """
    _validate_inputs(y_true, y_pred)
    if not y_true:
        return float("nan")
    y_true_arr, y_pred_arr = (
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_pred, dtype=np.float64),
    )
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
