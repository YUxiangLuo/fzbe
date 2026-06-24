# -*- coding: utf-8 -*-
"""
Common evaluation metrics for time series forecasting models.

Optimized with NumPy for vectorized calculations and improved numerical stability.
"""

from typing import List, Union

import numpy as np

# Type alias for input flexibility
ArrayLike = Union[List[float], np.ndarray]

# Epsilon for numerical stability in MAPE calculation
EPS = 1e-12


def _validate_inputs(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """Validate inputs for metric functions."""
    if len(y_true) == 0:
        return  # Handled in functions
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: True values (list or numpy array)
        y_pred: Predicted values (list or numpy array)

    Returns:
        RMSE value, or NaN if input is empty
    """
    _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    y_true_arr, y_pred_arr = (
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_pred, dtype=np.float64),
    )
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: True values (list or numpy array)
        y_pred: Predicted values (list or numpy array)

    Returns:
        MAE value, or NaN if input is empty
    """
    _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    y_true_arr, y_pred_arr = (
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_pred, dtype=np.float64),
    )
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True values (list or numpy array)
        y_pred: Predicted values (list or numpy array)

    Returns:
        MAPE value in percentage, or NaN if input is empty
    """
    _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    y_true_arr, y_pred_arr = (
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_pred, dtype=np.float64),
    )
    diff = np.abs(y_true_arr - y_pred_arr)
    denom = np.where(np.abs(y_true_arr) > EPS, np.abs(y_true_arr), EPS)
    return float(100.0 * np.mean(diff / denom))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculate R-squared (coefficient of determination).

    Args:
        y_true: True values (list or numpy array)
        y_pred: Predicted values (list or numpy array)

    Returns:
        R² value, or NaN if input is empty
    """
    _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    y_true_arr, y_pred_arr = (
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_pred, dtype=np.float64),
    )
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    return float(1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0)
