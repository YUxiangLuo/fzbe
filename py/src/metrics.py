# -*- coding: utf-8 -*-
"""
Common evaluation metrics for time series forecasting models.

Optimized with NumPy for vectorized calculations and improved numerical stability.
"""

from typing import List, Optional, Union

import numpy as np

# Type alias for input flexibility
ArrayLike = Union[List[float], np.ndarray]

# Epsilon for zero-value and constant-series handling.
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


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> Optional[float]:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True values (list or numpy array)
        y_pred: Predicted values (list or numpy array)

    Returns:
        MAPE value in percentage, None when any actual value is zero or close
        enough to zero to make a percentage unstable, or NaN if input is empty.
        Returning None prevents an undefined partial-subset percentage from
        being presented to students as the MAPE of the complete evaluation set.
    """
    _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    y_true_arr, y_pred_arr = (
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_pred, dtype=np.float64),
    )
    if np.any(np.abs(y_true_arr) <= EPS):
        return None
    diff = np.abs(y_true_arr - y_pred_arr)
    return float(100.0 * np.mean(diff / np.abs(y_true_arr)))


def format_optional_metric(value: Optional[float], precision: int = 4) -> str:
    """Format a nullable metric for logs without converting undefined to zero."""
    return "N/A" if value is None else f"{value:.{precision}f}"


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
    if ss_tot <= EPS:
        return 1.0 if ss_res <= EPS else 0.0
    return float(1.0 - ss_res / ss_tot)
