# -*- coding: utf-8 -*-
"""
Common evaluation metrics for time series forecasting models.
"""
import math
from typing import List, Union

# Epsilon for numerical stability in MAPE calculation
EPS = 1e-12


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value, or NaN if input is empty
    """
    if not y_true:
        return float("nan")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def mae(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value, or NaN if input is empty
    """
    if not y_true:
        return float("nan")
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def mape(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE value in percentage, or NaN if input is empty
    """
    if not y_true:
        return float("nan")
    return 100.0 * sum(
        abs((a - b) / (a if abs(a) > EPS else EPS))
        for a, b in zip(y_true, y_pred)
    ) / len(y_true)


def r2(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate R-squared (coefficient of determination).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² value, or NaN if input is empty
    """
    if not y_true:
        return float("nan")
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean_y) ** 2 for a in y_true)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
