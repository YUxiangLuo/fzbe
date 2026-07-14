from __future__ import annotations

from typing import Iterable, List, Dict, Any
import numpy as np
from statsmodels.tsa.stattools import adfuller


def _ensure_1d_float(values: Iterable[float]) -> np.ndarray:
    raw_values = values if isinstance(values, np.ndarray) else list(values)
    arr = np.asarray(raw_values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input values must be 1-D numeric sequence.")
    if not np.isfinite(arr).all():
        raise ValueError("Input contains NaN or inf; remove before ADF.")
    return arr


def adf_fixed_d(values: Iterable[float],
                diff_orders: Iterable[int] = (0, 1, 2),
                regression: str = "c",
                alpha: float = 0.05,
                ndigits: int = 3) -> List[Dict[str, Any]]:
    """
    Run ADF for fixed differencing orders and return a structured result.

    Example:
        >>> adf_fixed_d([1, 2, 3, 4, 5], diff_orders=(0, 1))
        [{'diff_order': 0, 'p_value': 0.99, 'stationary': False, ...},
         {'diff_order': 1, 'p_value': 0.01, 'stationary': True, ...}]

    Args:
        values: Time series values.
        diff_orders: Differencing orders to test (default: 0,1,2).
        regression: ADF regression type ('n', 'c', 'ct', 'ctt').
        alpha: Significance level for stationary test (default 0.05).
        ndigits: Rounding digits for statistic/critical_values (p_value kept precise).

    Returns:
        List of dicts, one per diff_order.
    """
    orders = tuple(diff_orders)
    if not orders:
        raise ValueError("diff_orders cannot be empty.")
    if any(
        isinstance(order, (bool, np.bool_))
        or not isinstance(order, (int, np.integer))
        or int(order) < 0
        for order in orders
    ):
        raise ValueError("diff_orders must contain only non-negative integers.")
    normalized_orders = tuple(int(order) for order in orders)
    if len(set(normalized_orders)) != len(normalized_orders):
        raise ValueError("diff_orders cannot contain duplicates.")

    vals = _ensure_1d_float(values)
    max_order = max(normalized_orders)
    min_len = max_order + 1
    if len(vals) < min_len:
        raise ValueError(f"Input length {len(vals)} too short for max diff_order {max_order}; need >= {min_len}.")

    results: List[Dict[str, Any]] = []

    for d in normalized_orders:
        series = vals.copy()
        entry: Dict[str, Any] = {"diff_order": int(d)}
        if d > 0:
            if len(series) <= d:
                raise ValueError(f"Cannot diff order {d} on length {len(vals)}; need > {d} points.")
            series = np.diff(series, n=d)
        if len(series) < 4:
            entry.update({
                "error": f"Series too short ({len(series)} points) after d={d} for reliable ADF.",
                "statistic": None,
                "p_value": None,
                "used_lags": None,
                "n_obs": None,
                "stationary": None,
                "critical_values": None,
            })
            results.append(entry)
            continue

        try:
            stat, pval, used_lags, nobs, crit, _icbest = adfuller(
                series,
                regression=regression,
                autolag="AIC",
            )
            entry.update({
                "statistic": round(float(stat), ndigits),
                "p_value": float(pval),
                "used_lags": int(used_lags),
                "n_obs": int(nobs),
                "stationary": bool(pval < alpha),
                "critical_values": {k: round(float(v), ndigits) for k, v in crit.items()},
            })
        except Exception as e:
            entry.update({
                "error": f"ADF failed for d={d}: {str(e)}",
                "statistic": None,
                "p_value": None,
                "used_lags": None,
                "n_obs": None,
                "stationary": None,
                "critical_values": None,
            })
        results.append(entry)

    return results
