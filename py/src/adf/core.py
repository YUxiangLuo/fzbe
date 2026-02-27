from __future__ import annotations

from typing import Iterable, List, Dict, Any
import numpy as np
from statsmodels.tsa.stattools import adfuller


def _ensure_1d_float(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError("Input values must be 1-D numeric sequence.")
    if not np.isfinite(arr).all():  # IMPROVED: Check NaN/inf
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
        regression: ADF regression type ('c', 'ct', 'ctt', 'nc').
        alpha: Significance level for stationary test (default 0.05).
        ndigits: Rounding digits for statistic/critical_values (p_value kept precise).

    Returns:
        List of dicts, one per diff_order.
    """
    if not diff_orders:
        raise ValueError("diff_orders cannot be empty.")  # IMPROVED: Edge case
    vals = _ensure_1d_float(values)
    min_len = max(diff_orders) + 1
    if len(vals) < min_len:
        raise ValueError(f"Input length {len(vals)} too short for max diff_order {max(diff_orders)}; need >= {min_len}.")

    results: List[Dict[str, Any]] = []

    for d in diff_orders:
        series = vals.copy()
        entry: Dict[str, Any] = {"diff_order": int(d)}
        if d > 0:
            if len(series) <= d:
                raise ValueError(f"Cannot diff order {d} on length {len(vals)}; need > {d} points.")  # IMPROVED: Pre-check
            series = np.diff(series, n=d)
        if len(series) < 4:  # IMPROVED: ADF minimum for reliability
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
            stat, pval, used_lags, nobs, crit, icbest = adfuller(series, regression=regression)
            entry.update({
                "statistic": round(float(stat), ndigits),
                "p_value": float(pval),  # IMPROVED: Keep precise
                "used_lags": int(used_lags),
                "n_obs": int(nobs),
                "stationary": bool(pval < alpha),
                "critical_values": {k: round(float(v), ndigits) for k, v in crit.items()},
            })
        except Exception as e:
            # IMPROVED: More specific error message
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