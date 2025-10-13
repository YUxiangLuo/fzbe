from __future__ import annotations

from typing import Iterable, List, Dict, Any
import numpy as np
from statsmodels.tsa.stattools import adfuller


def _ensure_1d_float(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError("Input values must be 1-D numeric sequence.")
    return arr


def adf_fixed_d(values: Iterable[float],
                diff_orders: Iterable[int] = (0, 1, 2),
                regression: str = "c",
                alpha: float = 0.05,
                ndigits: int = 3) -> List[Dict[str, Any]]:
    """
    Run ADF for fixed differencing orders and return a structured result.
    """
    vals = _ensure_1d_float(values)
    results: List[Dict[str, Any]] = []

    for d in diff_orders:
        series = vals.copy()
        if d > 0:
            series = np.diff(series, n=d)

        entry: Dict[str, Any] = {"diff_order": int(d)}
        try:
            stat, pval, _lags, _nobs, crit, _icbest = adfuller(series, regression=regression)
            entry.update({
                "statistic": round(float(stat), ndigits),
                "p_value": round(float(pval), ndigits),
                "stationary": bool(pval < alpha),
                "critical_values": {
                    "1%": round(float(crit["1%"]), ndigits),
                    "5%": round(float(crit["5%"]), ndigits),
                    "10%": round(float(crit["10%"]), ndigits),
                },
            })
        except Exception as e:
            # Return a structured error for this diff order
            entry.update({
                "error": str(e),
                "statistic": None,
                "p_value": None,
                "stationary": None,
                "critical_values": None,
            })
        results.append(entry)

    return results
