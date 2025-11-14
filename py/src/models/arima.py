# src/models/arima.py
# -*- coding: utf-8 -*-
"""
ARIMA training/prediction with (p,q) grid-search and fixed d.
"""

import json
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from data_utils import load_data_from_json
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model

MODEL_TYPE = "arima"


# -----------------------------
# Utilities
# -----------------------------
def _validate_ranges(n: int, ts: int, te: int, es: int, ee: int) -> None:
    """Closed-interval semantics: indices are inclusive [start, end]."""
    if not (0 <= ts <= te < n):
        raise ValueError(f"[VALIDATION] train range invalid: ts={ts}, te={te}, n={n}")
    if not (0 <= es <= ee < n):
        raise ValueError(f"[VALIDATION] eval range invalid: es={es}, ee={ee}, n={n}")
    if es <= te:
        raise ValueError("[VALIDATION] eval_start must be > train_end.")


def _fit_arima(
    y_train: np.ndarray, d: int, p: int, q: int, y_eval_true: np.ndarray
) -> Tuple[Dict[str, Any], SARIMAXResults | None]:
    """
    Fit ARIMA model and return result dict and fitted object.
    """
    model = SARIMAX(y_train, order=(p, d, q), trend="c", enforce_stationarity=True, enforce_invertibility=True)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            res = model.fit(disp=False)
        converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
    except Exception as e:
        return {"p": p, "d": d, "q": q, "error": str(e)[:200], "converged": False}, None

    steps = len(y_eval_true)
    fcst = res.get_forecast(steps=steps)
    y_pred = fcst.predicted_mean.astype(np.float64)
    aic = float(res.aic) if res.aic is not None and np.isfinite(res.aic) else float("inf")

    return {
        "p": p, "d": d, "q": q,
        "rmse": rmse(y_eval_true, y_pred), "mae": mae(y_eval_true, y_pred),
        "mape": mape(y_eval_true, y_pred), "r2": r2(y_eval_true, y_pred),
        "aic": aic, "converged": converged, "y_pred": y_pred,
        "conf_int": fcst.conf_int().astype(np.float64),
    }, res


def _search_pq(
    y_train: np.ndarray, y_eval_true: np.ndarray, d: int, p_max: int, q_max: int
) -> Tuple[Dict[str, Any], SARIMAXResults, Dict[str, Any]]:
    """
    Search over p=[0..p_max], q=[0..q_max] with fixed d.
    """
    results, failures = [], []
    best_tuple, best_item, best_res_obj = None, None, None

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            item, res_obj = _fit_arima(y_train, d, p, q, y_eval_true)
            if "error" in item:
                failures.append(item)
            else:
                results.append(item)
                k = (item["rmse"], item["aic"], p + q, p)
                if best_tuple is None or k < best_tuple:
                    best_tuple, best_item, best_res_obj = k, item, res_obj

    if best_item is None:
        raise ValueError(f"[VALIDATION] All (p,q) fits failed for d={d}.")

    search_diag = {
        "candidates_tried": (p_max + 1) * (q_max + 1),
        "succeeded": len(results), "failed": len(failures),
    }
    return best_item, best_res_obj, search_diag


# ---------- predict (for ensemble) ----------
def predict(model_path: str, steps: int, data: str = None) -> List[Dict[str, float]]:
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0")
    res = load_model(model_path)
    fcst = res.get_forecast(steps=steps)
    return [
        {"prediction": p, "std_dev": s}
        for p, s in zip(fcst.predicted_mean.tolist(), fcst.se_mean.tolist())
    ]


# -----------------------------
# Public entry point
# -----------------------------
def run(args) -> None:
    pretty = bool(getattr(args, "pretty", False))
    if getattr(args, "load_model", None):
        _handle_prediction(args, pretty=pretty)
    else:
        _handle_training(args, pretty=pretty)


def _handle_training(args, pretty: bool) -> None:
    if not getattr(args, "data", None):
        raise ValueError("[VALIDATION] ARIMA training requires --data.")

    try:
        data = load_data_from_json(args.data)
        y, months, n = data.y, data.timestamps, data.n_obs
        ts, te, es, ee, d = map(int, [args.train_start, args.train_end, args.eval_start, args.eval_end, args.d])
        p_max, q_max = int(getattr(args, "p_max", 3)), int(getattr(args, "q_max", 3))
    except (ValueError, TypeError, AttributeError) as e:
        raise ValueError(f"Invalid arguments or data format: {e}")

    _validate_ranges(n, ts, te, es, ee)
    y_train = y[ts : te + 1].astype(np.float64)
    y_eval_true = y[es : ee + 1].astype(np.float64)

    best, best_res, search_summary = _search_pq(y_train, y_eval_true, d, p_max, q_max)

    if getattr(args, "save_model", None):
        dump_model(best_res, args.save_model)

    eval_pred_list = best.get("y_pred", np.array([])).tolist()
    out = {
        "mode": "arima_search_pq_fixed_d",
        "best_order": {"p": best["p"], "d": best["d"], "q": best["q"]},
        "metrics": {k: v for k, v in best.items() if k in ["rmse", "mae", "mape", "r2", "aic", "converged"]},
        "eval_y_true": y_eval_true.tolist(),
        "eval_predictions": eval_pred_list,
        "search_summary": search_summary,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))


def _handle_prediction(args, pretty: bool) -> None:
    res = load_model(args.load_model)
    steps = int(getattr(args, "forecast_steps", 0))
    if steps <= 0:
        raise ValueError("[VALIDATION] --forecast_steps must be > 0")

    fcst = res.get_forecast(steps=steps)
    y_pred = fcst.predicted_mean.astype(np.float64)
    se_mean = fcst.se_mean.astype(np.float64)

    predictions = [{"prediction": float(p), "std_dev": float(s)} for p, s in zip(y_pred, se_mean)]
    out = {
        "mode": "arima_loaded_predict",
        "order": res.model.order,
        "forecast_steps": steps,
        "predictions": predictions,
    }

    if getattr(args, "data", None):
        try:
            data = load_data_from_json(args.data)
            y_true = data.y
            if len(y_true) == len(y_pred):
                out["metrics"] = {
                    "rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred),
                    "mape": mape(y_true, y_pred), "r2": r2(y_true, y_pred),
                }
        except Exception as e:
            out["metrics_error"] = f"Failed to compute metrics: {e}"

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))