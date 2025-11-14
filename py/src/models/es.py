# py/src/models/es.py
# -*- coding: utf-8 -*-
"""
Exponential Smoothing (Simple Exponential Smoothing / SES) model.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import shared utilities
from data_utils import load_data_from_json
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model

MODEL_TYPE = "es"

# Type alias for input flexibility
ArrayLike = Union[List[float], np.ndarray]


# ---------- utilities ----------

def _format_predictions(preds: List[float], std_dev: float) -> List[Dict[str, float]]:
    """
    Format predictions with standard deviation for JSON output.
    """
    return [{"prediction": p, "std_dev": std_dev} for p in preds]


def _load_state(model_path: str) -> Dict[str, Any]:
    """
    Load and validate a saved ES model state.
    """
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid ES model file.")
    return state


# ---------- predict ----------

def predict(
    model_path: str,
    steps: int,
    data: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], float]:
    """
    Forecast using a saved SES model.

    For Simple Exponential Smoothing, all forecasts equal the last smoothed level.
    If future actual values are provided, they are used to update the level for
    subsequent predictions (online updating).
    """
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")
    state = state or _load_state(model_path)
    alpha = float(state["alpha"])
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"[VALIDATION] Invalid alpha={alpha}, must be in (0,1].")
    level = float(state["level"])
    std_dev_residuals = float(state.get("std_dev_residuals", 0.0))

    future_values: Optional[List[float]] = None
    if data:
        try:
            parsed_data = load_data_from_json(data)
            future_values = parsed_data.y.tolist()
        except (ValueError, TypeError):
            future_values = None

    preds: List[float] = []
    for idx in range(steps):
        # Prediction at time t is the current smoothed level
        pred = level
        preds.append(pred)

        # If actual value is available, update level for next prediction
        if future_values and idx < len(future_values):
            actual = future_values[idx]
            level = alpha * actual + (1.0 - alpha) * level

    return preds, std_dev_residuals


# ---------- train and evaluate ----------

def _fit_level(series: np.ndarray, train_range: range, alpha: float) -> float:
    """
    Fit the level using Simple Exponential Smoothing on the training range.
    """
    if len(train_range) < 1:
        raise ValueError("[VALIDATION] Training range must have at least one point.")
    level = float(series[train_range.start])
    for t in range(train_range.start + 1, train_range.stop):
        level = alpha * float(series[t]) + (1.0 - alpha) * level
    return level


def train_and_evaluate(
    sales: ArrayLike,
    months: List[str],
    train_range: range,
    eval_range: range,
    alpha: float,
) -> Tuple[Dict[str, Any], float, float, List[float], List[float]]:
    """
    Trains and evaluates the SES model.

    For Simple Exponential Smoothing, all future predictions equal the final
    smoothed level from the training period.
    """
    series = np.asarray(sales, dtype=np.float64)
    train_level = _fit_level(series, train_range, alpha)

    idxs = list(eval_range)
    if not idxs:
        return {}, 0.0, train_level, [], []

    # SES: All forecasts are the same - the final smoothed level
    preds = [train_level] * len(idxs)
    y_true = series[idxs]
    y_pred = np.array(preds, dtype=np.float64)

    residuals = y_true - y_pred
    # Use sample standard deviation (ddof=1) for prediction intervals
    std_dev_residuals = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
    return metrics, std_dev_residuals, train_level, y_true.tolist(), y_pred.tolist()


# ---------- public entry ----------

def run(args) -> None:
    """
    Main entry point for ES model training and prediction.
    """
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if load_path:
        state = _load_state(load_path)
        steps = None
        es = getattr(args, "eval_start", None)
        ee = getattr(args, "eval_end", None)

        fs = getattr(args, "forecast_steps", None)
        if fs is not None:
            steps = int(fs)
            if steps <= 0:
                raise ValueError("[VALIDATION] --forecast_steps must be > 0.")
        elif es is not None and ee is not None:
            try:
                steps = int(ee) - int(es) + 1
                if steps <= 0: raise ValueError()
            except (ValueError, TypeError):
                 raise ValueError("Invalid eval_start/eval_end range.")
        else:
            raise ValueError("Provide eval_start/eval_end or --forecast_steps.")

        preds, std_dev = predict(load_path, steps, getattr(args, "data", None), state=state)
        predictions_with_std = _format_predictions(preds, std_dev)

        out: Dict[str, Any] = {
            "mode": "es_loaded_predict",
            "alpha": float(state["alpha"]),
            "train_end_index": int(state["train_end_index"]),
            "forecast_steps": steps,
            "predictions": predictions_with_std,
        }
        if getattr(args, "data", None) and es is not None and ee is not None:
            _add_metrics_to_output(out, args.data, es, ee, preds)

        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # Training Mode
    if not hasattr(args, "data") or not args.data:
        raise ValueError("[VALIDATION] --data is required for training.")

    try:
        parsed_data = load_data_from_json(args.data)
        sales = parsed_data.y.tolist()
        months = parsed_data.timestamps or [str(i) for i in range(parsed_data.n_obs)]
        value_key = parsed_data.target_key
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"[DATA] Failed to parse --data: {e}")

    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)

    if not hasattr(args, "alpha") or args.alpha is None:
        raise ValueError("[VALIDATION] You must specify --alpha for SES.")
    try:
        alpha = float(args.alpha)
        if not (0.0 < alpha <= 1.0):
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("[VALIDATION] --alpha must be in (0,1].")

    _validate_training_args(sales, ts, te, es, ee)

    metrics, std_dev_residuals, final_level, y_true, y_pred = train_and_evaluate(
        sales, months, range(ts, te + 1), range(es, ee + 1), alpha
    )

    predictions_with_std = _format_predictions(y_pred, std_dev_residuals)

    out: Dict[str, Any] = {
        "mode": "ses_fixed_alpha",
        "data_points": len(sales),
        "alpha": alpha,
        "train_range": {"start_index": ts, "end_index": te, "months": months[ts : te + 1]},
        "evaluate_range": {"start_index": es, "end_index": ee, "months": months[es : ee + 1]},
        "metrics": metrics,
        "eval_y_true": y_true,
        "predictions": predictions_with_std, # Use the standardized key
    }

    if save_path:
        state = {
            "model_type": MODEL_TYPE, "alpha": alpha, "train_end_index": te,
            "level": final_level, "value_key": value_key,
            "std_dev_residuals": std_dev_residuals,
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))


def _validate_training_args(sales: ArrayLike, ts: int, te: int, es: int, ee: int) -> None:
    """
    Validate training and evaluation parameters.
    """
    n = len(sales)
    if not (0 <= ts <= te < n):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}]")
    if not (0 <= es <= ee < n):
        raise ValueError(f"[VALIDATION] Invalid eval range: [{es},{ee}]")
    if es <= te:
        raise ValueError("[VALIDATION] eval_start must be > train_end.")

def _add_metrics_to_output(out: Dict[str, Any], data: str, es: int, ee: int, preds: List[float]) -> None:
    try:
        parsed_data = load_data_from_json(data)
        series = parsed_data.y
        months = parsed_data.timestamps or [str(i) for i in range(parsed_data.n_obs)]
        y_true = series[int(es) : int(ee) + 1]
        if len(y_true) == len(preds):
            out["metrics"] = {
                "rmse": rmse(y_true, preds), "mae": mae(y_true, preds),
                "mape": mape(y_true, preds), "r2": r2(y_true, preds),
            }
            out["evaluate_months"] = months[int(es) : int(ee) + 1]
        else:
            out["metrics_error"] = "Length mismatch for evaluation."
    except (ValueError, TypeError) as e:
        out["metrics_error"] = f"Failed to calculate metrics: {e}"
