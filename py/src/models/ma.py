# src/models/ma.py
# -*- coding: utf-8 -*-
"""
Moving Average (MA) time series forecasting model.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from data_utils import ParsedData, load_data_from_json
from logger import log, get_log_contents
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model

MODEL_TYPE = "ma"

# Type alias for input flexibility
ArrayLike = Union[List[float], np.ndarray]


# ---------- core ----------
def moving_average_forecast(
    series: ArrayLike, eval_range: range, window: int, min_index: int = 0
) -> Tuple[List[int], List[float]]:
    """
    Generate one-step-ahead forecasts using moving average.
    """
    if window <= 0:
        raise ValueError("[VALIDATION] Moving average window must be greater than 0.")
    idxs, preds = [], []
    series = np.asarray(series)

    for t in eval_range:
        start = t - window
        if start < min_index:
            continue
        preds.append(float(np.mean(series[start:t])))
        idxs.append(t)
    return idxs, preds


def _format_predictions(preds: List[float], std_dev: float) -> List[Dict[str, float]]:
    """
    Format predictions with standard deviation for JSON output.
    """
    return [{"prediction": p, "std_dev": std_dev} for p in preds]


def _load_state(model_path: str) -> Dict[str, Any]:
    """
    Load and validate a saved MA model state.
    """
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid MA model file.")
    return state


def _forecast_future(
    history: List[float],
    window: int,
    steps: int,
    future_values: Optional[List[float]] = None,
) -> List[float]:
    """
    Generate multi-step-ahead forecasts.
    Uses deque for efficient sliding window operations.
    """
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    # Use deque with maxlen for O(1) append operations on sliding window
    initial_history = history[-window:] if window > 0 else list(history)
    if not initial_history:
        raise ValueError("[DATA] History is empty; cannot forecast.")

    buf = deque(initial_history, maxlen=window if window > 0 else None)
    preds = []
    future_iter = iter(future_values or [])

    for _ in range(steps):
        if not buf:
            raise ValueError("Insufficient history for moving average window.")

        avg = sum(buf) / len(buf)
        preds.append(avg)

        try:
            buf.append(next(future_iter))
        except StopIteration:
            buf.append(avg)

    return preds


# ---------- predict (for ensemble / CLI load) ----------
def predict(
    model_path: str,
    steps: int,
    data: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], float]:
    """
    Forecast future values using a saved MA model.
    """
    state = state or _load_state(model_path)
    window = int(state["window"])
    history = list(state["history"])
    std_dev_residuals = float(state.get("std_dev_residuals", 0.0))
    train_end = int(state["train_end_index"])

    future_values: Optional[List[float]] = None
    if data:
        try:
            parsed_data = load_data_from_json(data)
            series = parsed_data.y
            start = train_end + 1
            if start < len(series):
                future_values = series[start : start + steps].tolist()
        except (ValueError, TypeError):
            future_values = None

    predictions = _forecast_future(history, window, steps, future_values)
    return predictions, std_dev_residuals


def train_and_evaluate(
    sales: ArrayLike,
    months: List[str],
    train_range: range,
    eval_range: range,
    window: int,
) -> Tuple[Dict[str, Any], float, np.ndarray, np.ndarray]:
    """
    Evaluate the moving average model on a test set.
    """
    series = np.asarray(sales)

    # Validate that evaluation range has sufficient history
    # First possible prediction point needs at least 'window' points of history
    eval_list = list(eval_range)
    if eval_list:
        first_possible_eval_point = train_range.start + window
        if max(eval_list) < first_possible_eval_point:
            raise ValueError(
                f"[VALIDATION] Insufficient history for all evaluation points. "
                f"First possible evaluation point is {first_possible_eval_point}."
            )

    idxs, preds = moving_average_forecast(series, eval_range, window, min_index=train_range.start)

    if not idxs and eval_list:
        raise ValueError("[VALIDATION] No evaluation points could be generated.")

    # Handle empty eval range gracefully
    if not idxs:
        return {}, 0.0, np.array([]), np.array([])

    y_true = series[idxs]
    y_pred = np.array(preds)
    residuals = y_true - y_pred
    std_dev_residuals = float(np.std(residuals))

    metrics: Dict[str, Any] = {
        "window": window,
        "evaluated_points": len(y_true),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "evaluate_indices": idxs,
        "evaluate_months": [months[i] for i in idxs] if months else [],
    }
    return metrics, std_dev_residuals, y_true, y_pred


# ---------- public entry ----------
def run(args) -> None:
    """
    Main entry point for MA model training and prediction.
    """
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if load_path:
        state = _load_state(load_path)
        steps = int(getattr(args, "forecast_steps", 0))
        if steps <= 0:
            raise ValueError("[VALIDATION] --forecast_steps must be > 0.")

        preds, std_dev = predict(load_path, steps, getattr(args, "data", None), state=state)
        predictions_with_std = _format_predictions(preds, std_dev)

        out: Dict[str, Any] = {
            "mode": "ma_loaded_predict",
            "window": int(state["window"]),
            "train_end_index": int(state["train_end_index"]),
            "forecast_steps": steps,
            "predictions": predictions_with_std,
        }
        out["logs"] = get_log_contents()
        log.info("Prediction finished successfully.")
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # Training Mode
    if not hasattr(args, "data") or not args.data:
        raise ValueError("[VALIDATION] --data is required for training.")

    try:
        parsed_data = load_data_from_json(args.data)
        y = parsed_data.y
        timestamps = parsed_data.timestamps or [str(i) for i in range(parsed_data.n_obs)]
        log.info(f"Data loaded successfully. Number of observations: {parsed_data.n_obs}")

        ts, te = int(args.train_start), int(args.train_end)
        es, ee = int(args.eval_start), int(args.eval_end)

        window_arg = getattr(args, "window", None)
        window = int(window_arg) if window_arg is not None else 3
        log.debug(f"Training params: window={window}")

        _validate_training_args(y, ts, te, es, ee, window)

        log.info("Starting model training and evaluation...")
        metrics, std_dev_residuals, y_true, y_pred = train_and_evaluate(
            y, timestamps, range(ts, te + 1), range(es, ee + 1), window
        )

        predictions_with_std = _format_predictions(y_pred.tolist(), std_dev_residuals)

        out: Dict[str, Any] = {
            "mode": "fixed_window_only",
            "data_points": len(y),
            "train_range": {"start_index": ts, "end_index": te, "months": timestamps[ts : te + 1]},
            "evaluate_range": {"start_index": es, "end_index": ee, "months": timestamps[es : ee + 1]},
            "metrics": metrics,
            "eval_y_true": y_true.tolist(),
            "predictions": predictions_with_std,
        }

        if save_path:
            history_start = max(ts, te - window + 1)
            history = y[history_start : te + 1]
            state = {
                "model_type": MODEL_TYPE, "window": window, "train_end_index": te,
                "history": history.tolist(),
                "std_dev_residuals": std_dev_residuals,
            }
            dump_model(state, save_path)
            out["saved_model"] = save_path

        out["logs"] = get_log_contents()
        log.info("Run finished successfully.")
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))

    except Exception as e:
        log.error(f"An unexpected error occurred during MA training: {e}", exc_info=True)
        raise ValueError(f"[DATA] Failed to parse or process --data: {e}")


def _validate_training_args(sales: ArrayLike, ts: int, te: int, es: int, ee: int, w: int) -> None:
    """
    Validate training and evaluation parameters.
    """
    n_total = len(sales)
    if not (0 <= ts <= te < n_total):
        raise ValueError(f"[VALIDATION] Invalid training range [{ts}, {te}]")
    if not (es >= 0 and ee < n_total and es <= ee):
        raise ValueError(f"[VALIDATION] Invalid evaluation range [{es}, {ee}]")
    if es <= te:
        raise ValueError("[VALIDATION] Evaluation start must be after training end.")
    if w < 1:
        raise ValueError("[VALIDATION] --window must be >= 1")