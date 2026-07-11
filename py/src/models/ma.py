# src/models/ma.py
# -*- coding: utf-8 -*-
"""
Moving Average (MA) time series forecasting model.
"""

import json
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from data_utils import ParsedData, load_data_from_json
from logger import log, get_log_contents
from metrics import mae, mape, r2, rmse
from model_info import attach_method_metadata, ensure_state_metadata
from serialization import dump_model, load_model

MODEL_TYPE = "ma"
METHOD_NAME = "递推移动平均"
FORECAST_STRATEGY = "recursive_roll_forward"
IMPLEMENTATION_NOTES = [
    "教科书移动平均通常用最后一个窗口均值预测下一期。",
    "本系统预测多期未来销量时，将每期预测值滚入窗口继续递推，避免使用评估集或未来真实值。",
    "预测值附带的 std_dev 是评估段残差的常数标准差，不随预测步长增长，与 ARIMA 随步长展宽的置信区间口径不同。",
]

# Type alias for input flexibility
ArrayLike = Union[List[float], np.ndarray]


# ---------- core ----------
def moving_average_forecast(
    series: ArrayLike, eval_range: range, window: int, min_index: int = 0
) -> Tuple[List[int], List[float]]:
    """
    Generate one-step-ahead forecasts using moving average.
    """
    log.debug(f"moving_average_forecast called with window={window}, eval_range={eval_range}, min_index={min_index}")
    if window <= 0:
        raise ValueError("[VALIDATION] Moving average window must be greater than 0.")
    idxs, preds = [], []
    series = np.asarray(series)
    log.debug(f"Series length: {len(series)}")

    skipped_points = 0
    for t in eval_range:
        start = t - window
        if start < min_index:
            skipped_points += 1
            continue
        avg_value = float(np.mean(series[start:t]))
        preds.append(avg_value)
        idxs.append(t)
        log.debug(f"  t={t}: averaging series[{start}:{t}] = {avg_value:.4f}")

    if skipped_points > 0:
        log.warning(f"Skipped {skipped_points} evaluation points due to insufficient history (window={window})")
    log.info(f"Generated {len(preds)} predictions out of {len(list(eval_range))} requested evaluation points")

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
    return ensure_state_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)


def _forecast_future(
    history: List[float],
    window: int,
    steps: int,
) -> List[float]:
    """
    Generate multi-step-ahead forecasts.
    Each forecast is appended to the rolling window before the next step is
    generated, so no future or evaluation actuals are used.
    """
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    if window <= 0:
        raise ValueError("[VALIDATION] Moving average window must be greater than 0.")

    initial_history = list(history)[-window:]
    if len(initial_history) < window:
        raise ValueError(
            f"[DATA] History has {len(initial_history)} points, fewer than the moving average window ({window}); cannot forecast."
        )

    history_buffer: deque[float] = deque(initial_history, maxlen=window)
    preds: List[float] = []

    for _ in range(steps):
        avg = float(np.mean(history_buffer))
        preds.append(avg)
        history_buffer.append(avg)

    return preds


def _estimate_std_dev(residuals: ArrayLike, reference_series: ArrayLike) -> float:
    residuals_array = np.asarray(residuals, dtype=float)
    if residuals_array.size > 1:
        return float(np.std(residuals_array, ddof=1))

    reference_array = np.asarray(reference_series, dtype=float)
    if reference_array.size > 1:
        return float(np.std(reference_array, ddof=1))

    baseline = abs(float(np.mean(reference_array))) if reference_array.size else 0.0
    return float(max(baseline * 0.05, 1.0))


# ---------- predict (for ensemble / CLI load) ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], float]:
    """
    Forecast future values using a saved MA model.
    """
    state = state or _load_state(model_path)
    window = int(state["window"])
    history = list(state["history"])
    std_dev_residuals = float(state.get("std_dev_residuals", 0.0))

    predictions = _forecast_future(history, window, steps)
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
    log.info(f"Starting MA train_and_evaluate: window={window}, train_range={train_range}, eval_range={eval_range}")
    series = np.asarray(sales)
    log.debug(f"Total series length: {len(series)}")
    log.debug(f"Training data range: [{train_range.start}, {train_range.stop-1}], count={train_range.stop - train_range.start}")
    log.debug(f"Evaluation data range: [{eval_range.start}, {eval_range.stop-1}], count={eval_range.stop - eval_range.start}")

    eval_list = list(eval_range)
    if not eval_list:
        log.warning("Empty evaluation range, returning empty metrics")
        return {}, 0.0, np.array([]), np.array([])

    train_history = series[train_range]
    if len(train_history) < window:
        raise ValueError("[VALIDATION] Training range must be at least as large as the moving average window.")

    history_start = max(train_range.start, train_range.stop - window)
    forecast_steps = eval_range.stop - train_range.stop
    forecast_offset = eval_range.start - train_range.stop
    history = series[history_start:train_range.stop].tolist()
    all_forecasts = _forecast_future(history, window, forecast_steps)
    preds = all_forecasts[forecast_offset: forecast_offset + len(eval_list)]
    idxs = eval_list

    if len(preds) != len(idxs):
        raise ValueError("[VALIDATION] 无法为全部评估区间生成移动平均预测。")

    y_true = series[idxs]
    y_pred = np.array(preds)
    residuals = y_true - y_pred
    std_dev_residuals = _estimate_std_dev(residuals, train_history)

    log.info(f"Calculated residuals: mean={np.mean(residuals):.4f}, std_dev={std_dev_residuals:.4f}")

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
    log.info(f"Evaluation metrics computed: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}, R²={metrics['r2']:.4f}")

    return metrics, std_dev_residuals, y_true, y_pred


# ---------- public entry ----------
def run(args) -> None:
    """
    Main entry point for MA model training and prediction.
    """
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if getattr(args, "fit_only", False):
        if not hasattr(args, "data") or not args.data:
            raise ValueError("[VALIDATION] --data is required for fit-only mode.")
        if not save_path:
            raise ValueError("[VALIDATION] --save-model is required for fit-only mode.")

        state = _load_state(load_path) if load_path else {}
        parsed_data = load_data_from_json(args.data)
        y = parsed_data.y

        ts, te = int(args.train_start), int(args.train_end)
        window = int(state.get("window") if state else getattr(args, "window", 0))
        _validate_fit_args(y, ts, te, window)

        history_start = max(ts, te - window + 1)
        history = y[history_start : te + 1]
        std_dev_residuals = float(state.get("std_dev_residuals", _estimate_std_dev([], history)))

        next_state = {
            "model_type": MODEL_TYPE,
            "window": window,
            "train_end_index": te,
            "history": history.tolist(),
            "std_dev_residuals": std_dev_residuals,
        }
        attach_method_metadata(next_state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(next_state, save_path)

        out: Dict[str, Any] = {
            "mode": "ma_fit_only",
            "window": window,
            "train_end_index": te,
            "saved_model": save_path,
        }
        attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        out["logs"] = get_log_contents()
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    if load_path:
        state = _load_state(load_path)
        steps = int(getattr(args, "forecast_steps", 0))
        if steps <= 0:
            raise ValueError("[VALIDATION] --forecast_steps must be > 0.")

        log.info(f"Starting prediction with {steps} forecast steps")
        preds, std_dev = predict(load_path, steps, state=state)
        log.info(f"Generated {len(preds)} predictions")
        predictions_with_std = _format_predictions(preds, std_dev)

        out: Dict[str, Any] = {
            "mode": "ma_loaded_predict",
            "window": int(state["window"]),
            "train_end_index": int(state["train_end_index"]),
            "forecast_steps": steps,
            "predictions": predictions_with_std,
        }
        attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        log.info("Prediction finished successfully")
        out["logs"] = get_log_contents()
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
        if window_arg is not None:
            window = int(window_arg)
        else:
            # Auto-select window based on training data size (for educational small datasets)
            train_size = te - ts + 1
            window = max(3, min(12, train_size // 4))
            log.info(f"Auto-selected window={window} based on train_size={train_size}")
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
        attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)

        if save_path:
            log.info(f"Saving model to {save_path}")
            history_start = max(ts, te - window + 1)
            history = y[history_start : te + 1]
            log.debug(f"Saving history from index {history_start} to {te}, length={len(history)}")
            state = {
                "model_type": MODEL_TYPE, "window": window, "train_end_index": te,
                "history": history.tolist(),
                "std_dev_residuals": std_dev_residuals,
            }
            attach_method_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
            dump_model(state, save_path)
            log.info(f"Model saved successfully to {save_path}")
            out["saved_model"] = save_path

        out["logs"] = get_log_contents()
        log.info("Run finished successfully.")
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))

    except ValueError:
        raise
    except Exception as e:
        log.error(f"An unexpected error occurred during MA training: {e}", exc_info=True)
        raise ValueError(f"[DATA] Failed to parse or process --data: {e}")


def _validate_fit_args(sales: ArrayLike, ts: int, te: int, w: int) -> None:
    n_total = len(sales)

    if not (0 <= ts <= te < n_total):
        raise ValueError(f"[VALIDATION] Invalid training range [{ts}, {te}], must be within [0, {n_total-1}]")
    if w < 2:
        raise ValueError(f"[VALIDATION] Window must be >= 2, got {w}")

    train_size = te - ts + 1
    if train_size < w:
        raise ValueError(f"[VALIDATION] Training range must contain at least {w} points.")


def _validate_training_args(sales: ArrayLike, ts: int, te: int, es: int, ee: int, w: int) -> None:
    """
    Validate training and evaluation parameters.
    """
    log.debug(f"Validating training arguments: ts={ts}, te={te}, es={es}, ee={ee}, window={w}")
    n_total = len(sales)
    log.debug(f"Total data points: {n_total}")

    if not (0 <= ts <= te < n_total):
        log.error(f"Invalid training range: [{ts}, {te}], data size: {n_total}")
        raise ValueError(f"[VALIDATION] Invalid training range [{ts}, {te}], must be within [0, {n_total-1}]")
    if not (es >= 0 and ee < n_total and es <= ee):
        log.error(f"Invalid evaluation range: [{es}, {ee}], data size: {n_total}")
        raise ValueError(f"[VALIDATION] Invalid evaluation range [{es}, {ee}], must be within [0, {n_total-1}]")
    if es <= te:
        log.error(f"Evaluation start {es} must be after training end {te}")
        raise ValueError(f"[VALIDATION] Evaluation start ({es}) must be after training end ({te}).")
    if w < 2:
        log.error(f"Window {w} must be >= 2")
        raise ValueError(f"[VALIDATION] Window must be >= 2, got {w}")

    # Additional helpful validations
    train_size = te - ts + 1
    eval_size = ee - es + 1
    if train_size < w:
        log.error(f"Training range size {train_size} must be at least window {w}")
        raise ValueError(f"[VALIDATION] Training range must contain at least {w} points.")
    log.info(f"Validation passed: train_size={train_size}, eval_size={eval_size}, window={w}")
