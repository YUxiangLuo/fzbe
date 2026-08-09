# src/models/ma.py
# -*- coding: utf-8 -*-
"""
Moving Average (MA) time series forecasting model.
"""

import json
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from data_utils import load_data_from_json
from forecast_utils import (
    DEMAND_PREDICTION_DOMAIN,
    apply_prediction_domain,
    resolve_prediction_domain,
)
from logger import log, get_log_contents
from metrics import format_optional_metric, mae, mape, r2, rmse
from model_info import attach_method_metadata, ensure_state_metadata
from serialization import dump_model, load_model
from uncertainty import format_predictions, moving_average_profile, profile_for_steps

MODEL_TYPE = "ma"
METHOD_NAME = "递推移动平均"
FORECAST_STRATEGY = "recursive_roll_forward"
IMPLEMENTATION_NOTES = [
    "教科书移动平均通常用最后一个窗口均值预测下一期。",
    "本系统要求窗口 n >= 2；n=1 等同于朴素的上一期值预测，不能体现移动平均的平滑特性，因此不作为教学选项。",
    "本系统预测多期未来销量时，将每期预测值滚入窗口继续递推，避免使用评估集或未来真实值。",
    "移动平均原始输出通常在非负销量输入下自然非负；本系统仍将最终销量点预测按 max(0, y_hat) 截断，并用截断后的值计算残差与指标。该后处理不是教科书移动平均公式的一部分。",
    "std_dev 按训练段 rolling-origin 多步残差逐 horizon 估计；2.5%–97.5% 经验残差分位范围和生产用 99% 上侧压力偏移直接取同一组 actual-prediction 残差的经验分位数，因此会保留稳定偏差。该范围不声明 95% 覆盖保证；某个 horizon 样本不足时使用明确标记的一阶差分 sqrt(h) 正态近似回退。",
]

# Type alias for input flexibility
ArrayLike = Union[List[float], np.ndarray]


def _format_predictions(
    preds: List[float],
    std_dev: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Format predictions with standard deviation for JSON output.
    """
    profile = list(std_dev) if isinstance(std_dev, (list, tuple, np.ndarray)) else [float(std_dev)] * len(preds)
    return format_predictions(preds, profile, metadata or {"source": "empirical", "fallback": False})


def _load_state(model_path: str) -> Dict[str, Any]:
    """
    Load and validate a saved MA model state.
    """
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid MA model file.")
    try:
        window = int(state["window"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("[VALIDATION] Invalid MA model window.") from exc
    if window < 2:
        raise ValueError(f"[VALIDATION] Window must be >= 2, got {window}")
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

    if window < 2:
        raise ValueError(f"[VALIDATION] Window must be >= 2, got {window}")

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


# ---------- predict (for ensemble / CLI load) ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
    allow_negative_predictions: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Forecast future values using a saved MA model.
    """
    state = state or _load_state(model_path)
    window = int(state["window"])
    history = list(state["history"])

    prediction_domain = resolve_prediction_domain(
        state,
        allow_negative_predictions=allow_negative_predictions,
    )
    predictions = apply_prediction_domain(
        _forecast_future(history, window, steps),
        prediction_domain,
    )
    uncertainty_profile, _ = profile_for_steps(
        state.get("uncertainty_profile"),
        steps,
        state.get("reference_series", history),
        state.get("uncertainty_summary"),
    )
    return predictions.tolist(), uncertainty_profile


def train_and_evaluate(
    sales: ArrayLike,
    months: List[str],
    train_range: range,
    eval_range: range,
    window: int,
    prediction_domain: str = DEMAND_PREDICTION_DOMAIN,
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
    y_pred = apply_prediction_domain(preds, prediction_domain)
    residuals = y_true - y_pred
    one_step_profile, _ = moving_average_profile(train_history, window, 1)
    std_dev_residuals = float(one_step_profile[0])

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
    log.info(
        f"Evaluation metrics computed: RMSE={metrics['rmse']:.4f}, "
        f"MAE={metrics['mae']:.4f}, MAPE={format_optional_metric(metrics['mape'])}, "
        f"R²={metrics['r2']:.4f}"
    )

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
        prediction_domain = resolve_prediction_domain(
            state,
            allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
        )
        parsed_data = load_data_from_json(args.data)
        y = parsed_data.y

        ts, te = int(args.train_start), int(args.train_end)
        if state and state.get("window") is None:
            raise ValueError("[VALIDATION] Saved MA model is missing window.")
        window = int(state["window"] if state else getattr(args, "window", 0))
        _validate_fit_args(y, ts, te, window)

        history_start = max(ts, te - window + 1)
        history = y[history_start : te + 1]
        reference_series = np.asarray(y)[range(ts, te + 1)]
        profile, uncertainty_summary = moving_average_profile(reference_series, window, 12)
        std_dev_residuals = float(profile[0])

        next_state = {
            "model_type": MODEL_TYPE,
            "window": window,
            "train_end_index": te,
            "history": history.tolist(),
            "std_dev_residuals": std_dev_residuals,
            "uncertainty_profile": profile,
            "uncertainty_summary": uncertainty_summary,
            "reference_series": reference_series.tolist(),
            "prediction_domain": prediction_domain,
        }
        attach_method_metadata(next_state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(next_state, save_path)

        out: Dict[str, Any] = {
            "mode": "ma_fit_only",
            "window": window,
            "train_end_index": te,
            "saved_model": save_path,
        }
        fit_predict_steps = int(getattr(args, "fit_predict_steps", 0) or 0)
        if fit_predict_steps > 0:
            preds, resolved_profile = predict(
                save_path,
                fit_predict_steps,
                state=next_state,
                allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
            )
            _, resolved_summary = profile_for_steps(
                next_state["uncertainty_profile"],
                fit_predict_steps,
                next_state["reference_series"],
                next_state["uncertainty_summary"],
            )
            out["forecast_steps"] = fit_predict_steps
            out["predictions"] = _format_predictions(preds, resolved_profile, resolved_summary)
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
        prepared = state.get("prepared_predictions")
        if (
            not bool(getattr(args, "allow_negative_predictions", False))
            and isinstance(prepared, list)
            and len(prepared) >= steps
        ):
            predictions_with_std = [dict(point) for point in prepared[:steps]]
        else:
            preds, profile = predict(
                load_path,
                steps,
                state=state,
                allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
            )
            _, resolved_uncertainty_summary = profile_for_steps(
                state.get("uncertainty_profile"),
                steps,
                state.get("reference_series") or state.get("history") or [],
                state.get("uncertainty_summary") or {},
            )
            predictions_with_std = _format_predictions(preds, profile, resolved_uncertainty_summary)
        log.info(f"Generated {len(predictions_with_std)} predictions")

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
        prediction_domain = resolve_prediction_domain(
            allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
        )

        log.info("Starting model training and evaluation...")
        metrics, std_dev_residuals, y_true, y_pred = train_and_evaluate(
            y,
            timestamps,
            range(ts, te + 1),
            range(es, ee + 1),
            window,
            prediction_domain,
        )

        full_profile, uncertainty_summary = moving_average_profile(
            np.asarray(y)[range(ts, te + 1)],
            window,
            max(12, ee - te),
        )
        forecast_offset = es - te - 1
        evaluation_profile = full_profile[forecast_offset : forecast_offset + len(y_pred)]
        predictions_with_std = _format_predictions(
            y_pred.tolist(),
            evaluation_profile,
            {**uncertainty_summary, "horizon_start": forecast_offset + 1},
        )

        out: Dict[str, Any] = {
            "mode": "fixed_window_only",
            "data_points": len(y),
            "train_range": {"start_index": ts, "end_index": te, "months": timestamps[ts : te + 1]},
            "evaluate_range": {"start_index": es, "end_index": ee, "months": timestamps[es : ee + 1]},
            "metrics": metrics,
            "eval_y_true": y_true.tolist(),
            "predictions": predictions_with_std,
            "uncertainty_summary": uncertainty_summary,
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
                "std_dev_residuals": float(full_profile[0]),
                "uncertainty_profile": full_profile[:12],
                "uncertainty_summary": uncertainty_summary,
                "reference_series": np.asarray(y)[range(ts, te + 1)].tolist(),
                "prediction_domain": prediction_domain,
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
