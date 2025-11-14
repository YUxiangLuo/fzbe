# py/src/models/es.py
# -*- coding: utf-8 -*-
"""
Exponential Smoothing (Simple Exponential Smoothing / SES) model.

Implements level-based forecasting with:
- Training on specified data range
- Sequential evaluation with fixed alpha parameter
- Model persistence for future predictions
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import shared utilities
from data_utils import parse_data_from_json
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model

MODEL_TYPE = "es"


# ---------- utilities ----------


def _load_state(model_path: str) -> Dict[str, Any]:
    """
    Load and validate a saved ES model state.

    Args:
        model_path: Path to the saved model file

    Returns:
        Dictionary containing model state

    Raises:
        ValueError: If the model file is invalid or not an ES model
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
    Forecast using a saved SES model. If actual future data is provided, the
    level will be updated sequentially to mimic online smoothing.
    Returns a tuple of (predictions, std_dev_residuals).
    """
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")
    state = state or _load_state(model_path)
    alpha = float(state["alpha"])
    level = float(state["level"])
    std_dev_residuals = float(state.get("std_dev_residuals", 0.0))
    train_end = int(state["train_end_index"])

    future_values: Optional[List[float]] = None
    if data:
        try:
            # Determine target key from the provided data
            temp_data = json.loads(data)
            target_key = (
                "y"
                if isinstance(temp_data, list) and temp_data and "y" in temp_data[0]
                else "sales"
            )

            parsed_data = parse_data_from_json(data, target_key=target_key)
            future_values = parsed_data.y.tolist()
        except (json.JSONDecodeError, ValueError, KeyError, IndexError):
            # If data parsing fails, proceed without future values
            future_values = None

    preds: List[float] = []
    # Vectorized prediction loop with optional update
    for idx in range(steps):
        preds.append(level)
        if future_values and idx < len(future_values):
            actual = future_values[idx]
            level = alpha * actual + (1.0 - alpha) * level
    return preds, std_dev_residuals


# ---------- train and evaluate ----------


def _fit_level(series: np.ndarray, train_range: range, alpha: float) -> float:
    """
    Fit the level using Simple Exponential Smoothing on the training range.
    Optimized with NumPy for efficiency.
    """
    if len(train_range) < 1:
        raise ValueError("[VALIDATION] Training range must have at least one point.")
    level = series[train_range.start]
    for t in range(train_range.start + 1, train_range.stop):
        level = alpha * series[t] + (1.0 - alpha) * level
    return float(level)


def train_and_evaluate(
    sales: List[float],
    months: List[str],
    train_range: range,
    eval_range: range,
    alpha: float,
) -> Tuple[Dict[str, Any], float, float, List[float], List[float]]:
    """
    Trains the SES model strictly on the declared training segment, then
    evaluates sequentially over eval_range. Returns metrics, residual std dev,
    and the smoothed level at train_end (before consuming evaluation targets).
    """
    series = np.array(sales, dtype=np.float64)
    ts, te = train_range.start, train_range.stop - 1

    # Fit level on training slice
    train_level = _fit_level(series, train_range, alpha)

    # Sequential evaluation
    idxs = []
    preds = []
    for t in eval_range:
        if t <= te:
            continue
        preds.append(train_level)
        idxs.append(t)

    if not idxs:
        raise ValueError(
            "[VALIDATION] No evaluation points: set eval_start >= train_start + 1 to ensure at least one forecast."
        )

    y_true = series[idxs]
    y_pred = np.array(preds, dtype=np.float64)

    residuals = y_true - y_pred
    std_dev_residuals = np.std(residuals, ddof=0)

    metrics = {
        "rmse": rmse(y_true.tolist(), y_pred.tolist()),
        "mae": mae(y_true.tolist(), y_pred.tolist()),
        "mape": mape(y_true.tolist(), y_pred.tolist()),
        "r2": r2(y_true.tolist(), y_pred.tolist()),
    }

    return (
        metrics,
        float(std_dev_residuals),
        train_level,
        y_true.tolist(),
        y_pred.tolist(),
    )


# ---------- public entry ----------


def run(args) -> None:
    """
    Main entry point for ES model training and prediction.

    Supports two modes:
    1. Training mode: Requires --data, train/eval ranges, and --alpha
    2. Prediction mode: Requires --load_model and forecast steps

    Args:
        args: Namespace containing command-line arguments

    Raises:
        ValueError: For invalid arguments or data
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
        if fs is not None:  # 如果 forecast_steps 存在，优先使用它
            steps = int(fs)
            if steps <= 0:
                raise ValueError("[VALIDATION] --forecast_steps must be > 0.")
        elif (
            es is not None and ee is not None
        ):  # 否则，如果 eval_start/eval_end 存在，使用它们
            try:
                es_i, ee_i = int(es), int(ee)
                if ee_i >= es_i:
                    steps = ee_i - es_i + 1
            except (ValueError, TypeError):
                steps = None
            if steps is None:
                raise ValueError(
                    "[VALIDATION] Invalid eval_start/eval_end range for determining forecast steps."
                )
        else:  # 如果两者都不存在
            raise ValueError(
                "[VALIDATION] When using --load_model for ES, provide eval_start/eval_end or --forecast_steps."
            )

        preds, std_dev = predict(
            load_path, steps, getattr(args, "data", None), state=state
        )

        predictions_with_std = [{"prediction": p, "std_dev": std_dev} for p in preds]

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

    # Parse data
    if not hasattr(args, "data") or not args.data:
        raise ValueError("[VALIDATION] --data is required for training.")

    try:
        temp_data = json.loads(args.data)
        if not isinstance(temp_data, list) or not temp_data:
            raise ValueError("[DATA] --data must be a non-empty JSON array.")

        first_record = temp_data[0]
        if "y" in first_record:
            target_key = "y"
        elif "sales" in first_record:
            target_key = "sales"
        else:
            raise ValueError(
                "[DATA] Cannot determine target key. Expecting 'y' or 'sales'."
            )

        time_key = "month" if "month" in first_record else None

        parsed_data = parse_data_from_json(
            args.data, target_key=target_key, time_key=time_key
        )
        sales = parsed_data.y.tolist()
        months = (
            parsed_data.timestamps
            if parsed_data.timestamps is not None
            else [str(i) for i in range(parsed_data.n_obs)]
        )
        value_key = target_key

    except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
        raise ValueError(f"[DATA] Failed to parse --data: {e}")

    # Basic parameters
    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)

    # Alpha validation
    if not hasattr(args, "alpha") or args.alpha is None:
        raise ValueError("[VALIDATION] You must specify --alpha for SES.")
    try:
        alpha = float(args.alpha)
    except (ValueError, TypeError):
        raise ValueError("[VALIDATION] --alpha must be a float in (0,1].")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("[VALIDATION] --alpha must be in (0,1].")

    # Validation
    _validate_training_args(sales, ts, te, es, ee)

    metrics, std_dev_residuals, final_level, y_true, y_pred = train_and_evaluate(
        sales, months, range(ts, te + 1), range(es, ee + 1), alpha
    )

    out: Dict[str, Any] = {
        "mode": "ses_fixed_alpha",
        "data_points": len(sales),
        "alpha": alpha,
        "train_range": {
            "start_index": ts,
            "end_index": te,
            "months": months[ts : te + 1],
        },
        "evaluate_range": {
            "start_index": es,
            "end_index": ee,
            "months": months[es : ee + 1],
        },
        "metrics": metrics,
        "eval_y_true": [float(v) for v in y_true],
        "eval_predictions": [float(v) for v in y_pred],
        "notes": [
            "SES: forecast at t uses s_{t-1}.",
            "s_t = alpha*y_t + (1-alpha)*s_{t-1}; s_{train_start} = y_{train_start}.",
            "All indices use closed interval semantics: [start, end].",
        ],
    }

    if save_path:
        state = {
            "model_type": MODEL_TYPE,
            "alpha": alpha,
            "train_end_index": te,
            "level": final_level,
            "value_key": value_key,
            "std_dev_residuals": std_dev_residuals,
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))


def _validate_training_args(sales, ts, te, es, ee):
    """
    Validate arguments for training mode.
    """
    n = len(sales)
    if not (0 <= ts <= te < n):
        raise ValueError(
            f"[VALIDATION] Invalid train range: [{ts},{te}] within [0,{n - 1}]"
        )
    if not (0 <= es <= ee < n):
        raise ValueError(
            f"[VALIDATION] Invalid eval range: [{es},{ee}] within [0,{n - 1}]"
        )
    if te < ts:
        raise ValueError("[VALIDATION] train_end must be >= train_start")
    if ee < es:
        raise ValueError("[VALIDATION] eval_end must be >= eval_start")
    if es < ts + 1:
        raise ValueError(
            "[VALIDATION] No evaluation points: set eval_start >= train_start + 1"
        )


def _add_metrics_to_output(out, data, es, ee, preds):
    """
    Add evaluation metrics to output dictionary if possible.
    """
    try:
        temp_data = json.loads(data)
        target_key = (
            "y"
            if isinstance(temp_data, list) and temp_data and "y" in temp_data[0]
            else "sales"
        )
        time_key = (
            "month"
            if isinstance(temp_data, list) and temp_data and "month" in temp_data[0]
            else None
        )

        parsed_data = parse_data_from_json(
            data, target_key=target_key, time_key=time_key
        )
        series = parsed_data.y.tolist()
        months = (
            parsed_data.timestamps
            if parsed_data.timestamps is not None
            else [str(i) for i in range(parsed_data.n_obs)]
        )

        es_i, ee_i = int(es), int(ee)
        y_true = series[es_i : ee_i + 1]
        if len(y_true) == len(preds):
            out["metrics"] = {
                "rmse": rmse(y_true, preds),
                "mae": mae(y_true, preds),
                "mape": mape(y_true, preds),
                "r2": r2(y_true, preds),
            }
            out["evaluate_months"] = months[es_i : ee_i + 1]
        else:
            out["metrics_error"] = (
                "Length of true values and predictions do not match for evaluation."
            )
    except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
        out["metrics_error"] = f"Failed to calculate evaluation metrics: {e}"
