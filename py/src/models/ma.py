# src/models/ma.py
# -*- coding: utf-8 -*-
"""
Moving Average (MA) time series forecasting model.

Implements fixed-window moving average with support for:
- Training and evaluation with configurable window size
- Model persistence and loading
- Future forecasting with optional ground truth updates
"""
import json
import sys
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Import shared utilities
from data_utils import parse_data_from_json
from serialization import dump_model, load_model
from metrics import rmse, mae, mape, r2

MODEL_TYPE = "ma"


# ---------- core ----------
def moving_average_forecast(
    series: List[float], eval_range: range, window: int, min_index: int = 0
) -> Tuple[List[int], List[float]]:
    """
    Generate one-step-ahead forecasts using moving average.

    For each time t in eval_range, computes the mean of series[t-window:t].
    Skips time points where the window would access data before min_index.

    Args:
        series: The full time series data
        eval_range: Range of indices to generate predictions for
        window: Number of past observations to average
        min_index: Minimum index that can be accessed (default 0)

    Returns:
        Tuple of (valid_indices, predictions)
        - valid_indices: List of indices where predictions were made
        - predictions: List of corresponding prediction values

    Raises:
        ValueError: If window <= 0
    """
    if window <= 0:
        raise ValueError("[VALIDATION] Moving average window must be greater than 0.")
    idxs: List[int] = []
    preds: List[float] = []
    for t in eval_range:
        if t - window < min_index:
            continue
        preds.append(sum(series[t - window : t]) / window)
        idxs.append(t)
    return idxs, preds


def _load_state(model_path: str) -> Dict[str, Any]:
    """
    Load and validate a saved MA model state.

    Args:
        model_path: Path to the saved model file

    Returns:
        Dictionary containing model state

    Raises:
        ValueError: If the model file is invalid or not an MA model
    """
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid MA model file.")
    return state


def _forecast_future(
    history: List[float], window: int, steps: int, future_values: Optional[List[float]] = None
) -> List[float]:
    """
    Generate multi-step-ahead forecasts using moving average.

    Args:
        history: Historical values to initialize the forecast
        window: Number of values to average (0 means use all available)
        steps: Number of steps to forecast
        future_values: Optional actual future values to update the window

    Returns:
        List of forecasted values

    Raises:
        ValueError: If steps <= 0 or history is empty
    """
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    buf = history[-window:] if window > 0 else history[:]
    if not buf:
        raise ValueError("[DATA] History is empty; cannot forecast.")
    preds: List[float] = []
    future_iter = iter(future_values) if future_values else iter([])
    for _ in range(steps):
        window_values = buf[-window:] if window > 0 else buf
        if not window_values:
            raise ValueError("[VALIDATION] Insufficient history to form a moving average window.")
        avg = sum(window_values) / len(window_values)
        preds.append(avg)
        try:
            actual = next(future_iter)
            buf.append(actual)
        except StopIteration:
            buf.append(avg)
        if window > 0 and len(buf) > window:
            buf = buf[-window:]
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
    If data (JSON) is provided and contains actual future values, they will be
    consumed to update the moving window. Otherwise forecasts rely on past history.
    Returns a tuple of (predictions, std_dev_residuals).
    """
    state = state or _load_state(model_path)
    window = int(state["window"])
    history = list(state["history"])
    std_dev_residuals = float(state.get("std_dev_residuals", 0.0))
    train_end = int(state["train_end_index"])
    if window < 1:
        raise ValueError("[VALIDATION] Saved MA model window must be >= 1.")
    if not history:
        raise ValueError("[DATA] Saved MA model history is empty.")
    if len(history) < window:
        raise ValueError("[VALIDATION] Saved MA model history is shorter than window size.")

    future_values: Optional[List[float]] = None
    if data:
        try:
            temp_data = json.loads(data)
            target_key = "y" if isinstance(temp_data, list) and temp_data and "y" in temp_data[0] else "sales"

            parsed_data = parse_data_from_json(data, target_key=target_key)
            series = parsed_data.y.tolist()

            start = train_end + 1
            if start < len(series):
                future_values = series[start : start + steps]
        except (json.JSONDecodeError, ValueError, KeyError, IndexError):
            # If data parsing fails, proceed without future values
            future_values = None

    predictions = _forecast_future(history, window, steps, future_values)
    return predictions, std_dev_residuals


def train_and_evaluate(
    sales: List[float], months: List[str], train_range: range, eval_range: range, window: int
) -> Tuple[Dict[str, Any], float, List[float], List[float]]:
    """
    Evaluate the moving average model on a test set.

    Args:
        sales: Full time series data
        months: Corresponding time labels
        train_range: Range of training indices (for min_index constraint)
        eval_range: Range of evaluation indices
        window: Moving average window size

    Returns:
        Tuple of (metrics_dict, std_dev_residuals, y_true, y_pred)

    Raises:
        ValueError: If no valid evaluation points can be generated
    """
    idxs, preds = moving_average_forecast(sales, eval_range, window, min_index=train_range.start)
    if not idxs:
        raise ValueError(
            "[VALIDATION] No evaluation points: choose a smaller --window or increase eval_start to ensure at least `window` points within the training span."
        )
    y_true = [sales[i] for i in idxs]
    y_pred = preds

    residuals = np.array(y_true) - np.array(y_pred)
    std_dev_residuals = np.std(residuals)

    metrics: Dict[str, Any] = {
        "window": window,
        "evaluated_points": len(y_true),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "evaluate_indices": idxs,
        "evaluate_months": [months[i] for i in idxs],
    }
    return metrics, float(std_dev_residuals), y_true, y_pred


# ---------- public entry ----------
def run(args) -> None:
    """
    Main entry point for MA model training and prediction.

    Supports two modes:
    1. Training mode: Requires --data, train/eval ranges, and --window
    2. Prediction mode: Requires --load_model and forecast steps

    Args:
        args: Namespace containing command-line arguments

    Raises:
        ValueError: For invalid arguments or data
    """
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    # 加载模式：只需模型 + 步数信息
    if load_path:
        state = _load_state(load_path)
        steps = None
        es = getattr(args, "eval_start", None)
        ee = getattr(args, "eval_end", None)

        fs = getattr(args, "forecast_steps", None)
        if fs is not None: # 如果 forecast_steps 存在，优先使用它
            steps = int(fs)
            if steps <= 0:
                raise ValueError("[VALIDATION] --forecast_steps must be > 0.")
        elif es is not None and ee is not None: # 否则，如果 eval_start/eval_end 存在，使用它们
            try:
                es_i, ee_i = int(es), int(ee)
                if ee_i >= es_i:
                    steps = ee_i - es_i + 1
            except Exception:
                steps = None
            if steps is None:
                raise ValueError("[VALIDATION] Invalid eval_start/eval_end range for determining forecast steps.")
        else: # 如果两者都不存在
            raise ValueError("[VALIDATION] When using --load_model for MA, provide eval_start/eval_end or --forecast_steps.")

        preds, std_dev = predict(load_path, steps, getattr(args, "data", None), state=state)
        
        # Create prediction objects with std_dev
        predictions_with_std = [{"prediction": p, "std_dev": std_dev} for p in preds]

        out: Dict[str, Any] = {
            "mode": "ma_loaded_predict",
            "window": int(state["window"]),
            "train_end_index": int(state["train_end_index"]),
            "forecast_steps": steps,
            "predictions": predictions_with_std,
        }

        if getattr(args, "data", None) and es is not None and ee is not None:
            try:
                temp_data = json.loads(args.data)
                target_key = "y" if isinstance(temp_data, list) and temp_data and "y" in temp_data[0] else "sales"
                time_key = "month" if isinstance(temp_data, list) and temp_data and "month" in temp_data[0] else None

                parsed_data = parse_data_from_json(args.data, target_key=target_key, time_key=time_key)
                series = parsed_data.y.tolist()
                months = parsed_data.timestamps if parsed_data.timestamps is not None else [str(i) for i in range(parsed_data.n_obs)]

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
                    out["metrics_error"] = "Length of true values and predictions do not match for evaluation."
            except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
                out["metrics_error"] = f"Failed to calculate evaluation metrics: {e}"

        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # 解析数据
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
            raise ValueError("[DATA] Cannot determine target key. Expecting 'y' or 'sales'.")
        
        time_key = "month" if "month" in first_record else None

        parsed_data = parse_data_from_json(args.data, target_key=target_key, time_key=time_key)
        sales = parsed_data.y.tolist()
        months = parsed_data.timestamps if parsed_data.timestamps is not None else [str(i) for i in range(parsed_data.n_obs)]
        value_key = target_key

    except Exception as e:
        raise ValueError(f"[DATA] Failed to parse --data: {e}")

    # 基本参数
    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)

    # 窗口（必填）
    if not hasattr(args, "window") or args.window is None:
        raise ValueError("[VALIDATION] You must specify --window to use moving average.")
    w = int(args.window)

    # -------- 合法性检查（闭区间） --------
    n_total = len(sales)
    if not (0 <= ts <= te < n_total):
        raise ValueError(f"[VALIDATION] Invalid training range: [{ts}, {te}] within [0, {n_total-1}]")
    if not (0 <= es <= ee < n_total):
        raise ValueError(
            f"[VALIDATION] Invalid evaluation range: [{es}, {ee}] within [0, {n_total-1}]"
        )
    if te < ts:
        raise ValueError("[VALIDATION] train_end must be >= train_start")
    if ee < es:
        raise ValueError("[VALIDATION] eval_end must be >= eval_start")
    if w < 1:
        raise ValueError("[VALIDATION] --window must be >= 1")
    train_len = te - ts + 1
    if w > train_len:
        raise ValueError(
            f"[VALIDATION] --window ({w}) cannot exceed training span length ({train_len})."
        )
    # 为确保对验证区间的首个点 es 至少在某些 t 上有足够历史，这里以 te+1 作为上限（保守做法）
    if w > te + 1:
        raise ValueError(
            f"[VALIDATION] --window ({w}) cannot exceed available history (train_end_index+1={te+1})."
        )

    # -------- 训练与评估 --------
    metrics, std_dev_residuals, y_true, y_pred = train_and_evaluate(
        sales, months, range(ts, te + 1), range(es, ee + 1), w
    )

    out: Dict[str, Any] = {
        "mode": "fixed_window_only",
        "data_points": n_total,
        "train_range": {"start_index": ts, "end_index": te, "months": months[ts:te+1]},
        "evaluate_range": {"start_index": es, "end_index": ee, "months": months[es:ee+1]},
        "metrics": metrics,
        "eval_y_true": [float(v) for v in y_true],
        "eval_predictions": [float(v) for v in y_pred],
        "notes": [
            "Fixed window mode only: must provide --window.",
            "Indices use closed interval semantics: [start, end].",
            "For each evaluation t, forecast = mean(y[t-window:t]).",
            "If t < window, that point is skipped due to insufficient history.",
        ],
    }

    if save_path:
        # Save the last `window` values from training data as history
        # Ensure we don't go before train_start
        history_start = max(ts, te - w + 1)
        history = sales[history_start : te + 1]

        state = {
            "model_type": MODEL_TYPE,
            "window": w,
            "train_end_index": te,
            "history": history,
            "value_key": value_key,
            "std_dev_residuals": std_dev_residuals,
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
