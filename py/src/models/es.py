# src/models/es.py
# -*- coding: utf-8 -*-
import json
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from joblib_compat import load_joblib, dump_joblib

MODEL_TYPE = "es"


# ----- metrics -----
def rmse(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def mae(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def mape(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    eps = 1e-12
    return (
        100.0
        * sum(
            abs((a - b) / (a if abs(a) > eps else eps)) for a, b in zip(y_true, y_pred)
        )
        / len(y_true)
    )


def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean_y) ** 2 for a in y_true)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


def _extract_series(arr: List[Dict[str, Any]]) -> Tuple[List[str], List[float], str]:
    if not arr:
        raise ValueError("`--data` must not be empty.")
    first = arr[0]
    value_key = None
    label_key = None
    if "sales" in first:
        value_key = "sales"
        label_key = "month" if "month" in first else None
    elif "y" in first:
        value_key = "y"
        label_key = "ts" if "ts" in first else None
    else:
        raise ValueError("Data items must contain 'sales' or 'y' field.")
    months = [str(item[label_key]) if label_key and label_key in item else str(i) for i, item in enumerate(arr)]
    sales = [float(item[value_key]) for item in arr]
    return months, sales, value_key


def _load_state(model_path: str) -> Dict[str, Any]:
    state = load_joblib(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("Invalid ES model file.")
    return state


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
        raise ValueError("steps must be > 0 for forecasting.")
    state = state or _load_state(model_path)
    alpha = float(state["alpha"])
    level = float(state["level"])
    std_dev_residuals = float(state.get("std_dev_residuals", 0.0))
    train_end = int(state["train_end_index"])

    future_values: Optional[List[float]] = None
    if data:
        try:
            arr = json.loads(data)
            _, series, _ = _extract_series(arr)
            start = train_end + 1
            if start < len(series):
                future_values = series[start : start + steps]
        except Exception:
            future_values = None

    preds: List[float] = []
    for idx in range(steps):
        preds.append(level)
        if future_values and idx < len(future_values):
            actual = future_values[idx]
            level = alpha * actual + (1.0 - alpha) * level
    return preds, std_dev_residuals


def train_and_evaluate(
    sales: List[float], months: List[str], train_range: range, eval_range: range, alpha: float
) -> Tuple[Dict[str, Any], float, float]:
    """
    Trains the SES model strictly on the declared training segment, then
    evaluates sequentially over eval_range. Returns metrics, residual std dev,
    and the smoothed level at train_end (before consuming evaluation targets).
    """
    ts = train_range.start
    te = train_range.stop - 1  # inclusive

    # --- Phase 1: fit level on training slice only ---
    level = float(sales[ts])
    for t in range(ts + 1, te + 1):
        level = alpha * float(sales[t]) + (1.0 - alpha) * level

    train_level = float(level)  # snapshot before evaluation

    # --- Phase 2: sequential evaluation (1-step ahead) ---
    idxs: List[int] = []
    preds: List[float] = []
    current_level = float(level)
    for t in eval_range:
        if t <= ts:
            # Defensive: evaluation must have at least one prior observation
            continue
        preds.append(current_level)
        idxs.append(t)
        current_level = alpha * float(sales[t]) + (1.0 - alpha) * current_level

    if not idxs:
        raise ValueError(
            "no evaluation points: set eval_start >= train_start + 1 to ensure at least one forecast."
        )

    y_true = [float(sales[i]) for i in idxs]
    y_pred = preds

    residuals = np.array(y_true) - np.array(y_pred)
    std_dev_residuals = np.std(residuals)

    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    return metrics, float(std_dev_residuals), train_level


# ----- core -----
def run(args) -> None:
    """
    入口参数（必须）：
      --data, --train_start, --train_end, --eval_start, --eval_end, --alpha
    可选：
      --pretty
    """
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if load_path:
        state = _load_state(load_path)
        steps = None
        es = getattr(args, "eval_start", None)
        ee = getattr(args, "eval_end", None)
        if es is not None and ee is not None:
            try:
                es_i, ee_i = int(es), int(ee)
                if ee_i >= es_i:
                    steps = ee_i - es_i + 1
            except Exception:
                steps = None
        if steps is None:
            fs = getattr(args, "forecast_steps", None)
            if fs is None:
                raise ValueError(
                    "When using --load_model for ES, provide eval_start/eval_end or --forecast_steps."
                )
            steps = int(fs)
            if steps <= 0:
                raise ValueError("--forecast_steps must be > 0.")

        preds, std_dev = predict(load_path, steps, getattr(args, "data", None), state=state)
        
        predictions_with_std = [{"prediction": p, "std_dev": std_dev} for p in preds]
        
        out: Dict[str, Any] = {
            "mode": "es_loaded_predict",
            "alpha": float(state["alpha"]),
            "train_end_index": int(state["train_end_index"]),
            "forecast_steps": steps,
            "predictions": predictions_with_std,
        }
        if getattr(args, "data", None) and es is not None and ee is not None:
            try:
                arr = json.loads(args.data)
                months, series, _ = _extract_series(arr)
                es_i, ee_i = int(es), int(ee)
                y_true = series[es_i : ee_i + 1]
                if len(y_true) == len(preds):
                    out["metrics"] = {
                        "rmse": rmse(y_true, preds),
                        "mae": mae(y_true, preds),
                        "mape": mape(y_true, preds),
                        "r2": r2_score(y_true, preds),
                    }
                    out["evaluate_months"] = months[es_i : ee_i + 1]
            except Exception:
                pass

        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # 解析数据
    try:
        arr = json.loads(args.data)
        if not isinstance(arr, list):
            raise ValueError("`--data` must be a JSON array.")
        months, sales, value_key = _extract_series(arr)
    except Exception as e:
        raise ValueError(f"Failed to parse --data: {e}")

    n = len(sales)
    ts, te = int(args.train_start), int(args.train_end)  # [ts, te] 闭区间
    es, ee = int(args.eval_start), int(args.eval_end)  # [es, ee] 闭区间

    # alpha 校验
    if not hasattr(args, "alpha") or args.alpha is None:
        raise ValueError("You must specify --alpha for SES.")
    try:
        alpha = float(args.alpha)
    except Exception:
        raise ValueError("--alpha must be a float in (0,1].")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("--alpha must be in (0,1].")

    # 范围校验（闭区间）
    if not (0 <= ts <= te < n):
        raise ValueError(f"Invalid train range: [{ts},{te}] within [0,{n-1}]")
    if not (0 <= es <= ee < n):
        raise ValueError(f"Invalid eval range: [{es},{ee}] within [0,{n-1}]")
    if te < ts:
        raise ValueError("train_end must be >= train_start")
    if ee < es:
        raise ValueError("eval_end must be >= eval_start")

    # 需要至少有一个"历史点"来预测 eval 段的首个点：
    # 对 t=es 预测用的是 s_{es-1}，因此要求 es >= ts+1
    if es < ts + 1:
        raise ValueError("no evaluation points: set eval_start >= train_start + 1")

    metrics, std_dev_residuals, final_level = train_and_evaluate(
        sales, months, range(ts, te + 1), range(es, ee + 1), alpha
    )

    out: Dict[str, Any] = {
        "mode": "ses_fixed_alpha",
        "data_points": n,
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
        dump_joblib(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
