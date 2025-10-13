# src/models/ma.py
# -*- coding: utf-8 -*-
import json
import math
from typing import List, Dict, Any, Tuple, Optional

from joblib_compat import load_joblib, dump_joblib

MODEL_TYPE = "ma"


# ---------- metrics ----------
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


# ---------- core ----------
def moving_average_forecast(
    series: List[float], eval_range: range, window: int
) -> Tuple[List[int], List[float]]:
    """
    对 eval_range 中的每个 t，使用 series[t-window:t] 的真实值做一步预测。
    t < window 的点跳过。
    返回：有效的索引列表、对应的预测值列表
    """
    idxs: List[int] = []
    preds: List[float] = []
    for t in eval_range:
        if t - window < 0:
            continue
        preds.append(sum(series[t - window : t]) / window)
        idxs.append(t)
    return idxs, preds


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
        raise ValueError("Each item must contain 'sales' or 'y' field.")

    months = [str(item[label_key]) if label_key and label_key in item else str(i) for i, item in enumerate(arr)]
    sales = [float(item[value_key]) for item in arr]
    return months, sales, value_key


def _load_state(model_path: str) -> Dict[str, Any]:
    state = load_joblib(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("Invalid MA model file.")
    return state


def _forecast_future(
    history: List[float], window: int, steps: int, future_values: Optional[List[float]] = None
) -> List[float]:
    if steps <= 0:
        raise ValueError("steps must be > 0 for forecasting.")

    buf = history[-window:] if window > 0 else history[:]
    if not buf:
        raise ValueError("History is empty; cannot forecast.")
    preds: List[float] = []
    future_iter = iter(future_values) if future_values else iter([])
    for _ in range(steps):
        window_values = buf[-window:] if window > 0 else buf
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
) -> List[float]:
    """
    Forecast future values using a saved MA model.
    If data (JSON) is provided and contains actual future values, they will be
    consumed to update the moving window. Otherwise forecasts rely on past history.
    """
    state = state or _load_state(model_path)
    window = int(state["window"])
    history = list(state["history"])
    train_end = int(state["train_end_index"])
    if window < 1:
        raise ValueError("Saved MA model window must be >= 1.")
    if not history:
        raise ValueError("Saved MA model history is empty.")
    if len(history) < window:
        raise ValueError("Saved MA model history is shorter than window size.")

    future_values: Optional[List[float]] = None
    if data:
        try:
            arr = json.loads(data)
            _, series, value_key = _extract_series(arr)
            start = train_end + 1
            if start < len(series):
                future_values = series[start : start + steps]
        except Exception:
            future_values = None

    return _forecast_future(history, window, steps, future_values)


# ---------- public entry ----------
def run(args) -> None:
    """
    固定窗口移动平均法。
    必须传入 --window；仅做合法性检查 + 单窗口评估。
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
                    "When using --load_model for MA, provide eval_start/eval_end or --forecast_steps."
                )
            steps = int(fs)
            if steps <= 0:
                raise ValueError("--forecast_steps must be > 0.")

        preds = predict(load_path, steps, getattr(args, "data", None), state=state)
        out: Dict[str, Any] = {
            "mode": "ma_loaded_predict",
            "window": int(state["window"]),
            "train_end_index": int(state["train_end_index"]),
            "forecast_steps": steps,
            "predictions": preds,
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
    except Exception as e:
        raise ValueError(f"Failed to parse --data: {e}")

    try:
        months, sales, value_key = _extract_series(arr)
    except Exception as e:
        raise ValueError(str(e))

    # 基本参数
    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)

    # 窗口（必填）
    if not hasattr(args, "window") or args.window is None:
        raise ValueError("You must specify --window to use moving average.")
    w = int(args.window)

    # -------- 合法性检查（闭区间） --------
    n_total = len(sales)
    if not (0 <= ts <= te < n_total):
        raise ValueError(f"Invalid training range: [{ts}, {te}] within [0, {n_total-1}]")
    if not (0 <= es <= ee < n_total):
        raise ValueError(
            f"Invalid evaluation range: [{es}, {ee}] within [0, {n_total-1}]"
        )
    if te < ts:
        raise ValueError("train_end must be >= train_start")
    if ee < es:
        raise ValueError("eval_end must be >= eval_start")
    if w < 1:
        raise ValueError("--window must be >= 1")
    # 为确保对验证区间的首个点 es 至少在某些 t 上有足够历史，这里以 te+1 作为上限（保守做法）
    if w > te + 1:
        raise ValueError(
            f"--window ({w}) cannot exceed available history (train_end_index+1={te+1})."
        )

    # -------- 预测（闭区间：range(es, ee+1)） --------
    eval_range = range(es, ee + 1)
    idxs, preds = moving_average_forecast(sales, eval_range, w)
    if not idxs:
        raise ValueError(
            "no evaluation points: choose a smaller --window or move eval_start to >= window"
        )
    y_true = [sales[i] for i in idxs]
    y_pred = preds

    metrics: Dict[str, Any] = {
        "window": w,
        "evaluated_points": len(y_true),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "evaluate_indices": idxs,
        "evaluate_months": [months[i] for i in idxs],
    }

    out: Dict[str, Any] = {
        "mode": "fixed_window_only",
        "data_points": n_total,
        "train_range": {"start_index": ts, "end_index": te, "months": months[ts:te+1]},
        "evaluate_range": {"start_index": es, "end_index": ee, "months": months[es:ee+1]},
        "metrics": metrics,
        "notes": [
            "Fixed window mode only: must provide --window.",
            "Indices use closed interval semantics: [start, end].",
            "For each evaluation t, forecast = mean(y[t-window:t]).",
            "If t < window, that point is skipped due to insufficient history.",
        ],
    }

    if save_path:
        history = sales[max(0, te - w + 1) : te + 1]
        state = {
            "model_type": MODEL_TYPE,
            "window": w,
            "train_end_index": te,
            "history": history,
            "value_key": value_key,
        }
        dump_joblib(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
