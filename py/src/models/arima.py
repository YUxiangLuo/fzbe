# src/models/arima.py
# -*- coding: utf-8 -*-
import json
import math
from typing import List, Dict, Any, Tuple

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib_compat import load_joblib, dump_joblib


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


# ---------- helpers ----------
def _fit_forecast_arima(y_train: np.ndarray, p: int, d: int, q: int, steps: int):
    """
    拟合 ARIMA(p,d,q) 并前瞻 steps 步；返回 (result, y_pred, aic)。
    """
    model = SARIMAX(
        y_train,
        order=(p, d, q),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    res = model.fit(disp=False)
    fcst = res.get_forecast(steps=steps)
    y_pred = fcst.predicted_mean.tolist()
    aic = float(res.aic) if res.aic is not None else float("inf")
    return res, y_pred, aic


def _search_pq(
    y_train: np.ndarray, y_true: List[float], d: int, p_max: int, q_max: int
):
    results: List[Dict[str, Any]] = []
    best_tuple = None  # (rmse, aic, p, q, res, y_pred)

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            try:
                res, y_pred, aic = _fit_forecast_arima(
                    y_train, p, d, q, steps=len(y_true)
                )
                cur_rmse = rmse(y_true, y_pred)
                item = {
                    "p": p,
                    "d": d,
                    "q": q,
                    "rmse": cur_rmse,
                    "mae": mae(y_true, y_pred),
                    "mape": mape(y_true, y_pred),
                    "r2": r2_score(y_true, y_pred),
                    "aic": aic,
                }
                results.append(item)
                if not math.isnan(cur_rmse):
                    key = (cur_rmse, aic, p + q, p)
                    if (best_tuple is None) or (key < best_tuple[0]):
                        best_tuple = (key, p, q, res, item)
            except Exception:
                # 忽略失败组合
                pass

    if best_tuple is None:
        raise ValueError(
            "All (p,q) candidates failed to fit; consider reducing d or ranges."
        )

    _, p_best, q_best, res_best, item_best = best_tuple
    return (p_best, q_best, res_best, item_best)


# ---------- predict (for ensemble) ----------
def predict(model_path: str, steps: int) -> List[float]:
    """
    纯预测函数，用于 ensemble 调用。
    加载已训练的 ARIMA 模型并预测 steps 步。
    
    Args:
        model_path: 模型文件路径
        steps: 预测步数
    
    Returns:
        预测值列表
    """
    res = load_joblib(model_path)
    fcst = res.get_forecast(steps=steps)
    return fcst.predicted_mean.tolist()


# ---------- run ----------
def run(args) -> None:
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    # --------- 模式2：加载模型并预测 ---------
    if load_path:
        res = load_joblib(load_path)

        # 尝试读取 (p,d,q)
        try:
            p_d_q = getattr(res.model, "order", None)
            if p_d_q is None:
                p_d_q = (None, None, None)
        except Exception:
            p_d_q = (None, None, None)

        # 预测步数（闭区间）：
        # 1) 若提供 eval 范围，steps = eval_end - eval_start + 1
        # 2) 否则用 --forecast_steps
        steps = None
        if hasattr(args, "eval_start") and hasattr(args, "eval_end"):
            try:
                es, ee = int(args.eval_start), int(args.eval_end)
                if ee >= es:
                    steps = ee - es + 1
            except Exception:
                pass
        if steps is None:
            fs = getattr(args, "forecast_steps", None)
            if fs is None:
                raise ValueError(
                    "When using --load_model, provide eval_start/eval_end or --forecast_steps."
                )
            steps = int(fs)
            if steps <= 0:
                raise ValueError("--forecast_steps must be > 0.")

        # 预测
        fcst = res.get_forecast(steps=steps)
        y_pred = fcst.predicted_mean.tolist()

        out: Dict[str, Any] = {
            "mode": "arima_loaded_predict",
            "order": {"p": p_d_q[0], "d": p_d_q[1], "q": p_d_q[2]},
            "forecast_steps": steps,
            "predictions": y_pred,
        }

        # 若用户给了 data + eval，可以顺带算指标（有真值才算）
        if (
            hasattr(args, "data")
            and hasattr(args, "eval_start")
            and hasattr(args, "eval_end")
        ):
            try:
                arr = json.loads(args.data)
                sales = [float(it["sales"]) for it in arr]
                y_true = sales[es : ee + 1]  # 闭区间切片
                if len(y_true) == len(y_pred) and len(y_true) > 0:
                    out["metrics"] = {
                        "rmse": rmse(y_true, y_pred),
                        "mae": mae(y_true, y_pred),
                        "mape": mape(y_true, y_pred),
                        "r2": r2_score(y_true, y_pred),
                    }
            except Exception:
                pass

        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # --------- 模式1：训练 +（可选保存）---------
    # 解析数据
    try:
        arr = json.loads(args.data)
        if not isinstance(arr, list):
            raise ValueError("`--data` must be a JSON array.")
        months: List[str] = [str(it["month"]) for it in arr]
        sales: List[float] = [float(it["sales"]) for it in arr]
    except Exception as e:
        raise ValueError(f"Failed to parse --data: {e}")

    n = len(sales)
    ts, te = int(args.train_start), int(args.train_end)  # [ts, te] 闭区间
    es, ee = int(args.eval_start), int(args.eval_end)  # [es, ee] 闭区间

    if not hasattr(args, "d") or args.d is None:
        raise ValueError("ARIMA requires --d (diff order).")
    d = int(args.d)
    if d < 0:
        raise ValueError("--d must be >= 0")

    p_max = int(getattr(args, "p_max", 3))
    q_max = int(getattr(args, "q_max", 3))
    if p_max < 0 or q_max < 0:
        raise ValueError("--p_max and --q_max must be >= 0")

    # 范围校验（闭区间）
    if not (0 <= ts <= te < n):
        raise ValueError(f"Invalid train range: [{ts},{te}] within [0,{n-1}]")
    if not (0 <= es <= ee < n):
        raise ValueError(f"Invalid eval range: [{es},{ee}] within [0,{n-1}]")
    if te < ts:
        raise ValueError("train_end must be >= train_start")
    if ee < es:
        raise ValueError("eval_end must be >= eval_start")

    y_train = np.asarray(sales[ts : te + 1], dtype=float)  # 闭区间切片
    y_true = sales[es : ee + 1]  # 闭区间切片
    h = len(y_true)
    if h <= 0:
        raise ValueError("evaluation length must be >= 1")
    if len(y_train) <= d:
        raise ValueError(
            "training length must be > d (after differencing still needs samples)"
        )

    # 搜索最优 (p,q)
    p_best, q_best, res_best, item_best = _search_pq(y_train, y_true, d, p_max, q_max)

    # （可选）保存模型
    if save_path:
        dump_joblib(res_best, save_path)

    out: Dict[str, Any] = {
        "mode": "arima_search_pq_fixed_d",
        "best_order": {"p": int(p_best), "d": int(d), "q": int(q_best)},
        "metrics": {
            "rmse": item_best["rmse"],
            "mae": item_best["mae"],
            "mape": item_best["mape"],
            "r2": item_best["r2"],
            "aic": item_best["aic"],
        },
        "notes": [
            "Trained on training segment only; best (p,q) by RMSE; AIC tiebreaker.",
            "Use --save_model to persist best model; later use --load_model to predict.",
            "All indices use closed interval semantics: [start, end].",
        ],
    }

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
