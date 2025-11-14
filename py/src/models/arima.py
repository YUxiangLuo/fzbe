# src/models/arima.py
# -*- coding: utf-8 -*-
"""
ARIMA training/prediction with (p,q) grid-search and fixed d (provided by caller).

Design goals for this revision (Oct 2025):
- Keep CLI/args contract the same (d is required from front-end; p_max/q_max control search).
- Keep the top-level output structure compatible. Only *add* extra fields for diagnostics.
- Be robust to bad input data (NaN/inf/empty), and make evaluation range semantics explicit.
"""
from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Import shared utilities
from data_utils import parse_data_from_json
from serialization import dump_model, load_model
from metrics import rmse, mae, mape, r2

# Previous behavior: silence warnings (we keep this default to avoid output noise)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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
        raise ValueError(
            "[VALIDATION] eval_start must be > train_end for ARIMA to ensure genuine out-of-sample evaluation."
        )
    if (ee - es + 1) <= 0:
        raise ValueError("[VALIDATION] empty eval range")


def _determine_target_key(data_json_str: str) -> Tuple[str, str | None]:
    """
    Determine the target key and time key from JSON data string.

    Args:
        data_json_str: JSON string containing the data

    Returns:
        Tuple of (target_key, time_key) where target_key is either "y" or "sales",
        and time_key is "month" if present, else None

    Raises:
        ValueError: If data format is invalid or target key cannot be determined
    """
    try:
        temp_data = json.loads(data_json_str)
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
        return target_key, time_key

    except json.JSONDecodeError as e:
        raise ValueError(f"[DATA] Invalid JSON format: {e}")
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"[DATA] Failed to parse data structure: {e}")


def _fit_and_score(
    y_train: np.ndarray,
    y_eval_true: np.ndarray,
    d: int,
    p: int,
    q: int,
) -> Tuple[Dict[str, Any], Any]:
    """
    Fit ARIMA(p,d,q) on y_train (non-seasonal), forecast len(y_eval_true) steps,
    and compute metrics. Returns (result_dict, fitted_result_object).
    """
    model = SARIMAX(
        y_train,
        order=(p, d, q),
        seasonal_order=(0, 0, 0, 0),
        trend="c",  # keep existing default behavior
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    try:
        res = model.fit(disp=False)
        converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        # Common failures during ARIMA fitting: invalid params, singular matrix, convergence issues
        return {"p": p, "d": d, "q": q, "error": str(e)[:200], "converged": False}, None
    except Exception as e:
        # Catch any unexpected errors during grid search (should be rare)
        return {"p": p, "d": d, "q": q, "error": f"Unexpected: {str(e)[:180]}", "converged": False}, None

    steps = int(len(y_eval_true))
    fcst = res.get_forecast(steps=steps)
    y_pred = np.asarray(fcst.predicted_mean, dtype=np.float64)

    _aic = float(res.aic) if res.aic is not None and np.isfinite(res.aic) else float("inf")

    out = {
        "p": p,
        "d": d,
        "q": q,
        "rmse": float(rmse(y_eval_true.tolist(), y_pred.tolist())),
        "mae": float(mae(y_eval_true.tolist(), y_pred.tolist())),
        "mape": float(mape(y_eval_true.tolist(), y_pred.tolist())),
        "r2": float(r2(y_eval_true.tolist(), y_pred.tolist())),
        "aic": _aic,
        "converged": converged,
        "y_pred": y_pred,  # for diagnostics (not part of the original 'best' payload)
        "conf_int": np.asarray(fcst.conf_int(), dtype=np.float64),
    }
    return out, res


def _search_pq(
    y_train: np.ndarray,
    y_eval_true: np.ndarray,
    d: int,
    p_max: int,
    q_max: int,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """
    Search over p=[0..p_max], q=[0..q_max] with fixed d.
    Returns (best_result_dict, best_fitted_result, search_summary)
    """
    if len(y_train) <= d:
        raise ValueError(f"[VALIDATION] Training data length ({len(y_train)}) must be greater than d ({d}).")
    if len(y_train) < (8 + p_max + q_max):
        warnings.warn(
            f"Training sample ({len(y_train)}) may be too small for p_max={p_max}, q_max={q_max}.",
            RuntimeWarning,
        )

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    best_tuple = None
    best_item: Dict[str, Any] | None = None
    best_res_obj = None

    total = 0
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            total += 1
            item, res_obj = _fit_and_score(y_train, y_eval_true, d, p, q)
            if "error" in item: # Fit failed
                failures.append(item)
            else: # Fit succeeded
                results.append(item)
                k = (item["rmse"], item["aic"], p + q, p)  # preserve previous tie-break logic
                if best_tuple is None or k < best_tuple:
                    best_tuple = k
                    best_item = item
                    best_res_obj = res_obj

    if best_item is None:
        errors_preview = ', '.join([f"(p={f['p']},q={f['q']}): {f['error']}" for f in failures[:5]])
        raise ValueError(f"[VALIDATION] All (p,q) fits failed for d={d}. Preview: {errors_preview}")

    search_diag = {
        "candidates_tried": total,
        "succeeded": len(results),
        "failed": len(failures),
        "fit_failures_preview": failures[:10],
        "top_by_aic_preview": sorted(
            [{"p": r["p"], "q": r["q"], "aic": r["aic"]} for r in results],
            key=lambda x: x["aic"],
        )[:5],
    }
    return best_item, best_res_obj, search_diag


# ---------- predict (for ensemble) ----------
def predict(model_path: str, steps: int) -> List[Dict[str, float]]:
    """
    纯预测函数，用于 ensemble 调用。
    加载已训练的 ARIMA 模型并预测 steps 步。

    Args:
        model_path: 模型文件路径
        steps: 预测步数

    Returns:
        A list of prediction dictionaries, each containing 'prediction' and 'std_dev'.
    """
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0")
    res = load_model(model_path)
    fcst = res.get_forecast(steps=steps)

    predictions = fcst.predicted_mean.tolist()
    std_errors = fcst.se_mean.tolist()

    return [
        {"prediction": p, "std_dev": s} for p, s in zip(predictions, std_errors)
    ]


# -----------------------------
# Public entry point used by main.py
# -----------------------------
def run(args) -> None:
    """
    Two modes:
    - Train/search: when args.data is provided (and ranges are provided).
    - Predict: when args.load_model is provided (and either eval range is given or forecast_steps).
    Prints JSON to stdout (pretty if args.pretty).
    """
    pretty = bool(getattr(args, "pretty", False))
    load_path = getattr(args, "load_model", None)
    if load_path:
        _handle_prediction(args, pretty=pretty)
        return

    # Train/search mode (expects --data and ranges)
    if args.data is None:
        raise ValueError("[VALIDATION] ARIMA training requires --data.")
    req = ["train_start", "train_end", "eval_start", "eval_end", "d"]
    missing = [k for k in req if getattr(args, k, None) is None]
    if missing:
        raise ValueError(f"[VALIDATION] ARIMA training requires {', '.join('--'+k for k in missing)}")

    # Determine the target key ('y' is standard, 'sales' is for legacy)
    target_key, time_key = _determine_target_key(args.data)
    data = parse_data_from_json(args.data, target_key=target_key, time_key=time_key)
    y, months = data.y, data.timestamps
    n = data.n_obs

    ts = int(args.train_start)
    te = int(args.train_end)
    es = int(args.eval_start)
    ee = int(args.eval_end)
    _validate_ranges(n, ts, te, es, ee)

    y_train = y[ts : te + 1]
    y_eval_true = y[es : ee + 1]

    d = int(args.d)
    p_max = int(getattr(args, "p_max", 3))
    q_max = int(getattr(args, "q_max", 3))

    best, best_res, search_summary = _search_pq(y_train, y_eval_true, d, p_max, q_max)

    # Optional: save the best fitted result
    if getattr(args, "save_model", None):
        dump_model(best_res, args.save_model)

    # Assemble output (keep existing keys; only add new sections)
    best_predictions = best.get("y_pred")
    eval_y_true_list = [float(v) for v in y_eval_true.tolist()]
    eval_pred_list: List[float]
    if hasattr(best_predictions, "tolist"):
        eval_pred_list = [float(v) for v in best_predictions.tolist()]
    elif best_predictions is not None:
        eval_pred_list = [float(v) for v in best_predictions]
    else:
        eval_pred_list = []
    out: Dict[str, Any] = {
        "mode": "arima_search_pq_fixed_d",  # FIXED: Add for consistency
        "best_order": {  # FIXED: Use original "best_order" structure for compatibility
            "p": best["p"],
            "d": best["d"],
            "q": best["q"],
        },
        "metrics": {  # FIXED: Flat metrics for compatibility
            "rmse": best["rmse"],
            "mae": best["mae"],
            "mape": best["mape"],
            "r2": best["r2"],
            "aic": best["aic"],
            "converged": best.get("converged", True),
        },
        "eval_y_true": eval_y_true_list,
        "eval_predictions": eval_pred_list,
        "predictions": [
            {"prediction": float(p)} for p in eval_pred_list
        ],
        # New fields below are additive and safe for old front-ends to ignore:
        "search_summary": search_summary,
        "meta": {
            "train_range": [ts, te],
            "eval_range": [es, ee],
            "n_obs": n,
            "months_eval_slice": (months[es : ee + 1] if months else None),
        },
        "notes": [
            "Trained on training segment only; best (p,q) by RMSE; AIC tiebreaker.",
            "Use --save_model to persist best model; later use --load_model to predict.",
            "All indices use closed interval semantics: [start, end].",
        ],
    }

    # Diagnostics (optional; purely additive)
    try:
        out["diagnostics"] = {
            "y_pred_eval": best["y_pred"].tolist(),
            "conf_int": best["conf_int"].astype(float).tolist(),
        }
    except (KeyError, AttributeError, ValueError):
        # Skip diagnostics if data is not in expected format (non-critical)
        pass

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))


def _handle_prediction(args, pretty: bool) -> None:
    """
    Handle prediction mode (replaces _predict_only for +data metrics support).

    Behavior:
    - If eval_start/end are provided: we interpret them as absolute indices over the original series.
      * If both are within [0, nobs-1], we do in-sample get_prediction(start, end).
      * If both are >= nobs, we do out-of-sample get_forecast(steps = end - nobs + 1).
      * If the window crosses nobs-1, we split into in-sample [es .. nobs-1] + OOS [nobs .. ee].
    - Else we require --forecast_steps and do pure out-of-sample forecasting.

    Output:
    - Keeps original 'predictions' + 'conf_int' + 'steps' fields.
    - Adds a 'meta' section with diagnostic info (safe to ignore by old front-ends).
    """
    res = load_model(args.load_model)
    # Extract order from model for output (fallback to (0,0,0) if not available)
    try:
        order = res.model.order
    except AttributeError:
        # Model object may not have order attribute in some edge cases
        order = (0, 0, 0)
    nobs = int(getattr(res, "nobs", 0))

    # Optional: allow custom CI level if provided; default 0.95
    ci_level = float(getattr(args, "ci_level", 0.95))
    if not (0.0 < ci_level < 1.0):
        ci_level = 0.95
    alpha = 1.0 - ci_level

    have_eval_window = (
        getattr(args, "eval_start", None) is not None and
        getattr(args, "eval_end", None) is not None
    )

    predictions_list = []
    conf_list = []
    se_mean_list = []  # FIXED: Track se_mean for std_dev
    mode = None
    total_steps = 0
    eval_span = None

    if have_eval_window:
        es = int(args.eval_start)
        ee = int(args.eval_end)
        if ee < es:
            raise ValueError("[VALIDATION] eval_end must be >= eval_start")
        if es < 0:
            raise ValueError("[VALIDATION] eval_start must be >= 0")

        eval_span = [es, ee]

        # 3 cases: entirely in-sample, entirely OOS, or mixed
        if ee <= nobs - 1:
            # fully in-sample
            pred = res.get_prediction(start=es, end=ee)
            yhat = np.asarray(pred.predicted_mean, dtype=np.float64)
            ci = np.asarray(pred.conf_int(alpha=alpha), dtype=np.float64)
            se_mean = np.asarray(pred.se_mean, dtype=np.float64)  # FIXED: Get se_mean for in-sample
            predictions_list.append(yhat)
            conf_list.append(ci)
            se_mean_list.append(se_mean)
            total_steps = len(yhat)
            mode = "in_sample"

        elif es >= nobs:
            # fully out-of-sample
            steps = ee - nobs + 1
            if steps <= 0:
                raise ValueError("[VALIDATION] Computed steps <= 0; check eval range vs model nobs.")
            pred = res.get_forecast(steps=steps)
            yhat = np.asarray(pred.predicted_mean, dtype=np.float64)
            ci = np.asarray(pred.conf_int(alpha=alpha), dtype=np.float64)
            se_mean = np.asarray(pred.se_mean, dtype=np.float64)
            predictions_list.append(yhat)
            conf_list.append(ci)
            se_mean_list.append(se_mean)
            total_steps = steps
            mode = "out_of_sample"

        else:
            # mixed: [es .. nobs-1] in-sample + [nobs .. ee] OOS
            # part A: in-sample
            pred_a = res.get_prediction(start=es, end=nobs - 1)
            yhat_a = np.asarray(pred_a.predicted_mean, dtype=np.float64)
            ci_a = np.asarray(pred_a.conf_int(alpha=alpha), dtype=np.float64)
            se_mean_a = np.asarray(pred_a.se_mean, dtype=np.float64)

            # part B: out-of-sample
            steps_b = ee - nobs + 1
            pred_b = res.get_forecast(steps=steps_b)
            yhat_b = np.asarray(pred_b.predicted_mean, dtype=np.float64)
            ci_b = np.asarray(pred_b.conf_int(alpha=alpha), dtype=np.float64)
            se_mean_b = np.asarray(pred_b.se_mean, dtype=np.float64)

            predictions_list.append(yhat_a)
            predictions_list.append(yhat_b)
            conf_list.append(ci_a)
            conf_list.append(ci_b)
            se_mean_list.append(se_mean_a)
            se_mean_list.append(se_mean_b)

            total_steps = len(yhat_a) + len(yhat_b)
            mode = "mixed"

    else:
        # fallback: pure OOS forecast_steps
        if getattr(args, "forecast_steps", None) is None:
            raise ValueError("[VALIDATION] Prediction requires either eval_start/eval_end or --forecast_steps.")
        steps = int(args.forecast_steps)
        if steps <= 0:
            raise ValueError("[VALIDATION] --forecast_steps must be >= 1")

        pred = res.get_forecast(steps=steps)
        yhat = np.asarray(pred.predicted_mean, dtype=np.float64)
        ci = np.asarray(pred.conf_int(alpha=alpha), dtype=np.float64)
        se_mean = np.asarray(pred.se_mean, dtype=np.float64)
        predictions_list.append(yhat)
        conf_list.append(ci)
        se_mean_list.append(se_mean)
        total_steps = steps
        mode = "out_of_sample"

    # Concatenate parts if needed
    y_pred = np.concatenate(predictions_list, axis=0) if len(predictions_list) > 1 else predictions_list[0]
    ci = np.concatenate(conf_list, axis=0) if len(conf_list) > 1 else conf_list[0]
    se_mean_all = np.concatenate(se_mean_list, axis=0) if len(se_mean_list) > 1 else se_mean_list[0]

    # FIXED: Format predictions as original dict list with prediction + std_dev
    predictions_formatted = [
        {"prediction": float(p), "std_dev": float(s)} for p, s in zip(y_pred, se_mean_all)
    ]

    out = {
        "mode": "arima_loaded_predict",  # FIXED: Add for consistency
        "order": {"p": order[0], "d": order[1], "q": order[2]},  # FIXED: Add order extraction
        "forecast_steps": total_steps,  # FIXED: Use original key for compatibility
        "predictions": predictions_formatted,  # FIXED: Dict list with prediction + std_dev
        # FIXED: conf_int as additive (optional, for full CI)
        "conf_int": ci.astype(float).tolist(),
        # additive metadata (old front-ends can ignore safely)
        "meta": {
            "mode": mode,                # "in_sample" | "out_of_sample" | "mixed"
            "nobs": nobs,                # training sample length in the saved model
            "eval_range": eval_span,     # [eval_start, eval_end] if provided
            "ci_level": ci_level,
        },
        "notes": [
            "If eval_range is within [0, nobs-1], predictions are in-sample via get_prediction.",
            "If eval_range is beyond nobs-1, predictions are out-of-sample via get_forecast.",
            "Mixed windows are split into in-sample + out-of-sample and concatenated.",
        ],
    }

    # FIXED: Optional metrics if data + eval window
    if hasattr(args, "data") and args.data and have_eval_window:
        try:
            # Determine target key from the provided data
            target_key, _ = _determine_target_key(args.data)
            data = parse_data_from_json(args.data, target_key=target_key)
            y = data.y
            es, ee = int(args.eval_start), int(args.eval_end)
            y_true = y[es:ee+1]
            if len(y_true) == len(y_pred):
                out["metrics"] = {
                    "rmse": float(rmse(y_true.tolist(), y_pred.tolist())),
                    "mae": float(mae(y_true.tolist(), y_pred.tolist())),
                    "mape": float(mape(y_true.tolist(), y_pred.tolist())),
                    "r2": float(r2(y_true.tolist(), y_pred.tolist())),
                }
            else:
                out["metrics_error"] = f"[VALIDATION] Series length mismatch for metrics: y_true={len(y_true)}, pred={len(y_pred)}"
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            out["metrics_error"] = f"[DATA] Failed to compute metrics: {e}"

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
