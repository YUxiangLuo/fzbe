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

# Import shared utilities
from data_utils import parse_data_from_json
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

MODEL_TYPE = "arima"


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


def _fit_arima(
    y_train: np.ndarray, d: int, p: int, q: int, y_eval_true: np.ndarray
) -> Tuple[Dict[str, Any], SARIMAXResults | None]:
    """
    Fit ARIMA model with NumPy efficiency and return result dict and fitted object.

    Returns:
        Tuple of (result_dict, fitted_model) where fitted_model is None if fitting failed.
    """
    model = SARIMAX(
        y_train,
        order=(p, d, q),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    try:
        # Suppress convergence and user warnings during fitting to reduce noise
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            res = model.fit(disp=False)
        converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        # Common failures: invalid parameters, singular matrix, numerical issues
        return {"p": p, "d": d, "q": q, "error": str(e)[:200], "converged": False}, None
    except (AttributeError, TypeError, KeyError) as e:
        # Data structure issues or model specification problems
        return {
            "p": p,
            "d": d,
            "q": q,
            "error": f"Data/spec error: {str(e)[:180]}",
            "converged": False,
        }, None
    except Exception as e:
        # Fallback for other statsmodels exceptions (e.g., optimization failures)
        # We catch broadly here because statsmodels can raise various internal exceptions
        return {
            "p": p,
            "d": d,
            "q": q,
            "error": f"Unexpected: {type(e).__name__}: {str(e)[:150]}",
            "converged": False,
        }, None

    steps = len(y_eval_true)
    fcst = res.get_forecast(steps=steps)
    y_pred = fcst.predicted_mean.astype(np.float64)

    aic = (
        float(res.aic) if res.aic is not None and np.isfinite(res.aic) else float("inf")
    )

    out = {
        "p": p,
        "d": d,
        "q": q,
        "rmse": rmse(y_eval_true, y_pred),
        "mae": mae(y_eval_true, y_pred),
        "mape": mape(y_eval_true, y_pred),
        "r2": r2(y_eval_true, y_pred),
        "aic": aic,
        "converged": converged,
        "y_pred": y_pred,
        "conf_int": fcst.conf_int().astype(np.float64),
    }
    return out, res


def _determine_target_key(data_json_str: str) -> Tuple[str, str | None, list]:
    """
    Determine the target key and time key from JSON data string.
    Returns parsed records to avoid duplicate JSON parsing.

    Args:
        data_json_str: JSON string containing the data

    Returns:
        Tuple of (target_key, time_key, records) where:
        - target_key is either "y" or "sales"
        - time_key is "month" if present, else None
        - records is the parsed JSON list

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
            raise ValueError(
                "[DATA] Cannot determine target key. Expecting 'y' or 'sales'."
            )

        time_key = "month" if "month" in first_record else None
        return target_key, time_key, temp_data

    except json.JSONDecodeError as e:
        raise ValueError(f"[DATA] Invalid JSON format: {e}")
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"[DATA] Failed to parse data structure: {e}")


# _fit_arima is defined above, replacing _fit_and_score


def _search_pq(
    y_train: np.ndarray,
    y_eval_true: np.ndarray,
    d: int,
    p_max: int,
    q_max: int,
) -> Tuple[Dict[str, Any], SARIMAXResults, Dict[str, Any]]:
    """
    Search over p=[0..p_max], q=[0..q_max] with fixed d.

    Returns:
        Tuple of (best_result_dict, best_fitted_model, search_summary)
    """
    if len(y_train) <= d:
        raise ValueError(
            f"[VALIDATION] Training data length ({len(y_train)}) must be greater than d ({d})."
        )
    # Warning about potentially small sample size (8 is a heuristic minimum for stable estimation)
    if len(y_train) < (8 + p_max + q_max):
        warnings.warn(
            f"Training sample ({len(y_train)}) may be too small for p_max={p_max}, q_max={q_max}.",
            RuntimeWarning,
        )

    results = []
    failures = []
    best_tuple = None
    best_item = None
    best_res_obj = None

    total = 0
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            total += 1
            item, res_obj = _fit_arima(y_train, d, p, q, y_eval_true)
            if "error" in item:
                failures.append(item)
            else:
                results.append(item)
                k = (item["rmse"], item["aic"], p + q, p)
                if best_tuple is None or k < best_tuple:
                    best_tuple = k
                    best_item = item
                    best_res_obj = res_obj

    if best_item is None:
        errors_preview = ", ".join(
            [f"(p={f['p']},q={f['q']}): {f['error']}" for f in failures[:5]]
        )
        raise ValueError(
            f"[VALIDATION] All (p,q) fits failed for d={d}. Preview: {errors_preview}"
        )

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
def predict(model_path: str, steps: int, data: str = None) -> List[Dict[str, float]]:
    """
    Load and predict with ARIMA model. If data is provided, update predictions.

    Args:
        model_path: Model file path
        steps: Forecast steps
        data: Optional JSON data for update

    Returns:
        List of prediction dicts with 'prediction' and 'std_dev'.
    """
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0")
    res = load_model(model_path)
    fcst = res.get_forecast(steps=steps)

    predictions = fcst.predicted_mean.tolist()
    std_errors = fcst.se_mean.tolist()

    return [{"prediction": p, "std_dev": s} for p, s in zip(predictions, std_errors)]


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

    # Train mode
    _handle_training(args, pretty=pretty)


def _handle_training(args, pretty: bool) -> None:
    """
    Handle training and grid search for ARIMA.
    """
    if args.data is None:
        raise ValueError("[VALIDATION] ARIMA training requires --data.")
    req = ["train_start", "train_end", "eval_start", "eval_end", "d"]
    missing = [k for k in req if getattr(args, k, None) is None]
    if missing:
        raise ValueError(
            f"[VALIDATION] ARIMA training requires {', '.join('--' + k for k in missing)}"
        )

    target_key, time_key, records = _determine_target_key(args.data)
    data = parse_data_from_json(records, target_key=target_key, time_key=time_key)
    y = data.y
    months = data.timestamps
    n = data.n_obs

    ts = int(args.train_start)
    te = int(args.train_end)
    es = int(args.eval_start)
    ee = int(args.eval_end)
    _validate_ranges(n, ts, te, es, ee)

    y_train = y[ts : te + 1].astype(np.float64)
    y_eval_true = y[es : ee + 1].astype(np.float64)

    d = int(args.d)
    p_max = int(getattr(args, "p_max", 3))
    q_max = int(getattr(args, "q_max", 3))

    best, best_res, search_summary = _search_pq(y_train, y_eval_true, d, p_max, q_max)

    if getattr(args, "save_model", None):
        dump_model(best_res, args.save_model)

    best_predictions = best.get("y_pred")
    eval_y_true_list = y_eval_true.tolist()
    eval_pred_list = best_predictions.tolist() if best_predictions is not None else []

    out: Dict[str, Any] = {
        "mode": "arima_search_pq_fixed_d",
        "best_order": {
            "p": best["p"],
            "d": best["d"],
            "q": best["q"],
        },
        "metrics": {
            "rmse": best["rmse"],
            "mae": best["mae"],
            "mape": best["mape"],
            "r2": best["r2"],
            "aic": best["aic"],
            "converged": best.get("converged", True),
        },
        "eval_y_true": eval_y_true_list,
        "eval_predictions": eval_pred_list,
        "predictions": [{"prediction": p} for p in eval_pred_list],
        "search_summary": search_summary,
        "meta": {
            "train_range": [ts, te],
            "eval_range": [es, ee],
            "n_obs": n,
            "months_eval_slice": months[es : ee + 1] if months else None,
        },
        "notes": [
            "Trained on training segment only; best (p,q) by RMSE; AIC tiebreaker.",
            "Use --save_model to persist best model; later use --load_model to predict.",
            "All indices use closed interval semantics: [start, end].",
        ],
    }

    try:
        out["diagnostics"] = {
            "y_pred_eval": best_predictions.tolist(),
            "conf_int": best["conf_int"].tolist(),
        }
    except (KeyError, AttributeError, ValueError):
        pass

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))


def _handle_prediction(args, pretty: bool) -> None:
    """
    Handle prediction mode with NumPy optimizations.
    """
    res = load_model(args.load_model)
    try:
        order = res.model.order
    except AttributeError:
        order = (0, 0, 0)
    nobs = int(getattr(res, "nobs", 0))

    ci_level = float(getattr(args, "ci_level", 0.95))
    if not (0.0 < ci_level < 1.0):
        ci_level = 0.95
    alpha = 1.0 - ci_level

    have_eval_window = (
        getattr(args, "eval_start", None) is not None
        and getattr(args, "eval_end", None) is not None
    )

    predictions_list = []
    conf_list = []
    se_mean_list = []
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

        if ee <= nobs - 1:
            pred = res.get_prediction(start=es, end=ee)
            yhat = pred.predicted_mean.astype(np.float64)
            ci = pred.conf_int(alpha=alpha).astype(np.float64)
            se_mean = pred.se_mean.astype(np.float64)
            predictions_list.append(yhat)
            conf_list.append(ci)
            se_mean_list.append(se_mean)
            total_steps = len(yhat)
            mode = "in_sample"
        elif es >= nobs:
            steps = ee - nobs + 1
            if steps <= 0:
                raise ValueError(
                    "[VALIDATION] Computed steps <= 0; check eval range vs model nobs."
                )
            pred = res.get_forecast(steps=steps)
            yhat = pred.predicted_mean.astype(np.float64)
            ci = pred.conf_int(alpha=alpha).astype(np.float64)
            se_mean = pred.se_mean.astype(np.float64)
            predictions_list.append(yhat)
            conf_list.append(ci)
            se_mean_list.append(se_mean)
            total_steps = steps
            mode = "out_of_sample"
        else:
            pred_a = res.get_prediction(start=es, end=nobs - 1)
            yhat_a = pred_a.predicted_mean.astype(np.float64)
            ci_a = pred_a.conf_int(alpha=alpha).astype(np.float64)
            se_mean_a = pred_a.se_mean.astype(np.float64)

            steps_b = ee - nobs + 1
            pred_b = res.get_forecast(steps=steps_b)
            yhat_b = pred_b.predicted_mean.astype(np.float64)
            ci_b = pred_b.conf_int(alpha=alpha).astype(np.float64)
            se_mean_b = pred_b.se_mean.astype(np.float64)

            predictions_list.extend([yhat_a, yhat_b])
            conf_list.extend([ci_a, ci_b])
            se_mean_list.extend([se_mean_a, se_mean_b])

            total_steps = len(yhat_a) + len(yhat_b)
            mode = "mixed"
    else:
        if getattr(args, "forecast_steps", None) is None:
            raise ValueError(
                "[VALIDATION] Prediction requires either eval_start/eval_end or --forecast_steps."
            )
        steps = int(args.forecast_steps)
        if steps <= 0:
            raise ValueError("[VALIDATION] --forecast_steps must be >= 1")

        pred = res.get_forecast(steps=steps)
        yhat = pred.predicted_mean.astype(np.float64)
        ci = pred.conf_int(alpha=alpha).astype(np.float64)
        se_mean = pred.se_mean.astype(np.float64)
        predictions_list.append(yhat)
        conf_list.append(ci)
        se_mean_list.append(se_mean)
        total_steps = steps
        mode = "out_of_sample"

    y_pred = (
        np.concatenate(predictions_list)
        if len(predictions_list) > 1
        else predictions_list[0]
    )
    ci = np.concatenate(conf_list) if len(conf_list) > 1 else conf_list[0]
    se_mean_all = (
        np.concatenate(se_mean_list) if len(se_mean_list) > 1 else se_mean_list[0]
    )

    predictions_formatted = [
        {"prediction": float(p), "std_dev": float(s)}
        for p, s in zip(y_pred, se_mean_all)
    ]

    out = {
        "mode": "arima_loaded_predict",
        "order": {"p": order[0], "d": order[1], "q": order[2]},
        "forecast_steps": total_steps,
        "predictions": predictions_formatted,
        "conf_int": ci.tolist(),
        "meta": {
            "mode": mode,
            "nobs": nobs,
            "eval_range": eval_span,
            "ci_level": ci_level,
        },
        "notes": [
            "If eval_range is within [0, nobs-1], predictions are in-sample via get_prediction.",
            "If eval_range is beyond nobs-1, predictions are out-of-sample via get_forecast.",
            "Mixed windows are split into in-sample + out-of-sample and concatenated.",
        ],
    }

    if hasattr(args, "data") and args.data and have_eval_window:
        try:
            target_key, time_key, records = _determine_target_key(args.data)
            data = parse_data_from_json(records, target_key=target_key, time_key=time_key)
            y = data.y
            es, ee = int(args.eval_start), int(args.eval_end)
            y_true = y[es : ee + 1]
            if len(y_true) == len(y_pred):
                out["metrics"] = {
                    "rmse": rmse(y_true, y_pred),
                    "mae": mae(y_true, y_pred),
                    "mape": mape(y_true, y_pred),
                    "r2": r2(y_true, y_pred),
                }
            else:
                out["metrics_error"] = (
                    f"[VALIDATION] Series length mismatch for metrics: y_true={len(y_true)}, pred={len(y_pred)}"
                )
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            out["metrics_error"] = f"[DATA] Failed to compute metrics: {e}"

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
