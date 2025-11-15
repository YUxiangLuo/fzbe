#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA model with automatic order selection for p and q.
User specifies d, and the model searches for optimal p and q using AIC and BIC.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pmdarima as pm

from data_utils import ParsedData, load_data_from_json
from logger import log, get_log_contents
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model

MODEL_TYPE = "arima"


# ---------- utilities ----------
def _format_predictions(preds: List[float], conf_intervals: List[List[float]]) -> List[Dict[str, float]]:
    """
    Format predictions with confidence intervals for JSON output.
    Converts 95% confidence interval to approximate std_dev.
    """
    return [
        {"prediction": float(p), "std_dev": float((ci[1] - ci[0]) / 3.92)}
        for p, ci in zip(preds, conf_intervals)
    ]


def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved ARIMA model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid ARIMA model file.")
    return state


# ---------- predict ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[List[float]]]:
    """
    Forecast using a saved ARIMA model.

    Returns:
        predictions: List of forecasted values
        conf_int: List of [lower, upper] confidence interval bounds
    """
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    state = state or _load_state(model_path)
    model = state["model"]

    y_pred, conf_int = model.predict(n_periods=steps, return_conf_int=True)

    return y_pred.tolist(), conf_int.tolist()


# ---------- train and evaluate ----------
def train_and_evaluate(
    y: np.ndarray,
    timestamps: List[str],
    train_range: range,
    eval_range: range,
    d: int,
    max_p: int = 5,
    max_q: int = 5,
) -> Tuple[Dict[str, Any], Any, Dict[str, int], Dict[str, Any], List[float], List[float], List[List[float]]]:
    """
    Trains an ARIMA model by searching for the best p and q using both AIC and BIC,
    and evaluates the best resulting model.

    Args:
        y: Full time series
        timestamps: Timestamp labels
        train_range: Range of indices for training
        eval_range: Range of indices for evaluation
        d: Differencing order (fixed by user)
        max_p: Maximum AR order to search
        max_q: Maximum MA order to search

    Returns:
        metrics: Evaluation metrics dict
        best_model: The selected ARIMA model
        best_order: Dict with p, d, q
        search_summary: Summary of AIC/BIC model search
        y_true: True evaluation values
        y_pred: Predicted evaluation values
        conf_int: Confidence intervals
    """
    if d < 0:
        raise ValueError(f"[VALIDATION] d must be >= 0, got {d}")
    if max_p < 0 or max_q < 0:
        raise ValueError("[VALIDATION] max_p and max_q must be >= 0")

    y_train = y[train_range]
    y_eval = y[eval_range]

    if len(y_train) == 0:
        raise ValueError("[VALIDATION] Training range is empty")
    if len(y_eval) == 0:
        raise ValueError("[VALIDATION] Evaluation range is empty")

    # --- Run auto_arima for both AIC and BIC ---
    aic_model = pm.auto_arima(
        y_train, d=d, start_p=0, start_q=0, max_p=max_p, max_q=max_q,
        seasonal=False, stepwise=True, suppress_warnings=True,
        error_action="ignore", information_criterion="aic", trace=False,
    )

    bic_model = pm.auto_arima(
        y_train, d=d, start_p=0, start_q=0, max_p=max_p, max_q=max_q,
        seasonal=False, stepwise=True, suppress_warnings=True,
        error_action="ignore", information_criterion="bic", trace=False,
    )

    # --- Compare and select the best model ---
    aic_order = aic_model.order
    bic_order = bic_model.order

    aic_aic = aic_model.aic()
    aic_bic = aic_model.bic()
    bic_aic = bic_model.aic()
    bic_bic = bic_model.bic()

    # Selection strategy: compare AIC values first, use BIC as tiebreaker
    selection_criterion = "aic_optimized"
    best_model = aic_model

    if bic_aic < aic_aic:
        best_model = bic_model
        selection_criterion = "bic_optimized_better_aic"
    elif bic_aic == aic_aic and bic_bic < aic_bic:
        best_model = bic_model
        selection_criterion = "bic_optimized_tie_aic_better_bic"

    best_order = {"p": best_model.order[0], "d": best_model.order[1], "q": best_model.order[2]}

    # Search summary for transparency
    search_summary = {
        "aic_optimized_model": {
            "order": {"p": aic_order[0], "d": aic_order[1], "q": aic_order[2]},
            "aic": float(aic_aic),
            "bic": float(aic_bic),
        },
        "bic_optimized_model": {
            "order": {"p": bic_order[0], "d": bic_order[1], "q": bic_order[2]},
            "aic": float(bic_aic),
            "bic": float(bic_bic),
        },
        "selected_model": "aic_optimized" if best_model == aic_model else "bic_optimized",
        "selection_criterion": selection_criterion,
    }

    # --- Evaluate the best model ---
    y_pred, conf_int = best_model.predict(n_periods=len(eval_range), return_conf_int=True)
    y_true = y_eval.tolist()

    metrics = {
        "rmse": rmse(y_eval, y_pred),
        "mae": mae(y_eval, y_pred),
        "mape": mape(y_eval, y_pred),
        "r2": r2(y_eval, y_pred),
        "aic": float(best_model.aic()),
        "bic": float(best_model.bic()),
        "converged": bool(best_model.arima_res_.mle_retvals.get("converged", False)),
        "evaluated_points": len(y_eval),
    }

    return metrics, best_model, best_order, search_summary, y_true, y_pred.tolist(), conf_int.tolist()


# ---------- public entry ----------
def run(args) -> None:
    """
    Main entry point for ARIMA model training and prediction.
    """
    log.info(f"ARIMA run started with args: {vars(args)}")
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if load_path:
        log.info(f"Entering prediction mode with model: {load_path}")
        state = _load_state(load_path)
        steps = int(getattr(args, "forecast_steps", 0))
        if steps <= 0:
            log.error("forecast_steps must be > 0 for prediction mode.")
            raise ValueError("[VALIDATION] --forecast_steps must be > 0 for prediction mode.")

        preds, conf_int = predict(load_path, steps, state=state)
        predictions_with_std = _format_predictions(preds, conf_int)

        out: Dict[str, Any] = {
            "mode": "arima_loaded_predict",
            "order": state.get("order", {}),
            "forecast_steps": steps,
            "predictions": predictions_with_std,
        }
        out["logs"] = get_log_contents()
        log.info("Prediction finished successfully.")
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # Training Mode
    log.info("Entering training mode.")
    if not hasattr(args, "data") or not args.data:
        log.error("--data is required for training.")
        raise ValueError("[VALIDATION] --data is required for training.")

    try:
        parsed_data = load_data_from_json(args.data)
        y = parsed_data.y
        timestamps = parsed_data.timestamps or [str(i) for i in range(parsed_data.n_obs)]
        log.info(f"Data loaded successfully. Number of observations: {parsed_data.n_obs}")
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        log.error(f"Failed to parse --data: {e}", exc_info=True)
        raise ValueError(f"[DATA] Failed to parse --data: {e}")

    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)
    d = int(args.d)

    max_p_arg = getattr(args, "max_p", None)
    max_q_arg = getattr(args, "max_q", None)

    if max_p_arg is not None and max_q_arg is not None:
        max_p = int(max_p_arg)
        max_q = int(max_q_arg)
    else:
        # Auto-select max_p and max_q based on training data size (for educational small datasets)
        train_size = te - ts + 1
        auto_max = max(1, min(3, train_size // 4))
        max_p = int(max_p_arg) if max_p_arg is not None else auto_max
        max_q = int(max_q_arg) if max_q_arg is not None else auto_max
        if max_p_arg is None or max_q_arg is None:
            log.info(f"Auto-selected max_p={max_p}, max_q={max_q} based on train_size={train_size}")
    log.debug(f"Training params: d={d}, max_p={max_p}, max_q={max_q}")

    if not (0 <= ts <= te < len(y)):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}]")
    if not (0 <= es <= ee < len(y)):
        raise ValueError(f"[VALIDATION] Invalid eval range: [{es},{ee}]")
    if es <= te:
        raise ValueError("[VALIDATION] eval_start must be > train_end.")

    log.info("Starting model training and evaluation...")
    metrics, best_model, best_order, search_summary, y_true, y_pred, conf_int = train_and_evaluate(
        y, timestamps, range(ts, te + 1), range(es, ee + 1), d, max_p, max_q
    )
    log.info(f"Training finished. Best order: {best_order}, Metrics: {metrics}")
    log.debug(f"Search summary: {search_summary}")

    predictions_with_std = _format_predictions(y_pred, conf_int)

    out: Dict[str, Any] = {
        "mode": "arima_search_pq_fixed_d",
        "data_points": len(y),
        "d": d,
        "best_order": best_order,
        "train_range": {"start_index": ts, "end_index": te, "months": timestamps[ts : te + 1]},
        "evaluate_range": {"start_index": es, "end_index": ee, "months": timestamps[es : ee + 1]},
        "metrics": metrics,
        "eval_y_true": y_true,
        "predictions": predictions_with_std,
        "search_summary": search_summary,
    }

    if save_path:
        log.info(f"Saving model to {save_path}")
        state = {
            "model_type": MODEL_TYPE,
            "model": best_model,
            "order": best_order,
            "d": d,
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    out["logs"] = get_log_contents()
    log.info("Run finished successfully.")
    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
