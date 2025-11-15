# py/src/models/es.py
# -*- coding: utf-8 -*-
"""
Exponential Smoothing (ES) time series forecasting model.
"""
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing

from data_utils import ParsedData, load_data_from_json
from logger import log, get_log_contents
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model

MODEL_TYPE = "es"

# Type alias for input flexibility
ArrayLike = Union[List[float], np.ndarray]


def _format_predictions(preds: List[float], std_dev: float) -> List[Dict[str, float]]:
    """Format predictions with standard deviation for JSON output."""
    return [{"prediction": p, "std_dev": std_dev} for p in preds]


def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved ES model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid ES model file.")
    return state


def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], float]:
    """Forecast future values using a saved ES model."""
    state = state or _load_state(model_path)
    model = state["model"]
    std_dev_residuals = float(state.get("std_dev_residuals", 0.0))
    
    predictions = model.forecast(steps)
    return predictions.tolist(), std_dev_residuals


def train_and_evaluate(
    y: ArrayLike,
    timestamps: List[str],
    train_range: range,
    eval_range: range,
    alpha: float,
) -> Tuple[Dict[str, Any], Any, List[float], List[float], float]:
    """Train and evaluate the Exponential Smoothing model."""
    series = np.asarray(y)
    y_train = series[train_range]
    y_eval = series[eval_range]

    model = SimpleExpSmoothing(y_train, initialization_method="estimated").fit(
        smoothing_level=alpha, optimized=False
    )
    
    y_pred = model.forecast(len(eval_range))
    y_true = y_eval.tolist()

    residuals = np.array(y_true) - np.array(y_pred)
    std_dev_residuals = float(np.std(residuals))

    metrics = {
        "alpha": alpha,
        "evaluated_points": len(y_true),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
    return metrics, model, y_true, y_pred.tolist(), std_dev_residuals


def _validate_training_args(y: ArrayLike, ts: int, te: int, es: int, ee: int, alpha: float) -> None:
    """Validate training and evaluation parameters."""
    n_total = len(y)
    if not (0 <= ts <= te < n_total):
        raise ValueError(f"[VALIDATION] Invalid training range [{ts}, {te}]")
    if not (es >= 0 and ee < n_total and es <= ee):
        raise ValueError(f"[VALIDATION] Invalid evaluation range [{es}, {ee}]")
    if es <= te:
        raise ValueError("[VALIDATION] Evaluation start must be after training end.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("[VALIDATION] Alpha must be between 0 and 1.")


def run(args) -> None:
    """Main entry point for ES model training and prediction."""
    log.info(f"ES run started with args: {vars(args)}")
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if load_path:
        log.info(f"Entering prediction mode with model: {load_path}")
        try:
            state = _load_state(load_path)
            steps = int(getattr(args, "forecast_steps", 0))
            if steps <= 0:
                raise ValueError("[VALIDATION] --forecast_steps must be > 0.")

            preds, std_dev = predict(load_path, steps, state=state)
            predictions_with_std = _format_predictions(preds, std_dev)

            out: Dict[str, Any] = {
                "mode": "es_loaded_predict",
                "alpha": float(state["alpha"]),
                "train_end_index": int(state["train_end_index"]),
                "forecast_steps": steps,
                "predictions": predictions_with_std,
            }
            out["logs"] = get_log_contents()
            log.info("Prediction finished successfully.")
            print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
            return
        except Exception as e:
            log.error(f"An error occurred during prediction: {e}", exc_info=True)
            raise

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

        ts, te = int(args.train_start), int(args.train_end)
        es, ee = int(args.eval_start), int(args.eval_end)

        alpha_arg = getattr(args, "alpha", None)
        alpha = float(alpha_arg) if alpha_arg is not None else 0.5
        log.debug(f"Training params: alpha={alpha}")

        _validate_training_args(y, ts, te, es, ee, alpha)

        log.info("Starting model training and evaluation...")
        metrics, model, y_true, y_pred, std_dev_residuals = train_and_evaluate(
            y, timestamps, range(ts, te + 1), range(es, ee + 1), alpha
        )
        log.info(f"Training finished. Metrics: {metrics}")

        predictions_with_std = _format_predictions(y_pred, std_dev_residuals)

        out: Dict[str, Any] = {
            "mode": "fixed_alpha_only",
            "data_points": len(y),
            "alpha": alpha,
            "train_range": {"start_index": ts, "end_index": te, "months": timestamps[ts : te + 1]},
            "evaluate_range": {"start_index": es, "end_index": ee, "months": timestamps[es : ee + 1]},
            "metrics": metrics,
            "eval_y_true": y_true,
            "predictions": predictions_with_std,
        }

        if save_path:
            log.info(f"Saving model to {save_path}")
            state = {
                "model_type": MODEL_TYPE, "alpha": alpha, "train_end_index": te,
                "model": model, "std_dev_residuals": std_dev_residuals,
            }
            dump_model(state, save_path)
            out["saved_model"] = save_path

        out["logs"] = get_log_contents()
        log.info("Run finished successfully.")
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))

    except Exception as e:
        log.error(f"An error occurred during data processing or training: {e}", exc_info=True)
        raise ValueError(f"[DATA] Failed to parse or process --data: {e}")
