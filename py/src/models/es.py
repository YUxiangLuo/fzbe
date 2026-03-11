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
    log.debug(f"ES predict called with steps={steps}, model_path={model_path}")
    state = state or _load_state(model_path)
    model = state["model"]
    std_dev_residuals = float(state.get("std_dev_residuals", 0.0))
    alpha = float(state.get("alpha", 0.5))

    log.info(f"Generating {steps}-step forecast with alpha={alpha}")
    predictions = model.forecast(steps)
    log.info(f"Generated {len(predictions)} predictions")

    return predictions.tolist(), std_dev_residuals


def train_and_evaluate(
    y: ArrayLike,
    timestamps: List[str],
    train_range: range,
    eval_range: range,
    alpha: float,
) -> Tuple[Dict[str, Any], Any, List[float], List[float], float]:
    """Train and evaluate the Exponential Smoothing model."""
    log.info(f"Starting ES train_and_evaluate: alpha={alpha}, train_range={train_range}, eval_range={eval_range}")
    series = np.asarray(y)
    y_train = series[train_range]
    y_eval = series[eval_range]

    log.debug(f"Total series length: {len(series)}")
    log.debug(f"Training data range: [{train_range.start}, {train_range.stop-1}], count={len(y_train)}")
    log.debug(f"Evaluation data range: [{eval_range.start}, {eval_range.stop-1}], count={len(y_eval)}")
    log.debug(f"Training data: min={np.min(y_train):.4f}, max={np.max(y_train):.4f}, mean={np.mean(y_train):.4f}")

    log.info(f"Fitting SimpleExpSmoothing with alpha={alpha}")
    model = SimpleExpSmoothing(y_train, initialization_method="estimated").fit(
        smoothing_level=alpha, optimized=False
    )
    log.debug(f"Model fitted successfully")

    log.info(f"Forecasting {len(eval_range)} steps for evaluation")
    y_pred = model.forecast(len(eval_range))
    y_true = y_eval.tolist()

    residuals = np.array(y_true) - np.array(y_pred)
    std_dev_residuals = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    log.info(f"Calculated residuals: mean={np.mean(residuals):.4f}, std_dev={std_dev_residuals:.4f}")

    metrics = {
        "alpha": alpha,
        "evaluated_points": len(y_true),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "evaluate_indices": list(eval_range),
        "evaluate_months": [timestamps[i] for i in eval_range] if timestamps else [],
    }

    log.info(f"Evaluation metrics computed: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}, R²={metrics['r2']:.4f}")

    return metrics, model, y_true, y_pred.tolist(), std_dev_residuals


def _validate_training_args(y: ArrayLike, ts: int, te: int, es: int, ee: int, alpha: float) -> None:
    """Validate training and evaluation parameters."""
    log.debug(f"Validating training arguments: ts={ts}, te={te}, es={es}, ee={ee}, alpha={alpha}")
    n_total = len(y)
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
    if not (0.0 < alpha <= 1.0):
        log.error(f"Alpha {alpha} must be in (0, 1]")
        raise ValueError(f"[VALIDATION] Alpha must be in (0, 1], got {alpha}")

    # Additional helpful validations
    train_size = te - ts + 1
    eval_size = ee - es + 1
    log.info(f"Validation passed: train_size={train_size}, eval_size={eval_size}, alpha={alpha}")


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

            log.info(f"Starting prediction with {steps} forecast steps")
            preds, std_dev = predict(load_path, steps, state=state)
            predictions_with_std = _format_predictions(preds, std_dev)

            out: Dict[str, Any] = {
                "mode": "es_loaded_predict",
                "alpha": float(state["alpha"]),
                "train_end_index": int(state["train_end_index"]),
                "forecast_steps": steps,
                "predictions": predictions_with_std,
            }
            log.info("Prediction finished successfully")
            out["logs"] = get_log_contents()
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
            log.info(f"Model saved successfully to {save_path}")
            out["saved_model"] = save_path

        out["logs"] = get_log_contents()
        log.info("Run finished successfully.")
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))

    except Exception as e:
        log.error(f"An error occurred during data processing or training: {e}", exc_info=True)
        raise ValueError(f"[DATA] Failed to parse or process --data: {e}")
