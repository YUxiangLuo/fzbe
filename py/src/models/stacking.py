#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacking Ensemble Model for time series forecasting.
Combines predictions from a flexible set of base models using a meta-model.
"""
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

# Import shared utilities
from data_utils import ParsedData, load_data_from_json
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model

# Import base model functions
from . import arima, es, lstm, ma

MODEL_TYPE = "stacking"

# A mapping from model names to their functions for dynamic dispatch
MODEL_DISPATCHER = {
    "arima": {"train": arima.train_and_evaluate, "predict": arima.predict},
    "es": {"train": es.train_and_evaluate, "predict": es.predict},
    "ma": {"train": ma.train_and_evaluate, "predict": ma.predict},
    "lstm": {"train": lstm.train_and_evaluate, "predict": lstm.predict},
}


# ---------- utilities ----------
def _format_predictions(preds: List[float], std_dev: float = 0.0) -> List[Dict[str, float]]:
    """Format predictions with standard deviation for JSON output."""
    return [{"prediction": float(p), "std_dev": float(std_dev)} for p in preds]


def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved Stacking model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid Stacking model file.")
    return state


def _validate_training_args(
    parsed_data: ParsedData,
    ts: int,
    te: int,
    es: int,
    ee: int,
    models_to_use: List[str],
    model_params: Dict[str, Any],
) -> None:
    """Validate training and evaluation parameters for Stacking."""
    n = parsed_data.n_obs

    # Validate models list
    if not models_to_use:
        raise ValueError("[VALIDATION] models_to_use cannot be empty.")

    if len(models_to_use) < 2:
        raise ValueError("[VALIDATION] Stacking requires at least 2 base models.")

    # Validate all models are known
    for model_name in models_to_use:
        if model_name not in MODEL_DISPATCHER:
            raise ValueError(f"[VALIDATION] Unknown model: {model_name}")

    # Validate train range
    if not (0 <= ts <= te < n):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], n_obs={n}")

    # Validate eval range
    if not (0 <= es <= ee < n):
        raise ValueError(f"[VALIDATION] Invalid eval range: [{es},{ee}], n_obs={n}")

    # Eval must start after train
    if es <= te:
        raise ValueError("[VALIDATION] eval_start must be > train_end.")

    # Validate each model has required parameters
    for model_name in models_to_use:
        if model_name not in model_params:
            raise ValueError(f"[VALIDATION] Missing parameters for model: {model_name}")

        params = model_params[model_name]

        # Model-specific validation
        if model_name == "arima":
            if "d" not in params:
                raise ValueError("[VALIDATION] ARIMA requires 'd' parameter.")
            if params["d"] < 0:
                raise ValueError("[VALIDATION] ARIMA d must be >= 0.")

        elif model_name == "es":
            if "alpha" not in params:
                raise ValueError("[VALIDATION] ES requires 'alpha' parameter.")
            if not (0 < params["alpha"] <= 1):
                raise ValueError("[VALIDATION] ES alpha must be in (0, 1].")

        elif model_name == "ma":
            if "window" not in params:
                raise ValueError("[VALIDATION] MA requires 'window' parameter.")
            if params["window"] < 1:
                raise ValueError("[VALIDATION] MA window must be >= 1.")

        elif model_name == "lstm":
            if "look_back" not in params:
                raise ValueError("[VALIDATION] LSTM requires 'look_back' parameter.")
            if "epochs" not in params:
                raise ValueError("[VALIDATION] LSTM requires 'epochs' parameter.")
            if params["look_back"] < 1:
                raise ValueError("[VALIDATION] LSTM look_back must be >= 1.")
            if params["epochs"] < 1:
                raise ValueError("[VALIDATION] LSTM epochs must be >= 1.")
            if te - ts + 1 <= params["look_back"]:
                raise ValueError("[VALIDATION] Training range must be larger than LSTM look_back.")


# ---------- predict ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], float]:
    """Forecast using a saved Stacking model."""
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    state = state or _load_state(model_path)
    meta_model = state["meta_model"]
    base_model_states = state["base_model_states"]
    models_to_use = state["models"]
    std_dev_residuals = state.get("std_dev_residuals", 0.0)

    base_predictions = []
    for model_name in models_to_use:
        model_module = MODEL_DISPATCHER[model_name]
        model_state = base_model_states[model_name]

        # The LSTM model needs its Keras model loaded manually for direct state prediction
        if model_name == "lstm" and "keras_model" not in model_state:
             from tensorflow.keras.models import load_model as keras_load_model
             model_state["keras_model"] = keras_load_model(model_state["keras_model_path"])

        preds, _ = model_module["predict"](model_path=None, steps=steps, state=model_state)
        base_predictions.append(preds)

    X_meta_future = np.array(base_predictions).T
    final_predictions = meta_model.predict(X_meta_future)

    return final_predictions.tolist(), std_dev_residuals


# ---------- train and evaluate ----------
def train_and_evaluate(
    parsed_data: ParsedData,
    train_range: range,
    eval_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Any, List[float], List[float], float]:
    """Trains and evaluates the Stacking model.

    Returns:
        metrics: Evaluation metrics dictionary
        base_model_states: States of all base models
        meta_model: Trained meta-model (LinearRegression)
        y_true: True values
        y_pred: Predicted values
        std_dev_residuals: Standard deviation of residuals
    """

    base_model_predictions = []
    base_model_states = {}

    # --- Level 0: Train base models ---
    for model_name in models_to_use:
        model_module = MODEL_DISPATCHER[model_name]
        params = model_params[model_name]

        if model_name == "arima":
            # ARIMA returns: (metrics, model, order, search_summary, y_true, y_pred, conf_int)
            metrics, model, order, search_summary, y_true, y_pred, conf_int = model_module["train"](
                parsed_data.y, parsed_data.timestamps, train_range, eval_range,
                params["d"], params.get("max_p", 5), params.get("max_q", 5)
            )
            base_model_states[model_name] = {
                "model": model,
                "order": order,
                "d": params["d"]
            }
            base_model_predictions.append(y_pred)

        elif model_name == "es":
            # ES returns: (metrics, std_dev_residuals, train_level, y_true, y_pred)
            metrics, std_dev, level, y_true, y_pred = model_module["train"](
                parsed_data.y, parsed_data.timestamps, train_range, eval_range, params["alpha"]
            )
            base_model_states[model_name] = {
                "alpha": params["alpha"],
                "level": level
            }
            base_model_predictions.append(y_pred)

        elif model_name == "ma":
            # MA returns: (metrics, std_dev, y_true, y_pred)
            metrics, std_dev, y_true, y_pred = model_module["train"](
                parsed_data.y, parsed_data.timestamps, train_range, eval_range, params["window"]
            )
            history = parsed_data.y[train_range.stop - params["window"] : train_range.stop].tolist()
            base_model_states[model_name] = {
                "window": params["window"],
                "history": history,
                "std_dev_residuals": std_dev,
                "train_end_index": train_range.stop - 1
            }
            base_model_predictions.append(y_pred)

        elif model_name == "lstm":
            # LSTM returns: (metrics, model, preprocessor, y_true, y_pred, std_dev, history_df)
            metrics, model, preprocessor, y_true, y_pred, std_dev, history_df = model_module["train"](
                parsed_data, params["feature_types"], train_range, eval_range,
                params["look_back"], params["epochs"], params.get("normalization", "minmax")
            )
            # Save Keras model to a permanent location (not temp file)
            # The path will be derived from the final save_path in run()
            base_model_states[model_name] = {
                "keras_model": model,  # Store model object for now, will be saved in run()
                "preprocessor": preprocessor,
                "look_back": params["look_back"],
                "history_df": history_df,
                "std_dev_residuals": std_dev
            }
            base_model_predictions.append(y_pred)

    # --- Level 1: Train meta-model ---
    X_meta_train = np.array(base_model_predictions).T
    y_meta_train = parsed_data.y[eval_range]

    meta_model = LinearRegression()
    meta_model.fit(X_meta_train, y_meta_train)

    y_pred_final = meta_model.predict(X_meta_train)
    y_true_final = y_meta_train

    # Calculate residuals and standard deviation
    residuals = y_true_final - y_pred_final
    std_dev_residuals = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    metrics = {
        "rmse": rmse(y_true_final, y_pred_final),
        "mae": mae(y_true_final, y_pred_final),
        "mape": mape(y_true_final, y_pred_final),
        "r2": r2(y_true_final, y_pred_final),
    }

    return metrics, base_model_states, meta_model, y_true_final.tolist(), y_pred_final.tolist(), std_dev_residuals


# ---------- public entry ----------
def run(args) -> None:
    """Main entry point for Stacking model training and prediction."""
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if load_path:
        steps = int(getattr(args, "forecast_steps", 0))
        if steps <= 0:
            raise ValueError("[VALIDATION] --forecast_steps must be > 0.")
        
        preds, std_dev = predict(load_path, steps)
        predictions_with_std = _format_predictions(preds, std_dev)

        out = {
            "mode": "stacking_loaded_predict",
            "forecast_steps": steps,
            "predictions": predictions_with_std,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # Training Mode
    if not hasattr(args, "data") or not args.data:
        raise ValueError("[VALIDATION] --data is required for training mode.")

    parsed_data = load_data_from_json(args.data)
    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)
    models_to_use = [model.strip() for model in args.models.split(',')]

    model_params = {
        "arima": {
            "d": int(args.arima_d),
            "max_p": int(getattr(args, "arima_max_p", 5)),
            "max_q": int(getattr(args, "arima_max_q", 5))
        },
        "es": {"alpha": float(args.es_alpha)},
        "ma": {"window": int(args.ma_window)},
        "lstm": {
            "look_back": int(args.lstm_look_back),
            "epochs": int(args.lstm_epochs),
            "feature_types": json.loads(getattr(args, "lstm_feature_types", "{}"))
        }
    }

    # Validate all parameters
    _validate_training_args(parsed_data, ts, te, es, ee, models_to_use, model_params)

    metrics, base_model_states, meta_model, y_true, y_pred, std_dev = train_and_evaluate(
        parsed_data, range(ts, te + 1), range(es, ee + 1), models_to_use, model_params
    )

    predictions_with_std = _format_predictions(y_pred, std_dev)

    out: Dict[str, Any] = {
        "mode": "stacking_train_eval",
        "models": models_to_use,
        "metrics": metrics,
        "eval_y_true": y_true,
        "predictions": predictions_with_std,
    }

    if save_path:
        # Save LSTM Keras models to permanent files (not temp files)
        for model_name, model_state in base_model_states.items():
            if model_name == "lstm" and "keras_model" in model_state:
                keras_model = model_state.pop("keras_model")  # Remove from state
                keras_model_path = os.path.splitext(save_path)[0] + f"_lstm_keras_model.h5"
                keras_model.save(keras_model_path)
                model_state["keras_model_path"] = keras_model_path

        state = {
            "model_type": MODEL_TYPE,
            "models": models_to_use,
            "base_model_states": base_model_states,
            "meta_model": meta_model,
            "std_dev_residuals": std_dev,
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))