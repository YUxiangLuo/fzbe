#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted Average Ensemble Model using Inverse Variance weighting.
"""
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import shared utilities
from data_utils import ParsedData, load_data_from_json
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model

# Import base model functions
from . import arima, es, lstm, ma

MODEL_TYPE = "weighted_avg"

# A mapping from model names to their functions for dynamic dispatch
MODEL_DISPATCHER = {
    "arima": {"train": arima.train_and_evaluate, "predict": arima.predict},
    "es": {"train": es.train_and_evaluate, "predict": es.predict},
    "ma": {"train": ma.train_and_evaluate, "predict": ma.predict},
    "lstm": {"train": lstm.train_and_evaluate, "predict": lstm.predict},
}


# ---------- utilities ----------
def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved Weighted Average model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid Weighted Average model file.")
    return state


# ---------- predict ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], float]:
    """Forecast using a saved Weighted Average model."""
    state = state or _load_state(model_path)
    weights = state["weights"]
    base_model_states = state["base_model_states"]
    models_to_use = state["models"]

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

    X_future = np.array(base_predictions).T
    model_weights = np.array([weights[name] for name in models_to_use])
    final_predictions = np.sum(X_future * model_weights, axis=1)
    
    return final_predictions.tolist(), 0.0


# ---------- train and evaluate ----------
def train_and_evaluate(
    parsed_data: ParsedData,
    train_range: range,
    eval_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float], List[float], List[float]]:
    """Trains and evaluates the Weighted Average model."""
    
    base_model_predictions = {}
    base_model_states = {}
    temp_keras_paths = []
    y_true = parsed_data.y[eval_range]

    # --- Level 0: Train base models ---
    for model_name in models_to_use:
        model_module = MODEL_DISPATCHER[model_name]
        params = model_params[model_name]

        if model_name == "arima":
            _, model, _, y_pred, _ = model_module["train"](
                parsed_data.y, parsed_data.timestamps, train_range, eval_range, params["order"]
            )
            base_model_states[model_name] = {"model": model}
            base_model_predictions[model_name] = y_pred

        elif model_name == "es":
            _, _, level, _, y_pred = model_module["train"](
                parsed_data.y, parsed_data.timestamps, train_range, eval_range, params["alpha"]
            )
            base_model_states[model_name] = {"alpha": params["alpha"], "level": level}
            base_model_predictions[model_name] = y_pred

        elif model_name == "ma":
            _, std_dev, _, y_pred = model_module["train"](
                parsed_data.y, parsed_data.timestamps, train_range, eval_range, params["window"]
            )
            history = parsed_data.y[train_range.stop - params["window"] : train_range.stop].tolist()
            base_model_states[model_name] = {
                "window": params["window"], "history": history, 
                "std_dev_residuals": std_dev, "train_end_index": train_range.stop - 1
            }
            base_model_predictions[model_name] = y_pred

        elif model_name == "lstm":
            _, model, preprocessor, _, y_pred, std_dev, history_df = model_module["train"](
                parsed_data, params["feature_types"], train_range, eval_range,
                params["look_back"], params["epochs"]
            )
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                model.save(tmp.name)
                temp_keras_paths.append(tmp.name)
                base_model_states[model_name] = {
                    "preprocessor": preprocessor, "look_back": params["look_back"],
                    "history_df": history_df, "keras_model_path": tmp.name, "std_dev_residuals": std_dev
                }
            base_model_predictions[model_name] = y_pred

    # --- Calculate Inverse Variance Weights ---
    weights = {}
    total_inverse_variance = 0
    
    for name in models_to_use:
        residuals = y_true - np.array(base_model_predictions[name])
        variance = np.var(residuals)
        inverse_variance = 1 / (variance + 1e-9)
        weights[name] = inverse_variance
        total_inverse_variance += inverse_variance
        
    for name in models_to_use:
        weights[name] /= total_inverse_variance

    # --- Evaluate the final model ---
    X_eval = np.array([base_model_predictions[name] for name in models_to_use]).T
    model_weights = np.array([weights[name] for name in models_to_use])
    y_pred_final = np.sum(X_eval * model_weights, axis=1)

    metrics = {
        "rmse": rmse(y_true, y_pred_final),
        "mae": mae(y_true, y_pred_final),
        "mape": mape(y_true, y_pred_final),
        "r2": r2(y_true, y_pred_final),
    }

    return metrics, base_model_states, weights, y_true.tolist(), y_pred_final.tolist()


# ---------- public entry ----------
def run(args) -> None:
    """Main entry point for Weighted Average model."""
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if load_path:
        steps = int(getattr(args, "forecast_steps", 0))
        if steps <= 0:
            raise ValueError("[VALIDATION] --forecast_steps must be > 0.")
        
        preds, _ = predict(load_path, steps)
        out = {
            "mode": "weighted_avg_loaded_predict",
            "forecast_steps": steps,
            "predictions": [{"prediction": p, "std_dev": 0.0} for p in preds],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # Training Mode
    parsed_data = load_data_from_json(args.data)
    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)
    models_to_use = [model.strip() for model in args.models.split(',')]

    # Validate models
    for model_name in models_to_use:
        if model_name not in MODEL_DISPATCHER:
            raise ValueError(f"Unknown model in --models: {model_name}")

    model_params = {
        "arima": {"order": tuple(map(int, args.arima_order.split(',')))},
        "es": {"alpha": float(args.es_alpha)},
        "ma": {"window": int(args.ma_window)},
        "lstm": {
            "look_back": int(args.lstm_look_back),
            "epochs": int(args.lstm_epochs),
            "feature_types": json.loads(getattr(args, "lstm_feature_types", "{}"))
        }
    }

    metrics, base_model_states, weights, y_true, y_pred = train_and_evaluate(
        parsed_data, range(ts, te + 1), range(es, ee + 1), models_to_use, model_params
    )

    out: Dict[str, Any] = {
        "mode": "weighted_avg_train_eval",
        "models": models_to_use,
        "weights": weights,
        "metrics": metrics,
        "eval_y_true": y_true,
        "predictions": [{"prediction": p, "std_dev": 0.0} for p in y_pred],
    }

    if save_path:
        state = {
            "model_type": MODEL_TYPE,
            "models": models_to_use,
            "base_model_states": base_model_states,
            "weights": weights,
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
