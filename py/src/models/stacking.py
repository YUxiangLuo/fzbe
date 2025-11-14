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
def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved Stacking model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid Stacking model file.")
    return state


# ---------- predict ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], float]:
    """Forecast using a saved Stacking model."""
    state = state or _load_state(model_path)
    meta_model = state["meta_model"]
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

    X_meta_future = np.array(base_predictions).T
    final_predictions = meta_model.predict(X_meta_future)
    
    return final_predictions.tolist(), 0.0


# ---------- train and evaluate ----------
def train_and_evaluate(
    parsed_data: ParsedData,
    train_range: range,
    eval_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Any, List[float], List[float]]:
    """Trains and evaluates the Stacking model."""
    
    base_model_predictions = []
    base_model_states = {}
    temp_keras_paths = []

    # --- Level 0: Train base models ---
    for model_name in models_to_use:
        model_module = MODEL_DISPATCHER[model_name]
        params = model_params[model_name]
        
        if model_name == "arima":
            _, model, _, y_pred, _ = model_module["train"](
                parsed_data.y, parsed_data.timestamps, train_range, eval_range, params["order"]
            )
            base_model_states[model_name] = {"model": model}
            base_model_predictions.append(y_pred)

        elif model_name == "es":
            _, _, level, _, y_pred = model_module["train"](
                parsed_data.y, parsed_data.timestamps, train_range, eval_range, params["alpha"]
            )
            base_model_states[model_name] = {"alpha": params["alpha"], "level": level}
            base_model_predictions.append(y_pred)

        elif model_name == "ma":
            _, std_dev, _, y_pred = model_module["train"](
                parsed_data.y, parsed_data.timestamps, train_range, eval_range, params["window"]
            )
            history = parsed_data.y[train_range.stop - params["window"] : train_range.stop].tolist()
            base_model_states[model_name] = {
                "window": params["window"], "history": history, 
                "std_dev_residuals": std_dev, "train_end_index": train_range.stop - 1
            }
            base_model_predictions.append(y_pred)

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
            base_model_predictions.append(y_pred)

    # --- Level 1: Train meta-model ---
    X_meta_train = np.array(base_model_predictions).T
    y_meta_train = parsed_data.y[eval_range]

    meta_model = LinearRegression()
    meta_model.fit(X_meta_train, y_meta_train)

    y_pred_final = meta_model.predict(X_meta_train)
    y_true_final = y_meta_train

    metrics = {
        "rmse": rmse(y_true_final, y_pred_final),
        "mae": mae(y_true_final, y_pred_final),
        "mape": mape(y_true_final, y_pred_final),
        "r2": r2(y_true_final, y_pred_final),
    }

    return metrics, base_model_states, meta_model, y_true_final.tolist(), y_pred_final.tolist()


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
        
        preds, _ = predict(load_path, steps)
        out = {
            "mode": "stacking_loaded_predict",
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

    metrics, base_model_states, meta_model, y_true, y_pred = train_and_evaluate(
        parsed_data, range(ts, te + 1), range(es, ee + 1), models_to_use, model_params
    )

    out: Dict[str, Any] = {
        "mode": "stacking_train_eval",
        "models": models_to_use,
        "metrics": metrics,
        "eval_y_true": y_true,
        "predictions": [{"prediction": p, "std_dev": 0.0} for p in y_pred],
    }

    if save_path:
        state = {
            "model_type": MODEL_TYPE,
            "models": models_to_use,
            "base_model_states": base_model_states,
            "meta_model": meta_model,
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))