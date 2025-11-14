#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A flexible Boosting ensemble model using residual fitting.
"""
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import shared utilities
from data_utils import ParsedData, load_data_from_json
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model

# Import base model functions
from . import arima, es, lstm, ma

MODEL_TYPE = "boosting"

# A mapping from model names to their functions
MODEL_DISPATCHER = {
    "arima": {"train": arima.train_and_evaluate, "predict": arima.predict},
    "es": {"train": es.train_and_evaluate, "predict": es.predict},
    "ma": {"train": ma.train_and_evaluate, "predict": ma.predict},
    "lstm": {"train": lstm.train_and_evaluate, "predict": lstm.predict},
}


# ---------- utilities ----------
def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved Boosting model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid Boosting model file.")
    return state


# ---------- predict ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], float]:
    """Forecast using a saved Boosting model."""
    state = state or _load_state(model_path)
    model_chain = state["model_chain"]
    base_model_states = state["base_model_states"]

    final_predictions = np.zeros(steps)

    for model_name in model_chain:
        model_module = MODEL_DISPATCHER[model_name]
        model_state = base_model_states[model_name]
        
        # The LSTM model needs its Keras model loaded manually for direct state prediction
        if model_name == "lstm" and "keras_model" not in model_state:
             from tensorflow.keras.models import load_model as keras_load_model
             model_state["keras_model"] = keras_load_model(model_state["keras_model_path"])

        preds, _ = model_module["predict"](model_path=None, steps=steps, state=model_state)
        final_predictions += np.array(preds)

    return final_predictions.tolist(), 0.0


# ---------- train and evaluate ----------
def train_and_evaluate(
    parsed_data: ParsedData,
    train_range: range,
    eval_range: range,
    model_chain: List[str],
    model_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], List[float]]:
    """Trains and evaluates the Boosting model."""
    
    base_model_states = {}
    y_true_eval = parsed_data.y[eval_range]
    final_eval_preds = np.zeros(len(eval_range))
    
    current_parsed_data = parsed_data
    temp_keras_paths = []

    for model_name in model_chain:
        model_module = MODEL_DISPATCHER[model_name]
        params = model_params[model_name]

        # --- Train the current model and get its state ---
        # Note: We need predictions on both train and eval ranges to calculate residuals for the next stage
        full_range = range(train_range.start, eval_range.stop)
        
        if model_name in ["arima", "es", "ma"]:
            # These models can predict on a different range than they were trained on.
            # 1. Train on train_range
            if model_name == "arima":
                _, model, _, _, _ = model_module["train"](current_parsed_data.y, current_parsed_data.timestamps, train_range, range(0,0), params["order"])
                base_model_states[model_name] = {"model": model}
                # 2. Predict on full_range (train + eval)
                train_preds, _ = model.predict(n_periods=len(train_range), return_conf_int=False)
                eval_preds, _ = model.predict(n_periods=len(eval_range), return_conf_int=False)

            elif model_name == "es":
                _, _, level, _, _ = model_module["train"](current_parsed_data.y, current_parsed_data.timestamps, train_range, range(0,0), params["alpha"])
                base_model_states[model_name] = {"alpha": params["alpha"], "level": level}
                train_preds = [level] * len(train_range) # Simplification for training residuals
                eval_preds = [level] * len(eval_range)

            elif model_name == "ma":
                # MA needs careful handling of history for prediction
                _, std_dev, _, _ = model_module["train"](current_parsed_data.y, current_parsed_data.timestamps, train_range, range(0,0), params["window"])
                history = current_parsed_data.y[train_range.stop - params["window"] : train_range.stop].tolist()
                base_model_states[model_name] = {
                    "window": params["window"], "history": history, 
                    "std_dev_residuals": std_dev, "train_end_index": train_range.stop - 1
                }
                _, train_preds = ma.moving_average_forecast(current_parsed_data.y, train_range, params["window"])
                _, eval_preds = ma.moving_average_forecast(current_parsed_data.y, eval_range, params["window"])
                # Pad predictions to match range length
                train_preds = np.pad(train_preds, (len(train_range) - len(train_preds), 0), 'edge')
                eval_preds = np.pad(eval_preds, (len(eval_range) - len(eval_preds), 0), 'edge')

            y_pred_full = np.concatenate([train_preds, eval_preds])

        elif model_name == "lstm":
            # LSTM's train_and_evaluate is more monolithic, we call it once
            metrics, model, preprocessor, _, y_pred, std_dev, history_df = model_module["train"](
                current_parsed_data, params["feature_types"], train_range, eval_range,
                params["look_back"], params["epochs"]
            )
            # For LSTM, we don't easily get train predictions, so we'll make a simplification
            # and assume train residuals are zero for the next stage. This is a limitation.
            y_pred_full = np.pad(y_pred, (len(train_range), 0), 'constant')
            
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                model.save(tmp.name)
                temp_keras_paths.append(tmp.name)
                base_model_states[model_name] = {
                    "preprocessor": preprocessor, "look_back": params["look_back"],
                    "history_df": history_df, "keras_model_path": tmp.name, "std_dev_residuals": std_dev
                }
        
        # --- Update final predictions and prepare for the next iteration ---
        final_eval_preds += y_pred_full[len(train_range):]
        
        # Calculate residuals for the next model
        residuals = current_parsed_data.y - y_pred_full
        
        # Create a new ParsedData object with residuals as the target
        new_df = current_parsed_data.df.copy()
        new_df['residual_target'] = residuals
        
        current_parsed_data = ParsedData(df=new_df, target_key='residual_target', 
                                         time_key=current_parsed_data.time_key,
                                         feature_keys=current_parsed_data.feature_keys)


    # --- Final Evaluation ---
    metrics = {
        "rmse": rmse(y_true_eval, final_eval_preds),
        "mae": mae(y_true_eval, final_eval_preds),
        "mape": mape(y_true_eval, final_eval_preds),
        "r2": r2(y_true_eval, final_eval_preds),
    }
    
    return metrics, base_model_states, y_true_eval.tolist(), final_eval_preds.tolist()


# ---------- public entry ----------
def run(args) -> None:
    """Main entry point for the Boosting model."""
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if load_path:
        steps = int(getattr(args, "forecast_steps", 0))
        if steps <= 0:
            raise ValueError("[VALIDATION] --forecast_steps must be > 0.")
        
        preds, _ = predict(load_path, steps)
        out = {
            "mode": "boosting_loaded_predict",
            "forecast_steps": steps,
            "predictions": [{"prediction": p, "std_dev": 0.0} for p in preds],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # Training Mode
    parsed_data = load_data_from_json(args.data)
    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)
    model_chain = [model.strip() for model in args.model_chain.split(',')]

    # Validate model chain
    for model_name in model_chain:
        if model_name not in MODEL_DISPATCHER:
            raise ValueError(f"Unknown model in --model-chain: {model_name}")

    # Extract model parameters from args
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

    metrics, base_model_states, y_true, y_pred = train_and_evaluate(
        parsed_data, range(ts, te + 1), range(es, ee + 1), model_chain, model_params
    )

    out: Dict[str, Any] = {
        "mode": "boosting_train_eval",
        "model_chain": model_chain,
        "metrics": metrics,
        "eval_y_true": y_true,
        "predictions": [{"prediction": p, "std_dev": 0.0} for p in y_pred],
    }

    if save_path:
        state = {
            "model_type": MODEL_TYPE,
            "model_chain": model_chain,
            "base_model_states": base_model_states,
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
