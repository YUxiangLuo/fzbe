#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted Average Ensemble Model using Inverse Variance weighting.
Combines predictions from a flexible set of base models.
"""
import json
import os
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
def _format_predictions(preds: List[float], std_dev: float = 0.0) -> List[Dict[str, float]]:
    """Format predictions with standard deviation for JSON output."""
    return [{"prediction": float(p), "std_dev": float(std_dev)} for p in preds]


def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved Weighted Average model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid Weighted Average model file.")
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
    """Validate training and evaluation parameters for Weighted Average."""
    n = parsed_data.n_obs

    if not models_to_use:
        raise ValueError("[VALIDATION] models_to_use cannot be empty.")

    for model_name in models_to_use:
        if model_name not in MODEL_DISPATCHER:
            raise ValueError(f"[VALIDATION] Unknown model: {model_name}")

    if not (0 <= ts <= te < n):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], n_obs={n}")
    if not (0 <= es <= ee < n):
        raise ValueError(f"[VALIDATION] Invalid eval range: [{es},{ee}], n_obs={n}")
    if es <= te:
        raise ValueError("[VALIDATION] eval_start must be > train_end.")

    for model_name in models_to_use:
        if model_name not in model_params:
            raise ValueError(f"[VALIDATION] Missing parameters for model: {model_name}")
        params = model_params[model_name]
        if model_name == "arima":
            if "d" not in params: raise ValueError("[VALIDATION] ARIMA requires 'd' parameter.")
            if params["d"] < 0: raise ValueError("[VALIDATION] ARIMA d must be >= 0.")
        elif model_name == "es":
            if "alpha" not in params: raise ValueError("[VALIDATION] ES requires 'alpha' parameter.")
            if not (0 < params["alpha"] <= 1): raise ValueError("[VALIDATION] ES alpha must be in (0, 1].")
        elif model_name == "ma":
            if "window" not in params: raise ValueError("[VALIDATION] MA requires 'window' parameter.")
            if params["window"] < 1: raise ValueError("[VALIDATION] MA window must be >= 1.")
        elif model_name == "lstm":
            if "look_back" not in params: raise ValueError("[VALIDATION] LSTM requires 'look_back' parameter.")
            if "epochs" not in params: raise ValueError("[VALIDATION] LSTM requires 'epochs' parameter.")
            if params["look_back"] < 1: raise ValueError("[VALIDATION] LSTM look_back must be >= 1.")
            if params["epochs"] < 1: raise ValueError("[VALIDATION] LSTM epochs must be >= 1.")
            if te - ts + 1 <= params["look_back"]: raise ValueError("[VALIDATION] Training range must be larger than LSTM look_back.")


# ---------- predict ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], float]:
    """Forecast using a saved Weighted Average model."""
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    state = state or _load_state(model_path)
    weights = state["weights"]
    base_model_states = state["base_model_states"]
    models_to_use = state["models"]
    std_dev_residuals = state.get("std_dev_residuals", 0.0)

    base_predictions = []
    for model_name in models_to_use:
        model_module = MODEL_DISPATCHER[model_name]
        model_state = base_model_states[model_name]

        if model_name == "lstm" and "keras_model" not in model_state:
             from tensorflow.keras.models import load_model as keras_load_model
             model_state["keras_model"] = keras_load_model(model_state["keras_model_path"])

        preds, _ = model_module["predict"](model_path=None, steps=steps, state=model_state)
        base_predictions.append(preds)

    X_future = np.array(base_predictions).T
    model_weights = np.array([weights[name] for name in models_to_use])
    final_predictions = np.sum(X_future * model_weights, axis=1)
    
    return final_predictions.tolist(), std_dev_residuals


# ---------- train and evaluate ----------
def train_and_evaluate(
    parsed_data: ParsedData,
    train_range: range,
    eval_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float], List[float], List[float], float]:
    """Trains and evaluates the Weighted Average model."""
    
    base_model_predictions = {}
    base_model_states = {}
    y_true = parsed_data.y[eval_range]

    for model_name in models_to_use:
        model_module = MODEL_DISPATCHER[model_name]
        params = model_params[model_name]
        y_pred = []

        if model_name == "arima":
            _, model, order, _, _, y_pred, _ = model_module["train"](
                parsed_data.y, parsed_data.timestamps, train_range, eval_range,
                params["d"], params.get("max_p", 5), params.get("max_q", 5)
            )
            base_model_states[model_name] = {"model": model, "order": order, "d": params["d"]}
        elif model_name == "es":
            _, std_dev, level, _, y_pred = model_module["train"](
                parsed_data.y, parsed_data.timestamps, train_range, eval_range, params["alpha"]
            )
            base_model_states[model_name] = {
                "alpha": params["alpha"], "level": level, "std_dev_residuals": std_dev,
                "train_end_index": train_range.stop - 1
            }
        elif model_name == "ma":
            _, std_dev, _, y_pred_arr = model_module["train"](
                parsed_data.y, parsed_data.timestamps, train_range, eval_range, params["window"]
            )
            y_pred = y_pred_arr.tolist()
            history = parsed_data.y[train_range.stop - params["window"] : train_range.stop].tolist()
            base_model_states[model_name] = {
                "window": params["window"], "history": history, "std_dev_residuals": std_dev,
                "train_end_index": train_range.stop - 1
            }
        elif model_name == "lstm":
            _, model, preprocessor, _, y_pred, std_dev, history_df = model_module["train"](
                parsed_data, params.get("feature_types", {}), train_range, eval_range,
                params["look_back"], params["epochs"]
            )
            base_model_states[model_name] = {
                "keras_model": model, "preprocessor": preprocessor, "look_back": params["look_back"],
                "history_df": history_df, "std_dev_residuals": std_dev, "train_end_index": train_range.stop - 1
            }
        
        base_model_predictions[model_name] = y_pred

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

    X_eval = np.array([base_model_predictions[name] for name in models_to_use]).T
    model_weights = np.array([weights[name] for name in models_to_use])
    y_pred_final = np.sum(X_eval * model_weights, axis=1)

    residuals_final = y_true - y_pred_final
    std_dev_residuals = float(np.std(residuals_final, ddof=1)) if len(residuals_final) > 1 else 0.0

    metrics = {
        "rmse": rmse(y_true, y_pred_final), "mae": mae(y_true, y_pred_final),
        "mape": mape(y_true, y_pred_final), "r2": r2(y_true, y_pred_final),
    }

    return metrics, base_model_states, weights, y_true.tolist(), y_pred_final.tolist(), std_dev_residuals


# ---------- public entry ----------
def run(args) -> None:
    """Main entry point for Weighted Average model."""
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if load_path:
        steps = int(getattr(args, "forecast_steps", 0))
        preds, std_dev = predict(load_path, steps)
        out = {
            "mode": "weighted_avg_loaded_predict",
            "forecast_steps": steps,
            "predictions": _format_predictions(preds, std_dev),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # Training Mode
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

    _validate_training_args(parsed_data, ts, te, es, ee, models_to_use, model_params)

    metrics, base_model_states, weights, y_true, y_pred, std_dev = train_and_evaluate(
        parsed_data, range(ts, te + 1), range(es, ee + 1), models_to_use, model_params
    )

    out: Dict[str, Any] = {
        "mode": "weighted_avg_train_eval",
        "models": models_to_use,
        "weights": weights,
        "metrics": metrics,
        "eval_y_true": y_true,
        "predictions": _format_predictions(y_pred, std_dev),
    }

    if save_path:
        for model_name, model_state in base_model_states.items():
            if model_name == "lstm" and "keras_model" in model_state:
                keras_model = model_state.pop("keras_model")
                keras_model_path = os.path.splitext(save_path)[0] + f"_lstm_keras_model.h5"
                keras_model.save(keras_model_path)
                model_state["keras_model_path"] = keras_model_path

        state = {
            "model_type": MODEL_TYPE, "models": models_to_use,
            "base_model_states": base_model_states, "weights": weights,
            "std_dev_residuals": std_dev,
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))