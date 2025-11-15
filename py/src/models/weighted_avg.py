#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted Average Ensemble Model using Inverse Variance weighting.
Combines predictions from a flexible set of base models.

Uses subprocess to call base models for consistency with API behavior.
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
from subprocess_utils import run_base_model_subprocess, run_base_model_predict_subprocess, extract_predictions_from_result

MODEL_TYPE = "weighted_avg"


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

    # Supported base models
    SUPPORTED_BASE_MODELS = ["ma", "es", "arima", "lstm"]

    if not models_to_use:
        raise ValueError("[VALIDATION] models_to_use cannot be empty.")

    for model_name in models_to_use:
        if model_name not in SUPPORTED_BASE_MODELS:
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
    """Forecast using a saved Weighted Average model with subprocess calls."""
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    state = state or _load_state(model_path)
    weights = state["weights"]
    base_model_states = state["base_model_states"]
    models_to_use = state["models"]
    std_dev_residuals = state.get("std_dev_residuals", 0.0)

    base_predictions = []
    for model_name in models_to_use:
        model_state = base_model_states[model_name]
        model_file = model_state["model_file"]

        # Use subprocess to predict
        result = run_base_model_predict_subprocess(
            model_type=model_name,
            model_file=model_file,
            forecast_steps=steps,
            timeout=60,
        )

        # Extract predictions
        preds = extract_predictions_from_result(result)
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
    """Trains and evaluates the Weighted Average model using subprocess calls."""

    base_model_predictions = {}
    base_model_states = {}
    y_true = parsed_data.y[eval_range]

    # Convert parsed_data to list of dicts for subprocess
    data_dicts = []
    for i in range(parsed_data.n_obs):
        row = {parsed_data.target_key: float(parsed_data.y[i])}
        # Add feature columns if they exist
        if parsed_data.feature_keys:
            for feat_key in parsed_data.feature_keys:
                row[feat_key] = float(parsed_data.df.iloc[i][feat_key])
        data_dicts.append(row)

    # Train base models using subprocess
    for model_name in models_to_use:
        params = model_params[model_name]

        # Create a temporary file for this base model
        temp_model_file = os.path.join(
            tempfile.gettempdir(),
            f"weighted_avg_base_{model_name}_{os.getpid()}.pkl"
        )

        # Add save_model to params for subprocess
        params_with_save = params.copy()
        params_with_save["save_model"] = temp_model_file

        # Run base model in subprocess
        result = run_base_model_subprocess(
            model_type=model_name,
            data=data_dicts,
            train_start=train_range.start,
            train_end=train_range.stop - 1,
            eval_start=eval_range.start,
            eval_end=eval_range.stop - 1,
            model_params=params_with_save,
            timeout=120,
        )

        # Extract predictions from subprocess result
        y_pred = extract_predictions_from_result(result)
        base_model_predictions[model_name] = y_pred

        # Store the model file path and parameters
        base_model_states[model_name] = {
            "model_type": model_name,
            "model_file": temp_model_file,
            "params": params,
        }

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

    # Build model_params from the arguments passed by models.ts
    # models.ts sends: --d, --alpha, --window, --look-back, --epochs, --feature-types, etc.
    def get_arg(name, default, cast_type=None):
        """Safely get argument with default value"""
        val = getattr(args, name.replace("-", "_"), None)
        if val is None:
            return default
        if cast_type:
            return cast_type(val)
        return val

    model_params = {}

    if "arima" in models_to_use:
        model_params["arima"] = {
            "d": get_arg("d", 0, int),
            "max_p": get_arg("max_p", 5, int),
            "max_q": get_arg("max_q", 5, int)
        }

    if "es" in models_to_use:
        model_params["es"] = {"alpha": get_arg("alpha", 0.5, float)}

    if "ma" in models_to_use:
        model_params["ma"] = {"window": get_arg("window", 3, int)}

    if "lstm" in models_to_use:
        # Try to get feature_types from the argument
        feature_types_str = get_arg("feature_types", None)
        if feature_types_str:
            feature_types = json.loads(feature_types_str)
        else:
            # If feature_types not provided, try to infer from lstm-features
            lstm_features_str = get_arg("lstm_features", None)
            target_feature = get_arg("lstm_target_feature", "y")

            if lstm_features_str:
                # Build feature_types from feature list
                feature_names = [f.strip() for f in lstm_features_str.split(',')]
                feature_types = {feat: "numeric" for feat in feature_names}
                feature_types[target_feature] = "numeric"
            else:
                # Use auto-detection
                feature_types = {}

        model_params["lstm"] = {
            "look_back": get_arg("look_back", 3, int),
            "epochs": get_arg("epochs", 10, int),
            "feature_types": feature_types,
            "normalization": get_arg("normalization", "minmax")
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
        import shutil

        # Move base model files from temp to permanent location
        base_dir = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]

        for model_name, model_state in base_model_states.items():
            temp_file = model_state["model_file"]

            # Define permanent location
            perm_file = os.path.join(base_dir, f"{base_name}_base_{model_name}.pkl")

            # Move file from temp to permanent
            if os.path.exists(temp_file):
                shutil.move(temp_file, perm_file)

                # Also move LSTM Keras model if it exists
                if model_name == "lstm":
                    temp_keras = os.path.splitext(temp_file)[0] + "_lstm_keras_model.h5"
                    if os.path.exists(temp_keras):
                        perm_keras = os.path.join(base_dir, f"{base_name}_base_{model_name}_keras.h5")
                        shutil.move(temp_keras, perm_keras)

            # Update state with permanent path
            model_state["model_file"] = perm_file

        state = {
            "model_type": MODEL_TYPE, "models": models_to_use,
            "base_model_states": base_model_states, "weights": weights,
            "std_dev_residuals": std_dev,
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))