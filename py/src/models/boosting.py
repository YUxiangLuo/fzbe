#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdaBoost-style ensemble with Greedy Forward Selection for automatic chain building.

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

MODEL_TYPE = "boosting"


# ---------- utilities ----------
def _format_predictions(preds: List[float], std_dev: float = 0.0) -> List[Dict[str, float]]:
    """Format predictions with standard deviation for JSON output."""
    return [{"prediction": float(p), "std_dev": float(std_dev)} for p in preds]


def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved Boosting model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid Boosting model file.")
    return state


def _validate_training_args(
    parsed_data: ParsedData, ts: int, te: int, es: int, ee: int,
    models_to_use: List[str], model_params: Dict[str, Any]
) -> None:
    """Validate training and evaluation parameters."""
    n = parsed_data.n_obs

    # Supported base models
    SUPPORTED_BASE_MODELS = ["ma", "es", "arima", "lstm"]

    if not models_to_use:
        raise ValueError("[VALIDATION] models_to_use cannot be empty.")
    for name in models_to_use:
        if name not in SUPPORTED_BASE_MODELS:
            raise ValueError(f"[VALIDATION] Unknown model: {name}")
    if not (0 <= ts <= te < n and 0 <= es <= ee < n and es > te):
        raise ValueError("[VALIDATION] Invalid train/eval ranges.")
    # Model-specific param validation can be added here if needed


# ---------- predict ----------
def predict(
    model_path: str, steps: int, state: Optional[Dict[str, Any]] = None
) -> Tuple[List[float], float]:
    """Forecast using a saved AdaBoost model with subprocess calls."""
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    state = state or _load_state(model_path)
    model_chain = state["model_chain"]
    base_model_states = state["base_model_states"]
    model_weights = state["model_weights"]
    std_dev_residuals = state.get("std_dev_residuals", 0.0)

    final_predictions = np.zeros(steps)
    total_weight = sum(model_weights.values())

    for model_name in model_chain:
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
        final_predictions += np.array(preds) * (model_weights[model_name] / total_weight)

    return final_predictions.tolist(), std_dev_residuals


# ---------- train and evaluate ----------
def train_and_evaluate(
    parsed_data: ParsedData, train_range: range, eval_range: range,
    models_to_use: List[str], model_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any], Dict[str, float], List[float], List[float], float, Dict[str, Any]]:
    """
    Trains and evaluates the AdaBoost model with Greedy Forward Selection using a correct hold-out approach.

    Correct Boosting Flow:
    1. Split train_range into boost_train (80%) and boost_val (20%).
    2. Perform Greedy Forward Selection on boost_val:
        - Train base models on boost_train, predict on boost_val.
        - Select best model and calculate its weight based on boost_val error.
        - Update sample weights based on boost_val error.
    3. Retrain the selected model chain on the full train_range.
    4. Evaluate the final ensemble on the independent eval_range.
    """

    # === Step 1: Split train_range into boost_train and boost_val ===
    train_len = len(train_range)
    if train_len < 5:
        raise ValueError("[VALIDATION] train_range too small for boosting (需要至少5个点)")

    boost_split = max(3, int(train_len * 0.8))
    boost_train_range = range(train_range.start, train_range.start + boost_split)
    boost_val_range = range(train_range.start + boost_split, train_range.stop)

    print(f"[Boosting] 数据划分:")
    print(f"  Boost Train Range: {boost_train_range} ({len(boost_train_range)} points)")
    print(f"  Boost Val Range: {boost_val_range} ({len(boost_val_range)} points)")
    print(f"  Eval Range: {eval_range} ({len(eval_range)} points)")

    y_true_val = parsed_data.y[boost_val_range]
    sample_weights = np.ones(len(boost_val_range)) / len(boost_val_range)

    best_chain = []
    best_model_weights = {}
    best_overall_preds_val = np.zeros(len(boost_val_range))
    best_overall_rmse_val = float('inf')

    search_summary = {"stages": []}
    available_models = list(models_to_use)

    # Convert parsed_data to list of dicts for subprocess
    data_dicts = []
    for i in range(parsed_data.n_obs):
        row = {parsed_data.target_key: float(parsed_data.y[i])}
        if parsed_data.feature_keys:
            for feat_key in parsed_data.feature_keys:
                row[feat_key] = float(parsed_data.df.iloc[i][feat_key])
        data_dicts.append(row)

    # === Step 2: Greedy Forward Selection (using boost_train and boost_val) ===
    print(f"\n[Boosting Phase 1] Greedy selection on boost_val")
    for stage in range(len(models_to_use)):
        stage_results = []

        for model_name in available_models:
            params = model_params[model_name]
            result = run_base_model_subprocess(
                model_type=model_name,
                data=data_dicts,
                train_start=boost_train_range.start,
                train_end=boost_train_range.stop - 1,
                eval_start=boost_val_range.start,
                eval_end=boost_val_range.stop - 1,
                model_params=params,
                timeout=120,
            )
            y_pred_val = extract_predictions_from_result(result)
            errors = np.abs(np.array(y_pred_val) - y_true_val)
            weighted_error = np.sum(sample_weights * errors)
            weighted_error = min(weighted_error, 0.999)
            
            stage_results.append({
                "model_name": model_name,
                "weighted_error": weighted_error,
                "predictions": y_pred_val
            })

        best_model_this_stage = min(stage_results, key=lambda x: x["weighted_error"])
        winner_name = best_model_this_stage["model_name"]
        
        beta = best_model_this_stage["weighted_error"] / (1 - best_model_this_stage["weighted_error"] + 1e-9)
        model_weight = np.log(1 / (beta + 1e-9))

        current_preds_val = np.array(best_model_this_stage["predictions"])
        
        temp_preds_val = best_overall_preds_val.copy()
        if stage > 0:
             temp_preds_val = (temp_preds_val * sum(best_model_weights.values()) + current_preds_val * model_weight) / (sum(best_model_weights.values()) + model_weight)
        else:
            temp_preds_val = current_preds_val

        temp_rmse_val = rmse(y_true_val, temp_preds_val)

        if temp_rmse_val < best_overall_rmse_val:
            best_overall_rmse_val = temp_rmse_val
            best_overall_preds_val = temp_preds_val
            best_chain.append(winner_name)
            best_model_weights[winner_name] = model_weight
            
            errors = np.abs(current_preds_val - y_true_val)
            normalized_errors = errors / (np.max(errors) + 1e-9)
            sample_weights *= np.power(beta, 1 - normalized_errors)
            sample_weights /= np.sum(sample_weights)

            stage_summary = {
                "stage": stage + 1,
                "candidates": {r["model_name"]: r["weighted_error"] for r in stage_results},
                "winner": winner_name,
                "model_weight": model_weight,
                "chain_rmse_on_val": temp_rmse_val,
                "action": "Model Added"
            }
        else:
            stage_summary = {
                "stage": stage + 1,
                "candidates": {r["model_name"]: r["weighted_error"] for r in stage_results},
                "winner": winner_name,
                "action": "Model Skipped (did not improve validation RMSE)"
            }

        search_summary["stages"].append(stage_summary)
        if winner_name in available_models:
            available_models.remove(winner_name)
        if not available_models:
            break

    # === Step 3: Re-train final model chain on full train_range and evaluate on eval_range ===
    print(f"\n[Boosting Phase 2] Retraining on full train_range and final evaluation")
    best_states = {}
    final_eval_preds = np.zeros(len(eval_range))
    total_weight = sum(best_model_weights.values())

    if not best_chain: # Handle case where no models were added
        y_true_eval = parsed_data.y[eval_range]
        metrics = {"rmse": float('inf'), "mae": float('inf'), "mape": float('inf'), "r2": float('-inf')}
        return metrics, [], {}, {}, y_true_eval.tolist(), [0.0] * len(y_true_eval), 0.0, search_summary

    for model_name in best_chain:
        params = model_params[model_name]
        temp_model_file = os.path.join(
            tempfile.gettempdir(),
            f"boosting_base_{model_name}_{os.getpid()}.pkl"
        )
        params_with_save = params.copy()
        params_with_save["save_model"] = temp_model_file

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
        
        y_pred_eval = extract_predictions_from_result(result)
        final_eval_preds += np.array(y_pred_eval) * (best_model_weights[model_name] / total_weight)

        best_states[model_name] = {
            "model_type": model_name,
            "model_file": temp_model_file,
            "params": params,
        }

    y_true_eval = parsed_data.y[eval_range]
    metrics = {"rmse": rmse(y_true_eval, final_eval_preds), "mae": mae(y_true_eval, final_eval_preds),
               "mape": mape(y_true_eval, final_eval_preds), "r2": r2(y_true_eval, final_eval_preds)}

    final_residuals = y_true_eval - final_eval_preds
    std_dev_residuals = float(np.std(final_residuals, ddof=1)) if len(final_residuals) > 1 else 0.0

    return metrics, best_chain, best_states, best_model_weights, y_true_eval.tolist(), final_eval_preds.tolist(), std_dev_residuals, search_summary


# ---------- public entry ----------
def run(args) -> None:
    """Main entry point for the Boosting model."""
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if load_path:
        steps = int(getattr(args, "forecast_steps", 0))
        preds, std_dev = predict(load_path, steps)
        out = {"mode": "boosting_loaded_predict", "forecast_steps": steps,
               "predictions": _format_predictions(preds, std_dev)}
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

    # Auto-select conservative defaults based on training data size (for educational small datasets)
    train_size = te - ts + 1
    auto_window = max(2, min(3, train_size // 3))
    auto_max_pq = max(1, min(2, train_size // 4))
    auto_look_back = max(2, min(3, train_size // 3))
    auto_epochs = 20 if train_size <= 15 else 30  # Conservative for ensemble base models

    model_params = {}

    if "arima" in models_to_use:
        model_params["arima"] = {
            "d": get_arg("d", 0, int),
            "max_p": get_arg("max_p", auto_max_pq, int),
            "max_q": get_arg("max_q", auto_max_pq, int)
        }

    if "es" in models_to_use:
        model_params["es"] = {"alpha": get_arg("alpha", 0.5, float)}

    if "ma" in models_to_use:
        model_params["ma"] = {"window": get_arg("window", auto_window, int)}

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
            "look_back": get_arg("look_back", auto_look_back, int),
            "epochs": get_arg("epochs", auto_epochs, int),
            "feature_types": feature_types,
            "normalization": get_arg("normalization", "minmax")
        }

    _validate_training_args(parsed_data, ts, te, es, ee, models_to_use, model_params)

    metrics, chain, states, weights, y_true, y_pred, std_dev, summary = train_and_evaluate(
        parsed_data, range(ts, te + 1), range(es, ee + 1), models_to_use, model_params
    )

    out = {
        "mode": "boosting_greedy_search_adaboost",
        "best_model_chain": chain,
        "model_weights": weights,
        "metrics": metrics,
        "search_summary": summary,
        "eval_y_true": y_true,
        "predictions": _format_predictions(y_pred, std_dev),
    }

    if save_path:
        import shutil

        # Move base model files from temp to permanent location
        base_dir = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]

        for model_name, model_state in states.items():
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

        state = {"model_type": MODEL_TYPE, "model_chain": chain,
                 "base_model_states": states, "model_weights": weights, "std_dev_residuals": std_dev}
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))