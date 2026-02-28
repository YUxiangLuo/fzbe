#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient-like Boosting Ensemble for Heterogeneous Time Series Models.

Uses residual learning to implement boosting for heterogeneous models (ARIMA, LSTM, MA, ES).
Each model in the chain learns to predict the residual errors from previous models.

Algorithm:
1. Train first model to predict y
2. Calculate residual = y - prediction_1
3. Train second model to predict residual
4. Update: residual = residual - learning_rate * prediction_2
5. Repeat until no improvement or max_rounds reached
6. Final prediction = prediction_1 + learning_rate * (prediction_2 + prediction_3 + ...)

This approach allows boosting with models that don't support sample weights.
Uses subprocess to call base models for consistency with API behavior.
"""
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import shared utilities
from data_utils import (
    ParsedData,
    build_numeric_subprocess_records,
    build_subprocess_records,
    infer_feature_types,
    load_data_from_json,
)
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
    """Validate training and evaluation parameters for Boosting."""
    n = parsed_data.n_obs

    # Supported base models
    SUPPORTED_BASE_MODELS = ["ma", "es", "arima", "lstm"]

    if not models_to_use:
        raise ValueError("[VALIDATION] models_to_use cannot be empty.")

    for name in models_to_use:
        if name not in SUPPORTED_BASE_MODELS:
            raise ValueError(f"[VALIDATION] Unknown model: {name}")

    if not (0 <= ts <= te < n):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], n_obs={n}")
    if not (0 <= es <= ee < n):
        raise ValueError(f"[VALIDATION] Invalid eval range: [{es},{ee}], n_obs={n}")
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


def _create_residual_data(
    original_data: List[Dict[str, Any]],
    residual_values: np.ndarray,
    indices: range,
    target_key: str = "y"
) -> List[Dict[str, Any]]:
    """
    Create new dataset with residual as target.

    This allows training models to predict residuals without modifying base model code.
    """
    new_data = []
    for i, idx in enumerate(indices):
        new_row = original_data[idx].copy()
        new_row[target_key] = float(residual_values[i])
        new_data.append(new_row)

    # Pad with original data before and after the residual range
    # (base models need context, but will only evaluate on the specified range)
    full_data = original_data[:indices.start] + new_data + original_data[indices.stop:]
    return full_data


# ---------- predict ----------
def predict(
    model_path: str, steps: int, state: Optional[Dict[str, Any]] = None
) -> Tuple[List[float], float]:
    """Forecast using a saved Boosting model with subprocess calls."""
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    state = state or _load_state(model_path)
    model_chain = state["model_chain"]
    base_model_states = state["base_model_states"]
    learning_rate = state["learning_rate"]
    std_dev_residuals = state.get("std_dev_residuals", 0.0)

    # First model predicts y
    first_model = model_chain[0]
    first_state = base_model_states[first_model]["models"][0]
    first_file = first_state["model_file"]

    result = run_base_model_predict_subprocess(
        model_type=first_model,
        model_file=first_file,
        forecast_steps=steps,
        timeout=60,
    )
    final_predictions = np.array(extract_predictions_from_result(result))

    # Subsequent models predict residuals
    for i in range(1, len(model_chain)):
        model_name = model_chain[i]
        # Each model is only selected once, so models list has only one element at index 0
        model_state = base_model_states[model_name]["models"][0]
        model_file = model_state["model_file"]

        result = run_base_model_predict_subprocess(
            model_type=model_name,
            model_file=model_file,
            forecast_steps=steps,
            timeout=60,
        )

        residual_pred = np.array(extract_predictions_from_result(result))
        final_predictions += learning_rate * residual_pred

    return final_predictions.tolist(), std_dev_residuals


# ---------- train and evaluate ----------
def train_and_evaluate(
    parsed_data: ParsedData, train_range: range, eval_range: range,
    models_to_use: List[str], model_params: Dict[str, Any], learning_rate: float,
    max_rounds: int
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any], List[float], List[float], float, Dict[str, Any]]:
    """
    Trains and evaluates the Boosting model using Residual Learning.

    Correct Residual Boosting Flow:
    1. Split train_range into boost_train (80-85%) and boost_val (20-15%)
    2. Round 1: Train all candidate models to predict y on boost_train, evaluate on boost_val
    3. Select best model, add to chain
    4. For subsequent rounds:
       - Calculate residual = y - current_ensemble_prediction
       - Train all candidate models to predict residual
       - Select best model
       - If adding it improves RMSE on boost_val: add to chain
       - Else: early stopping
    5. Retrain entire chain on full train_range
    6. Evaluate on independent eval_range

    Args:
        learning_rate: Shrinkage parameter (0 < lr <= 1), typically 0.1-0.5
        max_rounds: Maximum number of models in chain
    """

    # === Step 1: Split train_range into boost_train and boost_val ===
    train_len = len(train_range)
    if train_len < 8:
        raise ValueError("[VALIDATION] train_range too small for boosting (需要至少8个点，推荐≥12点)")

    # For small datasets, use 85% for boost_train to ensure sufficient training data
    if train_len <= 15:
        boost_split = max(train_len - 3, int(train_len * 0.85))
    else:
        boost_split = max(3, int(train_len * 0.8))

    boost_train_range = range(train_range.start, train_range.start + boost_split)
    boost_val_range = range(train_range.start + boost_split, train_range.stop)

    print(f"[Boosting] 数据划分:", file=sys.stderr)
    print(f"  Boost Train Range: {boost_train_range} ({len(boost_train_range)} points)", file=sys.stderr)
    print(f"  Boost Val Range: {boost_val_range} ({len(boost_val_range)} points)", file=sys.stderr)
    print(f"  Eval Range: {eval_range} ({len(eval_range)} points)", file=sys.stderr)
    print(f"  Learning Rate: {learning_rate}", file=sys.stderr)
    print(f"  Max Rounds: {max_rounds}", file=sys.stderr)

    # Non-LSTM base models keep numeric-only rows; LSTM can use categorical features.
    numeric_data_dicts, _ = build_numeric_subprocess_records(parsed_data)
    lstm_data_dicts = numeric_data_dicts
    if "lstm" in models_to_use:
        lstm_feature_types = model_params.get("lstm", {}).get("feature_types", {})
        lstm_feature_keys = [k for k in lstm_feature_types.keys() if k != parsed_data.target_key]
        lstm_data_dicts, _ = build_subprocess_records(parsed_data, feature_keys=lstm_feature_keys)

    y_true_val = parsed_data.y[boost_val_range]
    current_residual_val = y_true_val.copy()  # Initial residual = y
    ensemble_pred_val = np.zeros(len(boost_val_range))

    model_chain = []
    chain_predictions_val = []  # Store each model's prediction for debugging
    search_summary = {"stages": [], "learning_rate": learning_rate}
    best_overall_rmse = float('inf')

    # Calculate dynamic timeout for LSTM models
    base_timeout = 120
    if "lstm" in models_to_use:
        lstm_epochs = model_params["lstm"].get("epochs", 20)
        base_timeout = max(120, lstm_epochs * 10)

    # === Step 2: Greedy Boosting Rounds ===
    print(f"\n[Boosting Phase 1] Residual learning with greedy selection", file=sys.stderr)

    for round_num in range(max_rounds):
        print(f"\n--- Round {round_num + 1}/{max_rounds} ---", file=sys.stderr)

        # Prepare training target description for this round.
        if round_num == 0:
            train_target_description = "original y"
        else:
            train_target_description = f"residual (round {round_num})"

        print(f"  Target: {train_target_description}", file=sys.stderr)
        print(f"  Current RMSE on val: {rmse(y_true_val, ensemble_pred_val) if round_num > 0 else 'N/A'}", file=sys.stderr)

        # Train all candidate models
        stage_results = []
        available_models = [m for m in models_to_use if m not in model_chain]

        if not available_models:
            print(f"  ⚠️  No more candidate models available", file=sys.stderr)
            break

        for model_name in available_models:
            params = model_params[model_name]

            if round_num == 0:
                # First round: train on original data
                result = run_base_model_subprocess(
                    model_type=model_name,
                    data=lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
                    train_start=boost_train_range.start,
                    train_end=boost_train_range.stop - 1,
                    eval_start=boost_val_range.start,
                    eval_end=boost_val_range.stop - 1,
                    model_params=params,
                    timeout=base_timeout,
                )
            else:
                # Subsequent rounds: train on residual
                # Create dataset where boost_train predicts current residual
                residual_train = current_residual_val  # Approximate
                residual_data = _create_residual_data(
                    lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
                    residual_train,
                    boost_val_range,
                    parsed_data.target_key,
                )

                result = run_base_model_subprocess(
                    model_type=model_name,
                    data=residual_data,
                    train_start=boost_train_range.start,
                    train_end=boost_train_range.stop - 1,
                    eval_start=boost_val_range.start,
                    eval_end=boost_val_range.stop - 1,
                    model_params=params,
                    timeout=base_timeout,
                )

            y_pred_val = np.array(extract_predictions_from_result(result))

            # Calculate what the new ensemble would predict
            if round_num == 0:
                temp_ensemble_pred = y_pred_val
            else:
                temp_ensemble_pred = ensemble_pred_val + learning_rate * y_pred_val

            # Calculate RMSE
            temp_rmse = rmse(y_true_val, temp_ensemble_pred)

            stage_results.append({
                "model_name": model_name,
                "rmse": temp_rmse,
                "predictions": y_pred_val,
                "ensemble_pred": temp_ensemble_pred
            })

            print(f"  {model_name}: RMSE={temp_rmse:.4f}", file=sys.stderr)

        # Select best model
        best_result = min(stage_results, key=lambda x: x["rmse"])
        winner_name = best_result["model_name"]
        winner_rmse = best_result["rmse"]
        winner_pred = best_result["predictions"]
        winner_ensemble_pred = best_result["ensemble_pred"]

        # Greedy selection: only add if it improves
        if winner_rmse < best_overall_rmse:
            # Improvement! Add to chain
            model_chain.append(winner_name)
            chain_predictions_val.append(winner_pred)
            ensemble_pred_val = winner_ensemble_pred

            # Update residual for next round
            if round_num == 0:
                current_residual_val = y_true_val - ensemble_pred_val
            else:
                current_residual_val = current_residual_val - learning_rate * winner_pred

            improvement = best_overall_rmse - winner_rmse if best_overall_rmse != float('inf') else 0
            best_overall_rmse = winner_rmse

            stage_summary = {
                "round": round_num + 1,
                "candidates": {r["model_name"]: r["rmse"] for r in stage_results},
                "winner": winner_name,
                "winner_rmse": winner_rmse,
                "improvement": improvement,
                "action": "✅ Model Added",
                "chain": model_chain.copy()
            }

            print(f"  ✅ Added {winner_name} to chain (RMSE: {winner_rmse:.4f}, improvement: {improvement:.4f})", file=sys.stderr)
        else:
            # No improvement, early stopping
            stage_summary = {
                "round": round_num + 1,
                "candidates": {r["model_name"]: r["rmse"] for r in stage_results},
                "winner": winner_name,
                "winner_rmse": winner_rmse,
                "current_best_rmse": best_overall_rmse,
                "action": "🛑 Early Stopping (no improvement)",
                "chain": model_chain.copy()
            }

            print(f"  🛑 Early stopping: {winner_name} (RMSE: {winner_rmse:.4f}) does not improve {best_overall_rmse:.4f}", file=sys.stderr)
            search_summary["stages"].append(stage_summary)
            break

        search_summary["stages"].append(stage_summary)

    # === Step 3: Retrain final chain on full train_range ===
    print(f"\n[Boosting Phase 2] Retraining {len(model_chain)} models on full train_range", file=sys.stderr)

    if not model_chain:
        # No models were added
        y_true_eval = parsed_data.y[eval_range]
        metrics = {"rmse": float('inf'), "mae": float('inf'), "mape": float('inf'), "r2": float('-inf')}
        return metrics, [], {}, y_true_eval.tolist(), [0.0] * len(y_true_eval), 0.0, search_summary

    base_model_states = {}
    final_eval_preds = np.zeros(len(eval_range))
    current_residual_eval = parsed_data.y[eval_range].copy()

    for round_idx, model_name in enumerate(model_chain):
        params = model_params[model_name]
        temp_model_file = os.path.join(
            tempfile.gettempdir(),
            f"boosting_base_{model_name}_round{round_idx}_{os.getpid()}.pkl"
        )
        params_with_save = params.copy()
        params_with_save["save_model"] = temp_model_file

        if round_idx == 0:
            # First model: predict original y
            print(f"  Round {round_idx + 1}: Training {model_name} to predict y", file=sys.stderr)
            result = run_base_model_subprocess(
                model_type=model_name,
                data=lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
                train_start=train_range.start,
                train_end=train_range.stop - 1,
                eval_start=eval_range.start,
                eval_end=eval_range.stop - 1,
                model_params=params_with_save,
                timeout=base_timeout,
            )

            y_pred_eval = np.array(extract_predictions_from_result(result))
            final_eval_preds = y_pred_eval
            current_residual_eval = parsed_data.y[eval_range] - final_eval_preds
        else:
            # Subsequent models: predict residual
            print(f"  Round {round_idx + 1}: Training {model_name} to predict residual", file=sys.stderr)

            # Create residual dataset for training
            residual_data_train = _create_residual_data(
                lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
                current_residual_eval,
                eval_range,
                parsed_data.target_key,
            )

            # Train on train_range to predict residual, evaluate on eval_range
            result = run_base_model_subprocess(
                model_type=model_name,
                data=residual_data_train,
                train_start=train_range.start,
                train_end=train_range.stop - 1,
                eval_start=eval_range.start,
                eval_end=eval_range.stop - 1,
                model_params=params_with_save,
                timeout=base_timeout,
            )

            residual_pred_eval = np.array(extract_predictions_from_result(result))
            final_eval_preds += learning_rate * residual_pred_eval
            current_residual_eval -= learning_rate * residual_pred_eval

        # Store model state
        if model_name not in base_model_states:
            base_model_states[model_name] = {"models": []}

        base_model_states[model_name]["models"].append({
            "model_type": model_name,
            "model_file": temp_model_file,
            "params": params,
            "round": round_idx,
            "role": "predict_y" if round_idx == 0 else "predict_residual"
        })

        print(f"    ✓ RMSE on eval: {rmse(parsed_data.y[eval_range], final_eval_preds):.4f}", file=sys.stderr)

    # === Step 4: Final evaluation ===
    print(f"\n[Boosting Phase 3] Final evaluation on eval_range", file=sys.stderr)
    y_true_eval = parsed_data.y[eval_range]

    metrics = {
        "rmse": rmse(y_true_eval, final_eval_preds),
        "mae": mae(y_true_eval, final_eval_preds),
        "mape": mape(y_true_eval, final_eval_preds),
        "r2": r2(y_true_eval, final_eval_preds)
    }

    final_residuals = y_true_eval - final_eval_preds
    std_dev_residuals = float(np.std(final_residuals, ddof=1)) if len(final_residuals) > 1 else 0.0

    print(f"  ✓ Final RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}", file=sys.stderr)
    print(f"  ✓ Model chain: {' → '.join(model_chain)}", file=sys.stderr)

    return metrics, model_chain, base_model_states, y_true_eval.tolist(), final_eval_preds.tolist(), std_dev_residuals, search_summary


# ---------- public entry ----------
def run(args) -> None:
    """Main entry point for the Boosting model."""
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if load_path:
        steps = int(getattr(args, "forecast_steps", 0))
        preds, std_dev = predict(load_path, steps)
        out = {
            "mode": "boosting_loaded_predict",
            "forecast_steps": steps,
            "predictions": _format_predictions(preds, std_dev)
        }
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # Training Mode
    parsed_data = load_data_from_json(args.data)
    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)
    models_to_use = [model.strip() for model in args.models.split(',')]

    # Boosting-specific parameters
    def get_arg(name, default, cast_type=None):
        """Safely get argument with default value"""
        val = getattr(args, name.replace("-", "_"), None)
        if val is None:
            return default
        if cast_type:
            return cast_type(val)
        return val

    learning_rate = get_arg("learning_rate", 0.3, float)
    max_rounds = get_arg("max_rounds", len(models_to_use), int)

    # Validate learning_rate
    if not (0 < learning_rate <= 1):
        raise ValueError("[VALIDATION] learning_rate must be in (0, 1]")
    if max_rounds < 1:
        raise ValueError("[VALIDATION] max_rounds must be >= 1")

    # Auto-select conservative defaults based on boost_train size
    # Align with individual model auto-selection logic
    train_size = te - ts + 1
    boost_train_size = max(3, int(train_size * 0.85)) if train_size <= 15 else int(train_size * 0.8)

    auto_window = max(3, min(12, boost_train_size // 4))
    auto_look_back = max(3, min(12, boost_train_size // 4))

    # ARIMA: use tiered approach like in arima.py
    if boost_train_size <= 12:
        auto_max_pq = 1
    elif boost_train_size <= 20:
        auto_max_pq = 2
    else:
        auto_max_pq = max(1, min(5, boost_train_size // 6))

    auto_epochs = 20 if train_size <= 15 else 30

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
        feature_types_str = get_arg("feature_types", None)
        lstm_features_str = get_arg("lstm_features", None)
        target_feature = get_arg("lstm_target_feature", parsed_data.target_key)
        lstm_feature_names = (
            [f.strip() for f in lstm_features_str.split(",") if f.strip()]
            if lstm_features_str
            else parsed_data.feature_keys
        )

        if feature_types_str:
            feature_types = json.loads(feature_types_str)
        else:
            feature_types = infer_feature_types(
                parsed_data=parsed_data,
                target_key=target_feature,
                feature_keys=lstm_feature_names,
            )

        if lstm_feature_names:
            allowed_keys = set(lstm_feature_names + [target_feature])
            feature_types = {k: v for k, v in feature_types.items() if k in allowed_keys}
        feature_types[target_feature] = "numeric"

        lstm_params = {
            "look_back": get_arg("look_back", auto_look_back, int),
            "epochs": get_arg("epochs", auto_epochs, int),
            "feature_types": feature_types,
            "normalization": get_arg("normalization", "minmax"),
            "lstm_target_feature": target_feature,
        }
        if lstm_features_str:
            lstm_params["lstm_features"] = lstm_features_str

        model_params["lstm"] = lstm_params

    _validate_training_args(parsed_data, ts, te, es, ee, models_to_use, model_params)

    metrics, chain, states, y_true, y_pred, std_dev, summary = train_and_evaluate(
        parsed_data, range(ts, te + 1), range(es, ee + 1), models_to_use, model_params,
        learning_rate, max_rounds
    )

    out = {
        "mode": "boosting_residual_learning",
        "models": models_to_use,
        "model_chain": chain,
        "learning_rate": learning_rate,
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

        for model_name, model_info in states.items():
            for idx, model_state in enumerate(model_info["models"]):
                temp_file = model_state["model_file"]

                # Define permanent location with round number
                perm_file = os.path.join(base_dir, f"{base_name}_base_{model_name}_round{idx}.pkl")

                # Move file from temp to permanent
                if os.path.exists(temp_file):
                    shutil.move(temp_file, perm_file)

                    # Also move LSTM Keras model if it exists
                    if model_name == "lstm":
                        temp_keras = os.path.splitext(temp_file)[0] + "_lstm_keras_model.h5"
                        if os.path.exists(temp_keras):
                            perm_keras = os.path.join(base_dir, f"{base_name}_base_{model_name}_round{idx}_keras.h5")
                            shutil.move(temp_keras, perm_keras)

                # Update state with permanent path
                model_state["model_file"] = perm_file

        state = {
            "model_type": MODEL_TYPE,
            "model_chain": chain,
            "base_model_states": states,
            "learning_rate": learning_rate,
            "std_dev_residuals": std_dev
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
