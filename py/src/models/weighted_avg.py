#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted Average Ensemble Model using Inverse Variance weighting.
Combines predictions from a flexible set of base models.

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

    if len(models_to_use) < 2:
        raise ValueError("[VALIDATION] Weighted Average requires at least 2 base models.")

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
    """Trains and evaluates the Weighted Average model using Inverse Variance Weighting.

    Correct Inverse Variance Weighting Flow (教科书式):
    1. Split train_range into weight_train (80%) and weight_val (20%)
    2. Train base models on weight_train, predict on weight_val
    3. Calculate weights using weight_val residuals: w_i = 1 / (var_i + eps)
    4. Retrain base models on full train_range
    5. Evaluate weighted combination on independent eval_range

    Algorithm: Inverse Variance Weighting
    - weight_i = 1 / (variance_i + epsilon)
    - normalized_weight_i = weight_i / sum(weight_j)
    - final_prediction = sum(normalized_weight_i * prediction_i)
    """

    # === Step 1: Split train_range into weight_train and weight_val ===
    # This prevents data leakage: weights are calculated on "unseen" weight_val predictions
    train_len = len(train_range)
    if train_len < 8:
        raise ValueError("[VALIDATION] train_range too small for weighted average (需要至少8个点，推荐≥12点)")

    # For small datasets, use 85% for weight_train to ensure sufficient training data
    # For larger datasets, use standard 80% split
    if train_len <= 15:
        weight_split = max(train_len - 3, int(train_len * 0.85))  # Leave at least 3 points for weight_val
    else:
        weight_split = max(3, int(train_len * 0.8))  # At least 3 points for weight_train

    weight_train_range = range(train_range.start, train_range.start + weight_split)
    weight_val_range = range(train_range.start + weight_split, train_range.stop)

    print(f"[Weighted Average] 数据划分:", file=sys.stderr)
    print(f"  权重训练集: {weight_train_range} ({len(weight_train_range)} points)", file=sys.stderr)
    print(f"  权重验证集 (计算权重): {weight_val_range} ({len(weight_val_range)} points)", file=sys.stderr)
    print(f"  评估集: {eval_range} ({len(eval_range)} points)", file=sys.stderr)

    # Non-LSTM base models keep numeric-only rows; LSTM can use categorical features.
    numeric_data_dicts, _ = build_numeric_subprocess_records(parsed_data)
    lstm_data_dicts = numeric_data_dicts
    if "lstm" in models_to_use:
        lstm_feature_types = model_params.get("lstm", {}).get("feature_types", {})
        lstm_feature_keys = [k for k in lstm_feature_types.keys() if k != parsed_data.target_key]
        lstm_data_dicts, _ = build_subprocess_records(parsed_data, feature_keys=lstm_feature_keys)

    # === Step 2: Phase 1 - Train on weight_train, predict on weight_val ===
    print(f"\n[Weighted Average Phase 1] 训练基础模型并生成权重验证集预测", file=sys.stderr)
    weight_val_predictions = {}

    # Calculate dynamic timeout for LSTM models
    base_timeout = 120
    if "lstm" in models_to_use:
        lstm_epochs = model_params["lstm"].get("epochs", 20)
        base_timeout = max(120, lstm_epochs * 10)  # 10 seconds per epoch

    for model_name in models_to_use:
        params = model_params[model_name]
        params_phase1 = params.copy()  # Don't save phase1 models

        print(f"  训练 {model_name} 在 weight_train，预测 weight_val...", file=sys.stderr)
        result = run_base_model_subprocess(
            model_type=model_name,
            data=lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
            train_start=weight_train_range.start,
            train_end=weight_train_range.stop - 1,
            eval_start=weight_val_range.start,  # ✅ 在weight_val上预测（用于计算权重）
            eval_end=weight_val_range.stop - 1,
            model_params=params_phase1,
            timeout=base_timeout,
        )

        y_pred_val = extract_predictions_from_result(result)
        weight_val_predictions[model_name] = y_pred_val
        print(f"    ✓ 生成 {len(y_pred_val)} 个weight_val预测", file=sys.stderr)

    # === Step 3: Calculate weights using Inverse Variance Weighting ===
    print(f"\n[Weighted Average Phase 2] 计算权重（Inverse Variance Weighting）", file=sys.stderr)
    y_true_val = parsed_data.y[weight_val_range]  # ✅ weight_val的真实值
    weights = {}
    total_inverse_variance = 0.0

    print(f"  方差计算 (基于weight_val残差):", file=sys.stderr)
    for name in models_to_use:
        # ✅ 使用weight_val的残差计算方差（避免泄露）
        residuals = y_true_val - np.array(weight_val_predictions[name])
        variance = np.var(residuals)
        inverse_variance = 1.0 / (variance + 1e-9)  # ✅ 保持原有算法
        weights[name] = inverse_variance
        total_inverse_variance += inverse_variance
        print(f"    {name}: variance={variance:.6f}, inverse_variance={inverse_variance:.6f}", file=sys.stderr)

    # Normalize weights
    print(f"  权重归一化:", file=sys.stderr)
    for name in models_to_use:
        weights[name] /= total_inverse_variance  # ✅ 保持原有归一化算法
        print(f"    {name}: normalized_weight={weights[name]:.6f}", file=sys.stderr)

    # === Step 4: Phase 2 - Retrain on full train_range, predict on eval_range ===
    print(f"\n[Weighted Average Phase 3] 在完整训练集上重新训练基础模型", file=sys.stderr)
    eval_predictions = {}
    base_model_states = {}

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

        print(f"  重新训练 {model_name} 在完整train_range，预测eval_range...", file=sys.stderr)
        # Run base model in subprocess on FULL train_range
        result = run_base_model_subprocess(
            model_type=model_name,
            data=lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
            train_start=train_range.start,  # ✅ 完整训练集
            train_end=train_range.stop - 1,
            eval_start=eval_range.start,  # ✅ 独立评估集
            eval_end=eval_range.stop - 1,
            model_params=params_with_save,
            timeout=base_timeout,  # Use dynamic timeout
        )

        y_pred_eval = extract_predictions_from_result(result)
        eval_predictions[model_name] = y_pred_eval

        # Store the model file path and parameters
        base_model_states[model_name] = {
            "model_type": model_name,
            "model_file": temp_model_file,
            "params": params,
        }
        print(f"    ✓ 生成 {len(y_pred_eval)} 个eval预测", file=sys.stderr)

    # === Step 5: Final weighted prediction and evaluation on eval_range ===
    print(f"\n[Weighted Average Phase 4] 加权组合和最终评估", file=sys.stderr)
    X_eval = np.array([eval_predictions[name] for name in models_to_use]).T  # ✅ eval_range的预测
    model_weights = np.array([weights[name] for name in models_to_use])
    y_pred_final = np.sum(X_eval * model_weights, axis=1)  # ✅ 加权组合（保持原有算法）

    y_true = parsed_data.y[eval_range]  # ✅ 真实标签
    residuals_final = y_true - y_pred_final
    std_dev_residuals = float(np.std(residuals_final, ddof=1)) if len(residuals_final) > 1 else 0.0

    # ✅ 在独立的eval_range上计算指标
    timestamps = parsed_data.timestamps
    metrics = {
        "rmse": rmse(y_true, y_pred_final),
        "mae": mae(y_true, y_pred_final),
        "mape": mape(y_true, y_pred_final),
        "r2": r2(y_true, y_pred_final),
        "evaluate_indices": list(eval_range),
        "evaluate_months": [timestamps[i] for i in eval_range] if timestamps else [],
    }

    print(f"  ✓ 评估完成: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}", file=sys.stderr)

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

    # Auto-select defaults based on weight_train size
    # (base models will be trained on weight_train in Phase 1, which is ~80-85% of train_range)
    train_size = te - ts + 1
    weight_train_size = max(3, int(train_size * 0.85)) if train_size <= 15 else int(train_size * 0.8)

    # Use weight_train_size for parameter calculation to avoid phase 1 failures
    # Aligned with individual model auto-selection logic
    auto_window = max(3, min(12, weight_train_size // 4))
    auto_look_back = max(3, min(12, weight_train_size // 4))

    # ARIMA: use tiered approach like in arima.py
    if weight_train_size <= 12:
        auto_max_pq = 1
    elif weight_train_size <= 20:
        auto_max_pq = 2
    else:
        auto_max_pq = max(1, min(5, weight_train_size // 6))

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

                # Also move LSTM Keras model and update internal path
                if model_name == "lstm":
                    lstm_pkl = load_model(perm_file)
                    temp_keras = lstm_pkl.get("keras_model_path")
                    if not temp_keras or not os.path.exists(temp_keras):
                        raise FileNotFoundError(
                            f"LSTM Keras model file not found at {temp_keras}"
                        )
                    perm_keras = os.path.join(base_dir, f"{base_name}_base_{model_name}_keras.h5")
                    shutil.move(temp_keras, perm_keras)
                    lstm_pkl["keras_model_path"] = perm_keras
                    dump_model(lstm_pkl, perm_file)

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
