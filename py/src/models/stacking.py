#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacking Ensemble Model for time series forecasting.
Combines predictions from a flexible set of base models using a meta-model.

Uses subprocess to call base models for consistency with API behavior.
"""
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

# Import shared utilities
from data_utils import ParsedData, load_data_from_json
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model
from subprocess_utils import run_base_model_subprocess, run_base_model_predict_subprocess, extract_predictions_from_result

MODEL_TYPE = "stacking"


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

    # Supported base models
    SUPPORTED_BASE_MODELS = ["ma", "es", "arima", "lstm"]

    # Validate models list
    if not models_to_use:
        raise ValueError("[VALIDATION] models_to_use cannot be empty.")

    if len(models_to_use) < 2:
        raise ValueError("[VALIDATION] Stacking requires at least 2 base models.")

    # Validate all models are known
    for model_name in models_to_use:
        if model_name not in SUPPORTED_BASE_MODELS:
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
    """Forecast using a saved Stacking model with subprocess calls."""
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    state = state or _load_state(model_path)
    meta_model = state["meta_model"]
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
    """Trains and evaluates the Stacking model using the correct hold-out approach.

    Correct Stacking Flow (教科书式):
    1. Split train_range into level0_train (80%) and level1_train (20%)
    2. Train base models on level0_train, predict on level1_train
    3. Train meta-model on level1_train predictions (out-of-fold)
    4. Retrain base models on full train_range
    5. Evaluate on independent eval_range

    Returns:
        metrics: Evaluation metrics dictionary
        base_model_states: States of all base models (for saving/loading)
        meta_model: Trained meta-model (LinearRegression)
        y_true: True values from eval_range
        y_pred: Predicted values on eval_range
        std_dev_residuals: Standard deviation of residuals
    """

    # === Step 1: Split train_range into level0 and level1 ===
    # This prevents data leakage: meta-model trains on "unseen" level1 predictions
    train_len = len(train_range)
    if train_len < 8:
        raise ValueError("[VALIDATION] train_range too small for stacking (需要至少8个点，推荐≥12点)")

    # For small datasets, use 85% for level0 to ensure sufficient training data
    # For larger datasets, use standard 80% split
    if train_len <= 15:
        level0_split = max(train_len - 3, int(train_len * 0.85))  # Leave at least 3 points for level1
    else:
        level0_split = max(3, int(train_len * 0.8))  # At least 3 points for level0

    level0_train_range = range(train_range.start, train_range.start + level0_split)
    level1_train_range = range(train_range.start + level0_split, train_range.stop)

    print(f"[Stacking] 数据划分:", file=sys.stderr)
    print(f"  Level-0 训练集: {level0_train_range} ({len(level0_train_range)} points)", file=sys.stderr)
    print(f"  Level-1 训练集 (元模型): {level1_train_range} ({len(level1_train_range)} points)", file=sys.stderr)
    print(f"  评估集: {eval_range} ({len(eval_range)} points)", file=sys.stderr)

    # Convert parsed_data to list of dicts for subprocess
    data_dicts = []
    for i in range(parsed_data.n_obs):
        row = {parsed_data.target_key: float(parsed_data.y[i])}
        # Add feature columns if they exist
        if parsed_data.feature_keys:
            for feat_key in parsed_data.feature_keys:
                row[feat_key] = float(parsed_data.df.iloc[i][feat_key])
        data_dicts.append(row)

    # === Step 2: Phase 1 - Train base models on level0, predict on level1 ===
    print(f"\n[Stacking Phase 1] 训练基础模型（level0）并生成level1预测", file=sys.stderr)
    level1_predictions = []

    # Calculate dynamic timeout for LSTM models
    base_timeout = 120
    if "lstm" in models_to_use:
        lstm_epochs = model_params["lstm"].get("epochs", 20)
        base_timeout = max(120, lstm_epochs * 10)  # 10 seconds per epoch

    for model_name in models_to_use:
        params = model_params[model_name]

        # We don't save models from phase 1 (temporary training)
        params_phase1 = params.copy()
        # Don't save phase1 models

        print(f"  训练 {model_name} 在 level0，预测 level1...", file=sys.stderr)
        result = run_base_model_subprocess(
            model_type=model_name,
            data=data_dicts,
            train_start=level0_train_range.start,
            train_end=level0_train_range.stop - 1,
            eval_start=level1_train_range.start,  # ✅ 在level1上预测（out-of-fold）
            eval_end=level1_train_range.stop - 1,
            model_params=params_phase1,
            timeout=base_timeout,
        )

        y_pred_level1 = extract_predictions_from_result(result)
        level1_predictions.append(y_pred_level1)
        print(f"    ✓ 生成 {len(y_pred_level1)} 个level1预测", file=sys.stderr)

    # === Step 3: Train meta-model on level1 predictions ===
    print(f"\n[Stacking Phase 2] 训练元模型（在level1预测上）", file=sys.stderr)
    X_meta_train = np.array(level1_predictions).T  # ✅ level1的out-of-fold预测
    y_meta_train = parsed_data.y[level1_train_range]  # ✅ level1的标签

    meta_model = LinearRegression()
    meta_model.fit(X_meta_train, y_meta_train)  # ✅ 在独立的level1集上训练
    print(f"  ✓ 元模型训练完成 (输入维度: {X_meta_train.shape})", file=sys.stderr)

    # === Step 4: Phase 2 - Retrain base models on full train_range ===
    print(f"\n[Stacking Phase 3] 在完整训练集上重新训练基础模型", file=sys.stderr)
    eval_predictions = []
    base_model_states = {}

    for model_name in models_to_use:
        params = model_params[model_name]

        # Create a temporary file for this base model
        # Note: The file will be moved to proper location in run() if stacking model is saved
        temp_model_file = os.path.join(
            tempfile.gettempdir(),
            f"stacking_base_{model_name}_{os.getpid()}.pkl"
        )

        # Add save_model to params for subprocess
        params_with_save = params.copy()
        params_with_save["save_model"] = temp_model_file

        print(f"  重新训练 {model_name} 在完整train_range，预测eval_range...", file=sys.stderr)
        # Run base model in subprocess on FULL train_range
        result = run_base_model_subprocess(
            model_type=model_name,
            data=data_dicts,
            train_start=train_range.start,  # ✅ 完整训练集
            train_end=train_range.stop - 1,
            eval_start=eval_range.start,  # ✅ 独立评估集
            eval_end=eval_range.stop - 1,
            model_params=params_with_save,
            timeout=base_timeout,  # Use dynamic timeout
        )

        y_pred_eval = extract_predictions_from_result(result)
        eval_predictions.append(y_pred_eval)

        # Store the model file path and parameters
        base_model_states[model_name] = {
            "model_type": model_name,
            "model_file": temp_model_file,  # Will be relocated in run() if saved
            "params": params,
        }
        print(f"    ✓ 生成 {len(y_pred_eval)} 个eval预测", file=sys.stderr)

    # === Step 5: Final prediction and evaluation on eval_range ===
    print(f"\n[Stacking Phase 4] 元模型最终预测和评估", file=sys.stderr)
    X_meta_eval = np.array(eval_predictions).T  # ✅ eval_range的预测
    y_pred_final = meta_model.predict(X_meta_eval)  # ✅ 元模型在独立测试集上预测
    y_true_final = parsed_data.y[eval_range]  # ✅ 真实标签

    # Calculate residuals and standard deviation
    residuals = y_true_final - y_pred_final
    std_dev_residuals = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    # ✅ 在独立的eval_range上计算指标
    metrics = {
        "rmse": rmse(y_true_final, y_pred_final),
        "mae": mae(y_true_final, y_pred_final),
        "mape": mape(y_true_final, y_pred_final),
        "r2": r2(y_true_final, y_pred_final),
    }

    print(f"  ✓ 评估完成: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}", file=sys.stderr)

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

    # Auto-select conservative defaults based on level0 training data size
    # (base models will be trained on level0 in Phase 1, which is ~80-85% of train_range)
    train_size = te - ts + 1
    level0_size = max(3, int(train_size * 0.85)) if train_size <= 15 else int(train_size * 0.8)

    # Use level0_size for parameter calculation to avoid phase 1 failures
    # Align with individual model auto-selection logic
    auto_window = max(3, min(12, level0_size // 4))
    auto_look_back = max(3, min(12, level0_size // 4))

    # ARIMA: use tiered approach like in arima.py
    if level0_size <= 12:
        auto_max_pq = 1
    elif level0_size <= 20:
        auto_max_pq = 2
    else:
        auto_max_pq = max(1, min(5, level0_size // 6))

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
            "model_type": MODEL_TYPE,
            "models": models_to_use,
            "base_model_states": base_model_states,
            "meta_model": meta_model,
            "std_dev_residuals": std_dev,
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))