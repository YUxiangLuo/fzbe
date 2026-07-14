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
from data_utils import (
    ParsedData,
    infer_feature_types,
    load_data_from_json,
)
from ensemble_utils import (
    build_ensemble_subprocess_data,
    can_refit_source_artifact_for_internal_holdout,
    compute_phase1_train_size,
    estimate_residual_std,
    fit_and_predict_from_source_model,
    format_predictions,
    get_lstm_source_forecast_strategy,
    get_member_model_path,
    load_lstm_source_state,
    normalize_demand_forecast,
    parse_base_model_files_arg,
    parse_base_model_names,
    parse_feature_name_list,
    persist_member_model_artifact,
    prepare_lstm_source_model_for_requested_steps,
    resolve_base_model_artifact_configs,
    validate_base_model_names,
    validate_internal_split_base_params,
)
from metrics import mae, mape, r2, rmse
from model_info import attach_method_metadata, ensure_state_metadata
from serialization import dump_model, load_model
from subprocess_utils import (
    extract_predictions_from_result,
    run_base_model_fit_subprocess,
    run_base_model_predict_subprocess,
    run_base_model_subprocess,
)

MODEL_TYPE = "stacking"
METHOD_NAME = "时间顺序留出 Stacking 融合"
FORECAST_STRATEGY = "holdout_meta_weighted_blend"
IMPLEMENTATION_NOTES = [
    "教科书 Stacking 用基础模型预测作为二级特征，再训练元学习器组合预测。",
    "本系统按时间顺序拆分：训练点 n<=15 时 Level-0 长度为 max(n-3, floor(0.85n))（本项目最少 n=8，因而 Level-1 为 2-3 点），n>15 时 Level-0 为 floor(0.8n)；实体销量成员预测和最终组合结果均截断到不小于 0。",
    "Level-0 重训沿用 MA 窗口、ES 的 alpha，以及 LSTM 的 look_back、epochs、特征和归一化配置；LSTM 动态结构按当前子训练段与 Level-1 跨度依同一规则重算。ARIMA 只沿用用户固定的 d，并在 Level-0 重新执行 AIC/BIC stepwise 搜索，避免把完整训练段选出的 p、q 泄漏给 Level-1；独立评估再复用完整训练产物。",
    "元模型先拟合无截距、非负线性回归，再把系数归一化为和为 1 的组合权重；这是为了小样本稳定性和可解释性采用的凸混合近似，并非教科书对 Stacking 元学习器的必要限制。",
    "当 Level-1 样本数不大于基础模型数加 1、预测矩阵奇异、条件数非有限或大于 1e6 时，回退为 inverse-MAE 权重；此时实际退化为逆误差加权，原因记录在 meta_model 字段。",
    "std_dev 优先取独立评估段残差的样本标准差；不足 2 点时退回 Level-1 组合残差。该常数不随预测步长增长。",
]


# ---------- utilities ----------
_parse_feature_name_list = parse_feature_name_list
_load_lstm_source_state = load_lstm_source_state
_get_lstm_source_forecast_strategy = get_lstm_source_forecast_strategy
_prepare_lstm_source_model_for_requested_steps = prepare_lstm_source_model_for_requested_steps
_format_predictions = format_predictions
_compute_phase1_train_size = compute_phase1_train_size


def _normalize_meta_weights(raw_weights: np.ndarray, n_models: int) -> np.ndarray:
    """Normalize non-negative meta weights and fall back to uniform weights when degenerate."""
    weights = np.asarray(raw_weights, dtype=float).reshape(-1)
    if weights.size != n_models:
        raise ValueError(
            f"[VALIDATION] Meta-model weight size mismatch: expected {n_models}, got {weights.size}."
        )

    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 1e-12:
        return np.full(n_models, 1.0 / n_models, dtype=float)

    return weights / total


def _compute_inverse_mae_weights(X_meta_train: np.ndarray, y_meta_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute stable ensemble weights from level1 MAE."""
    abs_errors = np.abs(X_meta_train - y_meta_train[:, np.newaxis])
    model_mae = np.mean(abs_errors, axis=0)
    inverse_mae = 1.0 / (model_mae + 1e-9)
    return _normalize_meta_weights(inverse_mae, X_meta_train.shape[1]), model_mae


def _json_safe_optional_float(value: float) -> Optional[float]:
    """Return finite floats for JSON payloads, and None for inf/nan diagnostics."""
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def _fit_meta_model(
    X_meta_train: np.ndarray,
    y_meta_train: np.ndarray,
    model_names: List[str],
) -> Dict[str, Any]:
    """Fit a stable meta-model for stacking.

    Small level1 samples are common in this project. The ``n_models + 1`` cutoff is a
    conservative stability margin rather than the parameter count of the no-intercept
    regression. When that guard is hit, or the meta feature matrix is singular or
    strongly collinear, we fall back to inverse-MAE weights computed on level1
    predictions.
    """
    n_samples, n_models = X_meta_train.shape
    if n_models == 0:
        raise ValueError("[VALIDATION] Stacking meta-model requires at least one base model.")

    try:
        condition_number = float(np.linalg.cond(X_meta_train))
    except np.linalg.LinAlgError:
        condition_number = float("inf")

    fallback_reason = None
    if n_samples <= n_models + 1:
        fallback_reason = (
            f"meta_samples({n_samples}) <= stability_threshold({n_models + 1})"
        )
    elif not np.isfinite(condition_number):
        fallback_reason = "meta_feature_matrix is singular"
    elif condition_number > 1e6:
        fallback_reason = f"meta_feature_matrix is ill-conditioned ({condition_number:.2e})"

    if fallback_reason:
        weights, model_mae = _compute_inverse_mae_weights(X_meta_train, y_meta_train)
        return {
            "kind": "weighted_blend",
            "strategy": "inverse_mae_fallback",
            "fallback_reason": fallback_reason,
            "condition_number": _json_safe_optional_float(condition_number),
            "model_names": list(model_names),
            "weights": weights.tolist(),
            "level1_mae": model_mae.tolist(),
        }

    regression = LinearRegression(positive=True, fit_intercept=False)
    regression.fit(X_meta_train, y_meta_train)
    raw_coefficients = np.asarray(regression.coef_, dtype=float)
    weights = _normalize_meta_weights(raw_coefficients, n_models)

    return {
        "kind": "weighted_blend",
        "strategy": "normalized_positive_regression",
        "condition_number": _json_safe_optional_float(condition_number),
        "model_names": list(model_names),
        "weights": weights.tolist(),
        "raw_coefficients": raw_coefficients.tolist(),
    }


def _predict_with_meta_model(meta_model: Dict[str, Any], X_meta: np.ndarray) -> np.ndarray:
    """Predict with the persisted convex-blend meta-model."""
    X_meta = np.asarray(X_meta, dtype=float)
    if not isinstance(meta_model, dict) or meta_model.get("kind") != "weighted_blend":
        raise ValueError("[VALIDATION] Saved Stacking meta-model state is invalid.")
    weights = np.asarray(meta_model.get("weights", []), dtype=float).reshape(-1)
    if weights.size != X_meta.shape[1] or not np.isfinite(weights).all():
        raise ValueError(
            "[VALIDATION] Saved Stacking meta-model weights do not match base model count."
        )
    return X_meta @ weights


def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved Stacking model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid Stacking model file.")
    validate_base_model_names(state.get("models"), "Saved Stacking model")
    return ensure_state_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)


def _validate_fit_args(
    parsed_data: ParsedData,
    ts: int,
    te: int,
    state: Dict[str, Any],
    required_forecast_steps: int = 0,
) -> None:
    n = parsed_data.n_obs
    if not (0 <= ts <= te < n):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], n_obs={n}")

    models_to_use = validate_base_model_names(
        state.get("models"),
        "Saved Stacking model",
    )

    base_model_states = state.get("base_model_states")
    if not isinstance(base_model_states, dict):
        raise ValueError("[VALIDATION] Saved Stacking model is missing base model states.")

    if "meta_model" not in state:
        raise ValueError("[VALIDATION] Saved Stacking model is missing meta model state.")

    for model_name in models_to_use:
        model_state = base_model_states.get(model_name)
        if not isinstance(model_state, dict) or not isinstance(model_state.get("model_file"), str):
            raise ValueError(f"[VALIDATION] Missing saved state for base model: {model_name}")

        if model_name != "lstm":
            continue

        from models import lstm as lstm_model

        source_model_state = _load_lstm_source_state(
            model_state["model_file"],
            "Saved Stacking LSTM member",
        )

        target_feature = str(source_model_state.get("target_key") or parsed_data.target_key)
        if target_feature not in parsed_data.df.columns:
            raise ValueError(
                f"[VALIDATION] Stacking fit-only member lstm is missing target column: {target_feature}"
            )
        missing_feature_names = [
            feature_name
            for feature_name in source_model_state.get("feature_keys", []) or []
            if feature_name not in parsed_data.df.columns
        ]
        if missing_feature_names:
            raise ValueError(
                f"[VALIDATION] Stacking fit-only member lstm is missing feature columns: {missing_feature_names}"
            )

        look_back = int(source_model_state["look_back"])
        forecast_strategy = source_model_state.get(
            "forecast_strategy",
            lstm_model.LEGACY_FORECAST_STRATEGY,
        )
        fit_forecast_horizon = 0
        if forecast_strategy == lstm_model.DIRECT_FORECAST_STRATEGY:
            fit_forecast_horizon = (
                required_forecast_steps
                if required_forecast_steps > 0
                else lstm_model._resolve_production_forecast_horizon(state=source_model_state)
            )

        try:
            lstm_model._validate_fit_args(
                parsed_data,
                ts,
                te,
                look_back,
                fit_forecast_horizon,
                forecast_strategy,
            )
        except ValueError as exc:
            message = str(exc)
            if message.startswith("[VALIDATION] "):
                message = message[len("[VALIDATION] "):]
            raise ValueError(
                f"[VALIDATION] Stacking fit-only member lstm cannot be refit: {message}"
            ) from exc


def _validate_training_args(
    parsed_data: ParsedData,
    ts: int,
    te: int,
    es: int,
    ee: int,
    models_to_use: List[str],
    model_params: Dict[str, Any],
    base_model_files: Optional[Dict[str, str]] = None,
) -> None:
    """Validate training and evaluation parameters for Stacking."""
    n = parsed_data.n_obs

    validate_base_model_names(models_to_use, "Stacking")

    # Validate train range
    if not (0 <= ts <= te < n):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], n_obs={n}")

    # Validate eval range
    if not (0 <= es <= ee < n):
        raise ValueError(f"[VALIDATION] Invalid eval range: [{es},{ee}], n_obs={n}")

    # Eval must start after train
    if es <= te:
        raise ValueError("[VALIDATION] eval_start must be > train_end.")

    validate_internal_split_base_params(
        models_to_use, model_params, te - ts + 1, "stacking"
    )

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
            if params["window"] < 2:
                raise ValueError("[VALIDATION] MA window must be >= 2.")

        elif model_name == "lstm":
            source_model_file = (base_model_files or {}).get(model_name)
            source_model_state = None
            if source_model_file:
                source_model_state = _load_lstm_source_state(
                    source_model_file,
                    "Saved Stacking LSTM base model",
                )

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
            train_size = te - ts + 1
            level0_size = _compute_phase1_train_size(train_size)
            level1_size = train_size - level0_size
            if level0_size <= params["look_back"]:
                raise ValueError(
                    f"[VALIDATION] LSTM look_back ({params['look_back']}) is too large for "
                    f"the stacking internal training split ({level0_size} points). "
                    f"Please reduce look_back or increase training data."
                )
            target_feature = str(params.get("lstm_target_feature", parsed_data.target_key) or parsed_data.target_key)
            if target_feature not in parsed_data.df.columns:
                raise ValueError(f"[VALIDATION] LSTM target feature not found in dataset: {target_feature}")
            missing_feature_names = [
                feature_name
                for feature_name in _parse_feature_name_list(params.get("lstm_features"))
                if feature_name not in parsed_data.df.columns
            ]
            if missing_feature_names:
                raise ValueError(
                    f"[VALIDATION] LSTM feature columns not found in dataset: {missing_feature_names}"
                )

            if source_model_state is None:
                if level0_size < params["look_back"] + level1_size:
                    raise ValueError(
                        f"[VALIDATION] LSTM look_back ({params['look_back']}) is too large for "
                        f"the stacking internal training split ({level0_size} points) "
                        f"and level1 horizon ({level1_size} points). Please reduce look_back "
                        f"or increase training data."
                    )
                continue

            from models import lstm as lstm_model

            forecast_strategy = _get_lstm_source_forecast_strategy(source_model_state)
            fit_forecast_horizon = (
                level1_size
                if forecast_strategy == lstm_model.DIRECT_FORECAST_STRATEGY
                else 0
            )

            try:
                lstm_model._validate_fit_args(
                    parsed_data,
                    ts,
                    ts + level0_size - 1,
                    params["look_back"],
                    fit_forecast_horizon,
                    forecast_strategy,
                )
            except ValueError as exc:
                message = str(exc)
                if message.startswith("[VALIDATION] "):
                    message = message[len("[VALIDATION] "):]
                raise ValueError(
                    "[VALIDATION] Reused LSTM base model cannot be refit on the stacking "
                    f"level0 split: {message}"
                ) from exc


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
        base_predictions.append(normalize_demand_forecast(preds).tolist())

    X_meta_future = np.array(base_predictions).T
    final_predictions = _predict_with_meta_model(meta_model, X_meta_future)
    final_predictions = normalize_demand_forecast(final_predictions)

    return final_predictions.tolist(), std_dev_residuals


# ---------- train and evaluate ----------
def train_and_evaluate(
    parsed_data: ParsedData,
    train_range: range,
    eval_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Any],
    base_model_files: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Any, List[float], List[float], float]:
    """Trains and evaluates the Stacking model using a time-ordered hold-out approach.

    Teaching-practical Stacking Flow:
    1. Split train_range into level0_train (80%) and level1_train (20%)
    2. Train base models on level0_train, predict on level1_train
    3. Train meta-model on level1_train hold-out predictions
    4. Retrain base models on full train_range
    5. Evaluate on independent eval_range

    Returns:
        metrics: Evaluation metrics dictionary
        base_model_states: States of all base models (for saving/loading)
        meta_model: Trained meta-model state
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
    level0_split = _compute_phase1_train_size(train_len)

    level0_train_range = range(train_range.start, train_range.start + level0_split)
    level1_train_range = range(train_range.start + level0_split, train_range.stop)

    print("[Stacking] 数据划分:", file=sys.stderr)
    print(f"  Level-0 训练集: {level0_train_range} ({len(level0_train_range)} points)", file=sys.stderr)
    print(f"  Level-1 训练集 (元模型): {level1_train_range} ({len(level1_train_range)} points)", file=sys.stderr)
    print(f"  评估集: {eval_range} ({len(eval_range)} points)", file=sys.stderr)

    # Non-LSTM base models keep numeric-only rows; LSTM can use categorical features.
    numeric_data_dicts, lstm_data_dicts = build_ensemble_subprocess_data(
        parsed_data,
        models_to_use,
        model_params,
    )

    # === Step 2: Phase 1 - Train base models on level0, predict on level1 ===
    print("\n[Stacking Phase 1] 训练基础模型（level0）并生成level1预测", file=sys.stderr)
    level1_predictions = []

    # Calculate dynamic timeout for LSTM models
    base_timeout = 120
    if "lstm" in models_to_use:
        lstm_epochs = model_params["lstm"].get("epochs", 20)
        base_timeout = max(120, lstm_epochs * 10)  # 10 seconds per epoch

    for model_name in models_to_use:
        params = model_params[model_name]
        source_model_file = (base_model_files or {}).get(model_name)

        print(f"  训练 {model_name} 在 level0，预测 level1...", file=sys.stderr)
        if (
            source_model_file
            and can_refit_source_artifact_for_internal_holdout(model_name)
        ):
            adjusted_source_model_file = source_model_file
            temp_source_model_file = None
            try:
                if model_name == "lstm":
                    source_model_state = _load_lstm_source_state(
                        source_model_file,
                        "Saved Stacking LSTM base model",
                    )
                    adjusted_source_model_file, temp_source_model_file = _prepare_lstm_source_model_for_requested_steps(
                        source_model_file,
                        source_model_state,
                        requested_steps=len(level1_train_range),
                    )
                y_pred_level1 = fit_and_predict_from_source_model(
                    model_name=model_name,
                    source_model_file=adjusted_source_model_file,
                    data=lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
                    train_start=level0_train_range.start,
                    train_end=level0_train_range.stop - 1,
                    forecast_steps=len(level1_train_range),
                    timeout=base_timeout,
                )
            finally:
                if temp_source_model_file and os.path.exists(temp_source_model_file):
                    os.remove(temp_source_model_file)
        else:
            result = run_base_model_subprocess(
                model_type=model_name,
                data=lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
                train_start=level0_train_range.start,
                train_end=level0_train_range.stop - 1,
                eval_start=level1_train_range.start,
                eval_end=level1_train_range.stop - 1,
                model_params=params.copy(),
                timeout=base_timeout,
            )

            y_pred_level1 = extract_predictions_from_result(result)
        level1_predictions.append(normalize_demand_forecast(y_pred_level1).tolist())
        print(f"    ✓ 生成 {len(y_pred_level1)} 个level1预测", file=sys.stderr)

    # === Step 3: Train meta-model on level1 predictions ===
    print("\n[Stacking Phase 2] 训练元模型（在level1预测上）", file=sys.stderr)
    X_meta_train = np.column_stack(level1_predictions)  # ✅ level1 的时间顺序留出预测
    y_meta_train = parsed_data.y[level1_train_range]  # ✅ level1的标签

    meta_model = _fit_meta_model(X_meta_train, y_meta_train, models_to_use)

    print(f"  ✓ 元模型训练完成 (输入维度: {X_meta_train.shape})", file=sys.stderr)
    if isinstance(meta_model, dict):
        condition_number = meta_model.get("condition_number")
        condition_label = f"{condition_number:.2e}" if condition_number is not None else "n/a"
        print(
            f"  元模型策略: {meta_model['strategy']} "
            f"(condition={condition_label})",
            file=sys.stderr,
        )
        if meta_model.get("fallback_reason"):
            print(f"  稳定性回退: {meta_model['fallback_reason']}", file=sys.stderr)
        for model_name, weight in zip(meta_model["model_names"], meta_model["weights"]):
            print(f"    {model_name}: weight={float(weight):.6f}", file=sys.stderr)

    # === Step 4: Phase 2 - Retrain base models on full train_range ===
    print("\n[Stacking Phase 3] 在完整训练集上重新训练基础模型", file=sys.stderr)
    eval_predictions = []
    base_model_states = {}
    eval_forecast_steps = eval_range.stop - train_range.stop
    eval_forecast_offset = eval_range.start - train_range.stop

    for model_name in models_to_use:
        params = model_params[model_name]
        source_model_file = (base_model_files or {}).get(model_name)

        if source_model_file:
            if model_name == "lstm":
                source_model_state = _load_lstm_source_state(
                    source_model_file,
                    "Saved Stacking LSTM base model",
                )
                source_forecast_strategy = _get_lstm_source_forecast_strategy(source_model_state)
                source_forecast_horizon = int(source_model_state.get("forecast_horizon", 0) or 0)
                if (
                    source_forecast_strategy == "direct_multi_horizon"
                    and eval_forecast_steps > source_forecast_horizon
                ):
                    adjusted_source_model_file = source_model_file
                    temp_source_model_file = None
                    temp_model_file = os.path.join(
                        tempfile.gettempdir(),
                        f"stacking_base_{model_name}_{os.getpid()}.pkl"
                    )
                    try:
                        adjusted_source_model_file, temp_source_model_file = _prepare_lstm_source_model_for_requested_steps(
                            source_model_file,
                            source_model_state,
                            requested_steps=eval_forecast_steps,
                        )
                        run_base_model_fit_subprocess(
                            model_type=model_name,
                            data=lstm_data_dicts,
                            train_start=train_range.start,
                            train_end=train_range.stop - 1,
                            source_model_file=adjusted_source_model_file,
                            save_model_file=temp_model_file,
                            fit_forecast_steps=eval_forecast_steps,
                            timeout=base_timeout,
                        )
                    finally:
                        if temp_source_model_file and os.path.exists(temp_source_model_file):
                            os.remove(temp_source_model_file)

                    result = run_base_model_predict_subprocess(
                        model_type=model_name,
                        model_file=temp_model_file,
                        forecast_steps=eval_forecast_steps,
                        timeout=base_timeout,
                    )
                    y_pred_eval = extract_predictions_from_result(result)[
                        eval_forecast_offset: eval_forecast_offset + len(eval_range)
                    ]
                    eval_predictions.append(normalize_demand_forecast(y_pred_eval).tolist())
                    base_model_states[model_name] = {
                        "model_type": model_name,
                        "model_file": temp_model_file,
                        "params": params,
                        "artifact_origin": "temporary_training",
                    }
                    print(f"    ✓ 扩展horizon后生成 {len(y_pred_eval)} 个eval预测", file=sys.stderr)
                    continue

            print(f"  复用已完成的 {model_name} 基础模型产物，预测eval_range...", file=sys.stderr)
            result = run_base_model_predict_subprocess(
                model_type=model_name,
                model_file=source_model_file,
                forecast_steps=eval_forecast_steps,
                timeout=base_timeout,
            )
            y_pred_eval = extract_predictions_from_result(result)[
                eval_forecast_offset: eval_forecast_offset + len(eval_range)
            ]
            eval_predictions.append(normalize_demand_forecast(y_pred_eval).tolist())
            base_model_states[model_name] = {
                "model_type": model_name,
                "model_file": source_model_file,
                "params": params,
                "artifact_origin": "standalone_base",
            }
            print(f"    ✓ 生成 {len(y_pred_eval)} 个eval预测", file=sys.stderr)
            continue

        temp_model_file = os.path.join(
            tempfile.gettempdir(),
            f"stacking_base_{model_name}_{os.getpid()}.pkl"
        )

        params_with_save = params.copy()
        params_with_save["save_model"] = temp_model_file

        print(f"  重新训练 {model_name} 在完整train_range，预测eval_range...", file=sys.stderr)
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

        y_pred_eval = extract_predictions_from_result(result)
        eval_predictions.append(normalize_demand_forecast(y_pred_eval).tolist())

        base_model_states[model_name] = {
            "model_type": model_name,
            "model_file": temp_model_file,
            "params": params,
            "artifact_origin": "temporary_training",
        }
        print(f"    ✓ 生成 {len(y_pred_eval)} 个eval预测", file=sys.stderr)

    # === Step 5: Final prediction and evaluation on eval_range ===
    print("\n[Stacking Phase 4] 元模型最终预测和评估", file=sys.stderr)
    X_meta_eval = np.column_stack(eval_predictions)  # ✅ eval_range的预测
    y_pred_final = _predict_with_meta_model(meta_model, X_meta_eval)  # ✅ 元模型在独立测试集上预测
    y_pred_final = normalize_demand_forecast(y_pred_final)
    y_true_final = parsed_data.y[eval_range]  # ✅ 真实标签

    # Calculate residuals and standard deviation
    residuals = y_true_final - y_pred_final
    level1_blend = normalize_demand_forecast(
        _predict_with_meta_model(meta_model, X_meta_train)
    )
    level1_residuals = y_meta_train - level1_blend
    std_dev_residuals = estimate_residual_std(
        residuals,
        level1_residuals,
        parsed_data.y[train_range],
    )

    # ✅ 在独立的eval_range上计算指标
    timestamps = parsed_data.timestamps
    metrics = {
        "rmse": rmse(y_true_final, y_pred_final),
        "mae": mae(y_true_final, y_pred_final),
        "mape": mape(y_true_final, y_pred_final),
        "r2": r2(y_true_final, y_pred_final),
        "evaluate_indices": list(eval_range),
        "evaluate_months": [timestamps[i] for i in eval_range] if timestamps else [],
    }

    print(f"  ✓ 评估完成: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}", file=sys.stderr)

    return metrics, base_model_states, meta_model, y_true_final.tolist(), y_pred_final.tolist(), std_dev_residuals


# ---------- public entry ----------
def run(args) -> None:
    """Main entry point for Stacking model training and prediction."""
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if getattr(args, "fit_only", False):
        if not hasattr(args, "data") or not args.data:
            raise ValueError("[VALIDATION] --data is required for fit-only mode.")
        if not save_path:
            raise ValueError("[VALIDATION] --save-model is required for fit-only mode.")
        if not load_path:
            raise ValueError("[VALIDATION] --load-model is required for fit-only mode.")

        state = _load_state(load_path)
        parsed_data = load_data_from_json(args.data)
        ts, te = int(args.train_start), int(args.train_end)
        _validate_fit_args(
            parsed_data,
            ts,
            te,
            state,
            int(getattr(args, "fit_forecast_steps", 0) or 0),
        )

        models_to_use = state["models"]
        previous_base_model_states = state["base_model_states"]
        model_params = {
            model_name: previous_base_model_states[model_name].get("params", {})
            for model_name in models_to_use
        }
        numeric_data_dicts, lstm_data_dicts = build_ensemble_subprocess_data(
            parsed_data,
            models_to_use,
            model_params,
        )

        next_base_model_states = {}
        required_fit_forecast_steps = int(getattr(args, "fit_forecast_steps", 0) or 0) or None
        for model_name in models_to_use:
            previous_model_state = previous_base_model_states[model_name]
            previous_model_file = previous_model_state["model_file"]
            member_save_path = get_member_model_path(save_path, model_name)
            model_timeout = 120
            if model_name == "lstm":
                model_timeout = max(120, int(model_params.get("lstm", {}).get("epochs", 20)) * 10)

            run_base_model_fit_subprocess(
                model_type=model_name,
                data=lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
                train_start=ts,
                train_end=te,
                source_model_file=previous_model_file,
                save_model_file=member_save_path,
                fit_forecast_steps=required_fit_forecast_steps if model_name == "lstm" else None,
                timeout=model_timeout,
            )

            next_base_model_states[model_name] = {
                "model_type": model_name,
                "model_file": member_save_path,
                "params": model_params.get(model_name, {}),
            }

        next_state = {
            "model_type": MODEL_TYPE,
            "models": models_to_use,
            "base_model_states": next_base_model_states,
            "meta_model": state["meta_model"],
            "std_dev_residuals": float(state.get("std_dev_residuals", 0.0)),
        }
        attach_method_metadata(next_state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(next_state, save_path)

        out = {
            "mode": "stacking_fit_only",
            "models": models_to_use,
            "saved_model": save_path,
        }
        attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

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
        attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # Training Mode
    if not hasattr(args, "data") or not args.data:
        raise ValueError("[VALIDATION] --data is required for training mode.")

    parsed_data = load_data_from_json(args.data)
    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)
    models_to_use = parse_base_model_names(args.models, "Stacking")

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
    level0_size = _compute_phase1_train_size(train_size)

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

    # Keep the no-artifact fallback aligned with standalone LSTM training.
    auto_epochs = 20

    model_params = {}
    base_model_files = parse_base_model_files_arg(get_arg("base_model_files", None))

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
        lstm_feature_names = _parse_feature_name_list(lstm_features_str) if lstm_features_str else parsed_data.feature_keys

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

    model_params, base_model_files = resolve_base_model_artifact_configs(
        parsed_data,
        models_to_use,
        model_params,
        base_model_files,
        "Stacking",
    )

    # Validate all parameters
    _validate_training_args(parsed_data, ts, te, es, ee, models_to_use, model_params, base_model_files)

    metrics, base_model_states, meta_model, y_true, y_pred, std_dev = train_and_evaluate(
        parsed_data,
        range(ts, te + 1),
        range(es, ee + 1),
        models_to_use,
        model_params,
        base_model_files,
    )

    predictions_with_std = _format_predictions(y_pred, std_dev)

    out: Dict[str, Any] = {
        "mode": "stacking_train_eval",
        "models": models_to_use,
        "meta_model": meta_model,
        "metrics": metrics,
        "eval_y_true": y_true,
        "predictions": predictions_with_std,
    }
    attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)

    if save_path:
        for model_name, model_state in base_model_states.items():
            source_file = model_state["model_file"]
            perm_file = get_member_model_path(save_path, model_name)
            persist_member_model_artifact(
                model_name,
                source_file,
                perm_file,
                move_source=model_state.get("artifact_origin") == "temporary_training",
            )
            model_state["model_file"] = perm_file
            model_state.pop("artifact_origin", None)

        state = {
            "model_type": MODEL_TYPE,
            "models": models_to_use,
            "base_model_states": base_model_states,
            "meta_model": meta_model,
            "std_dev_residuals": std_dev,
        }
        attach_method_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
