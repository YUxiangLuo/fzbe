#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted Average Ensemble Model using inverse validation residual-MSE weighting.

This follows the inverse-variance weighting idea, but uses validation residual
MSE so systematic forecast bias is penalized as well as residual fluctuation.
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

MODEL_TYPE = "weighted_avg"
METHOD_NAME = "验证残差MSE倒数加权平均融合"
FORECAST_STRATEGY = "validation_residual_mse_weighted_average"
IMPLEMENTATION_NOTES = [
    "教科书组合预测常用逆方差或误差方差-协方差思想分配权重；当预测误差近似无偏时，MSE 与方差口径一致。",
    "本系统按时间顺序留出末段：训练点 n<=15 时前段长度为 max(n-3, floor(0.85n))（本项目最少 n=8，因而留 2-3 点），n>15 时前段为 floor(0.8n)；再按 1/(MSE+1e-9) 归一化为非负且和为 1 的权重。",
    "该规则会同时惩罚系统性偏差和残差波动，但没有估计模型误差之间的协方差；小验证段上的权重可能不稳定，这是兼容教学小数据的明确妥协。",
    "内部验证沿用 MA 窗口、ES 的 alpha，以及 LSTM 的 look_back、epochs、特征和归一化配置；LSTM 的动态隐藏单元、批大小和直接输出宽度按当前子训练段与验证跨度依照同一规则重算。ARIMA 只沿用用户固定的 d，并在内部前段重新执行 AIC/BIC stepwise 搜索，避免把完整训练段选出的 p、q 泄漏给留出段。",
    "基础模型随后使用完整训练区间的既有匹配产物或重新拟合产物生成评估预测；所有表示实体销量的成员预测会先截断到不小于 0，再按固定权重组合，组合结果也做同样的非负兜底。",
    "std_dev 优先取独立评估段残差的样本标准差；不足 2 点时退回内部权重验证段残差。该常数不随预测步长增长。",
]


# ---------- utilities ----------
_parse_feature_name_list = parse_feature_name_list
_load_lstm_source_state = load_lstm_source_state
_get_lstm_source_forecast_strategy = get_lstm_source_forecast_strategy
_prepare_lstm_source_model_for_requested_steps = prepare_lstm_source_model_for_requested_steps
_format_predictions = format_predictions
_compute_phase1_train_size = compute_phase1_train_size


def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved Weighted Average model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid Weighted Average model file.")
    validate_base_model_names(state.get("models"), "Saved Weighted Average model")
    return ensure_state_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)


def _compute_inverse_residual_mse_weights(
    y_true: np.ndarray,
    predictions_by_model: Dict[str, Any],
) -> Dict[str, float]:
    """
    Estimate inverse-error weights from validation residual MSE.

    This is equivalent to inverse-SSE weighting when every model is evaluated on
    the same validation segment length, and it penalizes systematic bias.
    """
    raw_weights: Dict[str, float] = {}
    total_inverse_error = 0.0

    for model_name, predictions in predictions_by_model.items():
        residuals = np.asarray(y_true, dtype=float) - np.asarray(predictions, dtype=float)
        residual_mse = float(np.mean(np.square(residuals)))
        inverse_error = 1.0 / (residual_mse + 1e-9)
        raw_weights[model_name] = inverse_error
        total_inverse_error += inverse_error

    if total_inverse_error <= 0:
        uniform_weight = 1.0 / max(1, len(raw_weights))
        return {model_name: uniform_weight for model_name in raw_weights}

    return {
        model_name: weight / total_inverse_error
        for model_name, weight in raw_weights.items()
    }


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
        "Saved Weighted Average model",
    )

    base_model_states = state.get("base_model_states")
    if not isinstance(base_model_states, dict):
        raise ValueError("[VALIDATION] Saved Weighted Average model is missing base model states.")

    for model_name in models_to_use:
        model_state = base_model_states.get(model_name)
        if not isinstance(model_state, dict) or not isinstance(model_state.get("model_file"), str):
            raise ValueError(f"[VALIDATION] Missing saved state for base model: {model_name}")

        if model_name != "lstm":
            continue

        from models import lstm as lstm_model

        source_model_state = load_model(model_state["model_file"])
        if not isinstance(source_model_state, dict) or source_model_state.get("model_type") != "lstm":
            raise ValueError("[VALIDATION] Saved Weighted Average LSTM member is invalid.")

        target_feature = str(source_model_state.get("target_key") or parsed_data.target_key)
        if target_feature not in parsed_data.df.columns:
            raise ValueError(
                f"[VALIDATION] Weighted Average fit-only member lstm is missing target column: {target_feature}"
            )
        missing_feature_names = [
            feature_name
            for feature_name in source_model_state.get("feature_keys", []) or []
            if feature_name not in parsed_data.df.columns
        ]
        if missing_feature_names:
            raise ValueError(
                f"[VALIDATION] Weighted Average fit-only member lstm is missing feature columns: {missing_feature_names}"
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
                f"[VALIDATION] Weighted Average fit-only member lstm cannot be refit: {message}"
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
    """Validate training and evaluation parameters for Weighted Average."""
    n = parsed_data.n_obs

    validate_base_model_names(models_to_use, "Weighted Average")

    if not (0 <= ts <= te < n):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], n_obs={n}")
    if not (0 <= es <= ee < n):
        raise ValueError(f"[VALIDATION] Invalid eval range: [{es},{ee}], n_obs={n}")
    if es <= te:
        raise ValueError("[VALIDATION] eval_start must be > train_end.")

    validate_internal_split_base_params(
        models_to_use, model_params, te - ts + 1, "weighted average"
    )

    for model_name in models_to_use:
        if model_name not in model_params:
            raise ValueError(f"[VALIDATION] Missing parameters for model: {model_name}")
        params = model_params[model_name]
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
                    "Saved Weighted Average LSTM base model",
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
            weight_train_size = _compute_phase1_train_size(train_size)
            weight_val_size = train_size - weight_train_size
            if weight_train_size <= params["look_back"]:
                raise ValueError(
                    f"[VALIDATION] LSTM look_back ({params['look_back']}) is too large for "
                    f"the weighted average internal training split ({weight_train_size} points). "
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
                if weight_train_size < params["look_back"] + weight_val_size:
                    raise ValueError(
                        f"[VALIDATION] LSTM look_back ({params['look_back']}) is too large for "
                        f"the weighted average internal training split ({weight_train_size} points) "
                        f"and validation horizon ({weight_val_size} points). Please reduce look_back "
                        f"or increase training data."
                    )
                continue

            from models import lstm as lstm_model

            forecast_strategy = _get_lstm_source_forecast_strategy(source_model_state)
            fit_forecast_horizon = (
                weight_val_size
                if forecast_strategy == lstm_model.DIRECT_FORECAST_STRATEGY
                else 0
            )

            try:
                lstm_model._validate_fit_args(
                    parsed_data,
                    ts,
                    ts + weight_train_size - 1,
                    params["look_back"],
                    fit_forecast_horizon,
                    forecast_strategy,
                )
            except ValueError as exc:
                message = str(exc)
                if message.startswith("[VALIDATION] "):
                    message = message[len("[VALIDATION] "):]
                raise ValueError(
                    "[VALIDATION] Reused LSTM base model cannot be refit on the weighted average "
                    f"internal training split: {message}"
                ) from exc


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
        base_predictions.append(normalize_demand_forecast(preds).tolist())

    X_future = np.array(base_predictions).T
    model_weights = np.array([weights[name] for name in models_to_use])
    final_predictions = np.sum(X_future * model_weights, axis=1)
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
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float], List[float], List[float], float]:
    """Trains and evaluates the Weighted Average model using inverse residual-MSE weighting.

    Time-ordered error-weighted flow:
    1. Split train_range into weight_train (80%) and weight_val (20%)
    2. Train base models on weight_train, predict on weight_val
    3. Calculate weights using weight_val residual MSE: w_i = 1 / (mse_i + eps)
    4. Retrain base models on full train_range
    5. Evaluate weighted combination on independent eval_range

    Algorithm: inverse validation residual-MSE weighting
    (an inverse-variance-inspired rule adapted for potentially biased forecasts)
    - weight_i = 1 / (residual_mse_i + epsilon)
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
    weight_split = _compute_phase1_train_size(train_len)

    weight_train_range = range(train_range.start, train_range.start + weight_split)
    weight_val_range = range(train_range.start + weight_split, train_range.stop)

    print("[Weighted Average] 数据划分:", file=sys.stderr)
    print(f"  权重训练集: {weight_train_range} ({len(weight_train_range)} points)", file=sys.stderr)
    print(f"  权重验证集 (计算权重): {weight_val_range} ({len(weight_val_range)} points)", file=sys.stderr)
    print(f"  评估集: {eval_range} ({len(eval_range)} points)", file=sys.stderr)

    # Non-LSTM base models keep numeric-only rows; LSTM can use categorical features.
    numeric_data_dicts, lstm_data_dicts = build_ensemble_subprocess_data(
        parsed_data,
        models_to_use,
        model_params,
    )

    # === Step 2: Phase 1 - Train on weight_train, predict on weight_val ===
    print("\n[Weighted Average Phase 1] 训练基础模型并生成权重验证集预测", file=sys.stderr)
    weight_val_predictions = {}

    # Calculate dynamic timeout for LSTM models
    base_timeout = 120
    if "lstm" in models_to_use:
        lstm_epochs = model_params["lstm"].get("epochs", 20)
        base_timeout = max(120, lstm_epochs * 10)  # 10 seconds per epoch

    for model_name in models_to_use:
        params = model_params[model_name]
        source_model_file = (base_model_files or {}).get(model_name)

        print(f"  训练 {model_name} 在 weight_train，预测 weight_val...", file=sys.stderr)
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
                        "Saved Weighted Average LSTM base model",
                    )
                    adjusted_source_model_file, temp_source_model_file = _prepare_lstm_source_model_for_requested_steps(
                        source_model_file,
                        source_model_state,
                        requested_steps=len(weight_val_range),
                    )
                y_pred_val = fit_and_predict_from_source_model(
                    model_name=model_name,
                    source_model_file=adjusted_source_model_file,
                    data=lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
                    train_start=weight_train_range.start,
                    train_end=weight_train_range.stop - 1,
                    forecast_steps=len(weight_val_range),
                    timeout=base_timeout,
                )
            finally:
                if temp_source_model_file and os.path.exists(temp_source_model_file):
                    os.remove(temp_source_model_file)
        else:
            result = run_base_model_subprocess(
                model_type=model_name,
                data=lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
                train_start=weight_train_range.start,
                train_end=weight_train_range.stop - 1,
                eval_start=weight_val_range.start,
                eval_end=weight_val_range.stop - 1,
                model_params=params.copy(),
                timeout=base_timeout,
            )

            y_pred_val = extract_predictions_from_result(result)
        weight_val_predictions[model_name] = normalize_demand_forecast(y_pred_val).tolist()
        print(f"    ✓ 生成 {len(y_pred_val)} 个weight_val预测", file=sys.stderr)

    # === Step 3: Calculate weights using validation residual MSE ===
    print("\n[Weighted Average Phase 2] 计算权重（验证集残差MSE倒数加权）", file=sys.stderr)
    y_true_val = parsed_data.y[weight_val_range]  # ✅ weight_val的真实值
    print("  误差计算 (基于weight_val残差MSE):", file=sys.stderr)
    weights = _compute_inverse_residual_mse_weights(y_true_val, weight_val_predictions)
    for name in models_to_use:
        residuals = y_true_val - np.asarray(weight_val_predictions[name], dtype=float)
        residual_mse = float(np.mean(np.square(residuals)))
        inverse_error = 1.0 / (residual_mse + 1e-9)
        print(f"    {name}: residual_mse={residual_mse:.6f}, inverse_error={inverse_error:.6f}", file=sys.stderr)
    print("  权重归一化:", file=sys.stderr)
    for name in models_to_use:
        print(f"    {name}: normalized_weight={weights[name]:.6f}", file=sys.stderr)

    # === Step 4: Phase 2 - Retrain on full train_range, predict on eval_range ===
    print("\n[Weighted Average Phase 3] 在完整训练集上重新训练基础模型", file=sys.stderr)
    eval_predictions = {}
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
                    "Saved Weighted Average LSTM base model",
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
                        f"weighted_avg_base_{model_name}_{os.getpid()}.pkl"
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
                    eval_predictions[model_name] = normalize_demand_forecast(y_pred_eval).tolist()
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
            eval_predictions[model_name] = normalize_demand_forecast(y_pred_eval).tolist()
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
            f"weighted_avg_base_{model_name}_{os.getpid()}.pkl"
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
        eval_predictions[model_name] = normalize_demand_forecast(y_pred_eval).tolist()

        base_model_states[model_name] = {
            "model_type": model_name,
            "model_file": temp_model_file,
            "params": params,
            "artifact_origin": "temporary_training",
        }
        print(f"    ✓ 生成 {len(y_pred_eval)} 个eval预测", file=sys.stderr)

    # === Step 5: Final weighted prediction and evaluation on eval_range ===
    print("\n[Weighted Average Phase 4] 加权组合和最终评估", file=sys.stderr)
    X_eval = np.array([eval_predictions[name] for name in models_to_use]).T  # ✅ eval_range的预测
    model_weights = np.array([weights[name] for name in models_to_use])
    y_pred_final = np.sum(X_eval * model_weights, axis=1)  # ✅ 加权组合（保持原有算法）
    y_pred_final = normalize_demand_forecast(y_pred_final)

    y_true = parsed_data.y[eval_range]  # ✅ 真实标签
    residuals_final = y_true - y_pred_final
    x_weight_val = np.array([weight_val_predictions[name] for name in models_to_use]).T
    weight_val_blend = np.sum(x_weight_val * model_weights, axis=1)
    weight_val_blend = normalize_demand_forecast(weight_val_blend)
    weight_val_residuals = y_true_val - weight_val_blend
    std_dev_residuals = estimate_residual_std(
        residuals_final,
        weight_val_residuals,
        parsed_data.y[train_range],
    )

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
            "weights": state["weights"],
            "std_dev_residuals": float(state.get("std_dev_residuals", 0.0)),
        }
        attach_method_metadata(next_state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(next_state, save_path)

        out = {
            "mode": "weighted_avg_fit_only",
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
        out = {
            "mode": "weighted_avg_loaded_predict",
            "forecast_steps": steps,
            "predictions": _format_predictions(preds, std_dev),
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
    models_to_use = parse_base_model_names(args.models, "Weighted Average")

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
    weight_train_size = _compute_phase1_train_size(train_size)

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
        "Weighted Average",
    )

    _validate_training_args(parsed_data, ts, te, es, ee, models_to_use, model_params, base_model_files)

    metrics, base_model_states, weights, y_true, y_pred, std_dev = train_and_evaluate(
        parsed_data,
        range(ts, te + 1),
        range(es, ee + 1),
        models_to_use,
        model_params,
        base_model_files,
    )

    out: Dict[str, Any] = {
        "mode": "weighted_avg_train_eval",
        "models": models_to_use,
        "weights": weights,
        "metrics": metrics,
        "eval_y_true": y_true,
        "predictions": _format_predictions(y_pred, std_dev),
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
            "model_type": MODEL_TYPE, "models": models_to_use,
            "base_model_states": base_model_states, "weights": weights,
            "std_dev_residuals": std_dev,
        }
        attach_method_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
