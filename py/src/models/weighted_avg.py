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
import math
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
    format_predictions,
    get_lstm_source_forecast_strategy,
    get_member_model_path,
    load_lstm_source_state,
    normalize_demand_forecast,
    parse_base_model_files_arg,
    parse_base_model_names,
    parse_feature_name_list,
    plan_ensemble_selection_calibration_split,
    predict_member_on_chronological_holdout,
    persist_member_model_artifact,
    prepare_lstm_source_model_for_requested_steps,
    require_explicit_lstm_feature_names,
    resolve_base_model_artifact_configs,
    validate_base_model_names,
)
from metrics import mae, mape, r2, rmse
from model_info import attach_method_metadata, ensure_state_metadata
from serialization import dump_model, load_model
from subprocess_utils import (
    extract_predictions_from_result,
    extract_uncertainty_profile_from_result,
    run_base_model_fit_subprocess,
    run_base_model_predict_subprocess,
    run_base_model_subprocess,
)
from uncertainty import ensemble_growth_profile

MODEL_TYPE = "weighted_avg"
METHOD_NAME = "选择残差 MSE 倒数加权平均融合"
FORECAST_STRATEGY = "validation_residual_mse_weighted_average"
IMPLEMENTATION_NOTES = [
    "教科书组合预测常用逆方差或误差方差-协方差思想分配权重；当预测误差近似无偏时，MSE 与方差口径一致。",
    "本系统按时间顺序拆分训练前缀、权重选择集和选择后校准集，并根据模型数、样本数及所有成员的最小可训练条件动态缩减留出段；先按目标与成员预测的共同绝对量级归一化，再按残差 MSE 倒数比例形成候选权重，最后按选择样本可靠度向等权组合收缩。精确零误差成员并列分享全部候选权重。",
    "该规则会同时惩罚系统性偏差和残差波动，但没有估计模型误差之间的协方差；可靠度收缩避免很短选择段产生近乎全押单一模型的权重。",
    "内部选择只沿用用户选择的 MA 窗口、ES alpha、ARIMA d 与 LSTM 特征/归一化；ARIMA 阶数和 LSTM look-back、容量、批大小、轮数均按当前子训练段重新推导，避免完整训练段信息泄漏。",
    "固定权重后，各成员仅用选择集之前及选择集的数据重新拟合，并在更晚的校准段生成组合残差；基础模型最后使用完整训练区间的既有匹配产物或重新拟合产物生成外层选择评估预测。",
    "std_dev、经验范围和 P99 压力诊断只使用选择后校准段的 actual-prediction 组合残差，并继承成员不确定性的逐 horizon 增长形状；外层选择评估真实值不参与校准。若样本不足以形成不重叠校准段，则明确回退到训练尺度，不复用权重选择残差。",
]


# ---------- utilities ----------
_parse_feature_name_list = parse_feature_name_list
_load_lstm_source_state = load_lstm_source_state
_get_lstm_source_forecast_strategy = get_lstm_source_forecast_strategy
_prepare_lstm_source_model_for_requested_steps = prepare_lstm_source_model_for_requested_steps
_format_predictions = format_predictions


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
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Estimate inverse-error weights from validation residual MSE.

    This is equivalent to inverse-SSE weighting when every model is evaluated on
    the same validation segment length, and it penalizes systematic bias.
    """
    if not predictions_by_model:
        raise ValueError("[VALIDATION] Weighted Average requires at least one prediction column.")

    targets = np.asarray(y_true, dtype=float)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("[VALIDATION] Weighted Average targets must be a non-empty one-dimensional array.")
    if not np.isfinite(targets).all():
        raise ValueError("[VALIDATION] Weighted Average targets must contain only finite values.")

    prediction_arrays: Dict[str, np.ndarray] = {}
    normalization_scale = float(np.max(np.abs(targets)))
    for model_name, predictions in predictions_by_model.items():
        values = np.asarray(predictions, dtype=float)
        if values.ndim != 1 or len(values) != len(targets):
            raise ValueError(
                "[VALIDATION] Weighted Average prediction columns must be one-dimensional "
                "and match the target length."
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"[VALIDATION] Weighted Average predictions for {model_name} must be finite."
            )
        prediction_arrays[model_name] = values
        normalization_scale = max(
            normalization_scale,
            float(np.max(np.abs(values))),
        )

    def stable_log_rms(actual: np.ndarray, predicted: np.ndarray) -> float | None:
        """Return log(RMS) without losing tiny residuals or overflowing subtraction."""
        log_magnitudes: list[float] = []
        for actual_value, predicted_value in zip(actual, predicted):
            actual_float = float(actual_value)
            predicted_float = float(predicted_value)
            if actual_float == predicted_float:
                continue

            difference = actual_float - predicted_float
            if math.isfinite(difference) and difference != 0.0:
                log_magnitudes.append(math.log(abs(difference)))
                continue

            element_scale = max(abs(actual_float), abs(predicted_float))
            scaled_difference = abs(
                actual_float / element_scale - predicted_float / element_scale
            )
            if element_scale == 0.0 or scaled_difference == 0.0:
                raise ValueError(
                    "[VALIDATION] Weighted Average residual magnitude could not be represented."
                )
            log_magnitudes.append(
                math.log(element_scale) + math.log(scaled_difference)
            )

        if not log_magnitudes:
            return None
        maximum_log = max(log_magnitudes)
        scaled_square_sum = math.fsum(
            math.exp(2.0 * (value - maximum_log))
            for value in log_magnitudes
        )
        log_rms = maximum_log + 0.5 * (
            math.log(scaled_square_sum) - math.log(len(actual))
        )
        if not math.isfinite(log_rms):
            raise ValueError("[VALIDATION] Weighted Average residual RMS is invalid.")
        return log_rms

    rms_logs = {
        name: stable_log_rms(targets, values)
        for name, values in prediction_arrays.items()
    }
    exact_zero_members = [
        name for name, values in prediction_arrays.items()
        if bool(np.array_equal(targets, values))
    ]
    model_count = len(rms_logs)
    if exact_zero_members:
        exact_weight = 1.0 / len(exact_zero_members)
        raw_weights = {
            name: exact_weight if name in exact_zero_members else 0.0
            for name in rms_logs
        }
        inverse_scores = dict(raw_weights)
        zero_policy = "equal_raw_weight_among_exact_zero_members"
    else:
        if any(value is None for value in rms_logs.values()):
            raise ValueError("[VALIDATION] Weighted Average nonzero residual RMS is missing.")
        minimum_log_rms = min(float(value) for value in rms_logs.values())
        inverse_scores = {
            name: float(np.exp(2.0 * (minimum_log_rms - float(log_rms))))
            for name, log_rms in rms_logs.items()
        }
        score_total = float(sum(inverse_scores.values()))
        if not np.isfinite(score_total) or score_total <= 0.0:
            raise ValueError("[VALIDATION] Weighted Average inverse-MSE scores are invalid.")
        raw_weights = {
            name: value / score_total
            for name, value in inverse_scores.items()
        }
        zero_policy = "no_exact_zero_member"

    reliability = len(targets) / (len(targets) + model_count + 1.0)
    uniform_weight = 1.0 / model_count
    weights = {
        name: reliability * raw_weights[name] + (1.0 - reliability) * uniform_weight
        for name in raw_weights
    }
    final_total = float(sum(weights.values()))
    if not np.isfinite(final_total) or final_total <= 0.0:
        raise ValueError("[VALIDATION] Weighted Average final weights are invalid.")
    weights = {name: value / final_total for name, value in weights.items()}
    if not np.isfinite(list(weights.values())).all():
        raise ValueError("[VALIDATION] Weighted Average final weights are not finite.")

    minimum_float_log = float(np.log(np.nextafter(0.0, 1.0)))
    maximum_float_log = float(np.log(np.finfo(float).max))

    def materialize_from_log(value_log: float | None) -> float | None:
        if value_log is None or value_log < minimum_float_log or value_log > maximum_float_log:
            return None
        value = float(np.exp(value_log))
        return value if np.isfinite(value) and value > 0.0 else None

    normalization_log = (
        math.log(normalization_scale) if normalization_scale > 0.0 else None
    )

    def normalized_log_rms(original_log_rms: float | None) -> float | None:
        if original_log_rms is None or normalization_log is None:
            return original_log_rms
        return original_log_rms - normalization_log

    normalized_rmses = {
        name: (
            0.0 if log_rms is None
            else materialize_from_log(normalized_log_rms(log_rms))
        )
        for name, log_rms in rms_logs.items()
    }
    normalized_mses = {
        name: (
            0.0 if log_rms is None
            else materialize_from_log(2.0 * normalized_log_rms(log_rms))
        )
        for name, log_rms in rms_logs.items()
    }

    def original_rmse_from_log(original_log_rms: float | None) -> float | None:
        if original_log_rms is None:
            return 0.0
        return materialize_from_log(original_log_rms)

    def original_mse_from_log(original_log_rms: float | None) -> float | None:
        if original_log_rms is None:
            return 0.0
        return materialize_from_log(2.0 * original_log_rms)

    residual_mses = {
        name: original_mse_from_log(log_rms)
        for name, log_rms in rms_logs.items()
    }
    prediction_matrix = np.asarray(
        [prediction_arrays[name] for name in raw_weights],
        dtype=float,
    ).T

    def convex_blend(blend_weights: np.ndarray) -> np.ndarray:
        blended: list[float] = []
        for row in prediction_matrix:
            row_scale = float(np.max(np.abs(row)))
            if row_scale == 0.0:
                blended.append(0.0)
                continue
            normalized_value = math.fsum(
                float(value / row_scale) * float(weight)
                for value, weight in zip(row, blend_weights)
            )
            normalized_value = min(1.0, max(-1.0, normalized_value))
            blended.append(row_scale * normalized_value)
        return np.asarray(blended, dtype=float)

    equal_weight_predictions = convex_blend(
        np.full(model_count, 1.0 / model_count, dtype=float)
    )
    selected_weight_predictions = convex_blend(
        np.asarray([weights[name] for name in raw_weights], dtype=float)
    )
    equal_weight_log_rms = stable_log_rms(targets, equal_weight_predictions)
    selected_weight_log_rms = stable_log_rms(targets, selected_weight_predictions)
    nonzero_member_logs = {
        name: log_rms
        for name, log_rms in rms_logs.items()
        if log_rms is not None
    }
    best_member_name = (
        exact_zero_members[0]
        if exact_zero_members
        else min(nonzero_member_logs, key=lambda name: float(nonzero_member_logs[name]))
    )
    return weights, {
        "residual_mse": residual_mses,
        "residual_mse_original": residual_mses,
        "residual_mse_normalized": normalized_mses,
        "residual_rms_normalized": normalized_rmses,
        "normalization_scale": normalization_scale,
        "inverse_mse_scores": inverse_scores,
        "raw_weights": raw_weights,
        "final_weights": weights,
        "reliability": float(reliability),
        "epsilon": 0.0,
        "epsilon_policy": "no_epsilon_normalized_inverse_ratio",
        "zero_error_policy": zero_policy,
        "exact_zero_members": exact_zero_members,
        "equal_weight_rmse": original_rmse_from_log(equal_weight_log_rms),
        "selected_weight_rmse": original_rmse_from_log(selected_weight_log_rms),
        "best_member": {
            "model_name": best_member_name,
            "rmse": original_rmse_from_log(rms_logs[best_member_name]),
        },
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

    plan_ensemble_selection_calibration_split(
        parsed_data,
        range(ts, te + 1),
        models_to_use,
        model_params,
        "weighted",
    )


# ---------- predict ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """Forecast using a saved Weighted Average model with subprocess calls."""
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    state = state or _load_state(model_path)
    weights = state["weights"]
    base_model_states = state["base_model_states"]
    models_to_use = state["models"]

    base_predictions = []
    member_profiles: Dict[str, List[float]] = {}
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
        member_profiles[model_name] = extract_uncertainty_profile_from_result(result)
        base_predictions.append(normalize_demand_forecast(preds).tolist())

    X_future = np.array(base_predictions).T
    model_weights = np.array([weights[name] for name in models_to_use])
    final_predictions = np.sum(X_future * model_weights, axis=1)
    final_predictions = normalize_demand_forecast(final_predictions)

    uncertainty_state = state.get("uncertainty") or {}
    profile, summary = ensemble_growth_profile(
        member_profiles,
        weights,
        uncertainty_state.get("calibration_residuals") or [],
        uncertainty_state.get("reference_series") or [],
        steps,
        calibration_source=str(
            uncertainty_state.get("calibration_source") or "internal_validation"
        ),
    )
    summary.setdefault("calibration_source", str(
        uncertainty_state.get("calibration_source") or "internal_validation"
    ))
    return final_predictions.tolist(), profile, summary


# ---------- train and evaluate ----------
def train_and_evaluate(
    parsed_data: ParsedData,
    train_range: range,
    eval_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Any],
    base_model_files: Optional[Dict[str, str]] = None,
) -> Tuple[
    Dict[str, Any], Dict[str, Any], Dict[str, float], List[float], List[float],
    List[float], Dict[str, Any], Dict[str, Any],
]:
    """Trains and evaluates the Weighted Average model using inverse residual-MSE weighting.

    Time-ordered error-weighted flow:
    1. Split train_range into chronological weight_train, weight_selection, and
       post-selection calibration ranges whenever feasible
    2. Train base models on weight_train, predict on weight_selection
    3. Calculate inverse-MSE candidates and shrink them toward equal weights
    4. Freeze weights, refit members through weight_selection, and calibrate
       uncertainty only on the later calibration range
    5. Retrain base models on full train_range
    6. Assess the fixed workflow on the outer selection-assessment eval_range

    Algorithm: inverse validation residual-MSE weighting
    (an inverse-variance-inspired rule adapted for potentially biased forecasts)
    - common_scale = max(abs(actual), abs(all member predictions))
    - normalized_mse_i = mean((actual/common_scale - prediction_i/common_scale)^2)
    - exact-zero members share all raw weight; otherwise inverse-MSE ratios are normalized
    - reliability = n_validation / (n_validation + model_count + 1)
    - weight_i = reliability * normalized_weight_i + (1 - reliability) / model_count
    - final_prediction = sum(weight_i * prediction_i)
    """

    # === Step 1: Split train_range into selection and post-selection calibration ===
    train_len = len(train_range)
    if train_len < 8:
        raise ValueError("[VALIDATION] train_range too small for weighted average (需要至少8个点，推荐≥12点)")

    split_plan = plan_ensemble_selection_calibration_split(
        parsed_data, train_range, models_to_use, model_params, "weighted"
    )
    prefix_size = int(split_plan["prefix_size"])
    selection_size = int(split_plan["selection_size"])
    calibration_size = int(split_plan["calibration_size"])
    weight_train_range = range(
        train_range.start,
        train_range.start + prefix_size,
    )
    weight_selection_range = range(
        weight_train_range.stop,
        weight_train_range.stop + selection_size,
    )
    calibration_range = range(
        weight_selection_range.stop,
        weight_selection_range.stop + calibration_size,
    )
    split_plan = {
        **split_plan,
        "ranges": {
            "prefix": {
                "start": weight_train_range.start,
                "end_exclusive": weight_train_range.stop,
            },
            "selection": {
                "start": weight_selection_range.start,
                "end_exclusive": weight_selection_range.stop,
            },
            "calibration": {
                "start": calibration_range.start,
                "end_exclusive": calibration_range.stop,
            },
        },
    }

    print("[Weighted Average] 数据划分:", file=sys.stderr)
    print(f"  权重训练集: {weight_train_range} ({len(weight_train_range)} points)", file=sys.stderr)
    print(f"  权重选择集: {weight_selection_range} ({len(weight_selection_range)} points)", file=sys.stderr)
    print(f"  选择后校准集: {calibration_range} ({len(calibration_range)} points)", file=sys.stderr)
    print(f"  外层选择评估集: {eval_range} ({len(eval_range)} points)", file=sys.stderr)

    # Non-LSTM base models keep numeric-only rows; LSTM can use categorical features.
    numeric_data_dicts, lstm_data_dicts = build_ensemble_subprocess_data(
        parsed_data,
        models_to_use,
        model_params,
    )

    # === Step 2: Train on prefix, predict on the weight-selection range ===
    print("\n[Weighted Average Phase 1] 训练基础模型并生成权重选择集预测", file=sys.stderr)
    weight_selection_predictions = {}

    # Calculate dynamic timeout for LSTM models
    base_timeout = 600 if "lstm" in models_to_use else 120

    for model_name in models_to_use:
        params = model_params[model_name]
        source_model_file = (base_model_files or {}).get(model_name)

        print(f"  训练 {model_name} 在 weight_train，预测 weight_selection...", file=sys.stderr)
        predictions = predict_member_on_chronological_holdout(
            model_name=model_name,
            params=params,
            source_model_file=source_model_file,
            numeric_data=numeric_data_dicts,
            lstm_data=lstm_data_dicts,
            train_range=weight_train_range,
            holdout_range=weight_selection_range,
            timeout=base_timeout,
        )
        weight_selection_predictions[model_name] = predictions
        print(f"    ✓ 生成 {len(predictions)} 个weight_selection预测", file=sys.stderr)

    # === Step 3: Calculate weights using validation residual MSE ===
    print("\n[Weighted Average Phase 2] 计算权重（验证集残差MSE倒数加权）", file=sys.stderr)
    y_true_val = parsed_data.y[weight_selection_range]
    print("  误差计算 (基于weight_selection残差MSE):", file=sys.stderr)
    weights, weight_diagnostics = _compute_inverse_residual_mse_weights(
        y_true_val, weight_selection_predictions
    )
    for name in models_to_use:
        residual_mse = weight_diagnostics["residual_mse"][name]
        normalized_mse = weight_diagnostics["residual_mse_normalized"][name]
        inverse_score = weight_diagnostics["inverse_mse_scores"][name]
        residual_label = "not-representable" if residual_mse is None else f"{residual_mse:.6f}"
        normalized_label = (
            "not-representable" if normalized_mse is None else f"{normalized_mse:.12g}"
        )
        print(
            f"    {name}: residual_mse={residual_label}, "
            f"normalized_mse={normalized_label}, inverse_score={inverse_score:.12g}",
            file=sys.stderr,
        )
    print("  可靠度收缩后的权重:", file=sys.stderr)
    for name in models_to_use:
        print(f"    {name}: normalized_weight={weights[name]:.6f}", file=sys.stderr)

    # === Step 4: Freeze weights and calibrate on a later, disjoint range ===
    calibration_residuals = np.asarray([], dtype=float)
    calibration_source = "unavailable_post_selection_calibration"
    if calibration_size > 0:
        calibration_predictions: Dict[str, List[float]] = {}
        print("\n[Weighted Average Phase 3] 固定权重并生成选择后校准预测", file=sys.stderr)
        for model_name in models_to_use:
            calibration_predictions[model_name] = (
                predict_member_on_chronological_holdout(
                    model_name=model_name,
                    params=model_params[model_name],
                    source_model_file=(base_model_files or {}).get(model_name),
                    numeric_data=numeric_data_dicts,
                    lstm_data=lstm_data_dicts,
                    train_range=range(train_range.start, calibration_range.start),
                    holdout_range=calibration_range,
                    timeout=base_timeout,
                )
            )
        x_calibration = np.asarray(
            [calibration_predictions[name] for name in models_to_use],
            dtype=float,
        ).T
        calibration_blend = normalize_demand_forecast(
            np.sum(x_calibration * np.asarray(
                [weights[name] for name in models_to_use],
                dtype=float,
            ), axis=1)
        )
        calibration_residuals = (
            parsed_data.y[calibration_range] - calibration_blend
        )
        calibration_source = "post_selection_calibration"
    else:
        print(
            "\n[Weighted Average Phase 3] 样本不足，未复用权重选择残差；"
            "不确定性将回退到训练尺度",
            file=sys.stderr,
        )

    # === Step 5: Retrain on full train_range, predict on eval_range ===
    print("\n[Weighted Average Phase 4] 在完整训练集上重新训练基础模型", file=sys.stderr)
    eval_predictions = {}
    full_uncertainty_profiles = {}
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
                    full_uncertainty_profiles[model_name] = (
                        extract_uncertainty_profile_from_result(result)
                    )
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
            full_uncertainty_profiles[model_name] = (
                extract_uncertainty_profile_from_result(result)
            )
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
        profile_result = result
        if eval_forecast_offset > 0:
            profile_result = run_base_model_predict_subprocess(
                model_type=model_name,
                model_file=temp_model_file,
                forecast_steps=eval_forecast_steps,
                timeout=base_timeout,
            )
        full_uncertainty_profiles[model_name] = (
            extract_uncertainty_profile_from_result(profile_result)
        )
        eval_predictions[model_name] = normalize_demand_forecast(y_pred_eval).tolist()

        base_model_states[model_name] = {
            "model_type": model_name,
            "model_file": temp_model_file,
            "params": params,
            "artifact_origin": "temporary_training",
        }
        print(f"    ✓ 生成 {len(y_pred_eval)} 个eval预测", file=sys.stderr)

    # === Step 6: Fixed-workflow assessment on the outer evaluation range ===
    print("\n[Weighted Average Phase 5] 加权组合和外层选择评估", file=sys.stderr)
    X_eval = np.array([eval_predictions[name] for name in models_to_use]).T  # ✅ eval_range的预测
    model_weights = np.array([weights[name] for name in models_to_use])
    y_pred_final = np.sum(X_eval * model_weights, axis=1)  # ✅ 加权组合（保持原有算法）
    y_pred_final = normalize_demand_forecast(y_pred_final)

    y_true = parsed_data.y[eval_range]  # ✅ 真实标签
    full_uncertainty_profile, uncertainty_summary = ensemble_growth_profile(
        full_uncertainty_profiles,
        weights,
        calibration_residuals,
        parsed_data.y[train_range],
        eval_forecast_steps,
        calibration_source=calibration_source,
    )
    uncertainty_profile = full_uncertainty_profile[
        eval_forecast_offset: eval_forecast_offset + len(eval_range)
    ]
    # The outer range assesses the selected workflow; it is not an untouched
    # final test set because the UI can compare and select models using it.
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

    uncertainty_state = {
        "calibration_residuals": calibration_residuals.tolist(),
        "calibration_source": calibration_source,
        "reference_series": parsed_data.y[train_range].tolist(),
        "profile": full_uncertainty_profile,
        "summary": uncertainty_summary,
    }
    return (
        metrics,
        base_model_states,
        weights,
        y_true.tolist(),
        y_pred_final.tolist(),
        uncertainty_profile,
        weight_diagnostics,
        {"split": split_plan, "uncertainty": uncertainty_state},
    )


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
                model_timeout = 600

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
            "weight_diagnostics": state.get("weight_diagnostics", {}),
            "split_plan": state.get("split_plan", {}),
            "uncertainty": state.get("uncertainty", {}),
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
        state = _load_state(load_path)
        prepared = state.get("prepared_predictions")
        if isinstance(prepared, list) and len(prepared) >= steps:
            predictions = [dict(point) for point in prepared[:steps]]
        else:
            preds, uncertainty_profile, uncertainty_summary = predict(load_path, steps, state=state)
            predictions = _format_predictions(preds, uncertainty_profile, uncertainty_summary)
        out = {
            "mode": "weighted_avg_loaded_predict",
            "forecast_steps": steps,
            "predictions": predictions,
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

    # Build model_params from user-selected arguments. Hidden complexity is
    # derived independently for every chronological training segment.
    def get_arg(name, default, cast_type=None):
        """Safely get argument with default value"""
        val = getattr(args, name.replace("-", "_"), None)
        if val is None:
            return default
        if cast_type:
            return cast_type(val)
        return val

    model_params = {}
    base_model_files = parse_base_model_files_arg(get_arg("base_model_files", None))

    if "arima" in models_to_use:
        model_params["arima"] = {
            "d": get_arg("d", 0, int),
        }

    if "es" in models_to_use:
        model_params["es"] = {"alpha": get_arg("alpha", 0.5, float)}

    if "ma" in models_to_use:
        model_params["ma"] = {"window": get_arg("window", 3, int)}

    if "lstm" in models_to_use:
        feature_types_str = get_arg("feature_types", None)
        lstm_features_str = get_arg("lstm_features", None)
        target_feature = get_arg("lstm_target_feature", parsed_data.target_key)
        lstm_feature_names = require_explicit_lstm_feature_names(
            lstm_features_str,
            "Weighted Average",
        )

        if feature_types_str:
            feature_types = json.loads(feature_types_str)
        else:
            feature_types = infer_feature_types(
                parsed_data=parsed_data,
                target_key=target_feature,
                feature_keys=lstm_feature_names,
                inference_range=range(ts, te + 1),
            )

        allowed_keys = set(lstm_feature_names + [target_feature])
        feature_types = {k: v for k, v in feature_types.items() if k in allowed_keys}
        feature_types[target_feature] = "numeric"

        lstm_params = {
            "feature_types": feature_types,
            "normalization": get_arg("normalization", "minmax"),
            "lstm_target_feature": target_feature,
            "lstm_features": json.dumps(lstm_feature_names, ensure_ascii=False),
        }

        model_params["lstm"] = lstm_params

    model_params, base_model_files = resolve_base_model_artifact_configs(
        parsed_data,
        models_to_use,
        model_params,
        base_model_files,
        "Weighted Average",
        range(ts, te + 1),
    )

    _validate_training_args(parsed_data, ts, te, es, ee, models_to_use, model_params, base_model_files)

    (
        metrics,
        base_model_states,
        weights,
        y_true,
        y_pred,
        uncertainty_profile,
        weight_diagnostics,
        training_details,
    ) = train_and_evaluate(
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
        "weight_diagnostics": weight_diagnostics,
        "split_plan": training_details["split"],
        "metrics": metrics,
        "eval_y_true": y_true,
        "predictions": _format_predictions(
            y_pred,
            uncertainty_profile,
            {
                **training_details["uncertainty"]["summary"],
                "horizon_start": int(args.eval_start) - int(args.train_end),
            },
        ),
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
            "weight_diagnostics": weight_diagnostics,
            "split_plan": training_details["split"],
            "uncertainty": training_details["uncertainty"],
        }
        attach_method_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
