#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual Boosting Ensemble for Heterogeneous Time Series Models.

Stage 0 fits the original target. Subsequent stages fit the residuals left by the
current ensemble. Stage selection is performed on a held-out validation window,
while each selected stage is retrained on the full train_range for final eval and
artifact persistence.

Uses subprocess to call base models for consistency with API behavior.
"""
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from baselines import (
    baseline_fitted_values_for_range,
    baseline_forecast_for_range,
    choose_stage_zero_baseline,
    fit_baseline_state_for_range,
    forecast_from_baseline_state,
)
from data_utils import ParsedData, infer_feature_types, load_data_from_json
from ensemble_utils import (
    build_ensemble_subprocess_data,
    format_predictions,
    get_member_model_path,
    load_lstm_source_state,
    normalize_demand_forecast,
    parse_base_model_files_arg,
    parse_base_model_names,
    parse_feature_name_list,
    plan_ensemble_selection_calibration_split,
    persist_member_model_artifact,
    remove_member_model_artifact,
    require_explicit_lstm_feature_names,
    resolve_base_model_artifact_configs,
    validate_base_model_names,
)
from dynamic_config import derive_boosting_max_rounds
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

MODEL_TYPE = "boosting"
SUPPORTED_BASE_MODELS = ["ma", "es", "arima", "lstm"]
METHOD_NAME = "残差提升融合"
FORECAST_STRATEGY = "greedy_residual_additive_ensemble"
IMPLEMENTATION_NOTES = [
    "Boosting 的教科书核心是将多个弱学习器串行加到一个强学习器中，后续学习器重点修正前序误差。",
    "梯度提升在平方误差损失下等价于逐步拟合残差；本系统借鉴这一思想，但候选学习器是 MA、ES、ARIMA、LSTM 等时间序列模型。",
    "AdaBoost 通过样本权重实现错误关注；本系统的时间序列基础模型不支持样本权重，因此采用残差学习的加法模型。",
    "系统先在时间顺序选择段比较训练均值、naïve 和可用时的 seasonal naïve，并把胜者作为可保存、可预测的第 0 阶段；即使没有残差学习器达到改善阈值，也会返回有效的“保留基线/无提升”结果。",
    "第 0 阶段冻结后，每轮在同一选择段贪心地从全部残差候选中选择，同一类型可重复入选，最大轮数由 min(2×模型数, 选择点数-1) 决定。",
    "候选只沿用用户选择的 MA 窗口、ES alpha、ARIMA d 及 LSTM 特征/归一化。ARIMA 阶数和 LSTM 隐藏参数针对每轮当前数据段与目标重新推导。",
    "更新训练段残差时，若某阶段在序列开头还没有可定义的样本内拟合值（如 MA 尚未凑满完整窗口、LSTM 尚未凑满 look_back），该前缀按零修正处理，即保持原残差；系统不会改用较短窗口伪造拟合值。",
    "每个候选阶段以平方损失的一维非负线搜索求系数 gamma=max(0,<r,h>/<h,h>)，并完整应用该系数（后续 shrinkage 关闭，因子为 1.0）；短选择段要求至少 5% 相对改善，其余要求 1%，无足够改善即提前停止。选出的模型链在完整训练区间逐级重训并重新估计 gamma；若某级最优系数为零，则裁剪该级及其后的选择候选链。",
    "只有第 0 阶段和最终销量预测会截断到不小于 0，中间残差保持有符号。内部选择链先在更晚且不重叠的时间留出段之前重拟合并生成该段 actual-prediction 残差，随后才执行完整训练重拟合；只有两次最终保留的链结构一致时才采用该残差，否则明确回退到训练尺度。完整训练重拟合会纳入该校准段并重新估计参数，因此校准段不再是最终模型的独立留出；该经验范围不声明名义覆盖保证，外层选择评估真实值不参与校准。",
]

POST_LINE_SEARCH_SHRINKAGE = {
    "enabled": False,
    "factor": 1.0,
}


# ---------- utilities ----------
_parse_feature_name_list = parse_feature_name_list
_load_lstm_source_state = load_lstm_source_state
_format_predictions = format_predictions


def _load_state(model_path: str) -> Dict[str, Any]:
    state = load_model(model_path)
    if (
        not isinstance(state, dict)
        or state.get("model_type") != MODEL_TYPE
        or state.get("boosting_variant") != "residual"
    ):
        raise ValueError("[VALIDATION] Invalid Boosting model file.")
    state["post_line_search_shrinkage"] = get_post_line_search_shrinkage_metadata(
        state
    )
    return ensure_state_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)


def get_post_line_search_shrinkage_metadata(
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the explicit post-line-search policy, defaulting legacy artifacts.

    Boosting's fitted gamma is applied in full.  Keep this separate from
    optimizer learning rates and deprecated CLI controls so observability never
    implies that a neural-network or user-provided learning rate is involved.
    """
    if state is None or "post_line_search_shrinkage" not in state:
        return dict(POST_LINE_SEARCH_SHRINKAGE)
    candidate = state["post_line_search_shrinkage"]
    if (
        not isinstance(candidate, dict)
        or set(candidate) != {"enabled", "factor"}
        or candidate.get("enabled") is not False
        or isinstance(candidate.get("factor"), bool)
        or not isinstance(candidate.get("factor"), (int, float))
        or float(candidate["factor"]) != 1.0
    ):
        raise ValueError(
            "[VALIDATION] Saved Boosting model has invalid post-line-search shrinkage metadata."
        )
    return dict(POST_LINE_SEARCH_SHRINKAGE)


def _get_stage_coefficients(model_chain: List[str], state: Dict[str, Any]) -> List[float]:
    explicit_coefficients = state.get("stage_coefficients")
    if not isinstance(explicit_coefficients, list) or len(explicit_coefficients) != len(model_chain):
        raise ValueError("[VALIDATION] Saved Boosting model has invalid stage coefficients.")
    coefficients = np.asarray(explicit_coefficients, dtype=float).reshape(-1)
    if not np.isfinite(coefficients).all():
        raise ValueError("[VALIDATION] Saved Boosting model has non-finite stage coefficients.")
    return coefficients.tolist()


def _get_baseline_state(state: Dict[str, Any]) -> Dict[str, Any]:
    model_chain = state.get("model_chain")
    has_member_stages = isinstance(model_chain, list) and len(model_chain) > 0
    baseline = state.get("baseline")
    if isinstance(baseline, dict):
        # Validate eagerly so corrupt artifacts fail before any member subprocess.
        forecast_from_baseline_state(baseline, 1)
        if baseline.get("kind") == "zero" and not has_member_stages:
            raise ValueError(
                "[VALIDATION] Legacy zero-baseline Boosting artifacts require a non-empty stage chain."
            )
        return baseline
    # Revision-3 artifacts created before zero-stage baselines started from zero.
    # Keep their numerical replay possible while every newly saved artifact
    # persists an explicit, statistically meaningful baseline.
    if not has_member_stages:
        raise ValueError(
            "[VALIDATION] Legacy Boosting artifacts without a baseline require a non-empty stage chain."
        )
    return {"kind": "zero", "label": "旧版零起点"}


def _add_baseline_uncertainty_member(
    member_profiles: Dict[str, List[float]],
    coefficients: Dict[str, float],
    baseline_state: Dict[str, Any],
    steps: int,
    *,
    has_member_stages: bool,
) -> None:
    """Represent an explicit stage-zero baseline in ensemble growth.

    Revision-3 artifacts without a persisted baseline replay from zero.  Keep
    their non-empty-chain uncertainty behavior unchanged, while retaining the
    existing flat fallback member for an empty legacy chain.
    """
    has_explicit_baseline = baseline_state.get("kind") != "zero"
    if has_explicit_baseline or not has_member_stages:
        member_profiles["stage_0_baseline"] = [1.0] * steps
        coefficients["stage_0_baseline"] = 1.0


def _apply_stage_output_semantics(
    stage_idx: int,
    values: Any,
    *,
    residual_target: bool = False,
) -> np.ndarray:
    """Keep every residual correction signed; replay legacy stage 1 as demand."""
    predictions = np.asarray(values, dtype=float)
    if stage_idx == 0 and not residual_target:
        return normalize_demand_forecast(predictions)
    return predictions


def _nonnegative_line_search(residual: Any, prediction: Any) -> float:
    """Exact one-dimensional squared-error line search with gamma >= 0."""
    residual_values = np.asarray(residual, dtype=float).reshape(-1)
    prediction_values = np.asarray(prediction, dtype=float).reshape(-1)
    if residual_values.size != prediction_values.size:
        raise ValueError("[VALIDATION] Boosting line-search vectors have different lengths.")
    denominator = float(np.dot(prediction_values, prediction_values))
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
        return 0.0
    numerator = float(np.dot(residual_values, prediction_values))
    if not np.isfinite(numerator):
        return 0.0
    return max(0.0, numerator / denominator)


def _iter_stage_models(state: Dict[str, Any]):
    model_chain = state.get("model_chain")
    base_model_states = state.get("base_model_states")
    if not isinstance(model_chain, list):
        raise ValueError("[VALIDATION] Saved Boosting model is missing a valid model_chain.")
    if not isinstance(base_model_states, dict):
        raise ValueError("[VALIDATION] Saved Boosting model is missing base_model_states.")

    stage_coefficients = _get_stage_coefficients(model_chain, state)
    usage_counts: Dict[str, int] = {}

    for stage_idx, model_name in enumerate(model_chain):
        model_info = base_model_states.get(model_name)
        if not isinstance(model_info, dict):
            raise ValueError(f"[VALIDATION] Missing saved state for base model: {model_name}")

        models = model_info.get("models")
        if not isinstance(models, list):
            raise ValueError(f"[VALIDATION] Missing saved rounds for base model: {model_name}")

        usage_idx = usage_counts.get(model_name, 0)
        if usage_idx >= len(models):
            raise ValueError(f"[VALIDATION] Missing saved round {usage_idx} for base model: {model_name}")

        usage_counts[model_name] = usage_idx + 1
        yield stage_idx, model_name, float(stage_coefficients[stage_idx]), models[usage_idx]


def _build_residual_parsed_data(
    parsed_data: ParsedData,
    train_range: range,
    train_target_values: np.ndarray,
) -> ParsedData:
    df = parsed_data.df.copy().astype({parsed_data.target_key: float})
    target_col_idx = df.columns.get_loc(parsed_data.target_key)
    df.iloc[list(train_range), target_col_idx] = train_target_values
    return ParsedData(
        df=df,
        target_key=parsed_data.target_key,
        time_key=parsed_data.time_key,
        feature_keys=parsed_data.feature_keys,
    )


def _normalize_fitted_values(values: Any, expected_length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if np.isinf(arr).any():
        raise ValueError("[DATA] Boosting stage produced infinite in-sample fitted values.")
    # Leading NaNs are a normal warm-up representation for some differenced
    # models. They mean "no correction available" and therefore map to zero.
    arr = np.nan_to_num(arr, nan=0.0)
    if arr.size < expected_length:
        arr = np.concatenate([np.zeros(expected_length - arr.size), arr])
    elif arr.size > expected_length:
        arr = arr[-expected_length:]
    return arr


def _compute_ma_in_sample_fitted(train_target: np.ndarray, window: int) -> np.ndarray:
    fitted = np.zeros(len(train_target), dtype=float)
    # Match the standalone MA definition: a fitted value exists only after a
    # complete ``window`` of preceding observations is available.  Leaving the
    # warm-up prefix at zero means those observations do not update Boosting's
    # residual target through a shorter, otherwise different MA rule.
    for idx in range(window, len(train_target)):
        fitted[idx] = float(np.mean(train_target[idx - window:idx]))
    return fitted


def _compute_lstm_in_sample_fitted(
    model_file: str,
    parsed_data: ParsedData,
    train_range: range,
) -> np.ndarray:
    from models import lstm as lstm_model

    state = lstm_model._load_state(model_file)
    train_df = parsed_data.df.iloc[list(train_range)]
    preprocessor = state["preprocessor"]
    scaled_train = preprocessor.transform(train_df)
    if isinstance(scaled_train, csr_matrix):
        scaled_train = scaled_train.toarray()

    look_back = int(state["look_back"])
    fitted = np.zeros(len(train_df), dtype=float)
    if len(train_df) <= look_back:
        return fitted

    forecast_strategy = state.get("forecast_strategy", lstm_model.LEGACY_FORECAST_STRATEGY)
    if forecast_strategy == lstm_model.DIRECT_FORECAST_STRATEGY and int(state.get("forecast_horizon", 0) or 0) > 0:
        windows = np.asarray(
            [scaled_train[idx - look_back : idx, :] for idx in range(look_back, len(train_df))],
            dtype=float,
        )
        pred_scaled = state["keras_model"].predict(windows, verbose=0)
        if pred_scaled.ndim == 1:
            pred_scaled = pred_scaled.reshape(-1, 1)
        next_step_scaled = pred_scaled[:, 0]
    else:
        trainX, _ = lstm_model._create_autoregressive_dataset(scaled_train, look_back)
        if len(trainX) == 0:
            return fitted
        next_step_scaled = state["keras_model"].predict(trainX, verbose=0).reshape(-1)

    target_scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    n_numeric_features = target_scaler.n_features_in_
    dummy_array = np.zeros((len(next_step_scaled), n_numeric_features))
    dummy_array[:, 0] = next_step_scaled
    unscaled_predictions = target_scaler.inverse_transform(dummy_array)[:, 0]
    fitted[look_back: look_back + len(unscaled_predictions)] = unscaled_predictions
    return fitted


def _compute_in_sample_fitted_predictions(
    model_name: str,
    model_file: str,
    parsed_data: ParsedData,
    train_range: range,
) -> np.ndarray:
    train_target = np.asarray(parsed_data.y[train_range], dtype=float)

    if model_name == "ma":
        from models import ma as ma_model

        state = ma_model._load_state(model_file)
        return _compute_ma_in_sample_fitted(train_target, int(state["window"]))

    if model_name == "es":
        from models import es as es_model

        state = es_model._load_state(model_file)
        fitted = getattr(state["model"], "fittedvalues", [])
        return _normalize_fitted_values(fitted, len(train_target))

    if model_name == "arima":
        from models import arima as arima_model

        state = arima_model._load_state(model_file)
        model = state["model"]
        if hasattr(model, "predict_in_sample"):
            fitted = model.predict_in_sample()
        elif hasattr(model, "arima_res_") and hasattr(model.arima_res_, "fittedvalues"):
            fitted = model.arima_res_.fittedvalues
        else:
            fitted = []
        return _normalize_fitted_values(fitted, len(train_target))

    if model_name == "lstm":
        return _compute_lstm_in_sample_fitted(model_file, parsed_data, train_range)

    raise ValueError(f"[VALIDATION] Unsupported boosting base model: {model_name}")


def _make_temp_model_path(prefix: str) -> str:
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".pkl")
    os.close(fd)
    os.remove(temp_path)
    return temp_path


def _build_round_records(
    round_parsed_data: ParsedData,
    model_name: str,
    model_params: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return build_ensemble_subprocess_data(round_parsed_data, [model_name], {model_name: model_params})


def _get_model_timeout(model_name: str, params: Dict[str, Any]) -> int:
    if model_name == "lstm":
        return 600
    return 120


def _train_stage_model(
    model_name: str,
    round_numeric_records: List[Dict[str, Any]],
    round_lstm_records: List[Dict[str, Any]],
    train_range: range,
    eval_range: range,
    params: Dict[str, Any],
    timeout: int,
) -> Tuple[str, np.ndarray, List[float]]:
    temp_model_file = _make_temp_model_path(f"boosting_stage_{model_name}_")
    params_with_save = params.copy()
    params_with_save["save_model"] = temp_model_file
    # Every learner models a signed residual because the physical-demand
    # baseline is now an explicit zero stage.
    params_with_save["allow_negative_predictions"] = True

    result = run_base_model_subprocess(
        model_type=model_name,
        data=round_lstm_records if model_name == "lstm" else round_numeric_records,
        train_start=train_range.start,
        train_end=train_range.stop - 1,
        eval_start=train_range.stop,
        eval_end=eval_range.stop - 1,
        model_params=params_with_save,
        timeout=timeout,
    )
    forecast_offset = eval_range.start - train_range.stop
    full_predictions = np.asarray(extract_predictions_from_result(result), dtype=float)
    eval_predictions = full_predictions[forecast_offset: forecast_offset + len(eval_range)]
    return (
        temp_model_file,
        eval_predictions,
        extract_uncertainty_profile_from_result(result),
    )


def _refit_fixed_chain_for_holdout(
    parsed_data: ParsedData,
    train_range: range,
    holdout_range: range,
    model_chain: List[str],
    model_params: Dict[str, Any],
    baseline_kind: str,
    baseline_period: Optional[int],
) -> Tuple[np.ndarray, List[str], List[float]]:
    """Refit a fixed selected structure without reading holdout labels."""
    history = np.asarray(parsed_data.y[train_range], dtype=float)
    predictions = baseline_forecast_for_range(
        parsed_data,
        train_range,
        holdout_range,
        baseline_kind,
        baseline_period,
    )
    residual_train = history - baseline_fitted_values_for_range(
        parsed_data,
        train_range,
        baseline_kind,
        baseline_period,
    )
    retained_chain: List[str] = []
    coefficients: List[float] = []

    for stage_idx, model_name in enumerate(model_chain):
        params = model_params[model_name]
        timeout = _get_model_timeout(model_name, params)
        round_parsed_data = _build_residual_parsed_data(
            parsed_data,
            train_range,
            residual_train,
        )
        numeric_records, lstm_records = _build_round_records(
            round_parsed_data,
            model_name,
            params,
        )
        stage_model_file = ""
        try:
            (
                stage_model_file,
                stage_predictions,
                _stage_profile,
            ) = _train_stage_model(
                model_name,
                numeric_records,
                lstm_records,
                train_range,
                holdout_range,
                params,
                timeout,
            )
            fitted_train_predictions = _compute_in_sample_fitted_predictions(
                model_name,
                stage_model_file,
                round_parsed_data,
                train_range,
            )
            stage_predictions = _apply_stage_output_semantics(
                stage_idx,
                stage_predictions,
                residual_target=True,
            )
            fitted_train_predictions = _apply_stage_output_semantics(
                stage_idx,
                fitted_train_predictions,
                residual_target=True,
            )
            coefficient = _nonnegative_line_search(
                residual_train,
                fitted_train_predictions,
            )
            if coefficient <= np.finfo(float).eps:
                break
            retained_chain.append(model_name)
            coefficients.append(coefficient)
            predictions = predictions + coefficient * stage_predictions
            residual_train = (
                residual_train - coefficient * fitted_train_predictions
            )
        finally:
            if stage_model_file:
                remove_member_model_artifact(model_name, stage_model_file)

    return (
        normalize_demand_forecast(predictions),
        retained_chain,
        coefficients,
    )


def _prepare_selected_chain_calibration(
    parsed_data: ParsedData,
    train_range: range,
    calibration_range: range,
    selection_model_chain: List[str],
    model_params: Dict[str, Any],
    baseline_kind: str,
    baseline_period: Optional[int],
    *,
    prepared_before_full_refit: bool = True,
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """Generate calibration residuals from a frozen pre-calibration chain.

    The requested structure comes exclusively from the earlier selection phase.
    `_refit_fixed_chain_for_holdout` may trim that structure using only the
    pre-calibration prefix; it never reads calibration labels.  The caller must
    still compare the retained structure with the later full-training result
    before reusing these residuals.  A rolling-deployment recovery may invoke
    this helper after the full refit has already completed; the explicit flag
    keeps that chronology visible without changing the frozen inputs.
    """
    has_calibration_segment = len(calibration_range) > 0
    calibration_refit: Dict[str, Any] = {
        "requested_chain": list(selection_model_chain),
        "retained_chain": [],
        "coefficients": [],
        "matched_selection_structure": False,
        "matched_final_structure": False,
        "prepared_before_full_refit": prepared_before_full_refit,
        "preparation_mode": (
            "before_full_refit"
            if prepared_before_full_refit
            else "recovered_after_full_refit"
        ),
        "selection_chain_frozen_before_calibration_labels": True,
        "full_refit_includes_calibration_segment": has_calibration_segment,
        "final_refit_parameters_may_use_calibration_labels": (
            has_calibration_segment
        ),
        "final_refit_stage_coefficients_may_use_calibration_labels": bool(
            has_calibration_segment and selection_model_chain
        ),
        "coverage_claim": "none",
    }
    if not has_calibration_segment:
        return (
            np.asarray([], dtype=float),
            "unavailable_post_selection_calibration",
            calibration_refit,
        )

    (
        calibration_predictions,
        calibration_chain,
        calibration_coefficients,
    ) = _refit_fixed_chain_for_holdout(
        parsed_data,
        range(train_range.start, calibration_range.start),
        calibration_range,
        selection_model_chain,
        model_params,
        baseline_kind,
        baseline_period,
    )
    calibration_refit.update(
        {
            "retained_chain": calibration_chain,
            "coefficients": calibration_coefficients,
            "matched_selection_structure": (
                calibration_chain == selection_model_chain
            ),
        }
    )
    return (
        np.asarray(parsed_data.y[calibration_range], dtype=float)
        - calibration_predictions,
        "post_selection_calibration",
        calibration_refit,
    )


def _finalize_selected_chain_calibration(
    calibration_residuals: np.ndarray,
    calibration_source: str,
    calibration_refit: Dict[str, Any],
    final_model_chain: List[str],
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """Fail closed when the later full refit changes the calibrated structure."""
    finalized = dict(calibration_refit)
    retained_chain = list(finalized.get("retained_chain") or [])
    matched_final_structure = bool(
        calibration_source == "post_selection_calibration"
        and retained_chain == list(final_model_chain)
    )
    finalized["matched_final_structure"] = matched_final_structure

    if calibration_source != "post_selection_calibration":
        return np.asarray([], dtype=float), calibration_source, finalized
    if not matched_final_structure:
        return (
            np.asarray([], dtype=float),
            "unavailable_after_full_refit_chain_change",
            finalized,
        )
    return calibration_residuals, calibration_source, finalized


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

    stages = list(_iter_stage_models(state))

    for stage_idx, model_name, _coefficient, previous_model_state in stages:
        if model_name != "lstm":
            continue

        from models import lstm as lstm_model

        source_model_state = _load_lstm_source_state(
            previous_model_state["model_file"],
            f"Boosting fit-only stage {stage_idx + 1} LSTM source model",
        )

        target_feature = str(source_model_state.get("target_key") or parsed_data.target_key)
        if target_feature not in parsed_data.df.columns:
            raise ValueError(
                f"[VALIDATION] Boosting fit-only stage {stage_idx + 1} (lstm) is missing target column: {target_feature}"
            )
        missing_feature_names = [
            feature_name
            for feature_name in source_model_state.get("feature_keys", []) or []
            if feature_name not in parsed_data.df.columns
        ]
        if missing_feature_names:
            raise ValueError(
                f"[VALIDATION] Boosting fit-only stage {stage_idx + 1} (lstm) is missing feature columns: {missing_feature_names}"
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
                f"[VALIDATION] Boosting fit-only stage {stage_idx + 1} (lstm) cannot be refit: {message}"
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
    n = parsed_data.n_obs

    validate_base_model_names(models_to_use, "Boosting")

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
        "boosting",
    )


# ---------- predict ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    state = state or _load_state(model_path)
    baseline_state = _get_baseline_state(state)
    final_predictions = forecast_from_baseline_state(
        baseline_state,
        steps,
    )
    has_explicit_baseline = baseline_state.get("kind") != "zero"
    member_profiles: Dict[str, List[float]] = {}
    coefficients: Dict[str, float] = {}
    has_member_stage = False
    for _stage_idx, model_name, coefficient, model_state in _iter_stage_models(state):
        has_member_stage = True
        model_file = model_state["model_file"]
        result = run_base_model_predict_subprocess(
            model_type=model_name,
            model_file=model_file,
            forecast_steps=steps,
            allow_negative_predictions=True,
            timeout=60,
        )
        stage_predictions = _apply_stage_output_semantics(
            _stage_idx,
            extract_predictions_from_result(result),
            residual_target=has_explicit_baseline,
        )
        stage_key = f"stage_{_stage_idx}_{model_name}"
        member_profiles[stage_key] = extract_uncertainty_profile_from_result(result)
        coefficients[stage_key] = coefficient
        final_predictions += coefficient * stage_predictions

    _add_baseline_uncertainty_member(
        member_profiles,
        coefficients,
        baseline_state,
        steps,
        has_member_stages=has_member_stage,
    )

    uncertainty_state = state.get("uncertainty") or {}
    profile, summary = ensemble_growth_profile(
        member_profiles,
        coefficients,
        uncertainty_state.get("calibration_residuals") or [],
        uncertainty_state.get("reference_series") or [],
        steps,
        calibration_source=str(
            uncertainty_state.get("calibration_source")
            or "unavailable_post_selection_calibration"
        ),
    )
    summary.setdefault("calibration_source", str(
        uncertainty_state.get("calibration_source")
        or "unavailable_post_selection_calibration"
    ))
    return normalize_demand_forecast(final_predictions).tolist(), profile, summary


# ---------- train and evaluate ----------
def train_and_evaluate(
    parsed_data: ParsedData,
    train_range: range,
    eval_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Any],
    learning_rate: float,
    max_rounds: int,
) -> Tuple[
    Dict[str, Any], List[str], List[float], Dict[str, Any], List[float],
    List[float], List[float], Dict[str, Any], Dict[str, Any],
]:
    """Fit a stage-zero baseline plus a chain of signed-residual learners.

    Saved base-model artifacts are deliberately not accepted here. They predict
    demand, whereas every stage after the baseline models a signed residual, so
    there is nothing to replay. `run()` still consumes `--base-model-files` to
    derive the canonical member parameters and to validate the request.
    """
    train_len = len(train_range)
    if train_len < 8:
        raise ValueError("[VALIDATION] train_range too small for boosting (需要至少8个点，推荐≥12点)")

    split_plan = plan_ensemble_selection_calibration_split(
        parsed_data, train_range, models_to_use, model_params, "boosting"
    )
    prefix_size = int(split_plan["prefix_size"])
    selection_size = int(split_plan["selection_size"])
    calibration_size = int(split_plan["calibration_size"])
    boost_train_range = range(
        train_range.start,
        train_range.start + prefix_size,
    )
    boost_selection_range = range(
        boost_train_range.stop,
        boost_train_range.stop + selection_size,
    )
    calibration_range = range(
        boost_selection_range.stop,
        boost_selection_range.stop + calibration_size,
    )
    split_plan = {
        **split_plan,
        "ranges": {
            "prefix": {
                "start": boost_train_range.start,
                "end_exclusive": boost_train_range.stop,
            },
            "selection": {
                "start": boost_selection_range.start,
                "end_exclusive": boost_selection_range.stop,
            },
            "calibration": {
                "start": calibration_range.start,
                "end_exclusive": calibration_range.stop,
            },
        },
    }

    print("[Boosting] 数据划分:", file=sys.stderr)
    print(f"  Boost Train Range: {boost_train_range} ({len(boost_train_range)} points)", file=sys.stderr)
    print(f"  Boost Selection Range: {boost_selection_range} ({len(boost_selection_range)} points)", file=sys.stderr)
    print(f"  Post-selection Calibration Range: {calibration_range} ({len(calibration_range)} points)", file=sys.stderr)
    print(f"  Outer Selection Assessment: {eval_range} ({len(eval_range)} points)", file=sys.stderr)
    max_rounds = derive_boosting_max_rounds(
        len(models_to_use), len(boost_selection_range)
    )
    minimum_relative_improvement = 0.05 if len(boost_selection_range) < 5 else 0.01
    print(f"  Max Rounds: {max_rounds}", file=sys.stderr)

    (
        selected_baseline,
        baseline_fitted_train,
        baseline_selection_predictions,
        baseline_diagnostics,
    ) = choose_stage_zero_baseline(
        parsed_data,
        boost_train_range,
        boost_selection_range,
    )
    y_true_val = np.asarray(parsed_data.y[boost_selection_range], dtype=float)
    residual_train = (
        np.asarray(parsed_data.y[boost_train_range], dtype=float)
        - baseline_fitted_train
    )
    residual_val = y_true_val - baseline_selection_predictions
    print(
        f"  Stage 0 Baseline: {selected_baseline['label']} "
        f"(selection RMSE={baseline_diagnostics['selected_rmse']:.4f})",
        file=sys.stderr,
    )

    model_chain: List[str] = []
    stage_coefficients: List[float] = []
    search_summary: Dict[str, Any] = {
        "algorithm": "residual_boosting",
        "coefficient_strategy": "nonnegative_squared_error_line_search",
        "post_line_search_shrinkage": get_post_line_search_shrinkage_metadata(),
        "minimum_relative_improvement": minimum_relative_improvement,
        "max_rounds": max_rounds,
        "split_plan": split_plan,
        "baseline": baseline_diagnostics,
        "stages": [],
    }

    for round_idx in range(max_rounds):
        current_rmse = rmse(np.zeros_like(residual_val), residual_val)
        round_parsed_data = _build_residual_parsed_data(parsed_data, boost_train_range, residual_train)

        stage_results: List[Dict[str, Any]] = []
        for model_name in models_to_use:
            params = model_params[model_name]
            timeout = _get_model_timeout(model_name, params)
            round_numeric_records, round_lstm_records = _build_round_records(
                round_parsed_data,
                model_name,
                params,
            )
            temp_model_file, stage_predictions, _stage_profile = _train_stage_model(
                model_name,
                round_numeric_records,
                round_lstm_records,
                boost_train_range,
                boost_selection_range,
                params,
                timeout,
            )
            fitted_train_predictions = _compute_in_sample_fitted_predictions(
                model_name,
                temp_model_file,
                round_parsed_data,
                boost_train_range,
            )
            stage_predictions = _apply_stage_output_semantics(
                round_idx,
                stage_predictions,
                residual_target=True,
            )
            fitted_train_predictions = _apply_stage_output_semantics(
                round_idx,
                fitted_train_predictions,
                residual_target=True,
            )
            coefficient = _nonnegative_line_search(residual_val, stage_predictions)
            next_residual_val = residual_val - coefficient * stage_predictions
            next_rmse = rmse(np.zeros_like(next_residual_val), next_residual_val)
            stage_results.append({
                "model_name": model_name,
                "coefficient": coefficient,
                "rmse": next_rmse,
                "temp_model_file": temp_model_file,
                "fitted_train_predictions": fitted_train_predictions,
                "next_residual_val": next_residual_val,
            })
            print(f"  Round {round_idx + 1} candidate {model_name}: residual RMSE={next_rmse:.4f}", file=sys.stderr)

        best_result = min(stage_results, key=lambda result: result["rmse"])
        improvement = current_rmse - best_result["rmse"]
        relative_improvement = improvement / max(current_rmse, np.finfo(float).eps)
        stage_summary = {
            "round": round_idx + 1,
            "objective": "validation_residual_rmse",
            "previous_rmse": current_rmse,
            "candidates": {
                result["model_name"]: {
                    "rmse": result["rmse"],
                    "coefficient": result["coefficient"],
                }
                for result in stage_results
            },
            "winner": best_result["model_name"],
            "winner_rmse": best_result["rmse"],
            "coefficient": best_result["coefficient"],
            "improvement": improvement,
            "relative_improvement": relative_improvement,
        }

        if relative_improvement >= minimum_relative_improvement:
            model_chain.append(best_result["model_name"])
            stage_coefficients.append(best_result["coefficient"])
            residual_train = residual_train - best_result["coefficient"] * best_result["fitted_train_predictions"]
            residual_val = best_result["next_residual_val"]
            stage_summary["action"] = "selected"
            print(
                f"  ✓ Selected {best_result['model_name']} (residual RMSE: {best_result['rmse']:.4f})",
                file=sys.stderr,
            )
        else:
            stage_summary["action"] = "early_stop"
            print(
                f"  🛑 Early stopping: no candidate improves residual RMSE {current_rmse:.4f}",
                file=sys.stderr,
            )
            search_summary["stages"].append(stage_summary)
            for result in stage_results:
                remove_member_model_artifact(result["model_name"], result["temp_model_file"])
            break

        search_summary["stages"].append(stage_summary)
        for result in stage_results:
            remove_member_model_artifact(result["model_name"], result["temp_model_file"])

    if not model_chain:
        search_summary["selection_outcome"] = "baseline_retained_no_improvement"
        search_summary["termination_reason"] = (
            "no_residual_candidate_met_minimum_relative_improvement"
        )
    else:
        search_summary["selection_outcome"] = "residual_chain_selected"

    selection_model_chain = list(model_chain)
    selection_coefficients = list(stage_coefficients)
    search_summary["selection_model_chain"] = selection_model_chain
    baseline_kind = str(selected_baseline["kind"])
    baseline_period = selected_baseline.get("seasonal_period")
    (
        calibration_residuals,
        calibration_source,
        calibration_refit,
    ) = _prepare_selected_chain_calibration(
        parsed_data,
        train_range,
        calibration_range,
        selection_model_chain,
        model_params,
        baseline_kind,
        baseline_period,
    )
    refit_model_chain: List[str] = []
    stage_coefficients = []
    full_train_history = np.asarray(parsed_data.y[train_range], dtype=float)
    final_baseline_state = fit_baseline_state_for_range(
        parsed_data,
        train_range,
        baseline_kind,
        baseline_period,
    )
    final_eval_preds = baseline_forecast_for_range(
        parsed_data,
        train_range,
        eval_range,
        baseline_kind,
        baseline_period,
    )
    residual_full_train = (
        full_train_history
        - baseline_fitted_values_for_range(
            parsed_data,
            train_range,
            baseline_kind,
            baseline_period,
        )
    )
    y_true_eval = np.asarray(parsed_data.y[eval_range], dtype=float)
    timestamps = parsed_data.timestamps
    base_model_states: Dict[str, Any] = {}
    full_uncertainty_profiles: Dict[str, List[float]] = {}
    uncertainty_coefficients: Dict[str, float] = {}
    eval_forecast_steps = eval_range.stop - train_range.stop
    eval_forecast_offset = eval_range.start - train_range.stop

    for stage_idx, model_name in enumerate(selection_model_chain):
        params = model_params[model_name]
        timeout = _get_model_timeout(model_name, params)
        round_parsed_data = _build_residual_parsed_data(parsed_data, train_range, residual_full_train)
        # Every stage is a signed-residual learner: stage 0 is the statistical
        # baseline, so a saved base-model artifact (which predicts demand, not
        # residuals) can never be replayed here. `base_model_files` still
        # contributes to this run through `resolve_base_model_artifact_configs`,
        # which derives the canonical member parameters in `run()`.
        round_numeric_records, round_lstm_records = _build_round_records(
            round_parsed_data,
            model_name,
            params,
        )
        stage_model_file, stage_eval_predictions, stage_uncertainty_profile = _train_stage_model(
            model_name,
            round_numeric_records,
            round_lstm_records,
            train_range,
            eval_range,
            params,
            timeout,
        )

        fitted_train_predictions = _compute_in_sample_fitted_predictions(
            model_name,
            stage_model_file,
            round_parsed_data,
            train_range,
        )
        stage_eval_predictions = _apply_stage_output_semantics(
            stage_idx,
            stage_eval_predictions,
            residual_target=True,
        )
        fitted_train_predictions = _apply_stage_output_semantics(
            stage_idx,
            fitted_train_predictions,
            residual_target=True,
        )
        coefficient = _nonnegative_line_search(
            residual_full_train,
            fitted_train_predictions,
        )
        if coefficient <= np.finfo(float).eps:
            remove_member_model_artifact(model_name, stage_model_file)
            search_summary["refit_stop"] = {
                "stage": stage_idx + 1,
                "model_name": model_name,
                "reason": "nonpositive_full_refit_coefficient",
                "coefficient": coefficient,
            }
            break
        refit_model_chain.append(model_name)
        stage_coefficients.append(coefficient)
        final_eval_preds += coefficient * stage_eval_predictions
        residual_full_train = residual_full_train - coefficient * fitted_train_predictions
        stage_key = f"stage_{stage_idx}_{model_name}"
        full_uncertainty_profiles[stage_key] = stage_uncertainty_profile
        uncertainty_coefficients[stage_key] = coefficient

        base_model_states.setdefault(model_name, {"models": []})["models"].append({
            "model_type": model_name,
            "model_file": stage_model_file,
            "params": params,
            "round": stage_idx,
            "coefficient": coefficient,
        })

        print(
            f"  Stage {stage_idx + 1}: {model_name} coefficient={coefficient:.4f}, eval RMSE={rmse(y_true_eval, final_eval_preds):.4f}",
            file=sys.stderr,
        )

    model_chain = refit_model_chain
    (
        calibration_residuals,
        calibration_source,
        calibration_refit,
    ) = _finalize_selected_chain_calibration(
        calibration_residuals,
        calibration_source,
        calibration_refit,
        model_chain,
    )
    search_summary["calibration_refit"] = calibration_refit

    final_eval_preds = normalize_demand_forecast(final_eval_preds)
    _add_baseline_uncertainty_member(
        full_uncertainty_profiles,
        uncertainty_coefficients,
        final_baseline_state,
        eval_forecast_steps,
        has_member_stages=bool(model_chain),
    )
    full_uncertainty_profile, uncertainty_summary = ensemble_growth_profile(
        full_uncertainty_profiles,
        uncertainty_coefficients,
        calibration_residuals,
        parsed_data.y[train_range],
        eval_forecast_steps,
        calibration_source=calibration_source,
    )
    uncertainty_profile = full_uncertainty_profile[
        eval_forecast_offset: eval_forecast_offset + len(eval_range)
    ]
    metrics = {
        "rmse": rmse(y_true_eval, final_eval_preds),
        "mae": mae(y_true_eval, final_eval_preds),
        "mape": mape(y_true_eval, final_eval_preds),
        "r2": r2(y_true_eval, final_eval_preds),
        "evaluate_indices": list(eval_range),
        "evaluate_months": [timestamps[i] for i in eval_range] if timestamps else [],
    }

    print(f"  ✓ Final RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}", file=sys.stderr)
    chain_label = " → ".join(model_chain) if model_chain else "无提升（保留第 0 阶段基线）"
    print(f"  ✓ Model chain: {chain_label}", file=sys.stderr)
    search_summary["selection_coefficients"] = selection_coefficients
    search_summary["refit_model_chain"] = model_chain
    search_summary["refit_coefficients"] = stage_coefficients
    if selection_model_chain and not model_chain:
        search_summary["refit_outcome"] = "baseline_retained_after_refit_trim"
    elif model_chain:
        search_summary["refit_outcome"] = "residual_chain_retained"
    else:
        search_summary["refit_outcome"] = "baseline_retained_no_improvement"

    return (
        metrics,
        model_chain,
        stage_coefficients,
        base_model_states,
        y_true_eval.tolist(),
        final_eval_preds.tolist(),
        uncertainty_profile,
        search_summary,
        {
            "split": split_plan,
            "baseline": final_baseline_state,
            "uncertainty": {
                "calibration_residuals": calibration_residuals.tolist(),
                "calibration_source": calibration_source,
                "reference_series": parsed_data.y[train_range].tolist(),
                "profile": full_uncertainty_profile,
                "summary": uncertainty_summary,
            },
        },
    )


# ---------- public entry ----------
def run(args) -> None:
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
        train_range = range(ts, te + 1)
        _validate_fit_args(
            parsed_data,
            ts,
            te,
            state,
            int(getattr(args, "fit_forecast_steps", 0) or 0),
        )

        next_base_model_states: Dict[str, Any] = {}
        required_fit_forecast_steps = int(getattr(args, "fit_forecast_steps", 0) or 0) or None
        saved_baseline = _get_baseline_state(state)
        train_history = np.asarray(parsed_data.y[train_range], dtype=float)
        if saved_baseline.get("kind") == "zero":
            next_baseline = dict(saved_baseline)
            residual_train = train_history.copy()
            residual_chain = False
        else:
            baseline_kind = str(saved_baseline["kind"])
            baseline_period = saved_baseline.get("seasonal_period")
            next_baseline = fit_baseline_state_for_range(
                parsed_data,
                train_range,
                baseline_kind,
                baseline_period,
            )
            residual_train = train_history - baseline_fitted_values_for_range(
                parsed_data,
                train_range,
                baseline_kind,
                baseline_period,
            )
            residual_chain = True

        for stage_idx, model_name, coefficient, previous_model_state in _iter_stage_models(state):
            params = previous_model_state.get("params", {})
            timeout = _get_model_timeout(model_name, params)
            round_parsed_data = _build_residual_parsed_data(parsed_data, train_range, residual_train)
            round_numeric_records, round_lstm_records = _build_round_records(
                round_parsed_data,
                model_name,
                params,
            )
            member_save_path = get_member_model_path(save_path, model_name, stage_idx)
            run_base_model_fit_subprocess(
                model_type=model_name,
                data=round_lstm_records if model_name == "lstm" else round_numeric_records,
                train_start=ts,
                train_end=te,
                source_model_file=previous_model_state["model_file"],
                save_model_file=member_save_path,
                fit_forecast_steps=required_fit_forecast_steps if model_name == "lstm" else None,
                allow_negative_predictions=True,
                timeout=timeout,
            )
            fitted_train_predictions = _compute_in_sample_fitted_predictions(
                model_name,
                member_save_path,
                round_parsed_data,
                train_range,
            )
            # Use exactly the same stage-output semantics as evaluation and
            # prediction before constructing the next residual target.
            fitted_train_predictions = _apply_stage_output_semantics(
                stage_idx,
                fitted_train_predictions,
                residual_target=residual_chain,
            )
            residual_train = residual_train - coefficient * fitted_train_predictions
            next_base_model_states.setdefault(model_name, {"models": []})["models"].append({
                "model_type": model_name,
                "model_file": member_save_path,
                "params": params,
                "round": stage_idx,
                "coefficient": coefficient,
            })

        next_state = {
            "model_type": MODEL_TYPE,
            "boosting_variant": "residual",
            "post_line_search_shrinkage": get_post_line_search_shrinkage_metadata(
                state
            ),
            "model_chain": state["model_chain"],
            "stage_coefficients": _get_stage_coefficients(state["model_chain"], state),
            "base_model_states": next_base_model_states,
            "baseline": next_baseline,
            "split_plan": state.get("split_plan", {}),
            "uncertainty": state.get("uncertainty", {}),
        }
        attach_method_metadata(next_state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(next_state, save_path)

        out = {
            "mode": "boosting_fit_only",
            "post_line_search_shrinkage": get_post_line_search_shrinkage_metadata(
                state
            ),
            "model_chain": state["model_chain"],
            "baseline": next_baseline,
            "saved_model": save_path,
        }
        attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    if load_path:
        steps = int(getattr(args, "forecast_steps", 0))
        state = _load_state(load_path)
        prepared = state.get("prepared_predictions")
        if isinstance(prepared, list) and len(prepared) >= steps:
            predictions = [dict(point) for point in prepared[:steps]]
        else:
            preds, uncertainty_profile, uncertainty_summary = predict(load_path, steps, state=state)
            predictions = _format_predictions(preds, uncertainty_profile, uncertainty_summary)
        out = {
            "mode": "boosting_loaded_predict",
            "post_line_search_shrinkage": get_post_line_search_shrinkage_metadata(
                state
            ),
            "forecast_steps": steps,
            "predictions": predictions,
        }
        attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    if not hasattr(args, "data") or not args.data:
        raise ValueError("[VALIDATION] --data is required for training mode.")

    parsed_data = load_data_from_json(args.data)
    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)
    train_range = range(ts, te + 1)
    eval_range = range(es, ee + 1)
    models_to_use = parse_base_model_names(args.models, "Boosting")

    def get_arg(name, default, cast_type=None):
        val = getattr(args, name.replace("-", "_"), None)
        if val is None:
            return default
        if cast_type:
            return cast_type(val)
        return val

    model_params: Dict[str, Dict[str, Any]] = {}
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
            "Boosting",
        )

        if feature_types_str:
            feature_types = json.loads(feature_types_str)
        else:
            feature_types = infer_feature_types(
                parsed_data=parsed_data,
                target_key=target_feature,
                feature_keys=lstm_feature_names,
                inference_range=train_range,
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
        "Boosting",
        train_range,
    )

    _validate_training_args(parsed_data, ts, te, es, ee, models_to_use, model_params, base_model_files)

    (
        metrics,
        model_chain,
        stage_coefficients,
        base_model_states,
        y_true,
        y_pred,
        uncertainty_profile,
        summary,
        training_details,
    ) = train_and_evaluate(
        parsed_data,
        train_range,
        eval_range,
        models_to_use,
        model_params,
        0.0,
        0,
    )

    out = {
        "mode": "boosting_residual_learning",
        "post_line_search_shrinkage": get_post_line_search_shrinkage_metadata(),
        "models": models_to_use,
        "baseline": training_details["baseline"],
        "model_chain": model_chain,
        "stage_coefficients": stage_coefficients,
        "split_plan": training_details["split"],
        "metrics": metrics,
        "search_summary": summary,
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
        for model_name, model_info in base_model_states.items():
            for idx, model_state in enumerate(model_info["models"]):
                source_file = model_state["model_file"]
                perm_file = get_member_model_path(save_path, model_name, int(model_state.get("round", idx)))
                persist_member_model_artifact(
                    model_name,
                    source_file,
                    perm_file,
                    # Every stage artifact is a freshly trained temporary file.
                    move_source=True,
                )
                model_state["model_file"] = perm_file

        state = {
            "model_type": MODEL_TYPE,
            "boosting_variant": "residual",
            "post_line_search_shrinkage": get_post_line_search_shrinkage_metadata(),
            "model_chain": model_chain,
            "stage_coefficients": stage_coefficients,
            "base_model_states": base_model_states,
            "baseline": training_details["baseline"],
            "split_plan": training_details["split"],
            "uncertainty": training_details["uncertainty"],
        }
        attach_method_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
