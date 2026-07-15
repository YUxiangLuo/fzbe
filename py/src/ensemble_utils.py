#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_utils import (
    ParsedData,
    build_numeric_subprocess_records,
    build_subprocess_records,
    infer_feature_types,
)
# Re-exported here because all ensemble modules share this boundary helper.
from forecast_utils import normalize_demand_forecast as normalize_demand_forecast
from serialization import dump_model, load_model
from subprocess_utils import (
    extract_predictions_from_result,
    run_base_model_fit_subprocess,
    run_base_model_predict_subprocess,
)
from uncertainty import format_predictions as format_uncertainty_predictions
from dynamic_config import (
    derive_arima_search_limit,
    derive_lstm_config,
    plan_internal_holdout,
)

SUPPORTED_BASE_MODEL_TYPES = frozenset({"ma", "es", "arima", "lstm"})


def validate_base_model_names(
    models_to_use: Any,
    context: str = "Ensemble",
) -> List[str]:
    if not isinstance(models_to_use, list):
        raise ValueError(f"[VALIDATION] {context} base models must be a list.")
    if any(not isinstance(model, str) or not model.strip() for model in models_to_use):
        raise ValueError(f"[VALIDATION] {context} contains an invalid base model name.")

    normalized = [model.strip() for model in models_to_use]
    if len(normalized) < 2:
        raise ValueError(f"[VALIDATION] {context} requires at least 2 base models.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"[VALIDATION] {context} base models must not contain duplicates.")

    unknown_models = [
        model for model in normalized if model not in SUPPORTED_BASE_MODEL_TYPES
    ]
    if unknown_models:
        raise ValueError(
            f"[VALIDATION] {context} contains unsupported base model: {unknown_models[0]}"
        )
    return normalized


def parse_base_model_names(raw_value: Any, context: str = "Ensemble") -> List[str]:
    if not isinstance(raw_value, str):
        raise ValueError(f"[VALIDATION] {context} requires --models.")
    return validate_base_model_names(
        [model.strip() for model in raw_value.split(",") if model.strip()],
        context,
    )


def parse_feature_name_list(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    return []


def require_explicit_lstm_feature_names(
    raw_value: Any,
    context: str = "LSTM",
) -> List[str]:
    """Parse an explicit LSTM feature selection without inventing defaults.

    An empty JSON/list value is meaningful: it selects a target-only LSTM.  A
    missing or blank value is ambiguous and must never be interpreted as every
    dataset column.
    """
    if raw_value is None or (
        isinstance(raw_value, str) and not raw_value.strip()
    ):
        raise ValueError(
            f"[VALIDATION] {context} requires an explicit LSTM feature list; "
            "use [] for a target-only LSTM."
        )
    return parse_feature_name_list(raw_value)


def load_lstm_source_state(model_file: str, context: str) -> Dict[str, Any]:
    source_model_state = load_model(model_file)
    if not isinstance(source_model_state, dict) or source_model_state.get("model_type") != "lstm":
        raise ValueError(f"[VALIDATION] {context} is invalid.")
    return source_model_state


def build_lstm_params_from_source_state(
    parsed_data: ParsedData,
    source_model_state: Dict[str, Any],
    fallback_params: Dict[str, Any],
) -> Dict[str, Any]:
    target_feature = str(
        source_model_state.get("target_key")
        or fallback_params.get("lstm_target_feature")
        or parsed_data.target_key
    )
    raw_feature_keys = source_model_state.get("feature_keys")
    if not isinstance(raw_feature_keys, list) or any(
        not isinstance(feature_name, str) or not feature_name.strip()
        for feature_name in raw_feature_keys
    ):
        raise ValueError("Saved LSTM artifact is missing an explicit feature list.")
    feature_keys = [
        feature_name.strip()
        for feature_name in raw_feature_keys
        if feature_name.strip() != target_feature
    ]
    raw_feature_types = source_model_state.get("feature_types")
    if isinstance(raw_feature_types, dict) and raw_feature_types:
        feature_types = {
            str(key): value
            for key, value in raw_feature_types.items()
            if key == target_feature or key in feature_keys
        }
    else:
        feature_types = infer_feature_types(
            parsed_data=parsed_data,
            target_key=target_feature,
            feature_keys=feature_keys,
        )
    feature_types[target_feature] = "numeric"

    params = {
        "feature_types": feature_types,
        "normalization": source_model_state.get(
            "normalization",
            fallback_params.get("normalization", "minmax"),
        ),
        "lstm_target_feature": target_feature,
        "lstm_features": json.dumps(feature_keys, ensure_ascii=False),
    }
    return params


def get_lstm_source_forecast_strategy(source_model_state: Dict[str, Any]) -> str:
    from models import lstm as lstm_model

    if source_model_state.get("forecast_strategy"):
        return str(source_model_state["forecast_strategy"])
    if int(source_model_state.get("forecast_horizon", 0) or 0) > 0:
        return lstm_model.DIRECT_FORECAST_STRATEGY
    return lstm_model.LEGACY_FORECAST_STRATEGY


def resolve_lstm_reuse_production_horizon(
    source_model_state: Dict[str, Any],
    requested_steps: int = 0,
) -> int:
    from models import lstm as lstm_model

    forecast_strategy = get_lstm_source_forecast_strategy(source_model_state)
    if forecast_strategy != lstm_model.DIRECT_FORECAST_STRATEGY:
        return 0

    saved_eval_horizon = int(source_model_state.get("forecast_horizon", 0) or 0)
    current_required_horizon = lstm_model._resolve_production_forecast_horizon(
        state=source_model_state,
        eval_forecast_horizon=saved_eval_horizon,
    )
    return max(current_required_horizon, int(requested_steps or 0))


def prepare_lstm_source_model_for_requested_steps(
    source_model_file: str,
    source_model_state: Dict[str, Any],
    requested_steps: int,
) -> Tuple[str, Optional[str]]:
    current_required_horizon = resolve_lstm_reuse_production_horizon(source_model_state)
    requested_required_horizon = resolve_lstm_reuse_production_horizon(
        source_model_state,
        requested_steps=requested_steps,
    )
    if requested_required_horizon <= current_required_horizon:
        return source_model_file, None

    adjusted_state = dict(source_model_state)
    adjusted_state["production_forecast_horizon"] = requested_required_horizon
    fd, adjusted_source_model_file = tempfile.mkstemp(
        prefix=f"ensemble_lstm_source_{os.getpid()}_",
        suffix=".pkl",
    )
    os.close(fd)
    try:
        dump_model(adjusted_state, adjusted_source_model_file)
    except Exception:
        try:
            os.remove(adjusted_source_model_file)
        except OSError:
            pass
        raise
    return adjusted_source_model_file, adjusted_source_model_file


def format_predictions(
    preds: List[float],
    std_dev: Any = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    profile = (
        [float(value) for value in std_dev]
        if isinstance(std_dev, (list, tuple, np.ndarray))
        else [float(std_dev)] * len(preds)
    )
    return format_uncertainty_predictions(
        preds,
        profile,
        metadata or {"source": "empirical", "fallback": False},
    )


def estimate_lstm_encoded_feature_count(
    parsed_data: ParsedData,
    params: Dict[str, Any],
    train_start: int,
    prefix_size: int,
) -> int:
    target = str(params.get("lstm_target_feature") or parsed_data.target_key)
    raw_features = require_explicit_lstm_feature_names(
        params.get("lstm_features") if "lstm_features" in params else None,
        "LSTM holdout planning",
    )
    feature_types = params.get("feature_types") or {}
    prefix = parsed_data.df.iloc[train_start : train_start + prefix_size]
    count = 1
    for feature in raw_features:
        if feature == target or feature not in prefix.columns:
            continue
        if feature_types.get(feature) == "categorical":
            count += max(1, int(prefix[feature].nunique(dropna=False)))
        else:
            count += 1
    return count


def plan_ensemble_internal_holdout(
    parsed_data: ParsedData,
    train_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Dict[str, Any]],
    purpose: str,
) -> Dict[str, Any]:
    models = validate_base_model_names(models_to_use, purpose.title())
    for model_name in models:
        if model_name != "lstm":
            continue
        params = model_params.get(model_name, {})
        require_explicit_lstm_feature_names(
            params.get("lstm_features") if "lstm_features" in params else None,
            f"{purpose.title()} holdout planning",
        )

    def is_feasible(prefix_size: int, validation_size: int) -> bool:
        try:
            for model_name in models:
                params = model_params.get(model_name, {})
                if model_name == "ma" and prefix_size < int(params.get("window", 0)):
                    return False
                if model_name == "es" and prefix_size < 2:
                    return False
                if model_name == "arima":
                    derive_arima_search_limit(prefix_size, int(params.get("d", 0)))
                if model_name == "lstm":
                    feature_count = estimate_lstm_encoded_feature_count(
                        parsed_data,
                        params,
                        train_range.start,
                        prefix_size,
                    )
                    derive_lstm_config(prefix_size, validation_size, feature_count)
            return True
        except (TypeError, ValueError):
            return False

    return plan_internal_holdout(
        purpose,
        len(train_range),
        len(models),
        is_feasible,
    )


def can_refit_source_artifact_for_internal_holdout(model_name: str) -> bool:
    """Whether fit-only can safely refit directly from the source artifact.

    MA and ES artifacts contain fixed user parameters, while LSTM fit-only mode
    carries over only user-facing feature/normalization choices and rebuilds all
    hidden configuration on the shorter prefix. An ARIMA artifact additionally
    contains p and q selected using the complete training range; reusing those
    orders would let the internal holdout influence its own prediction.  ARIMA
    therefore reruns the same fixed-d AICc search on the prefix.
    """
    return model_name in {"ma", "es", "lstm"}


def estimate_residual_std(
    calibration_residuals: Any,
    fallback_calibration_residuals: Any = None,
    reference_series: Any = None,
) -> float:
    """Estimate a finite sample scale from training-side calibration residuals.

    A one-point calibration segment cannot identify a sample standard deviation,
    so callers may provide a second chronological training-side residual series.
    The reference-series branch is the final small-data fallback. Independent
    evaluation labels must never be supplied to this helper.
    """
    for label, values in (
        ("primary calibration", calibration_residuals),
        ("fallback calibration", fallback_calibration_residuals),
        ("reference", reference_series),
    ):
        if values is None:
            continue
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size and not np.isfinite(array).all():
            raise ValueError(f"[DATA] {label} values contain NaN or infinity.")
        if array.size > 1:
            return float(np.std(array, ddof=1))

    reference = np.asarray(reference_series if reference_series is not None else [], dtype=float).reshape(-1)
    baseline = abs(float(np.mean(reference))) if reference.size else 0.0
    return float(max(baseline * 0.05, 1.0))


def parse_base_model_files_arg(raw_value: Any) -> Dict[str, str]:
    if raw_value is None or raw_value == "":
        return {}

    try:
        payload = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("[VALIDATION] --base-model-files must be valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("[VALIDATION] --base-model-files must be a JSON object.")

    parsed_files: Dict[str, str] = {}
    for model_name, model_file in payload.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "[VALIDATION] --base-model-files keys must be non-empty model names."
            )
        if not isinstance(model_file, str) or not model_file.strip():
            raise ValueError(
                f"[VALIDATION] --base-model-files entry for {model_name} must be a non-empty path."
            )
        parsed_files[model_name] = model_file
    return parsed_files


def _artifact_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"[VALIDATION] {label} must be an integer in the saved artifact.")
    return int(value)


def _artifact_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"[VALIDATION] {label} must be numeric in the saved artifact.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"[VALIDATION] {label} must be finite in the saved artifact.")
    return result


def resolve_base_model_artifact_configs(
    parsed_data: ParsedData,
    models_to_use: List[str],
    model_params: Dict[str, Dict[str, Any]],
    base_model_files: Dict[str, str],
    context: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Validate explicit source artifacts and make their fixed configuration authoritative.

    A selected model without a mapped artifact is trained from ``model_params``.
    Once an artifact is explicitly mapped, silently falling back to a fresh model
    would make the ensemble differ from the base-model run shown to the learner,
    so invalid paths and states are rejected here.

    ARIMA is deliberately special: only the user-fixed differencing order ``d``
    comes from the artifact. Search bounds and p/q are derived again on each
    internal prefix, avoiding holdout leakage from the full-data fit.
    """
    selected_models = set(models_to_use)
    canonical_params = {
        model_name: dict(params)
        for model_name, params in model_params.items()
    }
    validated_files: Dict[str, str] = {}

    for model_name, model_file in base_model_files.items():
        if model_name not in SUPPORTED_BASE_MODEL_TYPES:
            raise ValueError(
                f"[VALIDATION] {context} source artifact uses unsupported model: {model_name}."
            )
        if model_name not in selected_models:
            raise ValueError(
                f"[VALIDATION] {context} source artifact was supplied for unselected model: "
                f"{model_name}."
            )
        if not isinstance(model_file, str) or not model_file.strip() or not os.path.isfile(model_file):
            raise ValueError(
                f"[VALIDATION] {context} source artifact for {model_name} does not exist or is not a file."
            )

        try:
            source_state = load_model(model_file)
        except Exception as exc:
            raise ValueError(
                f"[VALIDATION] {context} source artifact for {model_name} cannot be read."
            ) from exc

        if not isinstance(source_state, dict) or source_state.get("model_type") != model_name:
            raise ValueError(
                f"[VALIDATION] {context} source artifact for {model_name} has a mismatched model type."
            )

        fallback_params = canonical_params.get(model_name)
        if fallback_params is None:
            raise ValueError(
                f"[VALIDATION] {context} is missing fallback parameters for {model_name}."
            )

        if model_name == "ma":
            if "window" not in source_state:
                raise ValueError(
                    f"[VALIDATION] {context} MA source artifact is missing window."
                )
            fallback_params["window"] = _artifact_integer(
                source_state["window"], f"{context} MA window"
            )
        elif model_name == "es":
            if "alpha" not in source_state:
                raise ValueError(
                    f"[VALIDATION] {context} ES source artifact is missing alpha."
                )
            fallback_params["alpha"] = _artifact_float(
                source_state["alpha"], f"{context} ES alpha"
            )
        elif model_name == "arima":
            artifact_d = source_state.get("d")
            if artifact_d is None:
                order = source_state.get("order")
                artifact_d = order.get("d") if isinstance(order, dict) else None
            if artifact_d is None:
                raise ValueError(
                    f"[VALIDATION] {context} ARIMA source artifact is missing d."
                )
            canonical_params[model_name] = {
                "d": _artifact_integer(artifact_d, f"{context} ARIMA d")
            }
        else:
            try:
                canonical_params[model_name] = build_lstm_params_from_source_state(
                    parsed_data,
                    source_state,
                    fallback_params,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"[VALIDATION] {context} LSTM source artifact has invalid training configuration."
                ) from exc

        validated_files[model_name] = model_file

    return canonical_params, validated_files


def build_ensemble_subprocess_data(
    parsed_data: ParsedData,
    models_to_use: List[str],
    model_params: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    numeric_data_dicts, _ = build_numeric_subprocess_records(parsed_data)
    lstm_data_dicts = numeric_data_dicts
    if "lstm" in models_to_use:
        lstm_params = model_params.get("lstm", {})
        lstm_feature_keys = require_explicit_lstm_feature_names(
            lstm_params.get("lstm_features") if "lstm_features" in lstm_params else None,
            "Ensemble LSTM member",
        )
        lstm_data_dicts, _ = build_subprocess_records(parsed_data, feature_keys=lstm_feature_keys)

    return numeric_data_dicts, lstm_data_dicts


def get_member_model_path(save_path: str, model_name: str, round_idx: int | None = None) -> str:
    base_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    suffix = f"_round{round_idx}" if round_idx is not None else ""
    return os.path.join(base_dir, f"{base_name}_base_{model_name}{suffix}.pkl")


def remove_member_model_artifact(model_name: str, model_path: str) -> None:
    if os.path.exists(model_path):
        os.remove(model_path)

    if model_name == "lstm":
        keras_path = os.path.splitext(model_path)[0] + "_keras_model.h5"
        if os.path.exists(keras_path):
            os.remove(keras_path)


def persist_member_model_artifact(
    model_name: str,
    source_model_path: str,
    destination_model_path: str,
    *,
    move_source: bool,
) -> None:
    if os.path.normcase(os.path.realpath(source_model_path)) == os.path.normcase(os.path.realpath(destination_model_path)):
        return
    os.makedirs(os.path.dirname(destination_model_path), exist_ok=True)
    transfer = shutil.move if move_source else shutil.copy2
    transfer(source_model_path, destination_model_path)

    if model_name != "lstm":
        return

    lstm_state = load_model(destination_model_path)
    source_keras_path = lstm_state.get("keras_model_path")
    if not source_keras_path or not os.path.exists(source_keras_path):
        raise FileNotFoundError(f"LSTM Keras model file not found at {source_keras_path}")

    destination_keras_path = os.path.splitext(destination_model_path)[0] + "_keras_model.h5"
    transfer(source_keras_path, destination_keras_path)
    lstm_state["keras_model_path"] = destination_keras_path
    dump_model(lstm_state, destination_model_path)


def fit_and_predict_from_source_model(
    model_name: str,
    source_model_file: str,
    data: List[Dict[str, Any]],
    train_start: int,
    train_end: int,
    forecast_steps: int,
    timeout: int,
) -> List[float]:
    temp_model_file = os.path.join(
        tempfile.gettempdir(),
        f"ensemble_refit_{model_name}_{os.getpid()}.pkl",
    )

    try:
        fit_result = run_base_model_fit_subprocess(
            model_type=model_name,
            data=data,
            train_start=train_start,
            train_end=train_end,
            source_model_file=source_model_file,
            save_model_file=temp_model_file,
            fit_forecast_steps=forecast_steps if model_name == "lstm" else None,
            predict_steps=forecast_steps,
            timeout=timeout,
        )
        return extract_predictions_from_result(fit_result)
    finally:
        remove_member_model_artifact(model_name, temp_model_file)
