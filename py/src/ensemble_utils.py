#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from data_utils import (
    ParsedData,
    build_numeric_subprocess_records,
    build_subprocess_records,
    infer_feature_types,
)
from serialization import dump_model, load_model
from subprocess_utils import (
    extract_predictions_from_result,
    run_base_model_fit_subprocess,
    run_base_model_predict_subprocess,
)

BaseModelType = Literal["ma", "es", "arima", "lstm"]


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
    feature_keys = [
        feature_name
        for feature_name in source_model_state.get("feature_keys", []) or []
        if feature_name and feature_name != target_feature
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
        "look_back": int(source_model_state["look_back"]),
        "epochs": int(source_model_state.get("epochs", fallback_params.get("epochs", 20))),
        "feature_types": feature_types,
        "normalization": source_model_state.get(
            "normalization",
            fallback_params.get("normalization", "minmax"),
        ),
        "lstm_target_feature": target_feature,
    }
    if feature_keys:
        params["lstm_features"] = json.dumps(feature_keys, ensure_ascii=False)
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


def format_predictions(preds: List[float], std_dev: float = 0.0) -> List[Dict[str, float]]:
    return [{"prediction": float(prediction), "std_dev": float(std_dev)} for prediction in preds]


def compute_phase1_train_size(train_size: int) -> int:
    """Return the chronological internal training size used by all ensembles."""
    if train_size <= 15:
        return max(train_size - 3, int(train_size * 0.85))
    return max(3, int(train_size * 0.8))


def validate_internal_split_base_params(
    models_to_use: List[str],
    model_params: Dict[str, Dict[str, Any]],
    train_size: int,
    context: str,
) -> int:
    phase1_size = compute_phase1_train_size(train_size)
    if "ma" in models_to_use:
        window = int(model_params.get("ma", {}).get("window", 0))
        if window > phase1_size:
            raise ValueError(
                f"[VALIDATION] MA window ({window}) exceeds the {context} internal "
                f"training split ({phase1_size} points)."
            )
    if "arima" in models_to_use:
        d = int(model_params.get("arima", {}).get("d", 0))
        if phase1_size < d + 2:
            raise ValueError(
                f"[VALIDATION] {context} internal training split ({phase1_size} points) "
                f"must contain at least d + 2 points for ARIMA (d={d})."
            )
    return phase1_size


def normalize_demand_forecast(values: Any) -> np.ndarray:
    """Normalize forecasts representing physical demand, never signed residuals."""
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def parse_base_model_files_arg(raw_value: Any) -> Dict[str, str]:
    if raw_value is None or raw_value == "":
        return {}

    try:
        payload = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("[VALIDATION] --base-model-files must be valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("[VALIDATION] --base-model-files must be a JSON object.")

    return {
        str(model_name): model_file
        for model_name, model_file in payload.items()
        if isinstance(model_name, str) and isinstance(model_file, str) and model_file
    }


def build_ensemble_subprocess_data(
    parsed_data: ParsedData,
    models_to_use: List[str],
    model_params: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    numeric_data_dicts, _ = build_numeric_subprocess_records(parsed_data)
    lstm_data_dicts = numeric_data_dicts
    if "lstm" in models_to_use:
        lstm_feature_types = model_params.get("lstm", {}).get("feature_types", {})
        lstm_feature_keys = [k for k in lstm_feature_types.keys() if k != parsed_data.target_key]
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
        run_base_model_fit_subprocess(
            model_type=model_name,
            data=data,
            train_start=train_start,
            train_end=train_end,
            source_model_file=source_model_file,
            save_model_file=temp_model_file,
            fit_forecast_steps=forecast_steps if model_name == "lstm" else None,
            timeout=timeout,
        )

        prediction_result = run_base_model_predict_subprocess(
            model_type=model_name,
            model_file=temp_model_file,
            forecast_steps=forecast_steps,
            timeout=timeout,
        )
        return extract_predictions_from_result(prediction_result)
    finally:
        remove_member_model_artifact(model_name, temp_model_file)
