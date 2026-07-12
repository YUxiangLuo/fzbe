#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Literal, Tuple

from data_utils import ParsedData, build_numeric_subprocess_records, build_subprocess_records
from durable_io import fsync_file, fsync_parent_directory
from serialization import dump_model, load_model
from subprocess_utils import (
    extract_predictions_from_result,
    run_base_model_fit_subprocess,
    run_base_model_predict_subprocess,
)

BaseModelType = Literal["ma", "es", "arima", "lstm"]


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


def _copy_artifact_atomically(source_path: str, destination_path: str) -> None:
    """Publish a complete copy and make the directory entry crash-durable."""

    if os.path.abspath(source_path) == os.path.abspath(destination_path):
        return

    destination_dir = os.path.dirname(destination_path)
    os.makedirs(destination_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination_path)}.",
        suffix=".tmp",
        dir=destination_dir,
    )
    os.close(fd)
    try:
        shutil.copy2(source_path, temp_path)
        fsync_file(temp_path)
        os.replace(temp_path, destination_path)
        fsync_parent_directory(destination_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def persist_member_model_artifact(
    model_name: str,
    source_model_path: str,
    destination_model_path: str,
    *,
    move_source: bool,
) -> None:
    if os.path.abspath(source_model_path) == os.path.abspath(destination_model_path):
        return

    source_keras_path = None
    destination_keras_path = None
    if model_name == "lstm":
        lstm_state = load_model(source_model_path)
        source_keras_path = lstm_state.get("keras_model_path")
        if not source_keras_path or not os.path.exists(source_keras_path):
            raise FileNotFoundError(f"LSTM Keras model file not found at {source_keras_path}")

        destination_keras_path = os.path.splitext(destination_model_path)[0] + "_keras_model.h5"
        _copy_artifact_atomically(source_keras_path, destination_keras_path)
        lstm_state["keras_model_path"] = destination_keras_path
        dump_model(lstm_state, destination_model_path)
    else:
        _copy_artifact_atomically(source_model_path, destination_model_path)

    # A move is intentionally implemented as durable copy-then-delete. If the
    # process dies before deletion, recovery sees a harmless duplicate instead
    # of a missing source and an incomplete destination.
    if move_source:
        if os.path.exists(source_model_path):
            os.remove(source_model_path)
            fsync_parent_directory(source_model_path)
        if (
            source_keras_path
            and destination_keras_path
            and os.path.abspath(source_keras_path) != os.path.abspath(destination_keras_path)
            and os.path.exists(source_keras_path)
        ):
            os.remove(source_keras_path)
            fsync_parent_directory(source_keras_path)


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
