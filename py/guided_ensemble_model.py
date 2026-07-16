#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-based ensemble training runner for the student guided training UI.

Each invocation executes one ensemble training stage, persists intermediate
state to a pickle file, and prints a compact JSON payload for the backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data_utils import infer_feature_types, load_data_from_json
from error_protocol import format_known_error
from ensemble_utils import (
    build_ensemble_subprocess_data,
    generate_rolling_origin_member_predictions,
    get_member_model_path,
    normalize_demand_forecast,
    parse_base_model_files_arg,
    parse_base_model_names,
    plan_ensemble_internal_holdout,
    persist_member_model_artifact,
    remove_member_model_artifact,
    require_explicit_lstm_feature_names,
    resolve_base_model_artifact_configs,
    rolling_origin_ranges,
)
from dynamic_config import derive_boosting_max_rounds
from logger import get_log_contents
from metrics import mae, mape, r2, rmse
from model_info import attach_method_metadata
from serialization import dump_model
from subprocess_utils import (
    extract_predictions_from_result,
    extract_uncertainty_profile_from_result,
    run_base_model_fit_subprocess,
    run_base_model_predict_subprocess,
    run_base_model_subprocess,
)
from uncertainty import ensemble_growth_profile
from models import boosting, stacking, weighted_avg


ENSEMBLE_MODULES = {
    "weighted_avg": weighted_avg,
    "stacking": stacking,
    "boosting": boosting,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_state(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _save_state(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp-{os.getpid()}"
    with open(temp_path, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _sha256_file(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_receipt(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp-{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _load_existing_receipt(path: str, attempt_id: str, step_id: str) -> Dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("attempt_id") != attempt_id or payload.get("step_id") != step_id:
        return None
    candidate_path = payload.get("candidate_state_path")
    if not isinstance(candidate_path, str) or _sha256_file(candidate_path) != payload.get("candidate_state_sha256"):
        return None
    return payload


def _get_arg(args, name: str, default: Any = None, cast_type=None):
    value = getattr(args, name.replace("-", "_"), None)
    if value is None:
        return default
    if cast_type:
        return cast_type(value)
    return value


def _parse_common_ranges(args) -> Tuple[int, int, int, int]:
    return int(args.train_start), int(args.train_end), int(args.eval_start), int(args.eval_end)


def _read_data_json(args) -> str:
    data_file = getattr(args, "data_file", None)
    if data_file:
        with open(data_file, "r", encoding="utf-8") as handle:
            return handle.read()
    if args.data:
        return args.data
    raise ValueError("[DATA] 训练数据快照缺失。")


def _range_to_state(range_value: range) -> Dict[str, int]:
    return {"start": int(range_value.start), "stop": int(range_value.stop)}


def _state_range(state: Dict[str, Any], key: str) -> range:
    range_state = state[key]
    return range(int(range_state["start"]), int(range_state["stop"]))


def _parse_models(args) -> List[str]:
    return parse_base_model_names(args.models, "Guided ensemble")


def _build_model_params(
    parsed_data,
    train_size: int,
    models_to_use: List[str],
    args,
) -> Dict[str, Dict[str, Any]]:
    auto_window = max(3, min(12, train_size // 4))
    model_params: Dict[str, Dict[str, Any]] = {}

    if "arima" in models_to_use:
        model_params["arima"] = {
            "d": _get_arg(args, "d", 0, int),
        }

    if "es" in models_to_use:
        model_params["es"] = {"alpha": _get_arg(args, "alpha", 0.5, float)}

    if "ma" in models_to_use:
        model_params["ma"] = {"window": _get_arg(args, "window", auto_window, int)}

    if "lstm" in models_to_use:
        feature_types_str = _get_arg(args, "feature_types", None)
        lstm_features_str = _get_arg(args, "lstm_features", None)
        target_feature = _get_arg(args, "lstm_target_feature", parsed_data.target_key)
        lstm_feature_names = require_explicit_lstm_feature_names(
            lstm_features_str,
            f"Guided {args.model_type}",
        )

        if feature_types_str:
            feature_types = json.loads(feature_types_str)
        else:
            feature_types = infer_feature_types(
                parsed_data=parsed_data,
                target_key=target_feature,
                feature_keys=lstm_feature_names,
            )

        allowed_keys = set(lstm_feature_names + [target_feature])
        feature_types = {key: val for key, val in feature_types.items() if key in allowed_keys}
        feature_types[target_feature] = "numeric"

        lstm_params = {
            "feature_types": feature_types,
            "normalization": _get_arg(args, "normalization", "minmax"),
            "lstm_target_feature": target_feature,
            "lstm_features": json.dumps(lstm_feature_names, ensure_ascii=False),
        }

        model_params["lstm"] = lstm_params

    return model_params


def _initialize_state(model_type: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    module = ENSEMBLE_MODULES[model_type]
    parsed_data = load_data_from_json(_read_data_json(args))
    ts, te, es_idx, ee = _parse_common_ranges(args)
    train_range = range(ts, te + 1)
    eval_range = range(es_idx, ee + 1)
    models_to_use = _parse_models(args)
    train_size = te - ts + 1
    base_model_files = parse_base_model_files_arg(_get_arg(args, "base_model_files", None))
    model_params = _build_model_params(
        parsed_data,
        train_size,
        models_to_use,
        args,
    )
    model_params, base_model_files = resolve_base_model_artifact_configs(
        parsed_data,
        models_to_use,
        model_params,
        base_model_files,
        f"Guided {model_type}",
    )

    module._validate_training_args(
        parsed_data,
        ts,
        te,
        es_idx,
        ee,
        models_to_use,
        model_params,
        base_model_files,
    )
    split_purpose = {
        "weighted_avg": "weighted",
        "stacking": "stacking",
        "boosting": "boosting",
    }[model_type]
    split_plan = plan_ensemble_internal_holdout(
        parsed_data,
        train_range,
        models_to_use,
        model_params,
        split_purpose,
    )

    state.clear()
    state.update({
        "model_type": model_type,
        "parsed_data": parsed_data,
        "train_range": _range_to_state(train_range),
        "eval_range": _range_to_state(eval_range),
        "models": models_to_use,
        "model_params": model_params,
        "base_model_files": base_model_files,
        "save_model": args.save_model,
        "artifact_dir": args.artifact_dir,
        "split_plan": split_plan,
    })

    if model_type == "boosting":
        state["max_rounds"] = derive_boosting_max_rounds(
            len(models_to_use), int(split_plan["calibration_size"])
        )
        state["minimum_relative_improvement"] = (
            0.05 if int(split_plan["calibration_size"]) < 5 else 0.01
        )

    return {
        "data_points": parsed_data.n_obs,
        "train_size": len(train_range),
        "evaluate_size": len(eval_range),
        "models": models_to_use,
        "reused_artifacts": sorted(base_model_files.keys()),
    }


def _base_timeout(models_to_use: List[str], model_params: Dict[str, Dict[str, Any]]) -> int:
    if "lstm" not in models_to_use:
        return 120
    return 600


def _rolling_oof_predictions(
    parsed_data,
    models_to_use: List[str],
    model_params: Dict[str, Dict[str, Any]],
    base_model_files: Dict[str, str],
    train_range: range,
    split_plan: Dict[str, Any],
    source_context: str,
) -> Tuple[Dict[str, List[float]], List[float], List[Dict[str, Any]]]:
    targets, predictions, fold_diagnostics = (
        generate_rolling_origin_member_predictions(
            parsed_data,
            train_range,
            models_to_use,
            model_params,
            base_model_files,
            split_plan,
            source_context,
        )
    )
    return (
        {name: values.tolist() for name, values in predictions.items()},
        targets.tolist(),
        fold_diagnostics,
    )


def _rolling_oof_fold_prediction(
    parsed_data,
    model_name: str,
    model_params: Dict[str, Dict[str, Any]],
    base_model_files: Dict[str, str],
    train_range: range,
    split_plan: Dict[str, Any],
    fold_index: int,
    source_context: str,
) -> Tuple[List[float], List[float], Dict[str, Any]] | None:
    """Run one member on one rolling-origin fold for a bounded checkpoint."""
    fold_ranges = rolling_origin_ranges(train_range, split_plan)
    if fold_index < 0:
        raise ValueError("[VALIDATION] Rolling-origin fold index must be positive.")
    if fold_index >= len(fold_ranges):
        return None

    prefix_range, validation_range = fold_ranges[fold_index]
    single_fold_plan = {
        **split_plan,
        "fold_count": 1,
        "calibration_size": len(validation_range),
        "folds": [{
            "fold_index": fold_index,
            "prefix_size": len(prefix_range),
            "validation_start_offset": validation_range.start - train_range.start,
            "validation_stop_offset": validation_range.stop - train_range.start,
            "validation_size": len(validation_range),
        }],
    }
    predictions, targets, fold_diagnostics = _rolling_oof_predictions(
        parsed_data,
        [model_name],
        model_params,
        base_model_files,
        train_range,
        single_fold_plan,
        source_context,
    )
    detail = dict(fold_diagnostics[0])
    detail["fold_index"] = fold_index
    return predictions[model_name], targets, detail


def _record_rolling_oof_fold(
    state: Dict[str, Any],
    storage_key: str,
    model_name: str,
    fold_index: int,
    prediction: List[float],
    targets: List[float],
    diagnostic: Dict[str, Any],
) -> None:
    model_folds = state.setdefault(storage_key, {}).setdefault(model_name, {})
    if fold_index in model_folds:
        raise ValueError("[VALIDATION] Rolling-origin member fold was already evaluated.")
    model_folds[fold_index] = prediction

    target_folds = state.setdefault("oof_target_folds", {})
    existing_targets = target_folds.get(fold_index)
    if existing_targets is not None and not np.array_equal(
        np.asarray(existing_targets, dtype=float),
        np.asarray(targets, dtype=float),
    ):
        raise ValueError("[VALIDATION] Rolling-origin target fold is inconsistent across members.")
    target_folds[fold_index] = targets
    _merge_fold_diagnostics(state, [diagnostic])


def _assemble_rolling_oof_folds(
    state: Dict[str, Any],
    storage_key: str,
    models_to_use: List[str],
) -> Tuple[Dict[str, List[float]], List[float]]:
    fold_count = len(state["split_plan"].get("folds") or []) or 1
    expected_folds = list(range(fold_count))
    stored_predictions = state.get(storage_key)
    target_folds = state.get("oof_target_folds")
    if not isinstance(stored_predictions, dict) or not isinstance(target_folds, dict):
        raise ValueError("[VALIDATION] Rolling-origin fold checkpoints are missing.")

    missing_targets = [index + 1 for index in expected_folds if index not in target_folds]
    if missing_targets:
        raise ValueError(
            "[VALIDATION] Rolling-origin target folds are incomplete: "
            + ", ".join(map(str, missing_targets))
        )

    assembled: Dict[str, List[float]] = {}
    for model_name in models_to_use:
        model_folds = stored_predictions.get(model_name)
        if not isinstance(model_folds, dict):
            raise ValueError(
                f"[VALIDATION] Rolling-origin predictions are missing for {model_name}."
            )
        missing = [index + 1 for index in expected_folds if index not in model_folds]
        if missing:
            raise ValueError(
                f"[VALIDATION] Rolling-origin folds are incomplete for {model_name}: "
                + ", ".join(map(str, missing))
            )
        assembled[model_name] = [
            value
            for index in expected_folds
            for value in model_folds[index]
        ]

    targets = [
        value
        for index in expected_folds
        for value in target_folds[index]
    ]
    return assembled, targets


def _merge_fold_diagnostics(
    state: Dict[str, Any],
    fold_diagnostics: List[Dict[str, Any]],
) -> None:
    existing = state.setdefault("fold_diagnostics", [])
    by_index = {
        int(item["fold_index"]): item
        for item in existing
        if isinstance(item, dict) and "fold_index" in item
    }
    for detail in fold_diagnostics:
        fold_index = int(detail["fold_index"])
        if fold_index not in by_index:
            by_index[fold_index] = {
                **detail,
                "member_rmse": dict(detail.get("member_rmse") or {}),
            }
        else:
            by_index[fold_index].setdefault("member_rmse", {}).update(
                detail.get("member_rmse") or {}
            )
    merged = [by_index[index] for index in sorted(by_index)]
    state["fold_diagnostics"] = merged
    state["split_plan"]["fold_diagnostics"] = merged


def _rolling_oof_targets(
    parsed_data,
    train_range: range,
    split_plan: Dict[str, Any],
) -> np.ndarray:
    return np.concatenate([
        np.asarray(parsed_data.y[validation_range], dtype=float)
        for _prefix_range, validation_range in rolling_origin_ranges(
            train_range,
            split_plan,
        )
    ])


def _eval_member_predictions(
    module,
    prefix: str,
    source_context: str,
    parsed_data,
    train_range: range,
    eval_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Dict[str, Any]],
    base_model_files: Dict[str, str],
    artifact_dir: str,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, Any]]:
    numeric_data_dicts, lstm_data_dicts = build_ensemble_subprocess_data(parsed_data, models_to_use, model_params)
    eval_predictions: Dict[str, List[float]] = {}
    full_profiles: Dict[str, List[float]] = {}
    base_model_states: Dict[str, Any] = {}
    eval_forecast_steps = eval_range.stop - train_range.stop
    eval_forecast_offset = eval_range.start - train_range.stop
    timeout = _base_timeout(models_to_use, model_params)

    for model_name in models_to_use:
        params = model_params[model_name]
        source_model_file = base_model_files.get(model_name)
        temp_model_file = os.path.join(artifact_dir, f"{prefix}_{model_name}_{os.getpid()}.pkl")

        if source_model_file:
            if model_name == "lstm":
                source_model_state = module._load_lstm_source_state(source_model_file, source_context)
                source_forecast_strategy = module._get_lstm_source_forecast_strategy(source_model_state)
                source_forecast_horizon = int(source_model_state.get("forecast_horizon", 0) or 0)
                if source_forecast_strategy == "direct_multi_horizon" and eval_forecast_steps > source_forecast_horizon:
                    adjusted_source_model_file = source_model_file
                    temp_source_model_file = None
                    try:
                        adjusted_source_model_file, temp_source_model_file = (
                            module._prepare_lstm_source_model_for_requested_steps(
                                source_model_file,
                                source_model_state,
                                requested_steps=eval_forecast_steps,
                            )
                        )
                        fit_result = run_base_model_fit_subprocess(
                            model_type=model_name,
                            data=lstm_data_dicts,
                            train_start=train_range.start,
                            train_end=train_range.stop - 1,
                            source_model_file=adjusted_source_model_file,
                            save_model_file=temp_model_file,
                            fit_forecast_steps=eval_forecast_steps,
                            predict_steps=eval_forecast_steps,
                            timeout=timeout,
                        )
                    finally:
                        if temp_source_model_file and os.path.exists(temp_source_model_file):
                            os.remove(temp_source_model_file)

                    y_pred_eval = extract_predictions_from_result(fit_result)[
                        eval_forecast_offset: eval_forecast_offset + len(eval_range)
                    ]
                    eval_predictions[model_name] = normalize_demand_forecast(y_pred_eval).tolist()
                    full_profiles[model_name] = (
                        extract_uncertainty_profile_from_result(fit_result)
                    )
                    base_model_states[model_name] = {
                        "model_type": model_name,
                        "model_file": temp_model_file,
                        "params": params,
                        "artifact_origin": "temporary_training",
                    }
                    continue

            result = run_base_model_predict_subprocess(
                model_type=model_name,
                model_file=source_model_file,
                forecast_steps=eval_forecast_steps,
                timeout=timeout,
            )
            y_pred_eval = extract_predictions_from_result(result)[
                eval_forecast_offset: eval_forecast_offset + len(eval_range)
            ]
            eval_predictions[model_name] = normalize_demand_forecast(y_pred_eval).tolist()
            full_profiles[model_name] = (
                extract_uncertainty_profile_from_result(result)
            )
            base_model_states[model_name] = {
                "model_type": model_name,
                "model_file": source_model_file,
                "params": params,
                "artifact_origin": "standalone_base",
            }
            continue

        params_with_save = params.copy()
        params_with_save["save_model"] = temp_model_file
        result = run_base_model_subprocess(
            model_type=model_name,
            data=lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
            train_start=train_range.start,
            train_end=train_range.stop - 1,
            eval_start=train_range.stop,
            eval_end=eval_range.stop - 1,
            model_params=params_with_save,
            timeout=timeout,
        )
        full_predictions = extract_predictions_from_result(result)
        eval_predictions[model_name] = normalize_demand_forecast(
            full_predictions[eval_forecast_offset: eval_forecast_offset + len(eval_range)]
        ).tolist()
        full_profiles[model_name] = extract_uncertainty_profile_from_result(result)
        base_model_states[model_name] = {
            "model_type": model_name,
            "model_file": temp_model_file,
            "params": params,
            "artifact_origin": "temporary_training",
        }

    return eval_predictions, full_profiles, base_model_states


def _save_flat_ensemble_artifact(module, state: Dict[str, Any], extra_state: Dict[str, Any]) -> str:
    save_path = state["save_model"]
    base_model_states = state["base_model_states"]
    for model_name, model_state in base_model_states.items():
        source_file = model_state["model_file"]
        perm_file = get_member_model_path(save_path, model_name)
        persist_member_model_artifact(
            model_name,
            source_file,
            perm_file,
            move_source=False,
        )
        model_state["model_file"] = perm_file
        model_state.pop("artifact_origin", None)

    model_state = {
        "model_type": state["model_type"],
        "models": state["models"],
        "base_model_states": base_model_states,
        "split_plan": state["split_plan"],
        "uncertainty": state["uncertainty"],
        **extra_state,
    }
    attach_method_metadata(model_state, module.METHOD_NAME, module.FORECAST_STRATEGY, module.IMPLEMENTATION_NOTES)
    dump_model(model_state, save_path)
    return save_path


def _run_weighted(step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if step_id == "prepare_data":
        return _initialize_state("weighted_avg", args, state)

    parsed_data = state["parsed_data"]
    train_range = _state_range(state, "train_range")
    eval_range = _state_range(state, "eval_range")
    models_to_use = state["models"]
    model_params = state["model_params"]
    base_model_files = state["base_model_files"]

    fold_member_match = re.fullmatch(
        r"validation_prediction_fold_(\d+)_(ma|es|arima|lstm)",
        step_id,
    )
    if fold_member_match:
        fold_index = int(fold_member_match.group(1)) - 1
        model_name = fold_member_match.group(2)
        if model_name not in models_to_use:
            raise ValueError("[VALIDATION] Weighted member step is not part of this session.")
        fold_count = len(rolling_origin_ranges(train_range, state["split_plan"]))
        result = _rolling_oof_fold_prediction(
            parsed_data,
            model_name,
            model_params,
            base_model_files,
            train_range,
            state["split_plan"],
            fold_index,
            "Saved Weighted Average LSTM base model",
        )
        if result is None:
            return {
                "model": model_name,
                "fold": fold_index + 1,
                "fold_count": fold_count,
                "skipped": True,
            }
        prediction, targets, diagnostic = result
        _record_rolling_oof_fold(
            state,
            "weight_val_prediction_folds",
            model_name,
            fold_index,
            prediction,
            targets,
            diagnostic,
        )
        return {
            "model": model_name,
            "fold": fold_index + 1,
            "fold_count": fold_count,
            "predicted_points": len(prediction),
        }

    member_match = re.fullmatch(
        r"(validation_prediction|eval_prediction)_(ma|es|arima|lstm)",
        step_id,
    )
    if member_match:
        action, model_name = member_match.groups()
        if model_name not in models_to_use:
            raise ValueError("[VALIDATION] Weighted member step is not part of this session.")
        if action == "validation_prediction":
            predictions, targets, fold_diagnostics = _rolling_oof_predictions(
                parsed_data,
                [model_name],
                model_params,
                base_model_files,
                train_range,
                state["split_plan"],
                "Saved Weighted Average LSTM base model",
            )
            prediction = predictions[model_name]
            state.setdefault("weight_val_predictions", {})[model_name] = prediction
            state["oof_targets"] = targets
            _merge_fold_diagnostics(state, fold_diagnostics)
            return {"model": model_name, "predicted_points": len(prediction)}

        predictions, profiles, model_states = _eval_member_predictions(
            weighted_avg,
            "weighted_avg_eval_base",
            "Saved Weighted Average LSTM base model",
            parsed_data,
            train_range,
            eval_range,
            [model_name],
            model_params,
            base_model_files,
            state["artifact_dir"],
        )
        state.setdefault("eval_predictions", {}).update(predictions)
        state.setdefault("full_uncertainty_profiles", {}).update(profiles)
        state.setdefault("base_model_states", {}).update(model_states)
        return {
            "model": model_name,
            "predicted_points": len(predictions[model_name]),
            "reused_artifact": model_states[model_name].get("artifact_origin") == "standalone_base",
        }

    if step_id == "split_validation":
        split = int(state["split_plan"]["prefix_size"])
        weight_train_range = range(train_range.start, train_range.start + split)
        weight_val_range = range(train_range.start + split, train_range.stop)
        state["weight_train_range"] = _range_to_state(weight_train_range)
        state["weight_val_range"] = _range_to_state(weight_val_range)
        return {
            "train_size": len(train_range),
            "weight_train_size": len(weight_train_range),
            "weight_val_size": len(weight_val_range),
            "evaluate_size": len(eval_range),
            "split_plan": state["split_plan"],
        }

    if step_id == "validation_predictions":
        predictions, targets, fold_diagnostics = _rolling_oof_predictions(
            parsed_data,
            models_to_use,
            model_params,
            base_model_files,
            train_range,
            state["split_plan"],
            "Saved Weighted Average LSTM base model",
        )
        state["weight_val_predictions"] = predictions
        state["oof_targets"] = targets
        _merge_fold_diagnostics(state, fold_diagnostics)
        return {
            "predicted_points": {name: len(values) for name, values in predictions.items()},
            "models": models_to_use,
        }

    if step_id == "compute_weights":
        if "weight_val_prediction_folds" in state:
            predictions, targets = _assemble_rolling_oof_folds(
                state,
                "weight_val_prediction_folds",
                models_to_use,
            )
            state["weight_val_predictions"] = predictions
            state["oof_targets"] = targets
        predictions = state["weight_val_predictions"]
        y_true_val = np.asarray(
            state.get("oof_targets")
            or _rolling_oof_targets(parsed_data, train_range, state["split_plan"]),
            dtype=float,
        )
        weights, diagnostics = weighted_avg._compute_inverse_residual_mse_weights(
            y_true_val, predictions
        )
        state["weights"] = weights
        state["weight_diagnostics"] = diagnostics
        return {"weights": weights, "weight_diagnostics": diagnostics}

    if step_id == "eval_predictions":
        eval_predictions, full_profiles, base_model_states = _eval_member_predictions(
            weighted_avg,
            "weighted_avg_eval_base",
            "Saved Weighted Average LSTM base model",
            parsed_data,
            train_range,
            eval_range,
            models_to_use,
            model_params,
            base_model_files,
            state["artifact_dir"],
        )
        state["eval_predictions"] = eval_predictions
        state["full_uncertainty_profiles"] = full_profiles
        state["base_model_states"] = base_model_states
        return {
            "predicted_points": {name: len(values) for name, values in eval_predictions.items()},
            "reused_artifacts": [
                name for name, model_state in base_model_states.items()
                if model_state.get("artifact_origin") == "standalone_base"
            ],
        }

    if step_id == "blend_evaluate":
        eval_predictions = state["eval_predictions"]
        weights = state["weights"]
        x_eval = np.array([eval_predictions[name] for name in models_to_use]).T
        model_weights = np.array([weights[name] for name in models_to_use])
        y_pred_final = np.sum(x_eval * model_weights, axis=1)
        y_pred_final = normalize_demand_forecast(y_pred_final)
        y_true = parsed_data.y[eval_range]
        weight_val_predictions = state["weight_val_predictions"]
        x_weight_val = np.array([weight_val_predictions[name] for name in models_to_use]).T
        weight_val_blend = np.sum(x_weight_val * model_weights, axis=1)
        weight_val_blend = normalize_demand_forecast(weight_val_blend)
        oof_targets = np.asarray(
            state.get("oof_targets")
            or _rolling_oof_targets(parsed_data, train_range, state["split_plan"]),
            dtype=float,
        )
        weight_val_residuals = oof_targets - weight_val_blend
        eval_forecast_steps = eval_range.stop - train_range.stop
        eval_forecast_offset = eval_range.start - train_range.stop
        full_uncertainty_profile, uncertainty_summary = ensemble_growth_profile(
            state["full_uncertainty_profiles"],
            weights,
            weight_val_residuals,
            parsed_data.y[train_range],
            eval_forecast_steps,
            calibration_source="rolling_origin_oof",
        )
        uncertainty_profile = full_uncertainty_profile[
            eval_forecast_offset: eval_forecast_offset + len(eval_range)
        ]
        uncertainty_summary.setdefault("calibration_source", "rolling_origin_oof")
        timestamps = parsed_data.timestamps
        metrics = {
            "rmse": rmse(y_true, y_pred_final),
            "mae": mae(y_true, y_pred_final),
            "mape": mape(y_true, y_pred_final),
            "r2": r2(y_true, y_pred_final),
            "evaluate_indices": list(eval_range),
            "evaluate_months": [timestamps[i] for i in eval_range] if timestamps else [],
        }
        final_result = {
            "mode": "weighted_avg_train_eval",
            "models": models_to_use,
            "weights": weights,
            "weight_diagnostics": state["weight_diagnostics"],
            "split_plan": state["split_plan"],
            "metrics": metrics,
            "eval_y_true": y_true.tolist(),
            "predictions": weighted_avg._format_predictions(
                y_pred_final.tolist(),
                uncertainty_profile,
                {**uncertainty_summary, "horizon_start": eval_forecast_offset + 1},
            ),
        }
        attach_method_metadata(final_result, weighted_avg.METHOD_NAME, weighted_avg.FORECAST_STRATEGY, weighted_avg.IMPLEMENTATION_NOTES)
        state["metrics"] = metrics
        state["eval_y_true"] = y_true.tolist()
        state["y_pred"] = y_pred_final.tolist()
        state["uncertainty"] = {
            "calibration_residuals": weight_val_residuals.tolist(),
            "calibration_source": "rolling_origin_oof",
            "reference_series": parsed_data.y[train_range].tolist(),
            "profile": full_uncertainty_profile,
            "summary": uncertainty_summary,
        }
        state["final_result"] = final_result
        return {"metrics": metrics, "uncertainty_summary": uncertainty_summary}

    if step_id == "save_artifact":
        save_path = _save_flat_ensemble_artifact(
            weighted_avg,
            state,
            {
                "weights": state["weights"],
                "weight_diagnostics": state["weight_diagnostics"],
            },
        )
        final_result = dict(state["final_result"])
        final_result["saved_model"] = save_path
        final_result["logs"] = get_log_contents()
        return {"output": {"saved_model": save_path}, "final_result": final_result}

    raise ValueError(f"[VALIDATION] Unknown weighted average guided step: {step_id}")


def _run_stacking(step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if step_id == "prepare_data":
        return _initialize_state("stacking", args, state)

    parsed_data = state["parsed_data"]
    train_range = _state_range(state, "train_range")
    eval_range = _state_range(state, "eval_range")
    models_to_use = state["models"]
    model_params = state["model_params"]
    base_model_files = state["base_model_files"]

    fold_member_match = re.fullmatch(
        r"level1_prediction_fold_(\d+)_(ma|es|arima|lstm)",
        step_id,
    )
    if fold_member_match:
        fold_index = int(fold_member_match.group(1)) - 1
        model_name = fold_member_match.group(2)
        if model_name not in models_to_use:
            raise ValueError("[VALIDATION] Stacking member step is not part of this session.")
        fold_count = len(rolling_origin_ranges(train_range, state["split_plan"]))
        result = _rolling_oof_fold_prediction(
            parsed_data,
            model_name,
            model_params,
            base_model_files,
            train_range,
            state["split_plan"],
            fold_index,
            "Saved Stacking LSTM base model",
        )
        if result is None:
            return {
                "model": model_name,
                "fold": fold_index + 1,
                "fold_count": fold_count,
                "skipped": True,
            }
        prediction, targets, diagnostic = result
        _record_rolling_oof_fold(
            state,
            "level1_prediction_folds",
            model_name,
            fold_index,
            prediction,
            targets,
            diagnostic,
        )
        return {
            "model": model_name,
            "fold": fold_index + 1,
            "fold_count": fold_count,
            "predicted_points": len(prediction),
        }

    member_match = re.fullmatch(
        r"(level1_prediction|eval_prediction)_(ma|es|arima|lstm)",
        step_id,
    )
    if member_match:
        action, model_name = member_match.groups()
        if model_name not in models_to_use:
            raise ValueError("[VALIDATION] Stacking member step is not part of this session.")
        if action == "level1_prediction":
            predictions, targets, fold_diagnostics = _rolling_oof_predictions(
                parsed_data,
                [model_name],
                model_params,
                base_model_files,
                train_range,
                state["split_plan"],
                "Saved Stacking LSTM base model",
            )
            prediction = predictions[model_name]
            state.setdefault("level1_prediction_map", {})[model_name] = prediction
            state["oof_targets"] = targets
            _merge_fold_diagnostics(state, fold_diagnostics)
            return {"model": model_name, "predicted_points": len(prediction)}

        predictions, profiles, model_states = _eval_member_predictions(
            stacking,
            "stacking_eval_base",
            "Saved Stacking LSTM base model",
            parsed_data,
            train_range,
            eval_range,
            [model_name],
            model_params,
            base_model_files,
            state["artifact_dir"],
        )
        state.setdefault("eval_prediction_map", {}).update(predictions)
        state.setdefault("full_uncertainty_profiles", {}).update(profiles)
        state.setdefault("base_model_states", {}).update(model_states)
        return {
            "model": model_name,
            "predicted_points": len(predictions[model_name]),
            "reused_artifact": model_states[model_name].get("artifact_origin") == "standalone_base",
        }

    if step_id == "split_levels":
        split = int(state["split_plan"]["prefix_size"])
        level0_range = range(train_range.start, train_range.start + split)
        level1_range = range(train_range.start + split, train_range.stop)
        state["level0_range"] = _range_to_state(level0_range)
        state["level1_range"] = _range_to_state(level1_range)
        return {
            "train_size": len(train_range),
            "level0_size": len(level0_range),
            "level1_size": len(level1_range),
            "evaluate_size": len(eval_range),
            "split_plan": state["split_plan"],
        }

    if step_id == "level1_predictions":
        prediction_map, targets, fold_diagnostics = _rolling_oof_predictions(
            parsed_data,
            models_to_use,
            model_params,
            base_model_files,
            train_range,
            state["split_plan"],
            "Saved Stacking LSTM base model",
        )
        state["level1_predictions"] = [prediction_map[name] for name in models_to_use]
        state["oof_targets"] = targets
        _merge_fold_diagnostics(state, fold_diagnostics)
        return {
            "predicted_points": {name: len(values) for name, values in prediction_map.items()},
            "models": models_to_use,
        }

    if step_id == "fit_meta_model":
        if "level1_prediction_folds" in state:
            prediction_map, targets = _assemble_rolling_oof_folds(
                state,
                "level1_prediction_folds",
                models_to_use,
            )
            state["level1_prediction_map"] = prediction_map
            state["oof_targets"] = targets
        if "level1_prediction_map" in state:
            state["level1_predictions"] = [
                state["level1_prediction_map"][name] for name in models_to_use
            ]
        x_meta_train = np.column_stack(state["level1_predictions"])
        y_meta_train = np.asarray(
            state.get("oof_targets")
            or _rolling_oof_targets(parsed_data, train_range, state["split_plan"]),
            dtype=float,
        )
        meta_model = stacking._fit_meta_model(x_meta_train, y_meta_train, models_to_use)
        state["meta_model"] = meta_model
        return {
            "meta_model": meta_model,
            "input_shape": list(x_meta_train.shape),
        }

    if step_id == "eval_predictions":
        eval_prediction_map, full_profiles, base_model_states = _eval_member_predictions(
            stacking,
            "stacking_eval_base",
            "Saved Stacking LSTM base model",
            parsed_data,
            train_range,
            eval_range,
            models_to_use,
            model_params,
            base_model_files,
            state["artifact_dir"],
        )
        state["eval_predictions"] = [eval_prediction_map[name] for name in models_to_use]
        state["full_uncertainty_profiles"] = full_profiles
        state["base_model_states"] = base_model_states
        return {
            "predicted_points": {name: len(values) for name, values in eval_prediction_map.items()},
            "reused_artifacts": [
                name for name, model_state in base_model_states.items()
                if model_state.get("artifact_origin") == "standalone_base"
            ],
        }

    if step_id == "meta_evaluate":
        if "eval_prediction_map" in state:
            state["eval_predictions"] = [
                state["eval_prediction_map"][name] for name in models_to_use
            ]
        x_meta_eval = np.column_stack(state["eval_predictions"])
        y_pred_final = stacking._predict_with_meta_model(state["meta_model"], x_meta_eval)
        y_pred_final = normalize_demand_forecast(y_pred_final)
        y_true = parsed_data.y[eval_range]
        x_meta_train = np.column_stack(state["level1_predictions"])
        level1_blend = normalize_demand_forecast(
            stacking._predict_with_meta_model(state["meta_model"], x_meta_train)
        )
        oof_targets = np.asarray(
            state.get("oof_targets")
            or _rolling_oof_targets(parsed_data, train_range, state["split_plan"]),
            dtype=float,
        )
        level1_residuals = oof_targets - level1_blend
        coefficients = dict(zip(
            state["meta_model"]["model_names"],
            state["meta_model"]["coefficients"],
        ))
        eval_forecast_steps = eval_range.stop - train_range.stop
        eval_forecast_offset = eval_range.start - train_range.stop
        full_uncertainty_profile, uncertainty_summary = ensemble_growth_profile(
            state["full_uncertainty_profiles"],
            coefficients,
            level1_residuals,
            parsed_data.y[train_range],
            eval_forecast_steps,
            calibration_source="rolling_origin_oof",
        )
        uncertainty_profile = full_uncertainty_profile[
            eval_forecast_offset: eval_forecast_offset + len(eval_range)
        ]
        uncertainty_summary.setdefault("calibration_source", "rolling_origin_oof")
        timestamps = parsed_data.timestamps
        metrics = {
            "rmse": rmse(y_true, y_pred_final),
            "mae": mae(y_true, y_pred_final),
            "mape": mape(y_true, y_pred_final),
            "r2": r2(y_true, y_pred_final),
            "evaluate_indices": list(eval_range),
            "evaluate_months": [timestamps[i] for i in eval_range] if timestamps else [],
        }
        final_result = {
            "mode": "stacking_train_eval",
            "models": models_to_use,
            "meta_model": state["meta_model"],
            "split_plan": state["split_plan"],
            "metrics": metrics,
            "eval_y_true": y_true.tolist(),
            "predictions": stacking._format_predictions(
                y_pred_final.tolist(),
                uncertainty_profile,
                {**uncertainty_summary, "horizon_start": eval_forecast_offset + 1},
            ),
        }
        attach_method_metadata(final_result, stacking.METHOD_NAME, stacking.FORECAST_STRATEGY, stacking.IMPLEMENTATION_NOTES)
        state["metrics"] = metrics
        state["eval_y_true"] = y_true.tolist()
        state["y_pred"] = y_pred_final.tolist()
        state["uncertainty"] = {
            "calibration_residuals": level1_residuals.tolist(),
            "calibration_source": "rolling_origin_oof",
            "reference_series": parsed_data.y[train_range].tolist(),
            "profile": full_uncertainty_profile,
            "summary": uncertainty_summary,
        }
        state["final_result"] = final_result
        return {"metrics": metrics, "uncertainty_summary": uncertainty_summary}

    if step_id == "save_artifact":
        save_path = _save_flat_ensemble_artifact(stacking, state, {"meta_model": state["meta_model"]})
        final_result = dict(state["final_result"])
        final_result["saved_model"] = save_path
        final_result["logs"] = get_log_contents()
        return {"output": {"saved_model": save_path}, "final_result": final_result}

    raise ValueError(f"[VALIDATION] Unknown stacking guided step: {step_id}")


def _boosting_round_match(step_id: str) -> Tuple[int, str] | None:
    match = re.fullmatch(r"round_(\d+)_(evaluate_candidates|select_winner)", step_id)
    if not match:
        return None
    return int(match.group(1)) - 1, match.group(2)


def _pooled_guided_boosting_residuals(state: Dict[str, Any]) -> np.ndarray:
    folds = state.get("boosting_fold_states")
    if not isinstance(folds, list) or not folds:
        raise ValueError("[VALIDATION] Boosting rolling-origin fold state is missing.")
    return np.concatenate([
        np.asarray(fold["residual_val"], dtype=float)
        for fold in folds
    ])


def _evaluate_guided_boosting_candidate(
    parsed_data,
    state: Dict[str, Any],
    model_name: str,
    params: Dict[str, Any],
    round_idx: int,
) -> Dict[str, Any]:
    normalized_folds = [
        {
            **fold,
            "prefix_range": _state_range(fold, "prefix_range"),
            "validation_range": _state_range(fold, "validation_range"),
        }
        for fold in state["boosting_fold_states"]
    ]
    return boosting.evaluate_rolling_origin_candidate(
        parsed_data,
        normalized_folds,
        model_name,
        params,
        round_idx,
    )


def _normalized_guided_boosting_folds(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            **fold,
            "prefix_range": _state_range(fold, "prefix_range"),
            "validation_range": _state_range(fold, "validation_range"),
        }
        for fold in state["boosting_fold_states"]
    ]


def _finalize_guided_boosting_fold_candidates(
    state: Dict[str, Any],
    models_to_use: List[str],
) -> List[Dict[str, Any]]:
    normalized_folds = _normalized_guided_boosting_folds(state)
    stored_results = state.get("pending_candidate_fold_results")
    if not isinstance(stored_results, dict):
        raise ValueError("[VALIDATION] Boosting candidate fold checkpoints are missing.")

    stage_results = []
    for model_name in models_to_use:
        model_results = stored_results.get(model_name)
        if not isinstance(model_results, dict):
            raise ValueError(
                f"[VALIDATION] Boosting candidate folds are missing for {model_name}."
            )
        missing = [
            fold_index + 1
            for fold_index in range(len(normalized_folds))
            if fold_index not in model_results
        ]
        if missing:
            raise ValueError(
                f"[VALIDATION] Boosting candidate folds are incomplete for {model_name}: "
                + ", ".join(map(str, missing))
            )
        stage_results.append(boosting.finalize_rolling_origin_candidate(
            normalized_folds,
            model_name,
            [model_results[index] for index in range(len(normalized_folds))],
        ))
    return stage_results


def _run_boosting(step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if step_id == "prepare_data":
        return _initialize_state("boosting", args, state)

    parsed_data = state["parsed_data"]
    train_range = _state_range(state, "train_range")
    eval_range = _state_range(state, "eval_range")
    models_to_use = state["models"]
    model_params = state["model_params"]
    base_model_files = state["base_model_files"]

    fold_candidate_match = re.fullmatch(
        r"round_(\d+)_evaluate_fold_(\d+)_(ma|es|arima|lstm)",
        step_id,
    )
    if fold_candidate_match:
        round_idx = int(fold_candidate_match.group(1)) - 1
        fold_index = int(fold_candidate_match.group(2)) - 1
        model_name = fold_candidate_match.group(3)
        if model_name not in models_to_use:
            raise ValueError("[VALIDATION] Boosting candidate is not part of this session.")
        if round_idx != int(state.get("next_round_index", 0)):
            raise ValueError("[VALIDATION] Boosting candidate step is out of order.")

        normalized_folds = _normalized_guided_boosting_folds(state)
        residual_val = _pooled_guided_boosting_residuals(state)
        if state.get("pending_round_index") != round_idx:
            state["pending_round_index"] = round_idx
            state["pending_current_rmse"] = rmse(
                np.zeros_like(residual_val), residual_val
            )
            state["pending_candidate_fold_results"] = {}
            state.pop("pending_stage_results", None)

        if fold_index < 0:
            raise ValueError("[VALIDATION] Boosting fold index must be positive.")
        if fold_index >= len(normalized_folds):
            return {
                "round": round_idx + 1,
                "model": model_name,
                "fold": fold_index + 1,
                "fold_count": len(normalized_folds),
                "skipped": True,
            }

        model_results = state.setdefault(
            "pending_candidate_fold_results", {}
        ).setdefault(model_name, {})
        if fold_index in model_results:
            raise ValueError("[VALIDATION] Boosting candidate fold was already evaluated.")
        model_results[fold_index] = boosting.evaluate_rolling_origin_fold_candidate(
            parsed_data,
            normalized_folds[fold_index],
            model_name,
            model_params[model_name],
            round_idx,
        )
        return {
            "round": round_idx + 1,
            "model": model_name,
            "fold": fold_index + 1,
            "fold_count": len(normalized_folds),
            "predicted_points": len(normalized_folds[fold_index]["validation_range"]),
        }

    candidate_match = re.fullmatch(
        r"round_(\d+)_evaluate_(ma|es|arima|lstm)",
        step_id,
    )
    if candidate_match:
        round_idx = int(candidate_match.group(1)) - 1
        model_name = candidate_match.group(2)
        if model_name not in models_to_use:
            raise ValueError("[VALIDATION] Boosting candidate is not part of this session.")
        if round_idx != int(state.get("next_round_index", 0)):
            raise ValueError("[VALIDATION] Boosting candidate step is out of order.")
        residual_val = _pooled_guided_boosting_residuals(state)
        if state.get("pending_round_index") != round_idx:
            state["pending_round_index"] = round_idx
            state["pending_current_rmse"] = rmse(np.zeros_like(residual_val), residual_val)
            state["pending_stage_results"] = []
            state.pop("pending_candidate_fold_results", None)
        if any(
            result["model_name"] == model_name
            for result in state["pending_stage_results"]
        ):
            raise ValueError("[VALIDATION] Boosting candidate was already evaluated.")

        candidate_result = _evaluate_guided_boosting_candidate(
            parsed_data,
            state,
            model_name,
            model_params[model_name],
            round_idx,
        )
        state["pending_stage_results"].append(candidate_result)
        return {
            "round": round_idx + 1,
            "model": model_name,
            "previous_rmse": state["pending_current_rmse"],
            "rmse": candidate_result["rmse"],
            "coefficient": candidate_result["coefficient"],
            "fold_rmse": candidate_result["fold_rmse"],
        }

    if step_id == "split_residual_sets":
        planned_ranges = rolling_origin_ranges(train_range, state["split_plan"])
        boosting_fold_states = []
        for fold_index, (prefix_range, validation_range) in enumerate(planned_ranges):
            boosting_fold_states.append({
                "fold_index": fold_index,
                "prefix_range": _range_to_state(prefix_range),
                "validation_range": _range_to_state(validation_range),
                "residual_train": np.asarray(parsed_data.y[prefix_range], dtype=float),
                "residual_val": np.asarray(parsed_data.y[validation_range], dtype=float),
            })
        # Keep the legacy aliases for checkpoint inspection/UI compatibility;
        # all selection math below uses the complete fold collection.
        boost_train_range, boost_val_range = planned_ranges[-1]
        state["boost_train_range"] = _range_to_state(boost_train_range)
        state["boost_val_range"] = _range_to_state(boost_val_range)
        state["boosting_fold_states"] = boosting_fold_states
        state["residual_train"] = boosting_fold_states[-1]["residual_train"]
        state["residual_val"] = boosting_fold_states[-1]["residual_val"]
        state["validation_residual_snapshots"] = [
            _pooled_guided_boosting_residuals(state).copy()
        ]
        state["model_chain"] = []
        state["stage_coefficients"] = []
        state["next_round_index"] = 0
        state["search_summary"] = {
            "algorithm": "residual_boosting",
            "coefficient_strategy": "nonnegative_squared_error_line_search",
            "minimum_relative_improvement": state["minimum_relative_improvement"],
            "max_rounds": state["max_rounds"],
            "split_plan": state["split_plan"],
            "stages": [],
        }
        return {
            "boost_train_size": len(boost_train_range),
            "boost_val_size": len(boost_val_range),
            "fold_count": len(planned_ranges),
            "calibration_size": int(state["split_plan"]["calibration_size"]),
            "max_rounds": state["max_rounds"],
            "split_plan": state["split_plan"],
        }

    round_match = _boosting_round_match(step_id)
    if round_match:
        round_idx, action = round_match
        if round_idx != int(state.get("next_round_index", 0)):
            raise ValueError("[VALIDATION] Boosting round step is out of order.")

        residual_val = _pooled_guided_boosting_residuals(state)
        if action == "evaluate_candidates":
            current_rmse = rmse(np.zeros_like(residual_val), residual_val)
            stage_results = [
                _evaluate_guided_boosting_candidate(
                    parsed_data,
                    state,
                    model_name,
                    model_params[model_name],
                    round_idx,
                )
                for model_name in models_to_use
            ]

            state["pending_round_index"] = round_idx
            state["pending_current_rmse"] = current_rmse
            state["pending_stage_results"] = stage_results
            state.pop("pending_candidate_fold_results", None)
            return {
                "round": round_idx + 1,
                "previous_rmse": current_rmse,
                "candidates": {
                    result["model_name"]: {
                        "rmse": result["rmse"],
                        "coefficient": result["coefficient"],
                        "fold_rmse": result["fold_rmse"],
                    }
                    for result in stage_results
                },
            }

        pending_round = state.get("pending_round_index")
        if pending_round != round_idx:
            raise ValueError("[VALIDATION] Boosting candidate results are missing for this round.")

        if "pending_candidate_fold_results" in state:
            state["pending_stage_results"] = _finalize_guided_boosting_fold_candidates(
                state,
                models_to_use,
            )
        stage_results = state["pending_stage_results"]
        evaluated_models = {result["model_name"] for result in stage_results}
        if evaluated_models != set(models_to_use):
            missing = sorted(set(models_to_use) - evaluated_models)
            raise ValueError(
                "[VALIDATION] Boosting candidate results are incomplete: "
                + ", ".join(missing)
            )
        current_rmse = float(state["pending_current_rmse"])
        best_result = min(stage_results, key=lambda result: result["rmse"])
        improvement = current_rmse - best_result["rmse"]
        relative_improvement = improvement / max(current_rmse, np.finfo(float).eps)
        stage_summary = {
            "round": round_idx + 1,
            "objective": "validation_residual_rmse",
            "coefficient": best_result["coefficient"],
            "previous_rmse": current_rmse,
            "candidates": {
                result["model_name"]: {
                    "rmse": result["rmse"],
                    "coefficient": result["coefficient"],
                    "fold_rmse": result["fold_rmse"],
                }
                for result in stage_results
            },
            "winner": best_result["model_name"],
            "winner_rmse": best_result["rmse"],
            "improvement": improvement,
            "relative_improvement": relative_improvement,
        }

        if relative_improvement >= float(state["minimum_relative_improvement"]):
            state["model_chain"].append(best_result["model_name"])
            state["stage_coefficients"].append(best_result["coefficient"])
            for fold_index, fold in enumerate(state["boosting_fold_states"]):
                fold["residual_train"] = (
                    np.asarray(fold["residual_train"], dtype=float)
                    - best_result["coefficient"]
                    * best_result["fold_fitted_train_predictions"][fold_index]
                )
                fold["residual_val"] = best_result["next_residuals"][fold_index]
            state["residual_train"] = state["boosting_fold_states"][-1]["residual_train"]
            state["residual_val"] = state["boosting_fold_states"][-1]["residual_val"]
            state["validation_residual_snapshots"].append(
                _pooled_guided_boosting_residuals(state).copy()
            )
            state["next_round_index"] = round_idx + 1
            stage_summary["action"] = "selected"
            if round_idx + 1 >= int(state["max_rounds"]):
                stage_summary["stop_reason"] = "round_budget"
        else:
            state["early_stopped"] = True
            stage_summary["action"] = "early_stop"

        state["search_summary"]["stages"].append(stage_summary)
        state.pop("pending_round_index", None)
        state.pop("pending_current_rmse", None)
        state.pop("pending_stage_results", None)
        state.pop("pending_candidate_fold_results", None)

        return {
            **stage_summary,
            "model_chain": state["model_chain"],
        }

    if step_id == "initialize_refit":
        selection_model_chain = list(state["model_chain"])
        if not selection_model_chain:
            raise ValueError(
                "[DATA] Boosting 未能选出任何有效阶段：所有候选基础模型都无法降低验证残差 RMSE。"
            )
        state["search_summary"]["selection_model_chain"] = selection_model_chain
        state["selection_model_chain"] = selection_model_chain
        state["selection_coefficients"] = list(state["stage_coefficients"])
        state["refit_model_chain"] = []
        state["refit_stage_coefficients"] = []
        state["final_eval_preds"] = np.zeros(len(eval_range), dtype=float)
        state["residual_full_train"] = np.asarray(parsed_data.y[train_range], dtype=float).copy()
        state["base_model_states"] = {}
        state["full_uncertainty_profiles"] = {}
        state["uncertainty_coefficients"] = {}
        state["stage_eval_summaries"] = []
        return {
            "selection_model_chain": selection_model_chain,
            "stage_count": len(selection_model_chain),
        }

    refit_match = re.fullmatch(r"refit_stage_(\d+)_(ma|es|arima|lstm)", step_id)
    if refit_match:
        stage_idx = int(refit_match.group(1)) - 1
        model_name = refit_match.group(2)
        selection_model_chain = list(state.get("selection_model_chain") or [])
        if stage_idx >= len(selection_model_chain) or selection_model_chain[stage_idx] != model_name:
            raise ValueError("[VALIDATION] Boosting refit step does not match the selected chain.")
        if stage_idx != len(state.get("refit_model_chain") or []):
            raise ValueError("[VALIDATION] Boosting refit stage is out of order.")

        params = model_params[model_name]
        timeout = boosting._get_model_timeout(model_name, params)
        residual_full_train = np.asarray(state["residual_full_train"], dtype=float)
        final_eval_preds = np.asarray(state["final_eval_preds"], dtype=float)
        round_parsed_data = boosting._build_residual_parsed_data(
            parsed_data,
            train_range,
            residual_full_train,
        )
        artifact_origin = "temporary_training"
        eval_forecast_steps = eval_range.stop - train_range.stop
        eval_forecast_offset = eval_range.start - train_range.stop

        if stage_idx == 0 and model_name in base_model_files:
            source_model_file = base_model_files[model_name]
            if model_name == "lstm":
                source_model_state = boosting._load_lstm_source_state(
                    source_model_file,
                    "Saved Boosting LSTM base model",
                )
                source_strategy = boosting._get_lstm_source_forecast_strategy(source_model_state)
                source_horizon = int(source_model_state.get("forecast_horizon", 0) or 0)
                if source_strategy == "direct_multi_horizon" and eval_forecast_steps > source_horizon:
                    stage_model_file = os.path.join(
                        state["artifact_dir"],
                        f"boosting_stage_{model_name}_{stage_idx}_{os.getpid()}.pkl",
                    )
                    round_numeric_records, round_lstm_records = boosting._build_round_records(
                        round_parsed_data,
                        model_name,
                        params,
                    )
                    adjusted_source_model_file = source_model_file
                    temp_source_model_file = None
                    try:
                        adjusted_source_model_file, temp_source_model_file = (
                            boosting._prepare_lstm_source_model_for_requested_steps(
                                source_model_file,
                                source_model_state,
                                requested_steps=eval_forecast_steps,
                            )
                        )
                        fit_result = run_base_model_fit_subprocess(
                            model_type=model_name,
                            data=round_lstm_records,
                            train_start=train_range.start,
                            train_end=train_range.stop - 1,
                            source_model_file=adjusted_source_model_file,
                            save_model_file=stage_model_file,
                            fit_forecast_steps=eval_forecast_steps,
                            predict_steps=eval_forecast_steps,
                            allow_negative_predictions=True,
                            timeout=timeout,
                        )
                    finally:
                        if temp_source_model_file and os.path.exists(temp_source_model_file):
                            os.remove(temp_source_model_file)
                    full_predictions = np.asarray(extract_predictions_from_result(fit_result), dtype=float)
                    stage_eval_predictions = full_predictions[
                        eval_forecast_offset: eval_forecast_offset + len(eval_range)
                    ]
                    stage_uncertainty_profile = extract_uncertainty_profile_from_result(fit_result)
                else:
                    stage_model_file = source_model_file
                    stage_eval_predictions, stage_uncertainty_profile = boosting._predict_eval_segment(
                        model_name,
                        stage_model_file,
                        train_range,
                        eval_range,
                        timeout,
                    )
                    artifact_origin = "standalone_base"
            else:
                stage_model_file = source_model_file
                stage_eval_predictions, stage_uncertainty_profile = boosting._predict_eval_segment(
                    model_name,
                    stage_model_file,
                    train_range,
                    eval_range,
                    timeout,
                )
                artifact_origin = "standalone_base"
        else:
            round_numeric_records, round_lstm_records = boosting._build_round_records(
                round_parsed_data,
                model_name,
                params,
            )
            stage_model_file, stage_eval_predictions, stage_uncertainty_profile = boosting._train_stage_model(
                model_name,
                round_numeric_records,
                round_lstm_records,
                train_range,
                eval_range,
                params,
                timeout,
            )

        fitted_train_predictions = boosting._compute_in_sample_fitted_predictions(
            model_name,
            stage_model_file,
            round_parsed_data,
            train_range,
        )
        stage_eval_predictions = boosting._apply_stage_output_semantics(stage_idx, stage_eval_predictions)
        fitted_train_predictions = boosting._apply_stage_output_semantics(
            stage_idx,
            fitted_train_predictions,
        )
        coefficient = float(state["selection_coefficients"][stage_idx])

        final_eval_preds = final_eval_preds + coefficient * stage_eval_predictions
        residual_full_train = residual_full_train - coefficient * fitted_train_predictions
        state["refit_model_chain"].append(model_name)
        state["refit_stage_coefficients"].append(coefficient)
        state["final_eval_preds"] = final_eval_preds
        state["residual_full_train"] = residual_full_train
        stage_key = f"stage_{stage_idx}_{model_name}"
        state["full_uncertainty_profiles"][stage_key] = stage_uncertainty_profile
        state["uncertainty_coefficients"][stage_key] = coefficient
        state["base_model_states"].setdefault(model_name, {"models": []})["models"].append({
            "model_type": model_name,
            "model_file": stage_model_file,
            "params": params,
            "round": stage_idx,
            "coefficient": coefficient,
            "artifact_origin": artifact_origin,
        })
        stage_summary = {
            "round": stage_idx + 1,
            "model_name": model_name,
            "coefficient": coefficient,
            "rmse": rmse(np.asarray(parsed_data.y[eval_range], dtype=float), final_eval_preds),
        }
        state["stage_eval_summaries"].append(stage_summary)
        state["calibration_residuals"] = np.asarray(
            state["validation_residual_snapshots"][len(state["refit_model_chain"])],
            dtype=float,
        )
        state["eval_y_true"] = np.asarray(parsed_data.y[eval_range], dtype=float).tolist()
        return {"action": "refit_stage", **stage_summary}

    if step_id == "retrain_chain":
        selection_model_chain = list(state["model_chain"])
        selection_coefficients = list(state["stage_coefficients"])
        state["search_summary"]["selection_model_chain"] = selection_model_chain
        refit_model_chain: List[str] = []
        stage_coefficients: List[float] = list(selection_coefficients)
        y_true_eval = np.asarray(parsed_data.y[eval_range], dtype=float)
        if not selection_model_chain:
            raise ValueError(
                "[DATA] Boosting 未能选出任何有效阶段：所有候选基础模型在验证段的预测均无效，"
                "无法降低残差 RMSE。请检查所选数据区间的质量，或调整基础模型参数后重试。"
            )

        final_eval_preds = np.zeros(len(eval_range), dtype=float)
        residual_full_train = np.asarray(parsed_data.y[train_range], dtype=float).copy()
        base_model_states: Dict[str, Any] = {}
        full_uncertainty_profiles: Dict[str, List[float]] = {}
        uncertainty_coefficients: Dict[str, float] = {}
        stage_eval_summaries = []
        eval_forecast_offset = eval_range.start - train_range.stop

        for stage_idx, model_name in enumerate(selection_model_chain):
            params = model_params[model_name]
            timeout = boosting._get_model_timeout(model_name, params)
            round_parsed_data = boosting._build_residual_parsed_data(parsed_data, train_range, residual_full_train)
            artifact_origin = "temporary_training"

            if stage_idx == 0 and model_name in base_model_files:
                source_model_file = base_model_files[model_name]
                if model_name == "lstm":
                    source_model_state = boosting._load_lstm_source_state(source_model_file, "Saved Boosting LSTM base model")
                    source_forecast_strategy = boosting._get_lstm_source_forecast_strategy(source_model_state)
                    source_forecast_horizon = int(source_model_state.get("forecast_horizon", 0) or 0)
                    if source_forecast_strategy == "direct_multi_horizon" and (eval_range.stop - train_range.stop) > source_forecast_horizon:
                        adjusted_source_model_file = source_model_file
                        temp_source_model_file = None
                        stage_model_file = os.path.join(state["artifact_dir"], f"boosting_stage_{model_name}_{stage_idx}_{os.getpid()}.pkl")
                        round_numeric_records, round_lstm_records = boosting._build_round_records(round_parsed_data, model_name, params)
                        try:
                            adjusted_source_model_file, temp_source_model_file = (
                                boosting._prepare_lstm_source_model_for_requested_steps(
                                    source_model_file,
                                    source_model_state,
                                    requested_steps=eval_range.stop - train_range.stop,
                                )
                            )
                            fit_result = run_base_model_fit_subprocess(
                                model_type=model_name,
                                data=round_lstm_records,
                                train_start=train_range.start,
                                train_end=train_range.stop - 1,
                                source_model_file=adjusted_source_model_file,
                                save_model_file=stage_model_file,
                                fit_forecast_steps=eval_range.stop - train_range.stop,
                                predict_steps=eval_range.stop - train_range.stop,
                                allow_negative_predictions=True,
                                timeout=timeout,
                            )
                        finally:
                            if temp_source_model_file and os.path.exists(temp_source_model_file):
                                os.remove(temp_source_model_file)
                        full_predictions = np.asarray(
                            extract_predictions_from_result(fit_result),
                            dtype=float,
                        )
                        stage_eval_predictions = full_predictions[
                            eval_forecast_offset: eval_forecast_offset + len(eval_range)
                        ]
                        stage_uncertainty_profile = extract_uncertainty_profile_from_result(fit_result)
                    else:
                        stage_model_file = source_model_file
                        stage_eval_predictions, stage_uncertainty_profile = boosting._predict_eval_segment(
                            model_name,
                            stage_model_file,
                            train_range,
                            eval_range,
                            timeout,
                        )
                        artifact_origin = "standalone_base"
                else:
                    stage_model_file = source_model_file
                    stage_eval_predictions, stage_uncertainty_profile = boosting._predict_eval_segment(
                        model_name,
                        stage_model_file,
                        train_range,
                        eval_range,
                        timeout,
                    )
                    artifact_origin = "standalone_base"
            else:
                round_numeric_records, round_lstm_records = boosting._build_round_records(
                    round_parsed_data,
                    model_name,
                    params,
                )
                stage_model_file, stage_eval_predictions, stage_uncertainty_profile = boosting._train_stage_model(
                    model_name,
                    round_numeric_records,
                    round_lstm_records,
                    train_range,
                    eval_range,
                    params,
                    timeout,
                )

            fitted_train_predictions = boosting._compute_in_sample_fitted_predictions(
                model_name,
                stage_model_file,
                round_parsed_data,
                train_range,
            )
            stage_eval_predictions = boosting._apply_stage_output_semantics(
                stage_idx,
                stage_eval_predictions,
            )
            fitted_train_predictions = boosting._apply_stage_output_semantics(
                stage_idx,
                fitted_train_predictions,
            )
            coefficient = float(selection_coefficients[stage_idx])
            refit_model_chain.append(model_name)
            final_eval_preds += coefficient * stage_eval_predictions
            residual_full_train = residual_full_train - coefficient * fitted_train_predictions
            stage_key = f"stage_{stage_idx}_{model_name}"
            full_uncertainty_profiles[stage_key] = stage_uncertainty_profile
            uncertainty_coefficients[stage_key] = coefficient
            stage_rmse = rmse(y_true_eval, final_eval_preds)

            base_model_states.setdefault(model_name, {"models": []})["models"].append({
                "model_type": model_name,
                "model_file": stage_model_file,
                "params": params,
                "round": stage_idx,
                "coefficient": coefficient,
                "artifact_origin": artifact_origin,
            })
            stage_eval_summaries.append({
                "round": stage_idx + 1,
                "model_name": model_name,
                "coefficient": coefficient,
                "rmse": stage_rmse,
            })

        calibration_residuals = np.asarray(
            state["validation_residual_snapshots"][len(refit_model_chain)],
            dtype=float,
        )
        state["model_chain"] = refit_model_chain
        state["base_model_states"] = base_model_states
        state["final_eval_preds"] = final_eval_preds
        state["eval_y_true"] = y_true_eval.tolist()
        state["stage_eval_summaries"] = stage_eval_summaries
        state["selection_coefficients"] = selection_coefficients
        state["stage_coefficients"] = stage_coefficients
        state["full_uncertainty_profiles"] = full_uncertainty_profiles
        state["uncertainty_coefficients"] = uncertainty_coefficients
        state["calibration_residuals"] = calibration_residuals
        return {
            "model_chain": refit_model_chain,
            "stage_count": len(refit_model_chain),
            "stage_eval": stage_eval_summaries,
        }

    if step_id == "combine_evaluate":
        if "refit_model_chain" in state:
            if not state["refit_model_chain"]:
                raise ValueError("[MODEL] Boosting full-range refit produced no usable stages.")
            state["model_chain"] = list(state["refit_model_chain"])
            state["stage_coefficients"] = list(state["refit_stage_coefficients"])
        y_true_eval = np.asarray(state["eval_y_true"], dtype=float)
        final_eval_preds = normalize_demand_forecast(state["final_eval_preds"])
        calibration_residuals = np.asarray(state["calibration_residuals"], dtype=float)
        eval_forecast_steps = eval_range.stop - train_range.stop
        eval_forecast_offset = eval_range.start - train_range.stop
        full_uncertainty_profile, uncertainty_summary = ensemble_growth_profile(
            state["full_uncertainty_profiles"],
            state["uncertainty_coefficients"],
            calibration_residuals,
            parsed_data.y[train_range],
            eval_forecast_steps,
            calibration_source="rolling_origin_oof",
        )
        uncertainty_profile = full_uncertainty_profile[
            eval_forecast_offset: eval_forecast_offset + len(eval_range)
        ]
        uncertainty_summary.setdefault("calibration_source", "rolling_origin_oof")
        timestamps = parsed_data.timestamps
        metrics = {
            "rmse": rmse(y_true_eval, final_eval_preds),
            "mae": mae(y_true_eval, final_eval_preds),
            "mape": mape(y_true_eval, final_eval_preds),
            "r2": r2(y_true_eval, final_eval_preds),
            "evaluate_indices": list(eval_range),
            "evaluate_months": [timestamps[i] for i in eval_range] if timestamps else [],
        }
        final_result = {
            "mode": "boosting_residual_learning",
            "models": models_to_use,
            "model_chain": state["model_chain"],
            "stage_coefficients": state["stage_coefficients"],
            "split_plan": state["split_plan"],
            "metrics": metrics,
            "search_summary": state["search_summary"],
            "eval_y_true": y_true_eval.tolist(),
            "predictions": boosting._format_predictions(
                final_eval_preds.tolist(),
                uncertainty_profile,
                {**uncertainty_summary, "horizon_start": eval_forecast_offset + 1},
            ),
        }
        attach_method_metadata(final_result, boosting.METHOD_NAME, boosting.FORECAST_STRATEGY, boosting.IMPLEMENTATION_NOTES)
        state["metrics"] = metrics
        state["search_summary"]["selection_coefficients"] = state["selection_coefficients"]
        state["search_summary"]["refit_model_chain"] = state["model_chain"]
        state["search_summary"]["refit_coefficients"] = state["stage_coefficients"]
        state["uncertainty"] = {
            "calibration_residuals": calibration_residuals.tolist(),
            "calibration_source": "rolling_origin_oof",
            "reference_series": parsed_data.y[train_range].tolist(),
            "profile": full_uncertainty_profile,
            "summary": uncertainty_summary,
        }
        state["final_result"] = final_result
        return {"metrics": metrics, "uncertainty_summary": uncertainty_summary}

    if step_id == "save_artifact":
        save_path = state["save_model"]
        base_model_states = state["base_model_states"]
        for model_name, model_info in base_model_states.items():
            for idx, model_state in enumerate(model_info["models"]):
                source_file = model_state["model_file"]
                perm_file = get_member_model_path(save_path, model_name, int(model_state.get("round", idx)))
                persist_member_model_artifact(
                    model_name,
                    source_file,
                    perm_file,
                    move_source=False,
                )
                model_state["model_file"] = perm_file
                model_state.pop("artifact_origin", None)

        model_state = {
            "model_type": boosting.MODEL_TYPE,
            "boosting_variant": "residual",
            "model_chain": state["model_chain"],
            "stage_coefficients": state["stage_coefficients"],
            "base_model_states": base_model_states,
            "split_plan": state["split_plan"],
            "uncertainty": state["uncertainty"],
        }
        attach_method_metadata(model_state, boosting.METHOD_NAME, boosting.FORECAST_STRATEGY, boosting.IMPLEMENTATION_NOTES)
        dump_model(model_state, save_path)
        final_result = dict(state["final_result"])
        final_result["saved_model"] = save_path
        final_result["logs"] = get_log_contents()
        return {"output": {"saved_model": save_path}, "final_result": final_result}

    raise ValueError(f"[VALIDATION] Unknown boosting guided step: {step_id}")


def run_step(model_type: str, step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if model_type == "weighted_avg":
        return _run_weighted(step_id, args, state)
    if model_type == "stacking":
        return _run_stacking(step_id, args, state)
    if model_type == "boosting":
        return _run_boosting(step_id, args, state)
    raise ValueError(f"[VALIDATION] Unsupported guided ensemble model type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one guided ensemble model training stage.")
    parser.add_argument("model_type", type=str)
    parser.add_argument("step_id", type=str)
    parser.add_argument("--input-state-file")
    parser.add_argument("--output-state-file", required=True)
    parser.add_argument("--receipt-file", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--data", type=str)
    parser.add_argument("--data-file", type=str)
    parser.add_argument("--train-start", type=int)
    parser.add_argument("--train-end", type=int)
    parser.add_argument("--eval-start", type=int)
    parser.add_argument("--eval-end", type=int)
    parser.add_argument("--save-model", type=str)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--models", type=str)
    parser.add_argument("--window", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--d", type=int)
    parser.add_argument("--max-p", type=int)
    parser.add_argument("--max-q", type=int)
    parser.add_argument("--look-back", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--feature-types", type=str)
    parser.add_argument("--lstm-features", type=str)
    parser.add_argument("--lstm-target-feature", type=str)
    parser.add_argument("--normalization", type=str, choices=["minmax", "zscore"])
    parser.add_argument("--base-model-files", type=str)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-rounds", type=int)
    args = parser.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)
    existing_receipt = _load_existing_receipt(
        args.receipt_file,
        args.attempt_id,
        args.step_id,
    )
    if existing_receipt is not None:
        print(json.dumps(existing_receipt, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    state = _load_state(args.input_state_file)
    step_result = run_step(args.model_type, args.step_id, args, state)

    final_result = None
    if isinstance(step_result, dict) and "final_result" in step_result:
        final_result = step_result["final_result"]
        output = step_result.get("output", {})
    else:
        output = step_result

    _save_state(args.output_state_file, state)
    payload = {
        "protocol_version": 4,
        "attempt_id": args.attempt_id,
        "step_id": args.step_id,
        "output": _jsonable(output),
        "input_state_path": args.input_state_file,
        "input_state_sha256": _sha256_file(args.input_state_file),
        "candidate_state_path": args.output_state_file,
        "candidate_state_sha256": _sha256_file(args.output_state_file),
    }
    if final_result is not None:
        payload["final_result"] = _jsonable(final_result)

    _write_receipt(args.receipt_file, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        known_error = format_known_error(exc)
        if known_error:
            print(known_error, file=sys.stderr, flush=True)
            sys.exit(1)
        raise
