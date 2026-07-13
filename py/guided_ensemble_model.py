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
    fit_and_predict_from_source_model,
    get_member_model_path,
    normalize_demand_forecast,
    parse_base_model_files_arg,
    persist_member_model_artifact,
    remove_member_model_artifact,
)
from logger import get_log_contents
from metrics import mae, mape, r2, rmse
from model_info import attach_method_metadata
from serialization import dump_model
from subprocess_utils import (
    extract_predictions_from_result,
    run_base_model_fit_subprocess,
    run_base_model_predict_subprocess,
    run_base_model_subprocess,
)
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
    models = [model.strip() for model in str(args.models or "").split(",") if model.strip()]
    if len(models) < 2:
        raise ValueError("[VALIDATION] 融合模型至少需要两个基础模型。")
    return models


def _auto_max_pq(phase_train_size: int) -> int:
    if phase_train_size <= 12:
        return 1
    if phase_train_size <= 20:
        return 2
    return max(1, min(5, phase_train_size // 6))


def _build_model_params(
    module,
    parsed_data,
    train_size: int,
    phase_train_size: int,
    models_to_use: List[str],
    base_model_files: Dict[str, str],
    args,
    lstm_context: str,
) -> Dict[str, Dict[str, Any]]:
    auto_window = max(3, min(12, phase_train_size // 4))
    auto_look_back = max(3, min(12, phase_train_size // 4))
    auto_epochs = 20 if train_size <= 15 else 30
    model_params: Dict[str, Dict[str, Any]] = {}

    if "arima" in models_to_use:
        model_params["arima"] = {
            "d": _get_arg(args, "d", 0, int),
            "max_p": _get_arg(args, "max_p", _auto_max_pq(phase_train_size), int),
            "max_q": _get_arg(args, "max_q", _auto_max_pq(phase_train_size), int),
        }

    if "es" in models_to_use:
        model_params["es"] = {"alpha": _get_arg(args, "alpha", 0.5, float)}

    if "ma" in models_to_use:
        model_params["ma"] = {"window": _get_arg(args, "window", auto_window, int)}

    if "lstm" in models_to_use:
        feature_types_str = _get_arg(args, "feature_types", None)
        lstm_features_str = _get_arg(args, "lstm_features", None)
        target_feature = _get_arg(args, "lstm_target_feature", parsed_data.target_key)
        lstm_feature_names = (
            module._parse_feature_name_list(lstm_features_str)
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
            feature_types = {key: val for key, val in feature_types.items() if key in allowed_keys}
        feature_types[target_feature] = "numeric"

        lstm_params = {
            "look_back": _get_arg(args, "look_back", auto_look_back, int),
            "epochs": _get_arg(args, "epochs", auto_epochs, int),
            "feature_types": feature_types,
            "normalization": _get_arg(args, "normalization", "minmax"),
            "lstm_target_feature": target_feature,
        }
        if lstm_features_str:
            lstm_params["lstm_features"] = lstm_features_str

        source_lstm_file = base_model_files.get("lstm")
        if source_lstm_file and os.path.exists(source_lstm_file):
            source_lstm_state = module._load_lstm_source_state(source_lstm_file, lstm_context)
            lstm_params = module._build_lstm_params_from_source_state(
                parsed_data,
                source_lstm_state,
                lstm_params,
            )

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
    phase_train_size = module._compute_phase1_train_size(train_size)
    base_model_files = parse_base_model_files_arg(_get_arg(args, "base_model_files", None))
    model_params = _build_model_params(
        module,
        parsed_data,
        train_size,
        phase_train_size,
        models_to_use,
        base_model_files,
        args,
        f"Saved {model_type} LSTM base model",
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
    })

    if model_type == "boosting":
        learning_rate = _get_arg(args, "learning_rate", 0.3, float)
        max_rounds = _get_arg(args, "max_rounds", len(models_to_use), int)
        if not (0 < learning_rate <= 1):
            raise ValueError("[VALIDATION] learning_rate must be in (0, 1]")
        if max_rounds < 1:
            raise ValueError("[VALIDATION] max_rounds must be >= 1")
        state["learning_rate"] = learning_rate
        state["max_rounds"] = max_rounds

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
    return max(120, int(model_params.get("lstm", {}).get("epochs", 20)) * 10)


def _holdout_predictions(
    module,
    parsed_data,
    models_to_use: List[str],
    model_params: Dict[str, Dict[str, Any]],
    base_model_files: Dict[str, str],
    train_range: range,
    holdout_range: range,
    source_context: str,
) -> Dict[str, List[float]]:
    numeric_data_dicts, lstm_data_dicts = build_ensemble_subprocess_data(parsed_data, models_to_use, model_params)
    predictions: Dict[str, List[float]] = {}
    timeout = _base_timeout(models_to_use, model_params)

    for model_name in models_to_use:
        params = model_params[model_name]
        source_model_file = base_model_files.get(model_name)
        if source_model_file and os.path.exists(source_model_file):
            adjusted_source_model_file = source_model_file
            temp_source_model_file = None
            try:
                if model_name == "lstm":
                    source_model_state = module._load_lstm_source_state(source_model_file, source_context)
                    adjusted_source_model_file, temp_source_model_file = (
                        module._prepare_lstm_source_model_for_requested_steps(
                            source_model_file,
                            source_model_state,
                            requested_steps=len(holdout_range),
                        )
                    )
                predictions[model_name] = normalize_demand_forecast(fit_and_predict_from_source_model(
                    model_name=model_name,
                    source_model_file=adjusted_source_model_file,
                    data=lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
                    train_start=train_range.start,
                    train_end=train_range.stop - 1,
                    forecast_steps=len(holdout_range),
                    timeout=timeout,
                )).tolist()
            finally:
                if temp_source_model_file and os.path.exists(temp_source_model_file):
                    os.remove(temp_source_model_file)
            continue

        result = run_base_model_subprocess(
            model_type=model_name,
            data=lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
            train_start=train_range.start,
            train_end=train_range.stop - 1,
            eval_start=holdout_range.start,
            eval_end=holdout_range.stop - 1,
            model_params=params.copy(),
            timeout=timeout,
        )
        predictions[model_name] = normalize_demand_forecast(
            extract_predictions_from_result(result)
        ).tolist()

    return predictions


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
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    numeric_data_dicts, lstm_data_dicts = build_ensemble_subprocess_data(parsed_data, models_to_use, model_params)
    eval_predictions: Dict[str, List[float]] = {}
    base_model_states: Dict[str, Any] = {}
    eval_forecast_steps = eval_range.stop - train_range.stop
    eval_forecast_offset = eval_range.start - train_range.stop
    timeout = _base_timeout(models_to_use, model_params)

    for model_name in models_to_use:
        params = model_params[model_name]
        source_model_file = base_model_files.get(model_name)
        temp_model_file = os.path.join(artifact_dir, f"{prefix}_{model_name}_{os.getpid()}.pkl")

        if source_model_file and os.path.exists(source_model_file):
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
                        run_base_model_fit_subprocess(
                            model_type=model_name,
                            data=lstm_data_dicts,
                            train_start=train_range.start,
                            train_end=train_range.stop - 1,
                            source_model_file=adjusted_source_model_file,
                            save_model_file=temp_model_file,
                            timeout=timeout,
                        )
                    finally:
                        if temp_source_model_file and os.path.exists(temp_source_model_file):
                            os.remove(temp_source_model_file)

                    result = run_base_model_predict_subprocess(
                        model_type=model_name,
                        model_file=temp_model_file,
                        forecast_steps=eval_forecast_steps,
                        timeout=timeout,
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
            eval_start=eval_range.start,
            eval_end=eval_range.stop - 1,
            model_params=params_with_save,
            timeout=timeout,
        )
        eval_predictions[model_name] = normalize_demand_forecast(
            extract_predictions_from_result(result)
        ).tolist()
        base_model_states[model_name] = {
            "model_type": model_name,
            "model_file": temp_model_file,
            "params": params,
            "artifact_origin": "temporary_training",
        }

    return eval_predictions, base_model_states


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
        "std_dev_residuals": state["std_dev_residuals"],
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

    if step_id == "split_validation":
        split = weighted_avg._compute_phase1_train_size(len(train_range))
        weight_train_range = range(train_range.start, train_range.start + split)
        weight_val_range = range(train_range.start + split, train_range.stop)
        state["weight_train_range"] = _range_to_state(weight_train_range)
        state["weight_val_range"] = _range_to_state(weight_val_range)
        return {
            "train_size": len(train_range),
            "weight_train_size": len(weight_train_range),
            "weight_val_size": len(weight_val_range),
            "evaluate_size": len(eval_range),
        }

    if step_id == "validation_predictions":
        predictions = _holdout_predictions(
            weighted_avg,
            parsed_data,
            models_to_use,
            model_params,
            base_model_files,
            _state_range(state, "weight_train_range"),
            _state_range(state, "weight_val_range"),
            "Saved Weighted Average LSTM base model",
        )
        state["weight_val_predictions"] = predictions
        return {
            "predicted_points": {name: len(values) for name, values in predictions.items()},
            "models": models_to_use,
        }

    if step_id == "compute_weights":
        weight_val_range = _state_range(state, "weight_val_range")
        predictions = state["weight_val_predictions"]
        y_true_val = parsed_data.y[weight_val_range]
        weights = weighted_avg._compute_inverse_residual_mse_weights(y_true_val, predictions)
        residual_mse = {
            name: float(np.mean(np.square(y_true_val - np.asarray(predictions[name], dtype=float))))
            for name in models_to_use
        }
        state["weights"] = weights
        return {"weights": weights, "residual_mse": residual_mse}

    if step_id == "eval_predictions":
        eval_predictions, base_model_states = _eval_member_predictions(
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
        y_true = parsed_data.y[eval_range]
        residuals = y_true - y_pred_final
        std_dev = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
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
            "metrics": metrics,
            "eval_y_true": y_true.tolist(),
            "predictions": weighted_avg._format_predictions(y_pred_final.tolist(), std_dev),
        }
        attach_method_metadata(final_result, weighted_avg.METHOD_NAME, weighted_avg.FORECAST_STRATEGY, weighted_avg.IMPLEMENTATION_NOTES)
        state["metrics"] = metrics
        state["eval_y_true"] = y_true.tolist()
        state["y_pred"] = y_pred_final.tolist()
        state["std_dev_residuals"] = std_dev
        state["final_result"] = final_result
        return {"metrics": metrics, "std_dev_residuals": std_dev}

    if step_id == "save_artifact":
        save_path = _save_flat_ensemble_artifact(weighted_avg, state, {"weights": state["weights"]})
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

    if step_id == "split_levels":
        split = stacking._compute_phase1_train_size(len(train_range))
        level0_range = range(train_range.start, train_range.start + split)
        level1_range = range(train_range.start + split, train_range.stop)
        state["level0_range"] = _range_to_state(level0_range)
        state["level1_range"] = _range_to_state(level1_range)
        return {
            "train_size": len(train_range),
            "level0_size": len(level0_range),
            "level1_size": len(level1_range),
            "evaluate_size": len(eval_range),
        }

    if step_id == "level1_predictions":
        prediction_map = _holdout_predictions(
            stacking,
            parsed_data,
            models_to_use,
            model_params,
            base_model_files,
            _state_range(state, "level0_range"),
            _state_range(state, "level1_range"),
            "Saved Stacking LSTM base model",
        )
        state["level1_predictions"] = [prediction_map[name] for name in models_to_use]
        return {
            "predicted_points": {name: len(values) for name, values in prediction_map.items()},
            "models": models_to_use,
        }

    if step_id == "fit_meta_model":
        level1_range = _state_range(state, "level1_range")
        x_meta_train = np.column_stack(state["level1_predictions"])
        y_meta_train = parsed_data.y[level1_range]
        meta_model = stacking._fit_meta_model(x_meta_train, y_meta_train, models_to_use)
        state["meta_model"] = meta_model
        return {
            "meta_model": meta_model,
            "input_shape": list(x_meta_train.shape),
        }

    if step_id == "eval_predictions":
        eval_prediction_map, base_model_states = _eval_member_predictions(
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
        state["base_model_states"] = base_model_states
        return {
            "predicted_points": {name: len(values) for name, values in eval_prediction_map.items()},
            "reused_artifacts": [
                name for name, model_state in base_model_states.items()
                if model_state.get("artifact_origin") == "standalone_base"
            ],
        }

    if step_id == "meta_evaluate":
        x_meta_eval = np.column_stack(state["eval_predictions"])
        y_pred_final = stacking._predict_with_meta_model(state["meta_model"], x_meta_eval)
        y_true = parsed_data.y[eval_range]
        residuals = y_true - y_pred_final
        std_dev = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
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
            "metrics": metrics,
            "eval_y_true": y_true.tolist(),
            "predictions": stacking._format_predictions(y_pred_final.tolist(), std_dev),
        }
        attach_method_metadata(final_result, stacking.METHOD_NAME, stacking.FORECAST_STRATEGY, stacking.IMPLEMENTATION_NOTES)
        state["metrics"] = metrics
        state["eval_y_true"] = y_true.tolist()
        state["y_pred"] = y_pred_final.tolist()
        state["std_dev_residuals"] = std_dev
        state["final_result"] = final_result
        return {"metrics": metrics, "std_dev_residuals": std_dev}

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


def _run_boosting(step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if step_id == "prepare_data":
        return _initialize_state("boosting", args, state)

    parsed_data = state["parsed_data"]
    train_range = _state_range(state, "train_range")
    eval_range = _state_range(state, "eval_range")
    models_to_use = state["models"]
    model_params = state["model_params"]
    base_model_files = state["base_model_files"]

    if step_id == "split_residual_sets":
        split = boosting._compute_phase1_train_size(len(train_range))
        boost_train_range = range(train_range.start, train_range.start + split)
        boost_val_range = range(train_range.start + split, train_range.stop)
        y_true_val = np.asarray(parsed_data.y[boost_val_range], dtype=float)
        state["boost_train_range"] = _range_to_state(boost_train_range)
        state["boost_val_range"] = _range_to_state(boost_val_range)
        state["residual_train"] = np.asarray(parsed_data.y[boost_train_range], dtype=float)
        state["residual_val"] = y_true_val
        state["model_chain"] = []
        state["stage_coefficients"] = []
        state["next_round_index"] = 0
        state["search_summary"] = {
            "algorithm": "residual_boosting",
            "learning_rate": state["learning_rate"],
            "stages": [],
        }
        return {
            "boost_train_size": len(boost_train_range),
            "boost_val_size": len(boost_val_range),
            "learning_rate": state["learning_rate"],
            "max_rounds": state["max_rounds"],
        }

    round_match = _boosting_round_match(step_id)
    if round_match:
        round_idx, action = round_match
        if round_idx != int(state.get("next_round_index", 0)):
            raise ValueError("[VALIDATION] Boosting round step is out of order.")

        boost_train_range = _state_range(state, "boost_train_range")
        boost_val_range = _state_range(state, "boost_val_range")
        residual_train = np.asarray(state["residual_train"], dtype=float)
        residual_val = np.asarray(state["residual_val"], dtype=float)
        coefficient = 1.0 if round_idx == 0 else float(state["learning_rate"])

        if action == "evaluate_candidates":
            current_rmse = float("inf") if round_idx == 0 else rmse(np.zeros_like(residual_val), residual_val)
            round_parsed_data = boosting._build_residual_parsed_data(parsed_data, boost_train_range, residual_train)
            stage_results: List[Dict[str, Any]] = []
            for model_name in models_to_use:
                params = model_params[model_name]
                timeout = boosting._get_model_timeout(model_name, params)
                round_numeric_records, round_lstm_records = boosting._build_round_records(
                    round_parsed_data,
                    model_name,
                    params,
                )
                temp_model_file, stage_predictions = boosting._train_stage_model(
                    model_name,
                    round_numeric_records,
                    round_lstm_records,
                    boost_train_range,
                    boost_val_range,
                    params,
                    timeout,
                )
                fitted_train_predictions = boosting._compute_in_sample_fitted_predictions(
                    model_name,
                    temp_model_file,
                    round_parsed_data,
                    boost_train_range,
                )
                if round_idx == 0:
                    stage_predictions = normalize_demand_forecast(stage_predictions)
                    fitted_train_predictions = normalize_demand_forecast(fitted_train_predictions)
                next_residual_val = residual_val - coefficient * stage_predictions
                next_rmse = rmse(np.zeros_like(next_residual_val), next_residual_val)
                remove_member_model_artifact(model_name, temp_model_file)
                stage_results.append({
                    "model_name": model_name,
                    "coefficient": coefficient,
                    "rmse": next_rmse,
                    "fitted_train_predictions": fitted_train_predictions,
                    "next_residual_val": next_residual_val,
                })

            state["pending_round_index"] = round_idx
            state["pending_current_rmse"] = current_rmse
            state["pending_stage_results"] = stage_results
            return {
                "round": round_idx + 1,
                "coefficient": coefficient,
                "previous_rmse": None if current_rmse == float("inf") else current_rmse,
                "candidates": {result["model_name"]: result["rmse"] for result in stage_results},
            }

        pending_round = state.get("pending_round_index")
        if pending_round != round_idx:
            raise ValueError("[VALIDATION] Boosting candidate results are missing for this round.")

        stage_results = state["pending_stage_results"]
        current_rmse = float(state["pending_current_rmse"])
        best_result = min(stage_results, key=lambda result: result["rmse"])
        improvement = float("inf") if current_rmse == float("inf") else current_rmse - best_result["rmse"]
        stage_summary = {
            "round": round_idx + 1,
            "objective": "validation_residual_rmse",
            "coefficient": coefficient,
            "previous_rmse": None if current_rmse == float("inf") else current_rmse,
            "candidates": {result["model_name"]: result["rmse"] for result in stage_results},
            "winner": best_result["model_name"],
            "winner_rmse": best_result["rmse"],
            "improvement": None if improvement == float("inf") else improvement,
        }

        if best_result["rmse"] < current_rmse - 1e-12:
            state["model_chain"].append(best_result["model_name"])
            state["stage_coefficients"].append(coefficient)
            state["residual_train"] = residual_train - coefficient * best_result["fitted_train_predictions"]
            state["residual_val"] = best_result["next_residual_val"]
            state["next_round_index"] = round_idx + 1
            stage_summary["action"] = "selected"
        else:
            state["early_stopped"] = True
            stage_summary["action"] = "early_stop"

        state["search_summary"]["stages"].append(stage_summary)
        state.pop("pending_round_index", None)
        state.pop("pending_current_rmse", None)
        state.pop("pending_stage_results", None)

        return {
            **stage_summary,
            "model_chain": state["model_chain"],
        }

    if step_id == "retrain_chain":
        model_chain = state["model_chain"]
        stage_coefficients = state["stage_coefficients"]
        y_true_eval = np.asarray(parsed_data.y[eval_range], dtype=float)
        if not model_chain:
            raise ValueError(
                "[DATA] Boosting 未能选出任何有效阶段：所有候选基础模型在验证段的预测均无效，"
                "无法降低残差 RMSE。请检查所选数据区间的质量，或调整基础模型参数后重试。"
            )

        final_eval_preds = np.zeros(len(eval_range), dtype=float)
        residual_full_train = np.asarray(parsed_data.y[train_range], dtype=float).copy()
        base_model_states: Dict[str, Any] = {}
        stage_eval_summaries = []

        for stage_idx, model_name in enumerate(model_chain):
            coefficient = stage_coefficients[stage_idx]
            params = model_params[model_name]
            timeout = boosting._get_model_timeout(model_name, params)
            round_parsed_data = boosting._build_residual_parsed_data(parsed_data, train_range, residual_full_train)
            artifact_origin = "temporary_training"

            if stage_idx == 0 and model_name in base_model_files and os.path.exists(base_model_files[model_name]):
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
                            run_base_model_fit_subprocess(
                                model_type=model_name,
                                data=round_lstm_records,
                                train_start=train_range.start,
                                train_end=train_range.stop - 1,
                                source_model_file=adjusted_source_model_file,
                                save_model_file=stage_model_file,
                                timeout=timeout,
                            )
                        finally:
                            if temp_source_model_file and os.path.exists(temp_source_model_file):
                                os.remove(temp_source_model_file)
                        stage_eval_predictions = boosting._predict_eval_segment(
                            model_name,
                            stage_model_file,
                            train_range,
                            eval_range,
                            timeout,
                        )
                    else:
                        stage_model_file = source_model_file
                        stage_eval_predictions = boosting._predict_eval_segment(
                            model_name,
                            stage_model_file,
                            train_range,
                            eval_range,
                            timeout,
                        )
                        artifact_origin = "standalone_base"
                else:
                    stage_model_file = source_model_file
                    stage_eval_predictions = boosting._predict_eval_segment(
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
                stage_model_file, stage_eval_predictions = boosting._train_stage_model(
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
            if stage_idx == 0:
                stage_eval_predictions = normalize_demand_forecast(stage_eval_predictions)
                fitted_train_predictions = normalize_demand_forecast(fitted_train_predictions)
            final_eval_preds += coefficient * stage_eval_predictions
            residual_full_train = residual_full_train - coefficient * fitted_train_predictions
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

        state["base_model_states"] = base_model_states
        state["final_eval_preds"] = final_eval_preds
        state["eval_y_true"] = y_true_eval.tolist()
        state["stage_eval_summaries"] = stage_eval_summaries
        return {"model_chain": model_chain, "stage_count": len(model_chain), "stage_eval": stage_eval_summaries}

    if step_id == "combine_evaluate":
        y_true_eval = np.asarray(state["eval_y_true"], dtype=float)
        final_eval_preds = normalize_demand_forecast(state["final_eval_preds"])
        final_residuals = y_true_eval - final_eval_preds
        std_dev = float(np.std(final_residuals, ddof=1)) if len(final_residuals) > 1 else 0.0
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
            "learning_rate": state["learning_rate"],
            "metrics": metrics,
            "search_summary": state["search_summary"],
            "eval_y_true": y_true_eval.tolist(),
            "predictions": boosting._format_predictions(final_eval_preds.tolist(), std_dev),
        }
        attach_method_metadata(final_result, boosting.METHOD_NAME, boosting.FORECAST_STRATEGY, boosting.IMPLEMENTATION_NOTES)
        state["metrics"] = metrics
        state["std_dev_residuals"] = std_dev
        state["final_result"] = final_result
        return {"metrics": metrics, "std_dev_residuals": std_dev}

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
            "learning_rate": state["learning_rate"],
            "std_dev_residuals": state["std_dev_residuals"],
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
    args, _ = parser.parse_known_args()

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
        "protocol_version": 2,
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
