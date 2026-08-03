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
    get_member_model_path,
    normalize_demand_forecast,
    parse_base_model_files_arg,
    parse_base_model_names,
    plan_ensemble_selection_calibration_split,
    predict_member_on_chronological_holdout,
    persist_member_model_artifact,
    remove_member_model_artifact,
    require_explicit_lstm_feature_names,
    resolve_base_model_artifact_configs,
    warn_ignored_boosting_controls,
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
    split_plan = plan_ensemble_selection_calibration_split(
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
            len(models_to_use), int(split_plan["validation_size"])
        )
        state["minimum_relative_improvement"] = (
            0.05 if int(split_plan["validation_size"]) < 5 else 0.01
        )

    initialized: Dict[str, Any] = {
        "data_points": parsed_data.n_obs,
        "train_size": len(train_range),
        "evaluate_size": len(eval_range),
        "models": models_to_use,
    }
    # Weighted average and Stacking predict the evaluation range from a saved
    # member artifact when one matches. Boosting cannot: after the stage-zero
    # baseline every stage learns a signed residual, so a saved demand-scale
    # artifact has nothing to contribute and each stage is retrained. Reporting
    # the available files as "复用产物" there would be a false teaching claim.
    if model_type != "boosting":
        initialized["reused_artifacts"] = sorted(base_model_files.keys())
    return initialized


def _base_timeout(models_to_use: List[str], model_params: Dict[str, Dict[str, Any]]) -> int:
    if "lstm" not in models_to_use:
        return 120
    return 600


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
        predictions[model_name] = predict_member_on_chronological_holdout(
            model_name=model_name,
            params=params,
            source_model_file=source_model_file,
            numeric_data=numeric_data_dicts,
            lstm_data=lstm_data_dicts,
            train_range=train_range,
            holdout_range=holdout_range,
            timeout=timeout,
        )

    return predictions


def _rolling_oof_predictions(
    module,
    parsed_data,
    models_to_use: List[str],
    model_params: Dict[str, Dict[str, Any]],
    base_model_files: Dict[str, str],
    folds: List[Dict[str, int]],
    source_context: str,
) -> Dict[str, List[float]]:
    predictions: Dict[str, List[float]] = {
        model_name: [] for model_name in models_to_use
    }
    for fold in folds:
        fold_predictions = _holdout_predictions(
            module,
            parsed_data,
            models_to_use,
            model_params,
            base_model_files,
            range(
                int(fold["train_start"]),
                int(fold["train_end_exclusive"]),
            ),
            range(
                int(fold["holdout_start"]),
                int(fold["holdout_end_exclusive"]),
            ),
            source_context,
        )
        for model_name in models_to_use:
            predictions[model_name].extend(fold_predictions[model_name])
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

    member_match = re.fullmatch(
        r"(validation_prediction|eval_prediction)_(ma|es|arima|lstm)",
        step_id,
    )
    if member_match:
        action, model_name = member_match.groups()
        if model_name not in models_to_use:
            raise ValueError("[VALIDATION] Weighted member step is not part of this session.")
        if action == "validation_prediction":
            prediction = _holdout_predictions(
                weighted_avg,
                parsed_data,
                [model_name],
                model_params,
                base_model_files,
                _state_range(state, "weight_train_range"),
                _state_range(state, "weight_val_range"),
                "Saved Weighted Average LSTM base model",
            )[model_name]
            state.setdefault("weight_val_predictions", {})[model_name] = prediction
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
        prefix_size = int(state["split_plan"]["prefix_size"])
        selection_size = int(state["split_plan"]["selection_size"])
        calibration_size = int(state["split_plan"]["calibration_size"])
        weight_train_range = range(
            train_range.start,
            train_range.start + prefix_size,
        )
        weight_val_range = range(
            weight_train_range.stop,
            weight_train_range.stop + selection_size,
        )
        calibration_range = range(
            weight_val_range.stop,
            weight_val_range.stop + calibration_size,
        )
        state["weight_train_range"] = _range_to_state(weight_train_range)
        state["weight_val_range"] = _range_to_state(weight_val_range)
        state["calibration_range"] = _range_to_state(calibration_range)
        state["split_plan"]["ranges"] = {
            "prefix": {
                "start": weight_train_range.start,
                "end_exclusive": weight_train_range.stop,
            },
            "selection": {
                "start": weight_val_range.start,
                "end_exclusive": weight_val_range.stop,
            },
            "calibration": {
                "start": calibration_range.start,
                "end_exclusive": calibration_range.stop,
            },
        }
        return {
            "train_size": len(train_range),
            "weight_train_size": len(weight_train_range),
            "weight_val_size": len(weight_val_range),
            "calibration_size": len(calibration_range),
            "evaluate_size": len(eval_range),
            "split_plan": state["split_plan"],
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
        calibration_range = _state_range(state, "calibration_range")
        calibration_residuals = np.asarray([], dtype=float)
        calibration_source = "unavailable_post_selection_calibration"
        if len(calibration_range) > 0:
            calibration_predictions = _holdout_predictions(
                weighted_avg,
                parsed_data,
                models_to_use,
                model_params,
                base_model_files,
                range(train_range.start, calibration_range.start),
                calibration_range,
                "Saved Weighted Average LSTM base model",
            )
            x_calibration = np.asarray(
                [calibration_predictions[name] for name in models_to_use],
                dtype=float,
            ).T
            calibration_blend = normalize_demand_forecast(
                np.sum(x_calibration * model_weights, axis=1)
            )
            calibration_residuals = (
                parsed_data.y[calibration_range] - calibration_blend
            )
            calibration_source = "post_selection_calibration"
            state["calibration_predictions"] = calibration_predictions
        eval_forecast_steps = eval_range.stop - train_range.stop
        eval_forecast_offset = eval_range.start - train_range.stop
        full_uncertainty_profile, uncertainty_summary = ensemble_growth_profile(
            state["full_uncertainty_profiles"],
            weights,
            calibration_residuals,
            parsed_data.y[train_range],
            eval_forecast_steps,
            calibration_source=calibration_source,
        )
        uncertainty_profile = full_uncertainty_profile[
            eval_forecast_offset: eval_forecast_offset + len(eval_range)
        ]
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
            "calibration_residuals": calibration_residuals.tolist(),
            "calibration_source": calibration_source,
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

    member_match = re.fullmatch(
        r"(level1_prediction|eval_prediction)_(ma|es|arima|lstm)",
        step_id,
    )
    if member_match:
        action, model_name = member_match.groups()
        if model_name not in models_to_use:
            raise ValueError("[VALIDATION] Stacking member step is not part of this session.")
        if action == "level1_prediction":
            prediction = _rolling_oof_predictions(
                stacking,
                parsed_data,
                [model_name],
                model_params,
                base_model_files,
                state["oof_folds"],
                "Saved Stacking LSTM base model",
            )[model_name]
            state.setdefault("level1_prediction_map", {})[model_name] = prediction
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
        prefix_size = int(state["split_plan"]["prefix_size"])
        selection_size = int(state["split_plan"]["selection_size"])
        calibration_size = int(state["split_plan"]["calibration_size"])
        level0_range = range(
            train_range.start,
            train_range.start + prefix_size,
        )
        level1_range = range(
            level0_range.stop,
            level0_range.stop + selection_size,
        )
        calibration_range = range(
            level1_range.stop,
            level1_range.stop + calibration_size,
        )
        oof_folds = [dict(fold) for fold in state["split_plan"]["oof_folds"]]
        state["level0_range"] = _range_to_state(level0_range)
        state["level1_range"] = _range_to_state(level1_range)
        state["calibration_range"] = _range_to_state(calibration_range)
        state["oof_folds"] = oof_folds
        state["split_plan"].update(
            {
                "oof_strategy": "expanding_window_rolling_origin",
                "oof_fold_count": len(oof_folds),
                "oof_folds": oof_folds,
                "ranges": {
                    "prefix": {
                        "start": level0_range.start,
                        "end_exclusive": level0_range.stop,
                    },
                    "selection": {
                        "start": level1_range.start,
                        "end_exclusive": level1_range.stop,
                    },
                    "calibration": {
                        "start": calibration_range.start,
                        "end_exclusive": calibration_range.stop,
                    },
                },
            }
        )
        return {
            "train_size": len(train_range),
            "level0_size": len(level0_range),
            "level1_size": len(level1_range),
            "calibration_size": len(calibration_range),
            "oof_fold_count": len(oof_folds),
            "evaluate_size": len(eval_range),
            "split_plan": state["split_plan"],
        }

    if step_id == "level1_predictions":
        prediction_map = _rolling_oof_predictions(
            stacking,
            parsed_data,
            models_to_use,
            model_params,
            base_model_files,
            state["oof_folds"],
            "Saved Stacking LSTM base model",
        )
        state["level1_predictions"] = [prediction_map[name] for name in models_to_use]
        return {
            "predicted_points": {name: len(values) for name, values in prediction_map.items()},
            "models": models_to_use,
        }

    if step_id == "fit_meta_model":
        level1_range = _state_range(state, "level1_range")
        if "level1_prediction_map" in state:
            state["level1_predictions"] = [
                state["level1_prediction_map"][name] for name in models_to_use
            ]
        x_meta_train = np.column_stack(state["level1_predictions"])
        y_meta_train = parsed_data.y[level1_range]
        meta_model = stacking._fit_meta_model(x_meta_train, y_meta_train, models_to_use)
        meta_model.update(
            {
                "training_strategy": "expanding_window_rolling_origin_oof",
                "oof_fold_count": len(state["oof_folds"]),
                "oof_row_count": int(x_meta_train.shape[0]),
            }
        )
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
        calibration_range = _state_range(state, "calibration_range")
        calibration_residuals = np.asarray([], dtype=float)
        calibration_source = "unavailable_post_selection_calibration"
        if len(calibration_range) > 0:
            calibration_prediction_map = _holdout_predictions(
                stacking,
                parsed_data,
                models_to_use,
                model_params,
                base_model_files,
                range(train_range.start, calibration_range.start),
                calibration_range,
                "Saved Stacking LSTM base model",
            )
            x_meta_calibration = np.column_stack(
                [
                    calibration_prediction_map[name]
                    for name in models_to_use
                ]
            )
            calibration_blend = normalize_demand_forecast(
                stacking._predict_with_meta_model(
                    state["meta_model"],
                    x_meta_calibration,
                )
            )
            calibration_residuals = (
                parsed_data.y[calibration_range] - calibration_blend
            )
            calibration_source = "post_selection_calibration"
            state["calibration_predictions"] = calibration_prediction_map
        coefficients = dict(zip(
            state["meta_model"]["model_names"],
            state["meta_model"]["coefficients"],
        ))
        eval_forecast_steps = eval_range.stop - train_range.stop
        eval_forecast_offset = eval_range.start - train_range.stop
        full_uncertainty_profile, uncertainty_summary = ensemble_growth_profile(
            state["full_uncertainty_profiles"],
            coefficients,
            calibration_residuals,
            parsed_data.y[train_range],
            eval_forecast_steps,
            calibration_source=calibration_source,
        )
        uncertainty_profile = full_uncertainty_profile[
            eval_forecast_offset: eval_forecast_offset + len(eval_range)
        ]
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
            "calibration_residuals": calibration_residuals.tolist(),
            "calibration_source": calibration_source,
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


def _run_boosting(step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if step_id == "prepare_data":
        return _initialize_state("boosting", args, state)

    parsed_data = state["parsed_data"]
    train_range = _state_range(state, "train_range")
    eval_range = _state_range(state, "eval_range")
    models_to_use = state["models"]
    model_params = state["model_params"]

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
        boost_train_range = _state_range(state, "boost_train_range")
        boost_val_range = _state_range(state, "boost_val_range")
        residual_train = np.asarray(state["residual_train"], dtype=float)
        residual_val = np.asarray(state["residual_val"], dtype=float)
        if state.get("pending_round_index") != round_idx:
            state["pending_round_index"] = round_idx
            state["pending_current_rmse"] = rmse(np.zeros_like(residual_val), residual_val)
            state["pending_stage_results"] = []
        if any(
            result["model_name"] == model_name
            for result in state["pending_stage_results"]
        ):
            raise ValueError("[VALIDATION] Boosting candidate was already evaluated.")

        params = model_params[model_name]
        timeout = boosting._get_model_timeout(model_name, params)
        round_parsed_data = boosting._build_residual_parsed_data(
            parsed_data,
            boost_train_range,
            residual_train,
        )
        round_numeric_records, round_lstm_records = boosting._build_round_records(
            round_parsed_data,
            model_name,
            params,
        )
        temp_model_file, stage_predictions, _stage_profile = boosting._train_stage_model(
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
        stage_predictions = boosting._apply_stage_output_semantics(
            round_idx,
            stage_predictions,
            residual_target=True,
        )
        fitted_train_predictions = boosting._apply_stage_output_semantics(
            round_idx,
            fitted_train_predictions,
            residual_target=True,
        )
        coefficient = boosting._nonnegative_line_search(residual_val, stage_predictions)
        next_residual_val = residual_val - coefficient * stage_predictions
        next_rmse = rmse(np.zeros_like(next_residual_val), next_residual_val)
        remove_member_model_artifact(model_name, temp_model_file)
        state["pending_stage_results"].append({
            "model_name": model_name,
            "coefficient": coefficient,
            "rmse": next_rmse,
            "fitted_train_predictions": fitted_train_predictions,
            "next_residual_val": next_residual_val,
        })
        return {
            "round": round_idx + 1,
            "model": model_name,
            "previous_rmse": state["pending_current_rmse"],
            "rmse": next_rmse,
            "coefficient": coefficient,
        }

    if step_id == "split_residual_sets":
        prefix_size = int(state["split_plan"]["prefix_size"])
        selection_size = int(state["split_plan"]["selection_size"])
        calibration_size = int(state["split_plan"]["calibration_size"])
        boost_train_range = range(
            train_range.start,
            train_range.start + prefix_size,
        )
        boost_val_range = range(
            boost_train_range.stop,
            boost_train_range.stop + selection_size,
        )
        calibration_range = range(
            boost_val_range.stop,
            boost_val_range.stop + calibration_size,
        )
        (
            selected_baseline,
            baseline_fitted_train,
            baseline_selection_predictions,
            baseline_diagnostics,
        ) = boosting.choose_stage_zero_baseline(
            parsed_data,
            boost_train_range,
            boost_val_range,
        )
        y_true_val = np.asarray(parsed_data.y[boost_val_range], dtype=float)
        state["boost_train_range"] = _range_to_state(boost_train_range)
        state["boost_val_range"] = _range_to_state(boost_val_range)
        state["calibration_range"] = _range_to_state(calibration_range)
        state["selection_baseline"] = selected_baseline
        state["baseline_diagnostics"] = baseline_diagnostics
        state["residual_train"] = (
            np.asarray(parsed_data.y[boost_train_range], dtype=float)
            - baseline_fitted_train
        )
        state["residual_val"] = (
            y_true_val - baseline_selection_predictions
        )
        state["model_chain"] = []
        state["stage_coefficients"] = []
        state["next_round_index"] = 0
        state["search_summary"] = {
            "algorithm": "residual_boosting",
            "coefficient_strategy": "nonnegative_squared_error_line_search",
            "post_line_search_shrinkage": (
                boosting.get_post_line_search_shrinkage_metadata()
            ),
            "minimum_relative_improvement": state["minimum_relative_improvement"],
            "max_rounds": state["max_rounds"],
            "split_plan": state["split_plan"],
            "baseline": baseline_diagnostics,
            "stages": [],
        }
        state["split_plan"]["ranges"] = {
            "prefix": {
                "start": boost_train_range.start,
                "end_exclusive": boost_train_range.stop,
            },
            "selection": {
                "start": boost_val_range.start,
                "end_exclusive": boost_val_range.stop,
            },
            "calibration": {
                "start": calibration_range.start,
                "end_exclusive": calibration_range.stop,
            },
        }
        return {
            "boost_train_size": len(boost_train_range),
            "boost_val_size": len(boost_val_range),
            "calibration_size": len(calibration_range),
            "baseline": baseline_diagnostics,
            "max_rounds": state["max_rounds"],
            "split_plan": state["split_plan"],
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
        if action == "evaluate_candidates":
            current_rmse = rmse(np.zeros_like(residual_val), residual_val)
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
                temp_model_file, stage_predictions, _stage_profile = boosting._train_stage_model(
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
                stage_predictions = boosting._apply_stage_output_semantics(
                    round_idx,
                    stage_predictions,
                    residual_target=True,
                )
                fitted_train_predictions = boosting._apply_stage_output_semantics(
                    round_idx,
                    fitted_train_predictions,
                    residual_target=True,
                )
                coefficient = boosting._nonnegative_line_search(
                    residual_val, stage_predictions
                )
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
                "previous_rmse": current_rmse,
                "candidates": {
                    result["model_name"]: {
                        "rmse": result["rmse"],
                        "coefficient": result["coefficient"],
                    }
                    for result in stage_results
                },
            }

        pending_round = state.get("pending_round_index")
        if pending_round != round_idx:
            raise ValueError("[VALIDATION] Boosting candidate results are missing for this round.")

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
            state["residual_train"] = residual_train - best_result["coefficient"] * best_result["fitted_train_predictions"]
            state["residual_val"] = best_result["next_residual_val"]
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

        return {
            **stage_summary,
            "model_chain": state["model_chain"],
        }

    if step_id == "initialize_refit":
        selection_model_chain = list(state["model_chain"])
        state["search_summary"]["selection_outcome"] = (
            "residual_chain_selected"
            if selection_model_chain
            else "baseline_retained_no_improvement"
        )
        state["search_summary"]["selection_model_chain"] = selection_model_chain
        state["selection_model_chain"] = selection_model_chain
        state["selection_coefficients"] = list(state["stage_coefficients"])
        state["refit_model_chain"] = []
        state["refit_stage_coefficients"] = []
        baseline_kind = str(state["selection_baseline"]["kind"])
        baseline_period = state["selection_baseline"].get("seasonal_period")
        (
            calibration_residuals,
            calibration_source,
            calibration_refit,
        ) = boosting._prepare_selected_chain_calibration(
            parsed_data,
            train_range,
            _state_range(state, "calibration_range"),
            selection_model_chain,
            model_params,
            baseline_kind,
            baseline_period,
        )
        state["calibration_candidate_residuals"] = calibration_residuals.tolist()
        state["calibration_candidate_source"] = calibration_source
        state["calibration_refit"] = calibration_refit
        state["baseline"] = boosting.fit_baseline_state_for_range(
            parsed_data,
            train_range,
            baseline_kind,
            baseline_period,
        )
        state["final_eval_preds"] = boosting.baseline_forecast_for_range(
            parsed_data,
            train_range,
            eval_range,
            baseline_kind,
            baseline_period,
        )
        state["residual_full_train"] = (
            np.asarray(parsed_data.y[train_range], dtype=float)
            - boosting.baseline_fitted_values_for_range(
                parsed_data,
                train_range,
                baseline_kind,
                baseline_period,
            )
        )
        state["base_model_states"] = {}
        state["full_uncertainty_profiles"] = {}
        state["uncertainty_coefficients"] = {}
        state["stage_eval_summaries"] = []
        state["eval_y_true"] = np.asarray(
            parsed_data.y[eval_range],
            dtype=float,
        ).tolist()
        return {
            "selection_model_chain": selection_model_chain,
            "baseline": state["baseline"],
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
        # Every stage is a signed-residual learner: stage 0 is the statistical
        # baseline, so a demand-scale base-model artifact cannot be replayed here.
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
            residual_target=True,
        )
        fitted_train_predictions = boosting._apply_stage_output_semantics(
            stage_idx,
            fitted_train_predictions,
            residual_target=True,
        )
        coefficient = boosting._nonnegative_line_search(
            residual_full_train,
            fitted_train_predictions,
        )
        if coefficient <= np.finfo(float).eps:
            boosting.remove_member_model_artifact(model_name, stage_model_file)
            state["search_summary"]["refit_stop"] = {
                "stage": stage_idx + 1,
                "model_name": model_name,
                "reason": "nonpositive_full_refit_coefficient",
                "coefficient": coefficient,
            }
            return {
                "action": "trim_refit_chain",
                "stage": stage_idx + 1,
                "model": model_name,
                "coefficient": coefficient,
            }

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
        })
        stage_summary = {
            "round": stage_idx + 1,
            "model_name": model_name,
            "coefficient": coefficient,
            "rmse": rmse(np.asarray(parsed_data.y[eval_range], dtype=float), final_eval_preds),
        }
        state["stage_eval_summaries"].append(stage_summary)
        state["eval_y_true"] = np.asarray(parsed_data.y[eval_range], dtype=float).tolist()
        return {"action": "refit_stage", **stage_summary}

    if step_id == "retrain_chain":
        selection_model_chain = list(state["model_chain"])
        selection_coefficients = list(state["stage_coefficients"])
        state["search_summary"]["selection_model_chain"] = selection_model_chain
        state["selection_model_chain"] = selection_model_chain
        refit_model_chain: List[str] = []
        stage_coefficients: List[float] = []
        y_true_eval = np.asarray(parsed_data.y[eval_range], dtype=float)
        baseline_kind = str(state["selection_baseline"]["kind"])
        baseline_period = state["selection_baseline"].get("seasonal_period")
        (
            calibration_residuals,
            calibration_source,
            calibration_refit,
        ) = boosting._prepare_selected_chain_calibration(
            parsed_data,
            train_range,
            _state_range(state, "calibration_range"),
            selection_model_chain,
            model_params,
            baseline_kind,
            baseline_period,
        )
        state["calibration_candidate_residuals"] = calibration_residuals.tolist()
        state["calibration_candidate_source"] = calibration_source
        state["calibration_refit"] = calibration_refit
        state["baseline"] = boosting.fit_baseline_state_for_range(
            parsed_data,
            train_range,
            baseline_kind,
            baseline_period,
        )
        final_eval_preds = boosting.baseline_forecast_for_range(
            parsed_data,
            train_range,
            eval_range,
            baseline_kind,
            baseline_period,
        )
        residual_full_train = (
            np.asarray(parsed_data.y[train_range], dtype=float)
            - boosting.baseline_fitted_values_for_range(
                parsed_data,
                train_range,
                baseline_kind,
                baseline_period,
            )
        )
        base_model_states: Dict[str, Any] = {}
        full_uncertainty_profiles: Dict[str, List[float]] = {}
        uncertainty_coefficients: Dict[str, float] = {}
        stage_eval_summaries = []

        for stage_idx, model_name in enumerate(selection_model_chain):
            params = model_params[model_name]
            timeout = boosting._get_model_timeout(model_name, params)
            round_parsed_data = boosting._build_residual_parsed_data(parsed_data, train_range, residual_full_train)
            # Every stage is a signed-residual learner: stage 0 is the statistical
            # baseline, so a demand-scale base-model artifact cannot be replayed here.
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
                residual_target=True,
            )
            fitted_train_predictions = boosting._apply_stage_output_semantics(
                stage_idx,
                fitted_train_predictions,
                residual_target=True,
            )
            coefficient = boosting._nonnegative_line_search(
                residual_full_train, fitted_train_predictions
            )
            if coefficient <= np.finfo(float).eps:
                boosting.remove_member_model_artifact(model_name, stage_model_file)
                state["search_summary"]["refit_stop"] = {
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
            stage_rmse = rmse(y_true_eval, final_eval_preds)

            base_model_states.setdefault(model_name, {"models": []})["models"].append({
                "model_type": model_name,
                "model_file": stage_model_file,
                "params": params,
                "round": stage_idx,
                "coefficient": coefficient,
            })
            stage_eval_summaries.append({
                "round": stage_idx + 1,
                "model_name": model_name,
                "coefficient": coefficient,
                "rmse": stage_rmse,
            })

        state["model_chain"] = refit_model_chain
        state["base_model_states"] = base_model_states
        state["final_eval_preds"] = final_eval_preds
        state["eval_y_true"] = y_true_eval.tolist()
        state["stage_eval_summaries"] = stage_eval_summaries
        state["selection_coefficients"] = selection_coefficients
        state["stage_coefficients"] = stage_coefficients
        state["full_uncertainty_profiles"] = full_uncertainty_profiles
        state["uncertainty_coefficients"] = uncertainty_coefficients
        return {
            "model_chain": refit_model_chain,
            "stage_count": len(refit_model_chain),
            "stage_eval": stage_eval_summaries,
        }

    if step_id == "combine_evaluate":
        if "refit_model_chain" in state:
            state["model_chain"] = list(state["refit_model_chain"])
            state["stage_coefficients"] = list(state["refit_stage_coefficients"])
        state["search_summary"]["post_line_search_shrinkage"] = (
            boosting.get_post_line_search_shrinkage_metadata()
        )
        y_true_eval = np.asarray(state["eval_y_true"], dtype=float)
        final_eval_preds = normalize_demand_forecast(state["final_eval_preds"])
        calibration_range = _state_range(state, "calibration_range")
        calibration_refit = state.get("calibration_refit")
        calibration_candidate_source = state.get("calibration_candidate_source")
        calibration_candidate_residuals = state.get(
            "calibration_candidate_residuals"
        )
        if (
            not isinstance(calibration_refit, dict)
            or not isinstance(calibration_candidate_source, str)
            or not isinstance(calibration_candidate_residuals, list)
        ):
            selection_model_chain = state.get("selection_model_chain")
            if not isinstance(selection_model_chain, list):
                summary_chain = state["search_summary"].get("selection_model_chain")
                selection_model_chain = (
                    summary_chain if isinstance(summary_chain, list) else None
                )
            if len(calibration_range) > 0 and selection_model_chain is not None:
                calibration_baseline = state.get("selection_baseline")
                if not isinstance(calibration_baseline, dict):
                    calibration_baseline = state["baseline"]
                (
                    prepared_residuals,
                    calibration_candidate_source,
                    calibration_refit,
                ) = boosting._prepare_selected_chain_calibration(
                    parsed_data,
                    train_range,
                    calibration_range,
                    list(selection_model_chain),
                    model_params,
                    str(calibration_baseline["kind"]),
                    calibration_baseline.get("seasonal_period"),
                    prepared_before_full_refit=False,
                )
                calibration_candidate_residuals = prepared_residuals.tolist()
            else:
                calibration_candidate_residuals = []
                calibration_candidate_source = (
                    "unavailable_post_selection_calibration"
                    if len(calibration_range) == 0
                    else "unavailable_missing_pre_refit_calibration"
                )
                calibration_refit = {
                    "requested_chain": list(selection_model_chain or []),
                    "retained_chain": [],
                    "coefficients": [],
                    "matched_selection_structure": False,
                    "matched_final_structure": False,
                    "prepared_before_full_refit": False,
                    "preparation_mode": "unavailable_missing_selection_chain",
                    "selection_chain_frozen_before_calibration_labels": False,
                    "full_refit_includes_calibration_segment": (
                        len(calibration_range) > 0
                    ),
                    "final_refit_parameters_may_use_calibration_labels": (
                        len(calibration_range) > 0
                    ),
                    "final_refit_stage_coefficients_may_use_calibration_labels": bool(
                        len(calibration_range) > 0 and state.get("model_chain")
                    ),
                    "coverage_claim": "none",
                }

        (
            calibration_residuals,
            calibration_source,
            calibration_refit,
        ) = boosting._finalize_selected_chain_calibration(
            np.asarray(calibration_candidate_residuals, dtype=float),
            calibration_candidate_source,
            calibration_refit,
            list(state["model_chain"]),
        )
        state["calibration_candidate_residuals"] = list(
            calibration_candidate_residuals
        )
        state["calibration_candidate_source"] = calibration_candidate_source
        state["calibration_refit"] = calibration_refit
        state["search_summary"]["calibration_refit"] = calibration_refit
        eval_forecast_steps = eval_range.stop - train_range.stop
        eval_forecast_offset = eval_range.start - train_range.stop
        boosting._add_baseline_uncertainty_member(
            state["full_uncertainty_profiles"],
            state["uncertainty_coefficients"],
            state["baseline"],
            eval_forecast_steps,
            has_member_stages=bool(state["model_chain"]),
        )
        full_uncertainty_profile, uncertainty_summary = ensemble_growth_profile(
            state["full_uncertainty_profiles"],
            state["uncertainty_coefficients"],
            calibration_residuals,
            parsed_data.y[train_range],
            eval_forecast_steps,
            calibration_source=calibration_source,
        )
        uncertainty_profile = full_uncertainty_profile[
            eval_forecast_offset: eval_forecast_offset + len(eval_range)
        ]
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
            "post_line_search_shrinkage": (
                boosting.get_post_line_search_shrinkage_metadata()
            ),
            "models": models_to_use,
            "baseline": state["baseline"],
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
        if state["model_chain"]:
            state["search_summary"]["refit_outcome"] = "residual_chain_retained"
        elif state.get("selection_model_chain"):
            state["search_summary"]["refit_outcome"] = (
                "baseline_retained_after_refit_trim"
            )
        else:
            state["search_summary"]["refit_outcome"] = (
                "baseline_retained_no_improvement"
            )
        state["uncertainty"] = {
            "calibration_residuals": calibration_residuals.tolist(),
            "calibration_source": calibration_source,
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

        model_state = {
            "model_type": boosting.MODEL_TYPE,
            "boosting_variant": "residual",
            "post_line_search_shrinkage": (
                boosting.get_post_line_search_shrinkage_metadata()
            ),
            "model_chain": state["model_chain"],
            "stage_coefficients": state["stage_coefficients"],
            "base_model_states": base_model_states,
            "baseline": state["baseline"],
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
    parser.add_argument("--max-p", type=int, help="Deprecated compatibility flag; the ARIMA p limit is data-derived")
    parser.add_argument("--max-q", type=int, help="Deprecated compatibility flag; the ARIMA q limit is data-derived")
    parser.add_argument("--look-back", type=int, help="Deprecated compatibility flag; LSTM look-back is data-derived")
    parser.add_argument("--epochs", type=int, help="Deprecated compatibility flag; LSTM epochs are data-derived")
    parser.add_argument("--feature-types", type=str)
    parser.add_argument("--lstm-features", type=str)
    parser.add_argument("--lstm-target-feature", type=str)
    parser.add_argument("--normalization", type=str, choices=["minmax", "zscore"])
    parser.add_argument("--base-model-files", type=str)
    parser.add_argument("--learning-rate", type=float, help="Deprecated compatibility flag; Boosting uses line search")
    parser.add_argument("--max-rounds", type=int, help="Deprecated compatibility flag; Boosting rounds are data-derived")
    args = parser.parse_args()

    if args.model_type == "boosting":
        warn_ignored_boosting_controls(args)

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
        "protocol_version": 3,
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
