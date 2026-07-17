#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpointed production refit runner.

Each invocation performs metadata initialization, one base-model refit, or the
final artifact assembly.  A refit invocation starts at most one nested model
process and stores forecasts for the prepared horizon in the production
artifact, so the later prediction request never fans out to member processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data_utils import load_data_from_json
from ensemble_utils import (
    build_ensemble_subprocess_data,
    get_member_model_path,
    normalize_demand_forecast,
    remove_member_model_artifact,
)
from error_protocol import format_known_error
from model_info import attach_method_metadata, ensure_state_metadata
from serialization import dump_model, load_model
from subprocess_utils import (
    extract_predictions_from_result,
    extract_uncertainty_profile_from_result,
    run_base_model_fit_subprocess,
)
from uncertainty import ensemble_growth_profile
from models import arima, boosting, es, lstm, ma, stacking, weighted_avg


BASE_MODULES = {"ma": ma, "es": es, "arima": arima, "lstm": lstm}
ENSEMBLE_MODULES = {
    "weighted_avg": weighted_avg,
    "stacking": stacking,
    "boosting": boosting,
}
ALL_MODELS = {**BASE_MODULES, **ENSEMBLE_MODULES}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_checkpoint(path: str | None) -> Dict[str, Any]:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, "rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise ValueError("[MODEL] Production preparation checkpoint is invalid.")
    return value


def _save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    with open(temporary, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_receipt(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _existing_receipt(path: str, attempt_id: str, step_id: str) -> Dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    candidate = payload.get("candidate_state_path")
    if (
        payload.get("protocol_version") != 1
        or payload.get("attempt_id") != attempt_id
        or payload.get("step_id") != step_id
        or not isinstance(candidate, str)
        or _sha256(candidate) != payload.get("candidate_state_sha256")
    ):
        return None
    return payload


def _read_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not value:
        raise ValueError("[DATA] Production preparation dataset snapshot is empty.")
    return value


def _base_timeout(model_name: str) -> int:
    return 600 if model_name == "lstm" else 120


def _prepared_points(result: Dict[str, Any], steps: int) -> List[Dict[str, Any]]:
    points = result.get("predictions")
    if not isinstance(points, list) or len(points) < steps:
        raise ValueError("[MODEL] Production refit did not return the prepared forecast horizon.")
    return [dict(point) for point in points[:steps]]


def _initialize(args, state: Dict[str, Any]) -> Dict[str, Any]:
    records = _read_records(args.data_file)
    source_state = load_model(args.source_model)
    if not isinstance(source_state, dict) or source_state.get("model_type") != args.model_type:
        raise ValueError("[MODEL] Backtest artifact does not match production model type.")
    source_module = ALL_MODELS[args.model_type]
    if args.model_type == "lstm":
        source_strategy = str(
            source_state.get("forecast_strategy")
            or lstm.DIRECT_FORECAST_STRATEGY
        )
        source_notes = lstm._metadata_for_strategy(source_strategy)[1]
    else:
        source_strategy = source_module.FORECAST_STRATEGY
        source_notes = source_module.IMPLEMENTATION_NOTES
    ensure_state_metadata(
        source_state,
        source_module.METHOD_NAME,
        source_strategy,
        source_notes,
    )
    if args.model_type == "lstm" and (
        source_strategy != lstm.DIRECT_FORECAST_STRATEGY
        or source_state.get("feature_timing") != lstm.FEATURE_TIMING
        or source_state.get("known_future_features") is not False
    ):
        raise ValueError(
            "[MODEL] Backtest LSTM artifact is not historical-only and must be retrained."
        )
    state.update({
        "model_type": args.model_type,
        "records": records,
        "parsed_data": load_data_from_json(json.dumps(records, ensure_ascii=False)),
        "source_model": args.source_model,
        "source_state": source_state,
        "save_model": args.save_model,
        "train_start": int(args.train_start),
        "train_end": int(args.train_end),
        "forecast_steps": int(args.forecast_steps),
        "member_predictions": {},
        "member_profiles": {},
        "member_states": {},
    })

    if args.model_type in BASE_MODULES:
        work_items = [{"id": "fit_model", "model": args.model_type}]
    elif args.model_type in {"weighted_avg", "stacking"}:
        models = list(source_state.get("models") or [])
        if len(models) < 2:
            raise ValueError("[MODEL] Ensemble artifact has no usable member list.")
        work_items = [{"id": f"fit_member_{name}", "model": name} for name in models]
    else:
        stages = []
        for stage_idx, model_name, coefficient, member_state in boosting._iter_stage_models(source_state):
            stages.append({
                "stage": int(stage_idx),
                "model": model_name,
                "coefficient": float(coefficient),
                "source_model": member_state["model_file"],
                "params": dict(member_state.get("params") or {}),
            })
        if not stages:
            raise ValueError("[MODEL] Boosting artifact has no usable model chain.")
        state["boosting_stages"] = stages
        state["residual_train"] = np.asarray(
            state["parsed_data"].y[range(state["train_start"], state["train_end"] + 1)],
            dtype=float,
        )
        state["future_prediction"] = np.zeros(state["forecast_steps"], dtype=float)
        state["refit_stages"] = []
        state["boosting_refit_terminated"] = False
        work_items = [
            {"id": f"fit_stage_{item['stage'] + 1}_{item['model']}", "model": item["model"]}
            for item in stages
        ]
    return {"work_items": work_items}


def _fit_base(args, state: Dict[str, Any]) -> Dict[str, Any]:
    model_name = args.model_type
    # A timed-out attempt may leave a partial pickle or LSTM sidecar at the
    # stable candidate path.  Every retry starts from the committed backtest
    # artifact, so discard that uncommitted output before refitting.
    remove_member_model_artifact(model_name, state["save_model"])
    result = run_base_model_fit_subprocess(
        model_type=model_name,
        data=state["records"],
        train_start=state["train_start"],
        train_end=state["train_end"],
        source_model_file=state["source_model"],
        save_model_file=state["save_model"],
        fit_forecast_steps=state["forecast_steps"] if model_name == "lstm" else None,
        predict_steps=state["forecast_steps"],
        timeout=_base_timeout(model_name),
    )
    state["prepared_predictions"] = _prepared_points(result, state["forecast_steps"])
    return {"model": model_name, "prepared_points": state["forecast_steps"]}


def _flat_member_source(source_state: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    member = (source_state.get("base_model_states") or {}).get(model_name)
    if not isinstance(member, dict) or not member.get("model_file"):
        raise ValueError(f"[MODEL] Ensemble source is missing member {model_name}.")
    return member


def _fit_flat_member(step_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    match = re.fullmatch(r"fit_member_(ma|es|arima|lstm)", step_id)
    if not match:
        raise ValueError("[VALIDATION] Invalid production ensemble member step.")
    model_name = match.group(1)
    source_member = _flat_member_source(state["source_state"], model_name)
    params = dict(source_member.get("params") or {})
    parsed_data = state["parsed_data"]
    numeric_records, lstm_records = build_ensemble_subprocess_data(
        parsed_data,
        [model_name],
        {model_name: params},
    )
    destination = get_member_model_path(state["save_model"], model_name)
    remove_member_model_artifact(model_name, destination)
    result = run_base_model_fit_subprocess(
        model_type=model_name,
        data=lstm_records if model_name == "lstm" else numeric_records,
        train_start=state["train_start"],
        train_end=state["train_end"],
        source_model_file=source_member["model_file"],
        save_model_file=destination,
        fit_forecast_steps=state["forecast_steps"] if model_name == "lstm" else None,
        predict_steps=state["forecast_steps"],
        timeout=_base_timeout(model_name),
    )
    points = _prepared_points(result, state["forecast_steps"])
    state["member_predictions"][model_name] = [float(point["prediction"]) for point in points]
    state["member_profiles"][model_name] = extract_uncertainty_profile_from_result(result)
    state["member_states"][model_name] = {
        "model_type": model_name,
        "model_file": destination,
        "params": params,
    }
    return {"model": model_name, "prepared_points": len(points)}


def _fit_boosting_stage(step_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    match = re.fullmatch(r"fit_stage_(\d+)_(ma|es|arima|lstm)", step_id)
    if not match:
        raise ValueError("[VALIDATION] Invalid production boosting stage.")
    stage_idx = int(match.group(1)) - 1
    model_name = match.group(2)
    stage = state["boosting_stages"][stage_idx]
    if stage["model"] != model_name:
        raise ValueError("[VALIDATION] Production boosting stage does not match source chain.")
    if state.get("boosting_refit_terminated"):
        raise ValueError("[VALIDATION] Production boosting refit chain has already been trimmed.")
    residual_train = np.asarray(state["residual_train"], dtype=float)
    parsed_data = state["parsed_data"]
    train_range = range(state["train_start"], state["train_end"] + 1)
    round_data = boosting._build_residual_parsed_data(parsed_data, train_range, residual_train)
    numeric_records, lstm_records = boosting._build_round_records(
        round_data,
        model_name,
        stage["params"],
    )
    destination = get_member_model_path(state["save_model"], model_name, stage_idx)
    remove_member_model_artifact(model_name, destination)
    result = run_base_model_fit_subprocess(
        model_type=model_name,
        data=lstm_records if model_name == "lstm" else numeric_records,
        train_start=state["train_start"],
        train_end=state["train_end"],
        source_model_file=stage["source_model"],
        save_model_file=destination,
        fit_forecast_steps=state["forecast_steps"] if model_name == "lstm" else None,
        predict_steps=state["forecast_steps"],
        allow_negative_predictions=True,
        timeout=_base_timeout(model_name),
    )
    points = _prepared_points(result, state["forecast_steps"])
    stage_predictions = boosting._apply_stage_output_semantics(
        stage_idx,
        np.asarray([point["prediction"] for point in points], dtype=float),
    )
    fitted = boosting._compute_in_sample_fitted_predictions(
        model_name,
        destination,
        round_data,
        train_range,
    )
    fitted = boosting._apply_stage_output_semantics(stage_idx, fitted)
    # Preserve the coefficient learned on the chronological validation holdout.
    # Full-range in-sample fitted values only construct the next residual target;
    # they never retune deployment coefficients.
    coefficient = float(stage["coefficient"])
    state["residual_train"] = residual_train - coefficient * fitted
    state["future_prediction"] = np.asarray(state["future_prediction"], dtype=float) + coefficient * stage_predictions
    key = f"stage_{stage_idx}_{model_name}"
    state["member_profiles"][key] = extract_uncertainty_profile_from_result(result)
    state.setdefault("uncertainty_coefficients", {})[key] = coefficient
    state["member_states"].setdefault(model_name, {"models": []})["models"].append({
        "model_type": model_name,
        "model_file": destination,
        "params": stage["params"],
        "round": stage_idx,
        "coefficient": coefficient,
    })
    state["refit_stages"].append({
        "stage": stage_idx,
        "model": model_name,
        "coefficient": coefficient,
    })
    return {"stage": stage_idx + 1, "model": model_name, "coefficient": coefficient}


def _finalize(state: Dict[str, Any]) -> Dict[str, Any]:
    model_type = state["model_type"]
    steps = state["forecast_steps"]
    if model_type in BASE_MODULES:
        artifact_state = load_model(state["save_model"])
        artifact_state["prepared_predictions"] = state["prepared_predictions"]
        artifact_state["prepared_forecast_steps"] = steps
        dump_model(artifact_state, state["save_model"])
        return {"saved_model": state["save_model"], "prepared_points": steps}

    source_state = state["source_state"]
    uncertainty = dict(source_state.get("uncertainty") or {})
    reference = uncertainty.get("reference_series") or state["parsed_data"].y[
        range(state["train_start"], state["train_end"] + 1)
    ].tolist()
    residuals = uncertainty.get("calibration_residuals") or []
    module = ENSEMBLE_MODULES[model_type]
    if model_type == "weighted_avg":
        models = list(source_state["models"])
        weights = dict(source_state["weights"])
        matrix = np.asarray([state["member_predictions"][name] for name in models], dtype=float).T
        predictions = normalize_demand_forecast(
            matrix @ np.asarray([weights[name] for name in models], dtype=float)
        )
        coefficients = weights
        extra = {
            "models": models,
            "weights": weights,
            "weight_diagnostics": source_state.get("weight_diagnostics", {}),
        }
    elif model_type == "stacking":
        models = list(source_state["models"])
        matrix = np.asarray([state["member_predictions"][name] for name in models], dtype=float).T
        predictions = normalize_demand_forecast(
            stacking._predict_with_meta_model(source_state["meta_model"], matrix)
        )
        coefficients = dict(zip(
            source_state["meta_model"]["model_names"],
            source_state["meta_model"]["coefficients"],
        ))
        extra = {"models": models, "meta_model": source_state["meta_model"]}
    else:
        refit_stages = list(state.get("refit_stages") or [])
        if not refit_stages:
            raise ValueError("[MODEL] Boosting production refit produced no usable stages.")
        predictions = normalize_demand_forecast(state["future_prediction"])
        coefficients = state["uncertainty_coefficients"]
        extra = {
            "model_chain": [item["model"] for item in refit_stages],
            "stage_coefficients": [item["coefficient"] for item in refit_stages],
            "boosting_variant": "residual",
        }

    calibration_source = str(
        uncertainty.get("calibration_source")
        or ("level1_holdout" if model_type == "stacking" else "internal_validation")
    )
    profile, summary = ensemble_growth_profile(
        state["member_profiles"],
        coefficients,
        residuals,
        reference,
        steps,
        calibration_source=calibration_source,
    )
    if model_type == "boosting":
        summary = boosting._label_boosting_uncertainty_contract(summary)
    prepared_predictions = module._format_predictions(predictions.tolist(), profile, summary)
    artifact_state = {
        "model_type": model_type,
        "base_model_states": state["member_states"],
        "split_plan": source_state.get("split_plan", {}),
        "uncertainty": {
            **uncertainty,
            "calibration_residuals": residuals,
            "calibration_source": summary.get("calibration_source", calibration_source),
            "profile": profile,
            "summary": summary,
        },
        "prepared_predictions": prepared_predictions,
        "prepared_forecast_steps": steps,
        **extra,
    }
    attach_method_metadata(
        artifact_state,
        module.METHOD_NAME,
        module.FORECAST_STRATEGY,
        module.IMPLEMENTATION_NOTES,
    )
    dump_model(artifact_state, state["save_model"])
    return {"saved_model": state["save_model"], "prepared_points": steps}


def run_step(args, state: Dict[str, Any]) -> Dict[str, Any]:
    if args.step_id == "initialize":
        return _initialize(args, state)
    if args.step_id == "fit_model":
        return _fit_base(args, state)
    if args.step_id.startswith("fit_member_"):
        return _fit_flat_member(args.step_id, state)
    if args.step_id.startswith("fit_stage_"):
        return _fit_boosting_stage(args.step_id, state)
    if args.step_id == "finalize":
        result = _finalize(state)
        return {"output": result, "final_result": result}
    raise ValueError(f"[VALIDATION] Unknown production preparation step: {args.step_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=sorted(ALL_MODELS))
    parser.add_argument("step_id")
    parser.add_argument("--input-state-file")
    parser.add_argument("--output-state-file", required=True)
    parser.add_argument("--receipt-file", required=True)
    parser.add_argument("--attempt-id", required=True)
    # Kept explicit because the shared guided-step launcher supplies the
    # session artifact directory to every checkpointed Python workflow.
    parser.add_argument("--artifact-dir")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--save-model", required=True)
    parser.add_argument("--train-start", type=int, required=True)
    parser.add_argument("--train-end", type=int, required=True)
    parser.add_argument("--forecast-steps", type=int, required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    existing = _existing_receipt(args.receipt_file, args.attempt_id, args.step_id)
    if existing is not None:
        print(json.dumps(existing, ensure_ascii=False, indent=2 if args.pretty else None))
        return
    state = _load_checkpoint(args.input_state_file)
    step_result = run_step(args, state)
    if "final_result" in step_result:
        final_result = step_result["final_result"]
        output = step_result.get("output", {})
    else:
        final_result = None
        output = step_result
    _save_checkpoint(args.output_state_file, state)
    payload = {
        "protocol_version": 1,
        "attempt_id": args.attempt_id,
        "step_id": args.step_id,
        "output": _jsonable(output),
        "input_state_path": args.input_state_file,
        "input_state_sha256": _sha256(args.input_state_file),
        "candidate_state_path": args.output_state_file,
        "candidate_state_sha256": _sha256(args.output_state_file),
    }
    if final_result is not None:
        payload["final_result"] = _jsonable(final_result)
    _write_receipt(args.receipt_file, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        known = format_known_error(exc)
        if known:
            print(known, file=sys.stderr, flush=True)
            sys.exit(1)
        raise
