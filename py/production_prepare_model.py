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
from model_info import attach_method_metadata
from serialization import dump_model, load_model
from subprocess_utils import (
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


def _load_validated_source_state(model_type: str, model_path: str) -> Dict[str, Any]:
    """Load the top-level artifact through the owning model's contract.

    Production preparation must not treat child-member validation as a
    substitute for validating the container that owns the selected weights,
    meta-model, Boosting chain, or baseline.  In particular, baseline-only
    Boosting has no child refit that could reject an obsolete container.

    LSTM's loader materializes its Keras sidecar for validation.  The sidecar
    object is not part of the pickle artifact contract and must not be copied
    into the resumable preparation checkpoint.
    """
    module = ALL_MODELS.get(model_type)
    if module is None:
        raise ValueError(f"[VALIDATION] Unknown production model type: {model_type}")
    source_state = dict(module._load_state(model_path))
    if model_type == "lstm":
        source_state.pop("keras_model", None)
    return source_state


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
    source_state = _load_validated_source_state(
        args.model_type,
        args.source_model,
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

    direct_finalize = False
    if args.model_type in BASE_MODULES:
        work_items = [{"id": "fit_model", "model": args.model_type}]
    elif args.model_type in {"weighted_avg", "stacking"}:
        models = list(source_state.get("models") or [])
        if len(models) < 2:
            raise ValueError("[MODEL] Ensemble artifact has no usable member list.")
        work_items = [{"id": f"fit_member_{name}", "model": name} for name in models]
    else:
        state["post_line_search_shrinkage"] = (
            boosting.get_post_line_search_shrinkage_metadata(source_state)
        )
        stages = []
        for stage_idx, model_name, coefficient, member_state in boosting._iter_stage_models(source_state):
            stages.append({
                "stage": int(stage_idx),
                "model": model_name,
                "coefficient": float(coefficient),
                "source_model": member_state["model_file"],
                "params": dict(member_state.get("params") or {}),
            })
        state["boosting_stages"] = stages
        train_history = np.asarray(
            state["parsed_data"].y[range(state["train_start"], state["train_end"] + 1)],
            dtype=float,
        )
        source_baseline = boosting._get_baseline_state(source_state)
        if source_baseline.get("kind") == "zero":
            state["baseline"] = dict(source_baseline)
            state["residual_train"] = train_history
            state["future_prediction"] = np.zeros(
                state["forecast_steps"],
                dtype=float,
            )
            state["residual_chain"] = False
        else:
            baseline_kind = str(source_baseline["kind"])
            baseline_period = source_baseline.get("seasonal_period")
            train_range = range(state["train_start"], state["train_end"] + 1)
            state["baseline"] = boosting.fit_baseline_state_for_range(
                state["parsed_data"],
                train_range,
                baseline_kind,
                baseline_period,
            )
            state["residual_train"] = (
                train_history
                - boosting.baseline_fitted_values_for_range(
                    state["parsed_data"],
                    train_range,
                    baseline_kind,
                    baseline_period,
                )
            )
            state["future_prediction"] = boosting.forecast_from_baseline_state(
                state["baseline"],
                state["forecast_steps"],
            )
            state["residual_chain"] = True
        state["refit_stages"] = []
        state["boosting_refit_terminated"] = False
        work_items = [
            {"id": f"fit_stage_{item['stage'] + 1}_{item['model']}", "model": item["model"]}
            for item in stages
        ]
        direct_finalize = not stages
    result = {"work_items": work_items}
    if direct_finalize:
        result["direct_finalize"] = True
    return result


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
        residual_target=bool(state.get("residual_chain")),
    )
    fitted = boosting._compute_in_sample_fitted_predictions(
        model_name,
        destination,
        round_data,
        train_range,
    )
    fitted = boosting._apply_stage_output_semantics(
        stage_idx,
        fitted,
        residual_target=bool(state.get("residual_chain")),
    )
    coefficient = float(boosting._nonnegative_line_search(residual_train, fitted))
    if coefficient <= np.finfo(float).eps:
        remove_member_model_artifact(model_name, destination)
        state["boosting_refit_terminated"] = True
        return {
            "stage": stage_idx + 1,
            "model": model_name,
            "coefficient": coefficient,
            "action": "trim_refit_chain",
            "trim_remaining": True,
        }
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
    reference = state["parsed_data"].y[
        range(state["train_start"], state["train_end"] + 1)
    ].tolist()
    residuals = uncertainty.get("calibration_residuals") or []
    calibration_source = str(
        uncertainty.get("calibration_source")
        or "unavailable_post_selection_calibration"
    )
    calibration_invalidation_reason = uncertainty.get(
        "calibration_invalidation_reason"
    )
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
        predictions = normalize_demand_forecast(state["future_prediction"])
        coefficients = dict(state.get("uncertainty_coefficients") or {})
        refit_chain = [item["model"] for item in refit_stages]
        source_chain = list(source_state.get("model_chain") or [])
        if refit_chain != source_chain:
            residuals = []
            calibration_invalidation_reason = (
                "unavailable_after_production_refit_chain_change"
            )
            calibration_source = "training_history"
        boosting._add_baseline_uncertainty_member(
            state["member_profiles"],
            coefficients,
            state["baseline"],
            steps,
            has_member_stages=bool(refit_chain),
        )
        extra = {
            "model_chain": refit_chain,
            "stage_coefficients": [item["coefficient"] for item in refit_stages],
            "boosting_variant": "residual",
            "post_line_search_shrinkage": (
                boosting.get_post_line_search_shrinkage_metadata(source_state)
            ),
            "baseline": state["baseline"],
        }
    profile, summary = ensemble_growth_profile(
        state["member_profiles"],
        coefficients,
        residuals,
        reference,
        steps,
        calibration_source=calibration_source,
    )
    prepared_predictions = module._format_predictions(predictions.tolist(), profile, summary)
    artifact_uncertainty = {
        **uncertainty,
        "calibration_residuals": residuals,
        "calibration_source": summary.get(
            "calibration_source",
            calibration_source,
        ),
        "profile": profile,
        "summary": summary,
    }
    if calibration_invalidation_reason:
        artifact_uncertainty["calibration_invalidation_reason"] = str(
            calibration_invalidation_reason
        )
    artifact_state = {
        "model_type": model_type,
        "base_model_states": state["member_states"],
        "split_plan": source_state.get("split_plan", {}),
        "uncertainty": artifact_uncertainty,
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
