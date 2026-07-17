#!/usr/bin/env python3
"""Smoke-test the resumable production preparation protocol.

The test covers MA, ARIMA, and LSTM production refits plus all three ensemble shapes.
Weighted averaging intentionally contains LSTM so the checkpoint runner must
preserve explicit feature selection, create the Keras sidecar, extend the
direct forecast horizon, and assemble a self-contained ensemble artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PY_ROOT = Path(__file__).resolve().parents[1]
RUN_MODEL = PY_ROOT / "run_model.py"
PREPARE_MODEL = PY_ROOT / "production_prepare_model.py"
CSV_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "core_models_smoke.csv"
BASE_MODELS = ("ma", "es", "arima", "lstm")
ENSEMBLE_MODELS = ("weighted_avg", "stacking", "boosting")
ENSEMBLE_MEMBERS = {
    "weighted_avg": ("ma", "lstm"),
    "stacking": ("ma", "es"),
    "boosting": ("ma", "es"),
}
LSTM_FEATURES = ("价格指数",)
FORECAST_STEPS = 6

sys.path.insert(0, str(PY_ROOT / "src"))
from models import boosting  # noqa: E402


def _run_json(command: list[str], label: str) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise AssertionError(f"{label} returned a non-object payload")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, dict):
        raise AssertionError(f"{path} did not contain a model state")
    return state


def _load_records() -> list[dict[str, str]]:
    with CSV_FIXTURE.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _train_sources(data_json: str, temp_path: Path) -> dict[str, Path]:
    common = [
        "--data", data_json,
        "--train-start", "0",
        "--train-end", "19",
        "--eval-start", "20",
        "--eval-end", "23",
    ]
    source_paths: dict[str, Path] = {}
    model_args = {
        "ma": ["--window", "3"],
        "es": ["--alpha", "0.4"],
        "arima": ["--d", "1"],
        "lstm": [
            "--normalization", "minmax",
            "--lstm-target-feature", "销售数量",
            "--lstm-features", json.dumps(LSTM_FEATURES, ensure_ascii=False),
        ],
    }
    for model_name in BASE_MODELS:
        source = temp_path / f"source-{model_name}.pkl"
        _run_json(
            [
                sys.executable,
                str(RUN_MODEL),
                model_name,
                *common,
                *model_args[model_name],
                "--save-model", str(source),
            ],
            f"train source {model_name}",
        )
        source_paths[model_name] = source

    for model_name in ENSEMBLE_MODELS:
        members = ENSEMBLE_MEMBERS[model_name]
        base_files = json.dumps(
            {name: str(source_paths[name]) for name in members},
            ensure_ascii=False,
        )
        lstm_args = model_args["lstm"] if "lstm" in members else []
        source = temp_path / f"source-{model_name}.pkl"
        _run_json(
            [
                sys.executable,
                str(RUN_MODEL),
                model_name,
                *common,
                "--models", ",".join(members),
                "--window", "3",
                "--alpha", "0.4",
                *lstm_args,
                "--base-model-files", base_files,
                "--save-model", str(source),
            ],
            f"train source {model_name}",
        )
        source_paths[model_name] = source
    return source_paths


def _invoke_prepare_step(
    *,
    model_type: str,
    step_id: str,
    attempt_index: int,
    data_file: Path,
    source_model: Path,
    production_model: Path,
    artifact_dir: Path,
    input_state: Path | None,
) -> tuple[dict[str, Any], Path]:
    candidate_state = artifact_dir / f"state-{attempt_index}-{step_id}.pkl"
    receipt_file = artifact_dir / f"receipt-{attempt_index}-{step_id}.json"
    command = [
        sys.executable,
        str(PREPARE_MODEL),
        model_type,
        step_id,
        "--output-state-file", str(candidate_state),
        "--receipt-file", str(receipt_file),
        "--attempt-id", f"{model_type}-{attempt_index}",
        "--artifact-dir", str(artifact_dir),
        "--data-file", str(data_file),
        "--source-model", str(source_model),
        "--save-model", str(production_model),
        "--train-start", "0",
        "--train-end", "23",
        "--forecast-steps", str(FORECAST_STEPS),
    ]
    if input_state is not None:
        command.extend(["--input-state-file", str(input_state)])
    receipt = _run_json(command, f"prepare {model_type}/{step_id}")
    if receipt.get("protocol_version") != 1:
        raise AssertionError(f"{model_type}/{step_id} returned the wrong protocol")
    if receipt.get("candidate_state_path") != str(candidate_state):
        raise AssertionError(f"{model_type}/{step_id} returned the wrong checkpoint path")
    if receipt.get("candidate_state_sha256") != _sha256(candidate_state):
        raise AssertionError(f"{model_type}/{step_id} returned a stale checkpoint digest")
    if input_state is None:
        if receipt.get("input_state_path") is not None or receipt.get("input_state_sha256") is not None:
            raise AssertionError("initialize unexpectedly claimed an input checkpoint")
    else:
        if receipt.get("input_state_path") != str(input_state):
            raise AssertionError(f"{model_type}/{step_id} lost its input checkpoint")
        if receipt.get("input_state_sha256") != _sha256(input_state):
            raise AssertionError(f"{model_type}/{step_id} lost its input digest")
    return receipt, candidate_state


def _assert_current_prediction_contract(payload: dict[str, Any], model_type: str) -> None:
    points = payload.get("predictions")
    if not isinstance(points, list) or len(points) != FORECAST_STEPS:
        raise AssertionError(
            f"{model_type} did not return the prepared {FORECAST_STEPS}-step horizon"
        )
    for index, point in enumerate(points):
        for key in ("prediction", "std_dev", "interval_lower", "interval_upper", "upper_error_p99"):
            value = point.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise AssertionError(f"{model_type}.predictions[{index}].{key} is invalid")
        if float(point["std_dev"]) < 0 or float(point["upper_error_p99"]) < 0:
            raise AssertionError(f"{model_type}.predictions[{index}] has a negative uncertainty scale")
        lower = float(point["interval_lower"])
        prediction = float(point["prediction"])
        upper = float(point["interval_upper"])
        if lower > upper:
            raise AssertionError(f"{model_type}.predictions[{index}] has an inverted interval")
        if model_type in {"arima", "lstm", "stacking", "boosting"}:
            expected_interval_kind = (
                "censored_nonnegative_model_prediction_interval"
                if model_type == "arima"
                else None
            )
            interval_kind = point.get("interval_kind")
            if (
                interval_kind != expected_interval_kind
                if expected_interval_kind is not None
                else not str(interval_kind).startswith("censored_nonnegative_")
            ):
                raise AssertionError(f"{model_type}.predictions[{index}] lost its demand interval semantics")
            if not 0.0 <= lower <= prediction <= upper:
                raise AssertionError(f"{model_type}.predictions[{index}] violates its demand interval")
        if model_type in {"lstm", "stacking", "boosting"} and (
            point.get("coverage_guarantee") is not False
            or point.get("upper_error_p99_kind") != "uncalibrated_estimate"
        ):
            raise AssertionError(f"{model_type}.predictions[{index}] overstates uncertainty calibration")
        if model_type in {"stacking", "boosting"} and point.get("uncertainty_source") == "empirical":
            expected_source = (
                "stacking_meta_fit_holdout_reused"
                if model_type == "stacking"
                else "boosting_selection_holdout_reused"
            )
            if (
                point.get("calibration_source") != expected_source
                or point.get("calibration_origins") != 1
                or point.get("interval_kind")
                != "censored_nonnegative_uncalibrated_empirical_residual_quantile"
            ):
                raise AssertionError(
                    f"{model_type}.predictions[{index}] lost its reused-holdout diagnostics"
                )
        if point.get("interval_level") != 0.95:
            raise AssertionError(f"{model_type}.predictions[{index}] lost its nominal 95% level")
        if not isinstance(point.get("uncertainty_source"), str) or not point["uncertainty_source"]:
            raise AssertionError(f"{model_type}.predictions[{index}] lost its uncertainty source")
        if not isinstance(point.get("calibration_source"), str) or not point["calibration_source"]:
            raise AssertionError(f"{model_type}.predictions[{index}] lost its calibration source")
        mean_error = point.get("calibration_mean_error")
        calibration_count = point.get("calibration_count")
        unavailable = mean_error is None and calibration_count is None
        available = (
            isinstance(mean_error, (int, float))
            and not isinstance(mean_error, bool)
            and math.isfinite(float(mean_error))
            and isinstance(calibration_count, int)
            and not isinstance(calibration_count, bool)
            and calibration_count > 0
        )
        if not unavailable and not available:
            raise AssertionError(f"{model_type}.predictions[{index}] has invalid bias diagnostics")


def _assert_inherited_parameters(model_type: str, source: Path, production: Path) -> None:
    source_state = _load_state(source)
    production_state = _load_state(production)
    if model_type == "ma":
        if production_state.get("window") != source_state.get("window"):
            raise AssertionError("MA production refit did not inherit the selected window")
        return
    if model_type == "arima":
        if production_state.get("model_spec") != source_state.get("model_spec"):
            raise AssertionError(
                "ARIMA production refit did not inherit the selected fixed-d specification"
            )
        if production_state.get("order") != source_state.get("order"):
            raise AssertionError("ARIMA production refit changed the selected order")
        return
    if model_type == "lstm":
        for key in ("normalization", "target_key", "feature_keys"):
            if production_state.get(key) != source_state.get(key):
                raise AssertionError(
                    f"LSTM production refit changed {key}: "
                    f"{source_state.get(key)!r} -> {production_state.get(key)!r}"
                )
        if production_state.get("feature_keys") != list(LSTM_FEATURES):
            raise AssertionError(
                "LSTM production refit did not preserve the explicitly selected feature list"
            )
        keras_path = Path(str(production_state.get("keras_model_path") or ""))
        if not keras_path.exists() or keras_path != production.with_name(
            f"{production.stem}_keras_model.h5"
        ):
            raise AssertionError("LSTM production refit did not create its canonical Keras sidecar")
        if int(production_state.get("production_forecast_horizon") or 0) < FORECAST_STEPS:
            raise AssertionError("LSTM production refit did not extend the direct forecast horizon")
        return
    if model_type in {"weighted_avg", "stacking"}:
        for member in source_state["models"]:
            source_params = source_state["base_model_states"][member].get("params") or {}
            production_params = production_state["base_model_states"][member].get("params") or {}
            if production_params != source_params:
                raise AssertionError(
                    f"{model_type} member {member} changed selected parameters: "
                    f"{source_params!r} -> {production_params!r}"
                )
        return

    source_stages = {
        int(stage): (name, dict(state.get("params") or {}))
        for stage, name, _coefficient, state in boosting._iter_stage_models(source_state)
    }
    production_stages = list(boosting._iter_stage_models(production_state))
    if not production_stages:
        raise AssertionError("Boosting production refit saved an empty chain")
    for stage, name, _coefficient, state in production_stages:
        expected = source_stages.get(int(stage))
        actual = (name, dict(state.get("params") or {}))
        if actual != expected:
            raise AssertionError(
                f"boosting stage {stage} changed selected parameters: {expected!r} -> {actual!r}"
            )


def _remove_production_members(production_model: Path) -> None:
    prefix = f"{production_model.stem}_base_"
    for path in production_model.parent.iterdir():
        if path.name.startswith(prefix):
            if path.is_dir():
                raise AssertionError(f"unexpected member directory: {path}")
            path.unlink()


def _prepare_model(
    model_type: str,
    data_file: Path,
    source_model: Path,
    temp_path: Path,
) -> dict[str, Any]:
    artifact_dir = temp_path / f"prepare-{model_type}"
    artifact_dir.mkdir(parents=True)
    production_model = temp_path / f"production-{model_type}.pkl"
    receipt, input_state = _invoke_prepare_step(
        model_type=model_type,
        step_id="initialize",
        attempt_index=0,
        data_file=data_file,
        source_model=source_model,
        production_model=production_model,
        artifact_dir=artifact_dir,
        input_state=None,
    )
    work_items = receipt.get("output", {}).get("work_items")
    if not isinstance(work_items, list) or not work_items:
        raise AssertionError(f"{model_type} initialize returned no work items")

    executed_steps: list[str] = []
    for attempt_index, item in enumerate(work_items, start=1):
        step_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(step_id, str):
            raise AssertionError(f"{model_type} initialize returned an invalid work item")
        receipt, input_state = _invoke_prepare_step(
            model_type=model_type,
            step_id=step_id,
            attempt_index=attempt_index,
            data_file=data_file,
            source_model=source_model,
            production_model=production_model,
            artifact_dir=artifact_dir,
            input_state=input_state,
        )
        executed_steps.append(step_id)
        if receipt.get("output", {}).get("trim_remaining") is True:
            break

    final_receipt, _ = _invoke_prepare_step(
        model_type=model_type,
        step_id="finalize",
        attempt_index=len(executed_steps) + 1,
        data_file=data_file,
        source_model=source_model,
        production_model=production_model,
        artifact_dir=artifact_dir,
        input_state=input_state,
    )
    if not isinstance(final_receipt.get("final_result"), dict):
        raise AssertionError(f"{model_type} finalize returned no final result")
    if not production_model.exists():
        raise AssertionError(f"{model_type} finalize did not save a production artifact")

    _assert_inherited_parameters(model_type, source_model, production_model)
    production_state = _load_state(production_model)
    if len(production_state.get("prepared_predictions") or []) != FORECAST_STEPS:
        raise AssertionError(f"{model_type} did not persist prepared predictions")
    if production_state.get("prepared_forecast_steps") != FORECAST_STEPS:
        raise AssertionError(f"{model_type} persisted the wrong prepared horizon")

    # Ensemble prediction must consume the prepared result and must not spawn
    # or depend on its member artifacts after preparation has completed.
    if model_type in ENSEMBLE_MODELS:
        _remove_production_members(production_model)
    prediction = _run_json(
        [
            sys.executable,
            str(RUN_MODEL),
            model_type,
            "--load-model", str(production_model),
            "--forecast-steps", str(FORECAST_STEPS),
        ],
        f"predict prepared {model_type}",
    )
    _assert_current_prediction_contract(prediction, model_type)
    return {
        "work_items": executed_steps,
        "calibration_sources": sorted({
            point["calibration_source"] for point in prediction["predictions"]
        }),
    }


def main() -> None:
    records = _load_records()
    data_json = json.dumps(records, ensure_ascii=False)
    summary: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="fangzhen-production-prepare-") as temp_dir:
        temp_path = Path(temp_dir)
        data_file = temp_path / "dataset.json"
        data_file.write_text(data_json, encoding="utf-8")
        source_paths = _train_sources(data_json, temp_path)
        for model_type in ("ma", "arima", "lstm", *ENSEMBLE_MODELS):
            summary[model_type] = _prepare_model(
                model_type,
                data_file,
                source_paths[model_type],
                temp_path,
            )
    print(json.dumps({"status": "ok", "models": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
