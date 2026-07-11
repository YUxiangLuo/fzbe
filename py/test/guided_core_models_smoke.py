#!/usr/bin/env python3
"""Smoke test for the guided-training path of the core single models (MA/ES/ARIMA).

The standalone MA/ES/ARIMA experiments train step-by-step via guided_model.py.
This drives each model's real guided step sequence end-to-end and verifies the
produced artifact is run_model.py-compatible (round-trip predict). Guards the
guided orchestration layer, which otherwise has no dedicated Python coverage.
"""
import csv
import json
import math
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

PY_ROOT = Path(__file__).resolve().parents[1]
GUIDED_MODEL = PY_ROOT / "guided_model.py"
RUN_MODEL = PY_ROOT / "run_model.py"
CSV_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "core_models_smoke.csv"

# Canonical guided step sequences (mirror GUIDED_TRAINING_STEPS in model-types.ts).
STEP_SEQUENCES = {
    "ma": ["prepare_data", "build_windows", "forecast", "estimate_residuals", "evaluate_metrics", "save_artifact"],
    "es": ["prepare_data", "fit_model", "forecast", "estimate_residuals", "evaluate_metrics", "save_artifact"],
    "arima": ["prepare_data", "aic_search", "bic_search", "select_model", "forecast", "evaluate_metrics", "save_artifact"],
}
MODEL_PARAMS = {
    "ma": ["--window", "3"],
    "es": ["--alpha", "0.4"],
    "arima": ["--d", "1", "--max-p", "2", "--max-q", "2"],
}
EVAL_START = 20
EVAL_END = 23
GAP = 2


def assert_all_finite(values, label):
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise AssertionError(f"{label}[{index}] is not finite: {value}")


def run_subprocess(command):
    return subprocess.run(command, capture_output=True, text=True, cwd=str(PY_ROOT))


def drive_guided_model(
    model_type,
    data_json,
    temp_path,
    *,
    eval_start=EVAL_START,
    eval_end=EVAL_END,
    run_label="contiguous",
):
    state_file = temp_path / f"{model_type}-{run_label}-state.pkl"
    artifact_dir = temp_path / f"{model_type}-{run_label}-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_model = temp_path / f"guided-{model_type}-{run_label}.pkl"
    expected_eval_length = eval_end - eval_start + 1

    common_args = [
        "--state-file", str(state_file),
        "--artifact-dir", str(artifact_dir),
        "--data", data_json,
        "--train-start", "0", "--train-end", "19",
        "--eval-start", str(eval_start), "--eval-end", str(eval_end),
        "--save-model", str(save_model),
        *MODEL_PARAMS[model_type],
    ]

    final_result = None
    step_outputs = {}
    for step in STEP_SEQUENCES[model_type]:
        completed = run_subprocess([sys.executable, str(GUIDED_MODEL), model_type, step, *common_args])
        if completed.returncode != 0:
            raise RuntimeError(
                f"guided {model_type} step '{step}' failed\nstderr:\n{completed.stderr[-1500:]}"
            )
        payload = json.loads(completed.stdout)
        step_outputs[step] = payload.get("output", {})
        if "final_result" in payload:
            final_result = payload["final_result"]

    if final_result is None:
        raise AssertionError(f"guided {model_type} produced no final_result")
    metrics = final_result["metrics"]
    for key in ("rmse", "mae", "mape", "r2"):
        if not math.isfinite(float(metrics[key])):
            raise AssertionError(f"{model_type} metric '{key}' is not finite: {metrics[key]}")
    eval_predictions = [float(item["prediction"]) for item in final_result["predictions"]]
    if len(eval_predictions) != expected_eval_length:
        raise AssertionError(
            f"{model_type}: expected {expected_eval_length} eval predictions, got {len(eval_predictions)}"
        )
    assert_all_finite(eval_predictions, f"{model_type} eval_predictions")

    with save_model.open("rb") as handle:
        saved_state = pickle.load(handle)
    if saved_state.get("model_type") != model_type:
        raise AssertionError(f"{model_type}: saved model_type mismatch: {saved_state.get('model_type')}")

    # Round-trip: guided artifact must be loadable & predictable by run_model.py.
    predict_completed = run_subprocess(
        [sys.executable, str(RUN_MODEL), model_type, "--load-model", str(save_model), "--forecast-steps", "3"]
    )
    if predict_completed.returncode != 0:
        raise RuntimeError(
            f"{model_type} round-trip predict failed\nstderr:\n{predict_completed.stderr[-1500:]}"
        )
    future_predictions = [
        float(item["prediction"]) for item in json.loads(predict_completed.stdout)["predictions"]
    ]
    if len(future_predictions) != 3:
        raise AssertionError(f"{model_type}: expected 3 future predictions, got {len(future_predictions)}")
    assert_all_finite(future_predictions, f"{model_type} future_predictions")
    return final_result, step_outputs


def assert_guided_single_point_rejected(model_type, data_json, temp_path):
    completed = run_subprocess([
        sys.executable,
        str(GUIDED_MODEL),
        model_type,
        "prepare_data",
        "--state-file", str(temp_path / f"{model_type}-single-point-state.pkl"),
        "--artifact-dir", str(temp_path / f"{model_type}-single-point-artifacts"),
        "--data", data_json,
        "--train-start", "0",
        "--train-end", "19",
        "--eval-start", "20",
        "--eval-end", "20",
        "--save-model", str(temp_path / f"guided-{model_type}-single-point.pkl"),
        *MODEL_PARAMS[model_type],
    ])
    if completed.returncode == 0 or "[VALIDATION]" not in completed.stderr:
        raise AssertionError(
            f"guided {model_type} accepted a single-point evaluation range\n"
            f"stdout:\n{completed.stdout[-1000:]}\nstderr:\n{completed.stderr[-1000:]}"
        )


def assert_guided_gap_aligned(model_type, contiguous_result, gapped_result):
    contiguous_tail = contiguous_result["predictions"][GAP:]
    gapped_predictions = gapped_result["predictions"]
    if len(contiguous_tail) != len(gapped_predictions):
        raise AssertionError(f"{model_type}: gapped prediction length mismatch")

    for index, (expected, actual) in enumerate(zip(contiguous_tail, gapped_predictions)):
        if not math.isclose(
            float(expected["prediction"]),
            float(actual["prediction"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise AssertionError(
                f"{model_type}: gapped prediction[{index}] is shifted: "
                f"{actual['prediction']} != {expected['prediction']}"
            )
        if model_type == "arima" and not math.isclose(
            float(expected["std_dev"]),
            float(actual["std_dev"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise AssertionError(
                f"ARIMA: gapped std_dev[{index}] is shifted: "
                f"{actual['std_dev']} != {expected['std_dev']}"
            )


def main():
    with CSV_FIXTURE.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    data_json = json.dumps(records, ensure_ascii=False)

    with tempfile.TemporaryDirectory(prefix="fangzhen-guided-core-") as temp_dir:
        temp_path = Path(temp_dir)
        contiguous_results = {}
        for model_type in ("ma", "es", "arima"):
            assert_guided_single_point_rejected(model_type, data_json, temp_path)
            contiguous_results[model_type] = drive_guided_model(model_type, data_json, temp_path)

        es_fit_output = contiguous_results["es"][1]["fit_model"]
        for key in ("initial_level", "final_level"):
            if not math.isfinite(float(es_fit_output[key])):
                raise AssertionError(f"ES fit output {key} is not finite: {es_fit_output[key]}")
        es_first_prediction = float(contiguous_results["es"][0]["predictions"][0]["prediction"])
        if not math.isclose(float(es_fit_output["final_level"]), es_first_prediction, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(
                "ES final_level must be the flat forecast level: "
                f"{es_fit_output['final_level']} != {es_first_prediction}"
            )

        for model_type in ("es", "arima"):
            gapped_result, gapped_step_outputs = drive_guided_model(
                model_type,
                data_json,
                temp_path,
                eval_start=EVAL_START + GAP,
                eval_end=EVAL_END,
                run_label="gapped",
            )
            contiguous_result, _ = contiguous_results[model_type]
            assert_guided_gap_aligned(model_type, contiguous_result, gapped_result)
            forecast_output = gapped_step_outputs["forecast"]
            if forecast_output.get("forecast_offset") != GAP:
                raise AssertionError(
                    f"{model_type}: expected forecast_offset={GAP}, got {forecast_output.get('forecast_offset')}"
                )
            if forecast_output.get("forecast_steps") != EVAL_END - 19:
                raise AssertionError(
                    f"{model_type}: forecast_steps did not reach the evaluation end: {forecast_output}"
                )

    print(json.dumps({
        "fixture": str(CSV_FIXTURE),
        "checked_models": ["ma", "es", "arima"],
        "checks": [
            "guided_steps",
            "final_metrics",
            "artifact_valid",
            "roundtrip_predict",
            "gapped_evaluation_alignment",
            "gapped_arima_uncertainty_alignment",
            "es_initial_and_final_levels",
            "single_point_evaluation_rejected",
        ],
        "status": "ok",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
