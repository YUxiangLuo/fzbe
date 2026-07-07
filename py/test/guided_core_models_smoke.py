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
EVAL_LENGTH = 4  # eval indices 20..23


def assert_all_finite(values, label):
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise AssertionError(f"{label}[{index}] is not finite: {value}")


def run_subprocess(command):
    return subprocess.run(command, capture_output=True, text=True, cwd=str(PY_ROOT))


def drive_guided_model(model_type, data_json, temp_path):
    state_file = temp_path / f"{model_type}-state.pkl"
    artifact_dir = temp_path / f"{model_type}-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_model = temp_path / f"guided-{model_type}.pkl"

    common_args = [
        "--state-file", str(state_file),
        "--artifact-dir", str(artifact_dir),
        "--data", data_json,
        "--train-start", "0", "--train-end", "19",
        "--eval-start", "20", "--eval-end", "23",
        "--save-model", str(save_model),
        *MODEL_PARAMS[model_type],
    ]

    final_result = None
    for step in STEP_SEQUENCES[model_type]:
        completed = run_subprocess([sys.executable, str(GUIDED_MODEL), model_type, step, *common_args])
        if completed.returncode != 0:
            raise RuntimeError(
                f"guided {model_type} step '{step}' failed\nstderr:\n{completed.stderr[-1500:]}"
            )
        payload = json.loads(completed.stdout)
        if "final_result" in payload:
            final_result = payload["final_result"]

    if final_result is None:
        raise AssertionError(f"guided {model_type} produced no final_result")
    metrics = final_result["metrics"]
    for key in ("rmse", "mae", "mape", "r2"):
        if not math.isfinite(float(metrics[key])):
            raise AssertionError(f"{model_type} metric '{key}' is not finite: {metrics[key]}")
    eval_predictions = [float(item["prediction"]) for item in final_result["predictions"]]
    if len(eval_predictions) != EVAL_LENGTH:
        raise AssertionError(f"{model_type}: expected {EVAL_LENGTH} eval predictions, got {len(eval_predictions)}")
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


def main():
    with CSV_FIXTURE.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    data_json = json.dumps(records, ensure_ascii=False)

    with tempfile.TemporaryDirectory(prefix="fangzhen-guided-core-") as temp_dir:
        temp_path = Path(temp_dir)
        for model_type in ("ma", "es", "arima"):
            drive_guided_model(model_type, data_json, temp_path)

    print(json.dumps({
        "fixture": str(CSV_FIXTURE),
        "checked_models": ["ma", "es", "arima"],
        "checks": ["guided_steps", "final_metrics", "artifact_valid", "roundtrip_predict"],
        "status": "ok",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
