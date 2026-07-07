#!/usr/bin/env python3
"""Smoke test for the guided-training path of the fusion models.

The weighted_avg / stacking / boosting experiments train step-by-step via
guided_ensemble_model.py (which spawns the base models through run_model.py).
This drives each fusion model's real guided step sequence end-to-end and verifies
the produced artifact is run_model.py-compatible (round-trip predict).

weighted_avg / stacking use fixed step sequences; boosting has dynamic per-round
steps (round_N_evaluate_candidates / round_N_select_winner) that repeat until a
round early-stops or maxRounds is reached — the same sequencing the backend builds.
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
GUIDED_ENSEMBLE = PY_ROOT / "guided_ensemble_model.py"
RUN_MODEL = PY_ROOT / "run_model.py"
CSV_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "core_models_smoke.csv"

# Fixed guided step sequences (mirror GUIDED_TRAINING_STEPS in model-types.ts).
STATIC_SEQUENCES = {
    "weighted_avg": ["prepare_data", "split_validation", "validation_predictions",
                     "compute_weights", "eval_predictions", "blend_evaluate", "save_artifact"],
    "stacking": ["prepare_data", "split_levels", "level1_predictions",
                 "fit_meta_model", "eval_predictions", "meta_evaluate", "save_artifact"],
}
BASE_MODELS = {
    "weighted_avg": "ma,es",
    "stacking": "ma,es",
    # Boosting needs a non-flat learner to reduce residuals; ARIMA provides it
    # (mirrors the known-good non-guided boosting_csv_smoke base set).
    "boosting": "ma,es,arima",
}
BOOSTING_MAX_ROUNDS = 3  # = len(boosting base models); backend uses max(1, #models)
EVAL_LENGTH = 4  # eval indices 20..23


def assert_all_finite(values, label):
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise AssertionError(f"{label}[{index}] is not finite: {value}")


def run_subprocess(command):
    return subprocess.run(command, capture_output=True, text=True, cwd=str(PY_ROOT))


def build_common_args(model_type, data_json, temp_path):
    save_model = temp_path / f"guided-{model_type}.pkl"
    common_args = [
        "--state-file", str(temp_path / f"{model_type}-state.pkl"),
        "--artifact-dir", str(temp_path / f"{model_type}-artifacts"),
        "--data", data_json,
        "--train-start", "0", "--train-end", "19",
        "--eval-start", "20", "--eval-end", "23",
        "--models", BASE_MODELS[model_type],
        "--window", "3",
        "--alpha", "0.4",
        "--d", "1", "--max-p", "2", "--max-q", "2",
        "--learning-rate", "0.3",
        "--max-rounds", str(BOOSTING_MAX_ROUNDS),
        "--save-model", str(save_model),
    ]
    (temp_path / f"{model_type}-artifacts").mkdir(parents=True, exist_ok=True)
    return common_args, save_model


def run_step(model_type, step_id, common_args):
    completed = run_subprocess([sys.executable, str(GUIDED_ENSEMBLE), model_type, step_id, *common_args])
    if completed.returncode != 0:
        raise RuntimeError(
            f"guided {model_type} step '{step_id}' failed\nstderr:\n{completed.stderr[-1800:]}"
        )
    return json.loads(completed.stdout)


def drive_boosting_steps(common_args):
    """Drive boosting's dynamic round steps, returning the collected final_result."""
    final_result = None

    def step(step_id):
        nonlocal final_result
        payload = run_step("boosting", step_id, common_args)
        if "final_result" in payload:
            final_result = payload["final_result"]
        return payload

    step("prepare_data")
    step("split_residual_sets")
    for round_number in range(1, BOOSTING_MAX_ROUNDS + 1):
        step(f"round_{round_number}_evaluate_candidates")
        selection = step(f"round_{round_number}_select_winner")
        if selection.get("output", {}).get("action") == "early_stop":
            break
    step("retrain_chain")
    step("combine_evaluate")
    step("save_artifact")
    return final_result


def drive_static_steps(model_type, common_args):
    final_result = None
    for step_id in STATIC_SEQUENCES[model_type]:
        payload = run_step(model_type, step_id, common_args)
        if "final_result" in payload:
            final_result = payload["final_result"]
    return final_result


def drive_guided_ensemble(model_type, data_json, temp_path):
    common_args, save_model = build_common_args(model_type, data_json, temp_path)

    if model_type == "boosting":
        final_result = drive_boosting_steps(common_args)
    else:
        final_result = drive_static_steps(model_type, common_args)

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

    # Round-trip: the guided ensemble artifact (with its base members) must be
    # loadable & predictable by run_model.py — the same path production uses.
    predict_completed = run_subprocess(
        [sys.executable, str(RUN_MODEL), model_type, "--load-model", str(save_model), "--forecast-steps", "3"]
    )
    if predict_completed.returncode != 0:
        raise RuntimeError(
            f"{model_type} round-trip predict failed\nstderr:\n{predict_completed.stderr[-1800:]}"
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

    with tempfile.TemporaryDirectory(prefix="fangzhen-guided-ensemble-") as temp_dir:
        temp_path = Path(temp_dir)
        for model_type in ("weighted_avg", "stacking", "boosting"):
            drive_guided_ensemble(model_type, data_json, temp_path)

    print(json.dumps({
        "fixture": str(CSV_FIXTURE),
        "base_models": BASE_MODELS,
        "checked_models": ["weighted_avg", "stacking", "boosting"],
        "checks": ["guided_steps", "final_metrics", "artifact_valid", "roundtrip_predict"],
        "status": "ok",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
