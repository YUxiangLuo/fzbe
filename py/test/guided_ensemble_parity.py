#!/usr/bin/env python3
"""End-to-end parity checks for regular and guided ensemble training."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any


PY_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_ROOT))

from guided_standalone_parity import (  # noqa: E402
    LSTM_FEATURES,
    LSTM_FIXTURE,
    load_fixture,
    maximum_delta,
    run_json,
)


RUN_MODEL = PY_ROOT / "run_model.py"
GUIDED_ENSEMBLE = PY_ROOT / "guided_ensemble_model.py"
MODELS = ("ma", "es", "arima", "lstm")
TRAIN_RANGE = (0, 27)
EVAL_RANGE = (30, 32)
TOLERANCE = 1e-5

STEP_SEQUENCES = {
    "weighted_avg": [
        "prepare_data",
        "split_validation",
        "validation_predictions",
        "compute_weights",
        "eval_predictions",
        "blend_evaluate",
        "save_artifact",
    ],
    "stacking": [
        "prepare_data",
        "split_levels",
        "level1_predictions",
        "fit_meta_model",
        "eval_predictions",
        "meta_evaluate",
        "save_artifact",
    ],
    "boosting": [
        "prepare_data",
        "split_residual_sets",
        "round_1_evaluate_candidates",
        "round_1_select_winner",
        "retrain_chain",
        "combine_evaluate",
        "save_artifact",
    ],
}


def train_base_models(data: str, temp_path: Path) -> dict[str, str]:
    base_files = {model: str(temp_path / f"base-{model}.pkl") for model in MODELS}
    common = [
        "--data", data,
        "--train-start", str(TRAIN_RANGE[0]),
        "--train-end", str(TRAIN_RANGE[1]),
        "--eval-start", str(EVAL_RANGE[0]),
        "--eval-end", str(EVAL_RANGE[1]),
    ]
    model_args = {
        "ma": ["--window", "4"],
        "es": ["--alpha", "0.35"],
        "arima": ["--d", "1", "--max-p", "1", "--max-q", "1"],
        "lstm": [
            "--look-back", "4",
            "--epochs", "1",
            "--normalization", "minmax",
            "--lstm-target-feature", "销售数量",
            "--lstm-features", json.dumps(LSTM_FEATURES, ensure_ascii=False),
        ],
    }
    for model_name in MODELS:
        run_json(
            [
                sys.executable,
                str(RUN_MODEL),
                model_name,
                *common,
                *model_args[model_name],
                "--save-model",
                base_files[model_name],
            ],
            f"base {model_name}",
        )
    return base_files


def ensemble_args(data: str, save_model: Path, base_files: dict[str, str]) -> list[str]:
    return [
        "--data", data,
        "--train-start", str(TRAIN_RANGE[0]),
        "--train-end", str(TRAIN_RANGE[1]),
        "--eval-start", str(EVAL_RANGE[0]),
        "--eval-end", str(EVAL_RANGE[1]),
        "--models", ",".join(MODELS),
        "--base-model-files", json.dumps(base_files),
        "--window", "4",
        "--alpha", "0.35",
        "--d", "1",
        "--max-p", "1",
        "--max-q", "1",
        "--look-back", "4",
        "--epochs", "1",
        "--normalization", "minmax",
        "--lstm-target-feature", "销售数量",
        "--lstm-features", json.dumps(LSTM_FEATURES, ensure_ascii=False),
        "--learning-rate", "0.3",
        "--max-rounds", "1",
        "--save-model", str(save_model),
    ]


def train_guided(
    model_type: str,
    args: list[str],
    temp_path: Path,
) -> dict[str, Any]:
    artifact_dir = temp_path / f"guided-{model_type}-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_state: Path | None = None
    final_result: dict[str, Any] | None = None

    for index, step_id in enumerate(STEP_SEQUENCES[model_type]):
        candidate_state = temp_path / f"guided-{model_type}-{index}-{step_id}.pkl"
        receipt_file = temp_path / f"guided-{model_type}-{index}-{step_id}.receipt.json"
        protocol_args = [
            "--output-state-file", str(candidate_state),
            "--receipt-file", str(receipt_file),
            "--attempt-id", f"{model_type}-{index}",
            "--artifact-dir", str(artifact_dir),
        ]
        if input_state is not None:
            protocol_args.extend(["--input-state-file", str(input_state)])
        payload = run_json(
            [
                sys.executable,
                str(GUIDED_ENSEMBLE),
                model_type,
                step_id,
                *protocol_args,
                *args,
            ],
            f"guided {model_type}/{step_id}",
        )
        input_state = candidate_state
        candidate_result = payload.get("final_result")
        if isinstance(candidate_result, dict):
            final_result = candidate_result

    if final_result is None:
        raise AssertionError(f"guided {model_type} produced no final result")
    return final_result


def compare_results(
    model_type: str,
    regular: dict[str, Any],
    guided: dict[str, Any],
) -> dict[str, float]:
    for key in ("mode", "models", "method_name", "forecast_strategy", "implementation_notes", "eval_y_true"):
        if regular.get(key) != guided.get(key):
            raise AssertionError(f"{model_type}.{key} differs")

    deltas = {
        "metrics": maximum_delta(
            [regular["metrics"][key] for key in ("rmse", "mae", "mape", "r2")],
            [guided["metrics"][key] for key in ("rmse", "mae", "mape", "r2")],
            f"{model_type}.metrics",
        ),
        "predictions": maximum_delta(
            [point["prediction"] for point in regular["predictions"]],
            [point["prediction"] for point in guided["predictions"]],
            f"{model_type}.predictions",
        ),
        "std_dev": maximum_delta(
            [point["std_dev"] for point in regular["predictions"]],
            [point["std_dev"] for point in guided["predictions"]],
            f"{model_type}.std_dev",
        ),
    }

    if model_type == "weighted_avg":
        deltas["weights"] = maximum_delta(
            [regular["weights"][model] for model in MODELS],
            [guided["weights"][model] for model in MODELS],
            "weighted_avg.weights",
        )
    elif model_type == "stacking":
        if regular["meta_model"]["strategy"] != guided["meta_model"]["strategy"]:
            raise AssertionError("stacking.meta_model.strategy differs")
        deltas["weights"] = maximum_delta(
            regular["meta_model"]["weights"],
            guided["meta_model"]["weights"],
            "stacking.meta_model.weights",
        )
    else:
        for key in ("model_chain", "stage_coefficients"):
            if regular[key] != guided[key]:
                raise AssertionError(f"boosting.{key} differs")

    for label, delta in deltas.items():
        if delta > TOLERANCE:
            raise AssertionError(f"{model_type}.{label} delta {delta} exceeds {TOLERANCE}")
    return deltas


def compare_loaded_predictions(model_type: str, regular_model: Path, guided_model: Path) -> float:
    predictions = []
    for label, model_path in (("regular", regular_model), ("guided", guided_model)):
        result = run_json(
            [
                sys.executable,
                str(RUN_MODEL),
                model_type,
                "--load-model", str(model_path),
                "--forecast-steps", "3",
            ],
            f"{label} {model_type} loaded prediction",
        )
        predictions.append([point["prediction"] for point in result["predictions"]])
    return maximum_delta(predictions[0], predictions[1], f"loaded {model_type}.predictions")


def main() -> None:
    data = load_fixture(LSTM_FIXTURE)
    summary: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="fangzhen-guided-ensemble-parity-") as temp_dir:
        temp_path = Path(temp_dir)
        base_files = train_base_models(data, temp_path)

        for model_type in STEP_SEQUENCES:
            regular_model = temp_path / f"regular-{model_type}.pkl"
            guided_model = temp_path / f"guided-{model_type}.pkl"
            regular_args = ensemble_args(data, regular_model, base_files)
            guided_args = ensemble_args(data, guided_model, base_files)
            regular = run_json(
                [sys.executable, str(RUN_MODEL), model_type, *regular_args],
                f"regular {model_type}",
            )
            guided = train_guided(model_type, guided_args, temp_path)
            deltas = compare_results(model_type, regular, guided)
            loaded_delta = compare_loaded_predictions(model_type, regular_model, guided_model)
            if loaded_delta > TOLERANCE:
                raise AssertionError(
                    f"loaded {model_type}.predictions delta {loaded_delta} exceeds {TOLERANCE}"
                )
            summary[model_type] = {
                "training": deltas,
                "loaded_predictions": loaded_delta,
            }

    print(json.dumps({"ensembles": summary, "status": "ok"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
