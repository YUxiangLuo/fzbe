#!/usr/bin/env python3
import csv
import json
import math
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path


PY_ROOT = Path(__file__).resolve().parents[1]
RUN_MODEL = PY_ROOT / "run_model.py"
CSV_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "core_models_smoke.csv"
BASE_MODELS = ["ma", "es", "arima"]
BASE_MODEL_TRAIN_ARGS = {
    "ma": ("--window", "3"),
    "es": ("--alpha", "0.4"),
    "arima": ("--d", "1", "--max-p", "2", "--max-q", "2"),
}


def load_csv_records(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def invoke_model(model_type: str, data_json: str | None, *extra_args: str) -> dict:
    command = [sys.executable, str(RUN_MODEL), model_type]
    if data_json is not None:
        command.extend(["--data", data_json])
    command.extend(extra_args)

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{model_type} CLI execution failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def extract_predictions(payload: dict) -> list[float]:
    return [float(item["prediction"]) for item in payload["predictions"]]


def extract_std_devs(payload: dict) -> list[float]:
    return [float(item["std_dev"]) for item in payload["predictions"]]


def assert_series_close(actual: list[float], expected: list[float], tol: float = 1e-6) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"length mismatch: actual={len(actual)}, expected={len(expected)}")
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
        if not math.isclose(actual_value, expected_value, rel_tol=tol, abs_tol=tol):
            raise AssertionError(
                f"value[{index}] mismatch: actual={actual_value}, expected={expected_value}"
            )


def assert_all_finite(values: list[float], label: str) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise AssertionError(f"{label}[{index}] is not finite: {value}")


def load_pickled_state(model_path: Path) -> dict:
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def train_base_model_artifacts(
    data_json: str,
    base_model_paths: dict[str, Path],
    train_start: int,
    train_end: int,
    eval_start: int,
    eval_end: int,
) -> None:
    for model_name in BASE_MODELS:
        invoke_model(
            model_name,
            data_json,
            "--train-start",
            str(train_start),
            "--train-end",
            str(train_end),
            "--eval-start",
            str(eval_start),
            "--eval-end",
            str(eval_end),
            *BASE_MODEL_TRAIN_ARGS[model_name],
            "--save-model",
            str(base_model_paths[model_name]),
            "--pretty",
        )


def iter_stage_models(state: dict):
    model_chain = state["model_chain"]
    explicit_coefficients = state.get("stage_coefficients")
    if isinstance(explicit_coefficients, list) and len(explicit_coefficients) == len(model_chain):
        stage_coefficients = [float(value) for value in explicit_coefficients]
    else:
        learning_rate = float(state.get("learning_rate", 0.3))
        stage_coefficients = [1.0 if idx == 0 else learning_rate for idx in range(len(model_chain))]

    usage_counts: dict[str, int] = {}
    for stage_idx, model_name in enumerate(model_chain):
        model_info = state["base_model_states"][model_name]
        usage_idx = usage_counts.get(model_name, 0)
        usage_counts[model_name] = usage_idx + 1
        yield stage_idx, model_name, stage_coefficients[stage_idx], model_info["models"][usage_idx]


def combine_stage_predictions(state: dict, forecast_steps: int) -> list[float]:
    combined: list[float] | None = None
    for _stage_idx, model_name, coefficient, model_state in iter_stage_models(state):
        stage_predictions = extract_predictions(
            invoke_model(
                model_name,
                None,
                "--load-model",
                model_state["model_file"],
                "--forecast-steps",
                str(forecast_steps),
                "--pretty",
            )
        )
        weighted_predictions = [coefficient * prediction for prediction in stage_predictions]
        if combined is None:
            combined = weighted_predictions
        else:
            combined = [left + right for left, right in zip(combined, weighted_predictions)]

    if combined is None:
        raise AssertionError("boosting state produced no stages")
    return combined


def main() -> None:
    records = load_csv_records(CSV_FIXTURE)
    data_json = json.dumps(records, ensure_ascii=False)
    sales = [float(row["销售数量"]) for row in records]
    timestamps = [f"{int(row['年份']):04d}-{int(row['月份']):02d}" for row in records]
    train_start = 0
    train_end = 19
    eval_start = 20
    eval_end = 23
    future_steps = 3

    with tempfile.TemporaryDirectory(prefix="fangzhen-boosting-csv-") as temp_dir:
        temp_path = Path(temp_dir)
        base_model_paths = {
            "ma": temp_path / "ma-backtest.pkl",
            "es": temp_path / "es-backtest.pkl",
            "arima": temp_path / "arima-backtest.pkl",
        }

        train_base_model_artifacts(
            data_json,
            base_model_paths,
            train_start,
            train_end,
            eval_start,
            eval_end,
        )

        boosting_model_backtest = temp_path / "boosting-backtest.pkl"
        train_output = invoke_model(
            "boosting",
            data_json,
            "--train-start",
            str(train_start),
            "--train-end",
            str(train_end),
            "--eval-start",
            str(eval_start),
            "--eval-end",
            str(eval_end),
            "--models",
            ",".join(BASE_MODELS),
            "--window",
            "3",
            "--alpha",
            "0.4",
            "--d",
            "1",
            "--max-p",
            "2",
            "--max-q",
            "2",
            "--learning-rate",
            "0.3",
            "--max-rounds",
            "3",
            "--base-model-files",
            json.dumps({name: str(path) for name, path in base_model_paths.items()}, ensure_ascii=False),
            "--save-model",
            str(boosting_model_backtest),
            "--pretty",
        )

        train_state = load_pickled_state(boosting_model_backtest)
        expected_eval_predictions = combine_stage_predictions(train_state, forecast_steps=4)
        actual_eval_predictions = extract_predictions(train_output)
        actual_eval_std_devs = extract_std_devs(train_output)
        assert train_output["mode"] == "boosting_residual_learning"
        assert train_output["models"] == BASE_MODELS
        assert train_output["metrics"]["evaluate_months"] == timestamps[20:24]
        assert train_output["eval_y_true"] == sales[20:24]
        assert train_output["search_summary"]["algorithm"] == "residual_boosting"
        assert len(train_output["model_chain"]) >= 1
        assert len(train_output["model_chain"]) == len(train_output["stage_coefficients"])
        assert math.isclose(float(train_output["stage_coefficients"][0]), 1.0, rel_tol=1e-9, abs_tol=1e-9)
        assert_series_close(actual_eval_predictions, expected_eval_predictions)
        assert_all_finite(actual_eval_std_devs, "train_std_devs")

        first_stage_idx, first_stage_model_name, first_stage_coefficient, first_stage_model_state = next(iter_stage_models(train_state))
        assert first_stage_idx == 0
        assert math.isclose(first_stage_coefficient, 1.0, rel_tol=1e-9, abs_tol=1e-9)
        assert Path(first_stage_model_state["model_file"]).exists()
        original_base_predictions = extract_predictions(
            invoke_model(
                first_stage_model_name,
                None,
                "--load-model",
                str(base_model_paths[first_stage_model_name]),
                "--forecast-steps",
                "4",
                "--pretty",
            )
        )
        first_stage_saved_predictions = extract_predictions(
            invoke_model(
                first_stage_model_name,
                None,
                "--load-model",
                first_stage_model_state["model_file"],
                "--forecast-steps",
                "4",
                "--pretty",
            )
        )
        assert_series_close(first_stage_saved_predictions, original_base_predictions)

        boosting_model_production = temp_path / "boosting-production.pkl"
        fit_output = invoke_model(
            "boosting",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "23",
            "--load-model",
            str(boosting_model_backtest),
            "--save-model",
            str(boosting_model_production),
            "--fit-only",
            "--pretty",
        )
        assert fit_output["mode"] == "boosting_fit_only"
        assert fit_output["model_chain"] == train_output["model_chain"]
        assert fit_output["saved_model"] == str(boosting_model_production)

        production_state = load_pickled_state(boosting_model_production)
        assert production_state["model_chain"] == train_state["model_chain"]
        assert production_state["stage_coefficients"] == train_state["stage_coefficients"]
        expected_future_predictions = combine_stage_predictions(production_state, forecast_steps=future_steps)

        predict_output_one = invoke_model(
            "boosting",
            None,
            "--load-model",
            str(boosting_model_production),
            "--forecast-steps",
            str(future_steps),
            "--pretty",
        )
        predict_output_two = invoke_model(
            "boosting",
            None,
            "--load-model",
            str(boosting_model_production),
            "--forecast-steps",
            str(future_steps),
            "--pretty",
        )

        future_predictions_one = extract_predictions(predict_output_one)
        future_predictions_two = extract_predictions(predict_output_two)
        future_std_devs_one = extract_std_devs(predict_output_one)
        future_std_devs_two = extract_std_devs(predict_output_two)
        assert predict_output_one["mode"] == "boosting_loaded_predict"
        assert predict_output_one["forecast_steps"] == future_steps
        assert_series_close(future_predictions_one, expected_future_predictions)
        assert_series_close(future_predictions_one, future_predictions_two)
        assert_series_close(future_std_devs_one, future_std_devs_two)
        assert_all_finite(future_std_devs_one, "future_std_devs")

    print(
        json.dumps(
            {
                "fixture": str(CSV_FIXTURE),
                "checked_modes": ["training", "fit_only", "loaded_predict"],
                "model": "boosting",
                "self_trained_base_models": BASE_MODELS,
                "reused_base_models": BASE_MODELS,
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
