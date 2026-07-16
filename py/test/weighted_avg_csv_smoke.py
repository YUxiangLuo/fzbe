#!/usr/bin/env python3
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


PY_ROOT = Path(__file__).resolve().parents[1]
RUN_MODEL = PY_ROOT / "run_model.py"
CSV_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "core_models_smoke.csv"
BASE_MODELS = ["ma", "es", "arima"]

from model_metadata_assertions import assert_method_metadata  # noqa: E402


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


def member_model_path(save_path: Path, model_name: str) -> str:
    return str(save_path.with_name(f"{save_path.stem}_base_{model_name}.pkl"))


def normalized_inverse_residual_mse_weights(
    predictions_by_model: dict[str, list[float]],
    y_true: list[float],
) -> dict[str, float]:
    scale = max(1.0, sum(value * value for value in y_true) / len(y_true))
    epsilon = max(1e-12, sys.float_info.epsilon * scale)
    raw_weights: dict[str, float] = {}
    total = 0.0
    for model_name, predictions in predictions_by_model.items():
        residuals = [target - prediction for target, prediction in zip(y_true, predictions)]
        residual_mse = sum(value * value for value in residuals) / len(residuals)
        inverse_error = 1.0 / (residual_mse + epsilon)
        raw_weights[model_name] = inverse_error
        total += inverse_error

    normalized = {model_name: weight / total for model_name, weight in raw_weights.items()}
    model_count = len(normalized)
    reliability = len(y_true) / (len(y_true) + model_count + 1.0)
    uniform = 1.0 / model_count
    return {
        model_name: reliability * weight + (1.0 - reliability) * uniform
        for model_name, weight in normalized.items()
    }


def weighted_sum(predictions_by_model: dict[str, list[float]], weights: dict[str, float]) -> list[float]:
    length = len(next(iter(predictions_by_model.values())))
    combined: list[float] = []
    for index in range(length):
        combined.append(sum(predictions_by_model[model_name][index] * weights[model_name] for model_name in weights))
    return combined


def main() -> None:
    synthetic_weights = normalized_inverse_residual_mse_weights(
        {
            "biased": [100.0, 100.0, 100.0, 100.0],
            "stable": [0.0, 1.0, 0.0, -1.0],
        },
        [0.0, 0.0, 0.0, 0.0],
    )
    if synthetic_weights["biased"] >= synthetic_weights["stable"]:
        raise AssertionError(
            f"expected constant large bias to receive lower weight, got {synthetic_weights}"
        )

    records = load_csv_records(CSV_FIXTURE)
    data_json = json.dumps(records, ensure_ascii=False)
    sales = [float(row["销售数量"]) for row in records]
    timestamps = [f"{int(row['年份']):04d}-{int(row['月份']):02d}" for row in records]
    train_start = 0
    train_end = 19
    eval_start = 20
    eval_end = 23
    future_steps = 3

    with tempfile.TemporaryDirectory(prefix="fangzhen-weighted-avg-csv-") as temp_dir:
        temp_path = Path(temp_dir)
        base_model_paths = {
            "ma": temp_path / "ma-backtest.pkl",
            "es": temp_path / "es-backtest.pkl",
            "arima": temp_path / "arima-backtest.pkl",
        }

        invoke_model(
            "ma",
            data_json,
            "--train-start",
            str(train_start),
            "--train-end",
            str(train_end),
            "--eval-start",
            str(eval_start),
            "--eval-end",
            str(eval_end),
            "--window",
            "3",
            "--save-model",
            str(base_model_paths["ma"]),
            "--pretty",
        )
        invoke_model(
            "es",
            data_json,
            "--train-start",
            str(train_start),
            "--train-end",
            str(train_end),
            "--eval-start",
            str(eval_start),
            "--eval-end",
            str(eval_end),
            "--alpha",
            "0.4",
            "--save-model",
            str(base_model_paths["es"]),
            "--pretty",
        )
        invoke_model(
            "arima",
            data_json,
            "--train-start",
            str(train_start),
            "--train-end",
            str(train_end),
            "--eval-start",
            str(eval_start),
            "--eval-end",
            str(eval_end),
            "--d",
            "1",
            "--max-p",
            "2",
            "--max-q",
            "2",
            "--save-model",
            str(base_model_paths["arima"]),
            "--pretty",
        )

        weight_val_predictions: dict[str, list[float]] = {
            model_name: [] for model_name in BASE_MODELS
        }
        eval_predictions: dict[str, list[float]] = {}
        # Independent oracle for the implementation's three expanding-window
        # rolling-origin folds: 0..7→8..11, 0..11→12..15, 0..15→16..19.
        for fold_index, (prefix_end, validation_start, validation_end) in enumerate((
            (7, 8, 11),
            (11, 12, 15),
            (15, 16, 19),
        )):
            for model_name in BASE_MODELS:
                if model_name == "arima":
                    fold_output = invoke_model(
                        "arima",
                        data_json,
                        "--train-start", str(train_start),
                        "--train-end", str(prefix_end),
                        "--eval-start", str(validation_start),
                        "--eval-end", str(validation_end),
                        "--d", "1",
                        "--pretty",
                    )
                else:
                    refit_model_path = temp_path / f"{model_name}-weight-fold-{fold_index}.pkl"
                    invoke_model(
                        model_name,
                        data_json,
                        "--train-start", str(train_start),
                        "--train-end", str(prefix_end),
                        "--load-model", str(base_model_paths[model_name]),
                        "--save-model", str(refit_model_path),
                        "--fit-only",
                        "--pretty",
                    )
                    fold_output = invoke_model(
                        model_name,
                        None,
                        "--load-model", str(refit_model_path),
                        "--forecast-steps", "4",
                        "--pretty",
                    )
                weight_val_predictions[model_name].extend(
                    extract_predictions(fold_output)
                )

        for model_name in BASE_MODELS:
            eval_predictions[model_name] = extract_predictions(
                invoke_model(
                    model_name,
                    None,
                    "--load-model", str(base_model_paths[model_name]),
                    "--forecast-steps",
                    "4",
                    "--pretty",
                )
            )

        expected_weights = normalized_inverse_residual_mse_weights(
            weight_val_predictions,
            sales[8:20],
        )
        expected_eval_predictions = weighted_sum(eval_predictions, expected_weights)

        weighted_model_backtest = temp_path / "weighted-avg-backtest.pkl"
        train_output = invoke_model(
            "weighted_avg",
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
            "--base-model-files",
            json.dumps({name: str(path) for name, path in base_model_paths.items()}, ensure_ascii=False),
            "--save-model",
            str(weighted_model_backtest),
            "--pretty",
        )

        actual_weights = {name: float(train_output["weights"][name]) for name in BASE_MODELS}
        actual_eval_predictions = extract_predictions(train_output)
        actual_eval_std_devs = extract_std_devs(train_output)
        assert train_output["mode"] == "weighted_avg_train_eval"
        assert_method_metadata(train_output, "weighted_avg")
        assert train_output["models"] == BASE_MODELS
        assert train_output["metrics"]["evaluate_months"] == timestamps[20:24]
        assert train_output["eval_y_true"] == sales[20:24]
        assert_series_close([sum(actual_weights.values())], [1.0])
        assert_series_close([actual_weights[name] for name in BASE_MODELS], [expected_weights[name] for name in BASE_MODELS])
        assert_series_close(actual_eval_predictions, expected_eval_predictions)
        assert_all_finite(actual_eval_std_devs, "train_std_devs")

        weighted_model_production = temp_path / "weighted-avg-production.pkl"
        fit_output = invoke_model(
            "weighted_avg",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "23",
            "--load-model",
            str(weighted_model_backtest),
            "--save-model",
            str(weighted_model_production),
            "--fit-only",
            "--pretty",
        )
        assert fit_output["mode"] == "weighted_avg_fit_only"
        assert_method_metadata(fit_output, "weighted_avg")
        assert fit_output["models"] == BASE_MODELS
        assert fit_output["saved_model"] == str(weighted_model_production)

        production_member_predictions = {
            model_name: extract_predictions(
                invoke_model(
                    model_name,
                    None,
                    "--load-model",
                    member_model_path(weighted_model_production, model_name),
                    "--forecast-steps",
                    str(future_steps),
                    "--pretty",
                )
            )
            for model_name in BASE_MODELS
        }
        expected_future_predictions = weighted_sum(production_member_predictions, actual_weights)

        predict_output_one = invoke_model(
            "weighted_avg",
            None,
            "--load-model",
            str(weighted_model_production),
            "--forecast-steps",
            str(future_steps),
            "--pretty",
        )
        predict_output_two = invoke_model(
            "weighted_avg",
            None,
            "--load-model",
            str(weighted_model_production),
            "--forecast-steps",
            str(future_steps),
            "--pretty",
        )

        future_predictions_one = extract_predictions(predict_output_one)
        future_predictions_two = extract_predictions(predict_output_two)
        future_std_devs_one = extract_std_devs(predict_output_one)
        future_std_devs_two = extract_std_devs(predict_output_two)
        assert predict_output_one["mode"] == "weighted_avg_loaded_predict"
        assert_method_metadata(predict_output_one, "weighted_avg")
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
                "model": "weighted_avg",
                "reused_base_models": BASE_MODELS,
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
