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


def normalized_inverse_variance_weights(
    predictions_by_model: dict[str, list[float]],
    y_true: list[float],
) -> dict[str, float]:
    raw_weights: dict[str, float] = {}
    total = 0.0
    for model_name, predictions in predictions_by_model.items():
        residuals = [target - prediction for target, prediction in zip(y_true, predictions)]
        residual_mean = sum(residuals) / len(residuals)
        variance = sum((value - residual_mean) ** 2 for value in residuals) / len(residuals)
        inverse_variance = 1.0 / (variance + 1e-9)
        raw_weights[model_name] = inverse_variance
        total += inverse_variance

    return {model_name: weight / total for model_name, weight in raw_weights.items()}


def weighted_sum(predictions_by_model: dict[str, list[float]], weights: dict[str, float]) -> list[float]:
    length = len(next(iter(predictions_by_model.values())))
    combined: list[float] = []
    for index in range(length):
        combined.append(sum(predictions_by_model[model_name][index] * weights[model_name] for model_name in weights))
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
    weight_train_end = 15
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

        weight_val_predictions: dict[str, list[float]] = {}
        eval_predictions: dict[str, list[float]] = {}
        for model_name in BASE_MODELS:
            refit_model_path = temp_path / f"{model_name}-weight-refit.pkl"
            invoke_model(
                model_name,
                data_json,
                "--train-start",
                str(train_start),
                "--train-end",
                str(weight_train_end),
                "--load-model",
                str(base_model_paths[model_name]),
                "--save-model",
                str(refit_model_path),
                "--fit-only",
                "--pretty",
            )
            weight_val_predictions[model_name] = extract_predictions(
                invoke_model(
                    model_name,
                    None,
                    "--load-model",
                    str(refit_model_path),
                    "--forecast-steps",
                    "4",
                    "--pretty",
                )
            )
            eval_predictions[model_name] = extract_predictions(
                invoke_model(
                    model_name,
                    None,
                    "--load-model",
                    str(base_model_paths[model_name]),
                    "--forecast-steps",
                    "4",
                    "--pretty",
                )
            )

        expected_weights = normalized_inverse_variance_weights(weight_val_predictions, sales[16:20])
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
