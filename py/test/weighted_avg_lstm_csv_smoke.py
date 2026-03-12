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
CSV_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lstm_rich_smoke.csv"
BASE_MODELS = ["ma", "es", "arima", "lstm"]
LSTM_FEATURE_KEYS = [
    "价格指数",
    "促销投入",
    "库存水平",
    "搜索热度",
    "竞争对手价格",
    "节假日强度",
    "渠道类型",
    "区域",
    "天气类型",
]
LSTM_FEATURE_TYPES = {
    "销售数量": "numeric",
    "价格指数": "numeric",
    "促销投入": "numeric",
    "库存水平": "numeric",
    "搜索热度": "numeric",
    "竞争对手价格": "numeric",
    "节假日强度": "numeric",
    "渠道类型": "categorical",
    "区域": "categorical",
    "天气类型": "categorical",
}
BASE_MODEL_TRAIN_ARGS = {
    "ma": ("--window", "3"),
    "es": ("--alpha", "0.4"),
    "arima": ("--d", "1", "--max-p", "2", "--max-q", "2"),
    "lstm": (
        "--look-back",
        "4",
        "--epochs",
        "2",
        "--normalization",
        "minmax",
        "--lstm-target-feature",
        "销售数量",
        "--lstm-features",
        ",".join(LSTM_FEATURE_KEYS),
        "--feature-types",
        json.dumps(LSTM_FEATURE_TYPES, ensure_ascii=False),
    ),
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


def combine_weighted_predictions(state: dict, forecast_steps: int) -> list[float]:
    weights = {name: float(weight) for name, weight in state["weights"].items()}
    combined = [0.0] * forecast_steps
    for model_name in state["models"]:
        member_state = state["base_model_states"][model_name]
        predictions = extract_predictions(
            invoke_model(
                model_name,
                None,
                "--load-model",
                member_state["model_file"],
                "--forecast-steps",
                str(forecast_steps),
                "--pretty",
            )
        )
        combined = [
            left + weights[model_name] * prediction
            for left, prediction in zip(combined, predictions)
        ]
    return combined


def main() -> None:
    records = load_csv_records(CSV_FIXTURE)
    data_json = json.dumps(records, ensure_ascii=False)
    timestamps = [f"{int(row['年份']):04d}-{int(row['月份']):02d}" for row in records]
    sales = [float(row["销售数量"]) for row in records]
    train_start = 0
    train_end = 27
    eval_start = 28
    eval_end = 35
    future_steps = 3

    with tempfile.TemporaryDirectory(prefix="fangzhen-weighted-avg-lstm-csv-") as temp_dir:
        temp_path = Path(temp_dir)
        base_model_paths = {model_name: temp_path / f"{model_name}-backtest.pkl" for model_name in BASE_MODELS}

        train_base_model_artifacts(
            data_json,
            base_model_paths,
            train_start,
            train_end,
            eval_start,
            eval_end,
        )
        assert (temp_path / "lstm-backtest_keras_model.h5").exists()

        weighted_model_backtest = temp_path / "weighted-avg-lstm-backtest.pkl"
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
            *BASE_MODEL_TRAIN_ARGS["lstm"],
            "--base-model-files",
            json.dumps({name: str(path) for name, path in base_model_paths.items()}, ensure_ascii=False),
            "--save-model",
            str(weighted_model_backtest),
            "--pretty",
        )

        train_state = load_pickled_state(weighted_model_backtest)
        actual_eval_predictions = extract_predictions(train_output)
        actual_eval_std_devs = extract_std_devs(train_output)
        expected_eval_predictions = combine_weighted_predictions(train_state, forecast_steps=8)
        actual_weights = {name: float(train_output["weights"][name]) for name in BASE_MODELS}
        assert train_output["mode"] == "weighted_avg_train_eval"
        assert train_output["models"] == BASE_MODELS
        assert train_output["metrics"]["evaluate_months"] == timestamps[28:36]
        assert train_output["eval_y_true"] == sales[28:36]
        assert_series_close([sum(actual_weights.values())], [1.0])
        assert set(actual_weights.keys()) == set(BASE_MODELS)
        assert all(weight >= 0 for weight in actual_weights.values())
        assert_series_close(actual_eval_predictions, expected_eval_predictions)
        assert_all_finite(actual_eval_std_devs, "train_std_devs")

        saved_lstm_member = Path(train_state["base_model_states"]["lstm"]["model_file"])
        if not saved_lstm_member.exists():
            raise AssertionError(f"saved weighted_avg LSTM member artifact missing: {saved_lstm_member}")
        saved_lstm_keras = saved_lstm_member.with_name(f"{saved_lstm_member.stem}_keras_model.h5")
        if not saved_lstm_keras.exists():
            raise AssertionError(f"saved weighted_avg LSTM keras artifact missing: {saved_lstm_keras}")

        weighted_model_production = temp_path / "weighted-avg-lstm-production.pkl"
        fit_output = invoke_model(
            "weighted_avg",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "35",
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

        production_state = load_pickled_state(weighted_model_production)
        expected_future_predictions = combine_weighted_predictions(production_state, forecast_steps=future_steps)

        production_lstm_member = Path(production_state["base_model_states"]["lstm"]["model_file"])
        if not production_lstm_member.exists():
            raise AssertionError(f"production weighted_avg LSTM member artifact missing: {production_lstm_member}")
        production_lstm_keras = production_lstm_member.with_name(f"{production_lstm_member.stem}_keras_model.h5")
        if not production_lstm_keras.exists():
            raise AssertionError(f"production weighted_avg LSTM keras artifact missing: {production_lstm_keras}")

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
                "feature_count": len(LSTM_FEATURE_KEYS),
                "model": "weighted_avg_lstm",
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
