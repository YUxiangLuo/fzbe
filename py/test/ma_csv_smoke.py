#!/usr/bin/env python3
import csv
import json
import math
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path


PY_ROOT = Path(__file__).resolve().parents[1]
RUN_MODEL = PY_ROOT / "run_model.py"
CSV_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "core_models_smoke.csv"


def load_csv_records(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def moving_average_roll(history: list[float], window: int, steps: int) -> list[float]:
    if len(history) < window:
        raise AssertionError("history shorter than moving-average window")

    buffer: deque[float] = deque(history[-window:], maxlen=window)
    predictions: list[float] = []
    for _ in range(steps):
        prediction = float(sum(buffer) / len(buffer))
        predictions.append(prediction)
        buffer.append(prediction)
    return predictions


def invoke_ma(data_json: str, *extra_args: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(RUN_MODEL), "ma", "--data", data_json, *extra_args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ma CLI execution failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def assert_series_close(actual: list[float], expected: list[float], tol: float = 1e-6) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"length mismatch: actual={len(actual)}, expected={len(expected)}")

    for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
        if not math.isclose(actual_value, expected_value, rel_tol=tol, abs_tol=tol):
            raise AssertionError(
                f"prediction[{index}] mismatch: actual={actual_value}, expected={expected_value}"
            )


def main() -> None:
    records = load_csv_records(CSV_FIXTURE)
    data_json = json.dumps(records, ensure_ascii=False)
    sales = [float(row["销售数量"]) for row in records]
    timestamps = [f"{int(row['年份']):04d}-{int(row['月份']):02d}" for row in records]

    with tempfile.TemporaryDirectory(prefix="fangzhen-ma-csv-") as temp_dir:
        temp_path = Path(temp_dir)
        backtest_model = temp_path / "ma-backtest.pkl"
        production_model = temp_path / "ma-production.pkl"

        train_output = invoke_ma(
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "19",
            "--eval-start",
            "20",
            "--eval-end",
            "23",
            "--window",
            "3",
            "--save-model",
            str(backtest_model),
            "--pretty",
        )

        expected_eval_predictions = moving_average_roll(sales[17:20], 3, 4)
        actual_eval_predictions = [float(item["prediction"]) for item in train_output["predictions"]]
        assert train_output["mode"] == "fixed_window_only"
        assert train_output["metrics"]["window"] == 3
        assert train_output["metrics"]["evaluate_months"] == timestamps[20:24]
        assert train_output["eval_y_true"] == sales[20:24]
        assert_series_close(actual_eval_predictions, expected_eval_predictions)

        fit_output = invoke_ma(
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "23",
            "--load-model",
            str(backtest_model),
            "--save-model",
            str(production_model),
            "--fit-only",
            "--pretty",
        )
        assert fit_output["mode"] == "ma_fit_only"
        assert fit_output["saved_model"] == str(production_model)

        predict_output = invoke_ma(
            data_json,
            "--load-model",
            str(production_model),
            "--forecast-steps",
            "3",
            "--pretty",
        )
        expected_future_predictions = moving_average_roll(sales[21:24], 3, 3)
        actual_future_predictions = [float(item["prediction"]) for item in predict_output["predictions"]]
        assert predict_output["mode"] == "ma_loaded_predict"
        assert predict_output["forecast_steps"] == 3
        assert_series_close(actual_future_predictions, expected_future_predictions)

    print(
        json.dumps(
            {
                "fixture": str(CSV_FIXTURE),
                "checked_modes": ["training", "fit_only", "loaded_predict"],
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
