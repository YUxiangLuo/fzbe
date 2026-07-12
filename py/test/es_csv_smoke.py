#!/usr/bin/env python3
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing


PY_ROOT = Path(__file__).resolve().parents[1]
RUN_MODEL = PY_ROOT / "run_model.py"
CSV_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "core_models_smoke.csv"
sys.path.insert(0, str(PY_ROOT / "src"))

from model_metadata_assertions import assert_method_metadata
from models import es as es_model


def load_csv_records(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def invoke_es(data_json: str, *extra_args: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(RUN_MODEL), "es", "--data", data_json, *extra_args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "es CLI execution failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def build_es_forecast(series: list[float], alpha: float, steps: int) -> list[float]:
    model = SimpleExpSmoothing(series, initialization_method="estimated").fit(
        smoothing_level=alpha,
        optimized=False,
    )
    return model.forecast(steps).tolist()


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
    alpha = 0.4

    with tempfile.TemporaryDirectory(prefix="fangzhen-es-csv-") as temp_dir:
        temp_path = Path(temp_dir)
        backtest_model = temp_path / "es-backtest.pkl"
        production_model = temp_path / "es-production.pkl"

        train_output = invoke_es(
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "19",
            "--eval-start",
            "20",
            "--eval-end",
            "23",
            "--alpha",
            str(alpha),
            "--save-model",
            str(backtest_model),
            "--pretty",
        )

        expected_eval_predictions = build_es_forecast(sales[0:20], alpha, 4)
        actual_eval_predictions = [float(item["prediction"]) for item in train_output["predictions"]]
        assert train_output["mode"] == "fixed_alpha_only"
        assert_method_metadata(train_output, "es")
        assert math.isclose(float(train_output["alpha"]), alpha, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(float(train_output["metrics"]["alpha"]), alpha, rel_tol=1e-9, abs_tol=1e-9)
        assert train_output["metrics"]["evaluate_months"] == timestamps[20:24]
        assert train_output["eval_y_true"] == sales[20:24]
        assert_series_close(actual_eval_predictions, expected_eval_predictions)

        gap_output = invoke_es(
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "19",
            "--eval-start",
            "22",
            "--eval-end",
            "23",
            "--alpha",
            str(alpha),
            "--pretty",
        )
        assert gap_output["metrics"]["evaluate_months"] == timestamps[22:24]
        assert gap_output["eval_y_true"] == sales[22:24]
        assert_series_close(
            [float(item["prediction"]) for item in gap_output["predictions"]],
            build_es_forecast(sales[0:20], alpha, 4)[2:4],
        )

        fit_output = invoke_es(
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
        assert fit_output["mode"] == "es_fit_only"
        assert_method_metadata(fit_output, "es")
        assert fit_output["saved_model"] == str(production_model)
        assert math.isclose(float(fit_output["alpha"]), alpha, rel_tol=1e-9, abs_tol=1e-9)

        predict_output = invoke_es(
            data_json,
            "--load-model",
            str(production_model),
            "--forecast-steps",
            "3",
            "--pretty",
        )
        expected_future_predictions = build_es_forecast(sales[0:24], alpha, 3)
        actual_future_predictions = [float(item["prediction"]) for item in predict_output["predictions"]]
        assert predict_output["mode"] == "es_loaded_predict"
        assert_method_metadata(predict_output, "es")
        assert predict_output["forecast_steps"] == 3
        assert predict_output["train_end_index"] == 23
        assert_series_close(actual_future_predictions, expected_future_predictions)

        class FakeFittedModel:
            params: dict[str, float] = {}

            def forecast(self, steps: int) -> np.ndarray:
                if steps != 4:
                    raise AssertionError(f"expected forecast through 2-point gap, got {steps} steps")
                return np.asarray([10.0, 11.0, 12.0, 13.0])

        class FakeSimpleExpSmoothing:
            def __init__(self, *_args, **_kwargs):
                pass

            def fit(self, **_kwargs) -> FakeFittedModel:
                return FakeFittedModel()

        original_simple_exp_smoothing = es_model.SimpleExpSmoothing
        es_model.SimpleExpSmoothing = FakeSimpleExpSmoothing
        try:
            _, _, _, sliced_predictions, _ = es_model.train_and_evaluate(
                [1, 2, 3, 4, 5, 6, 7],
                None,
                range(0, 3),
                range(5, 7),
                alpha,
            )
        finally:
            es_model.SimpleExpSmoothing = original_simple_exp_smoothing
        assert sliced_predictions == [12.0, 13.0]

    print(
        json.dumps(
            {
                "fixture": str(CSV_FIXTURE),
                "checked_modes": ["training", "fit_only", "loaded_predict", "evaluation_gap"],
                "model": "es",
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
