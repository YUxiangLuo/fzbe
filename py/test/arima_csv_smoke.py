#!/usr/bin/env python3
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pmdarima as pm


PY_ROOT = Path(__file__).resolve().parents[1]
RUN_MODEL = PY_ROOT / "run_model.py"
CSV_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "core_models_smoke.csv"

from model_metadata_assertions import assert_method_metadata  # noqa: E402


def load_csv_records(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def invoke_arima(data_json: str, *extra_args: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(RUN_MODEL), "arima", "--data", data_json, *extra_args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "arima CLI execution failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def run_arima(data_json: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUN_MODEL), "arima", "--data", data_json, *extra_args],
        capture_output=True,
        text=True,
        check=False,
    )


def select_reference_model(series: list[float], d: int, max_p: int, max_q: int):
    return pm.auto_arima(
        series,
        d=d,
        start_p=0,
        start_q=0,
        max_p=max_p,
        max_q=max_q,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        information_criterion="aicc",
        with_intercept="auto",
        maxiter=50,
        trace=False,
    )


NORMAL_95 = 1.959963984540054


def to_std_devs(conf_intervals: list[list[float]]) -> list[float]:
    return [float((interval[1] - interval[0]) / (2.0 * NORMAL_95)) for interval in conf_intervals]


def assert_series_close(actual: list[float], expected: list[float], tol: float = 1e-6) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"length mismatch: actual={len(actual)}, expected={len(expected)}")
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
        if not math.isclose(actual_value, expected_value, rel_tol=tol, abs_tol=tol):
            raise AssertionError(
                f"value[{index}] mismatch: actual={actual_value}, expected={expected_value}"
            )


def assert_error(completed: subprocess.CompletedProcess[str], expected_substring: str) -> None:
    if completed.returncode == 0:
        raise AssertionError(f"expected failure containing {expected_substring!r}, got success")
    if expected_substring not in completed.stderr:
        raise AssertionError(
            f"missing expected error substring {expected_substring!r}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def main() -> None:
    records = load_csv_records(CSV_FIXTURE)
    data_json = json.dumps(records, ensure_ascii=False)
    sales = [float(row["销售数量"]) for row in records]
    timestamps = [f"{int(row['年份']):04d}-{int(row['月份']):02d}" for row in records]
    d = 1
    max_p = 2
    max_q = 2

    with tempfile.TemporaryDirectory(prefix="fangzhen-arima-csv-") as temp_dir:
        temp_path = Path(temp_dir)
        backtest_model = temp_path / "arima-backtest.pkl"
        production_model = temp_path / "arima-production.pkl"

        train_output = invoke_arima(
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "19",
            "--eval-start",
            "20",
            "--eval-end",
            "23",
            "--d",
            str(d),
            "--max-p",
            str(max_p),
            "--max-q",
            str(max_q),
            "--save-model",
            str(backtest_model),
            "--pretty",
        )

        best_model = select_reference_model(sales[0:20], d, max_p, max_q)
        expected_eval_predictions, expected_conf_int = best_model.predict(n_periods=4, return_conf_int=True)
        expected_eval_std = to_std_devs(expected_conf_int.tolist())

        actual_eval_predictions = [float(item["prediction"]) for item in train_output["predictions"]]
        actual_eval_std = [float(item["std_dev"]) for item in train_output["predictions"]]
        expected_best_order = {
            "p": int(best_model.order[0]),
            "d": int(best_model.order[1]),
            "q": int(best_model.order[2]),
        }

        assert train_output["mode"] == "arima_search_pq_fixed_d"
        assert_method_metadata(train_output, "arima")
        assert train_output["best_order"] == expected_best_order
        assert train_output["metrics"]["evaluate_months"] == timestamps[20:24]
        assert train_output["eval_y_true"] == sales[20:24]
        search_summary = train_output["search_summary"]
        assert search_summary["selection_criterion"] == "aicc"
        assert search_summary["selected_order"] == expected_best_order
        assert search_summary["effective_train_size"] == 19
        assert search_summary["max_p"] == max_p
        assert search_summary["max_q"] == max_q
        assert isinstance(search_summary["diagnostics"], dict)
        assert_series_close(actual_eval_predictions, expected_eval_predictions.tolist())
        assert_series_close(actual_eval_std, expected_eval_std)
        assert all(
            point["calibration_mean_error"] is None
            and point["calibration_count"] is None
            for point in train_output["predictions"]
        )

        gap_output = invoke_arima(
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "19",
            "--eval-start",
            "22",
            "--eval-end",
            "23",
            "--d",
            str(d),
            "--max-p",
            str(max_p),
            "--max-q",
            str(max_q),
            "--pretty",
        )
        gap_reference = select_reference_model(sales[0:20], d, max_p, max_q)
        gap_predictions, gap_conf_int = gap_reference.predict(
            n_periods=4,
            return_conf_int=True,
        )
        assert gap_output["metrics"]["evaluate_months"] == timestamps[22:24]
        assert gap_output["eval_y_true"] == sales[22:24]
        assert_series_close(
            [float(item["prediction"]) for item in gap_output["predictions"]],
            gap_predictions.tolist()[2:4],
        )
        assert_series_close(
            [float(item["std_dev"]) for item in gap_output["predictions"]],
            to_std_devs(gap_conf_int.tolist())[2:4],
        )

        fit_output = invoke_arima(
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
        assert fit_output["mode"] == "arima_fit_only"
        assert_method_metadata(fit_output, "arima")
        assert fit_output["order"] == expected_best_order
        assert fit_output["train_end_index"] == 23
        assert fit_output["saved_model"] == str(production_model)

        production_reference = pm.ARIMA(
            order=(expected_best_order["p"], expected_best_order["d"], expected_best_order["q"]),
            suppress_warnings=True,
        )
        production_reference.fit(sales[0:24])
        expected_future_predictions, expected_future_conf_int = production_reference.predict(
            n_periods=3,
            return_conf_int=True,
        )
        expected_future_std = to_std_devs(expected_future_conf_int.tolist())

        predict_output = invoke_arima(
            data_json,
            "--load-model",
            str(production_model),
            "--forecast-steps",
            "3",
            "--pretty",
        )
        actual_future_predictions = [float(item["prediction"]) for item in predict_output["predictions"]]
        actual_future_std = [float(item["std_dev"]) for item in predict_output["predictions"]]
        assert predict_output["mode"] == "arima_loaded_predict"
        assert_method_metadata(predict_output, "arima")
        assert predict_output["order"] == expected_best_order
        assert predict_output["forecast_steps"] == 3
        assert_series_close(actual_future_predictions, expected_future_predictions.tolist())
        assert_series_close(actual_future_std, expected_future_std)
        assert all(
            point["calibration_mean_error"] is None
            and point["calibration_count"] is None
            for point in predict_output["predictions"]
        )

        short_train_result = run_arima(
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "1",
            "--eval-start",
            "2",
            "--eval-end",
            "3",
            "--d",
            "1",
            "--pretty",
        )
        assert_error(
            short_train_result,
            "[VALIDATION] ARIMA requires at least 6 effective training points after differencing (got train_size=2, d=1, effective=1).",
        )

        short_fit_result = run_arima(
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "1",
            "--load-model",
            str(backtest_model),
            "--save-model",
            str(temp_path / "arima-short-fit.pkl"),
            "--fit-only",
            "--pretty",
        )
        assert_error(
            short_fit_result,
            "[VALIDATION] ARIMA requires at least 6 effective training points after differencing (got train_size=2, d=1, effective=1). Context: ARIMA fit-only training range.",
        )

    print(
        json.dumps(
            {
                "fixture": str(CSV_FIXTURE),
                "checked_modes": ["training", "fit_only", "loaded_predict"],
                "model": "arima",
                "edge_cases": ["evaluation_gap", "train_short_window", "fit_only_short_window"],
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
