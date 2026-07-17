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
FEATURE_KEYS = [
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
sys.path.insert(0, str(PY_ROOT / "src"))

from data_utils import infer_feature_types, load_data_from_json  # noqa: E402
from dynamic_config import lstm_parameter_count  # noqa: E402
from model_metadata_assertions import assert_method_metadata  # noqa: E402


INFERRED_SELECTED_FEATURE_TYPES = {
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
INFERRED_AUTO_FEATURE_TYPES = {
    "销售数量": "numeric",
    "年份": "numeric",
    "月份": "categorical",
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
EDGE_CASE_INFERRED_TYPES = {
    "y": "numeric",
    "humidity": "numeric",
    "grid_load": "numeric",
    "promo_code": "categorical",
}


def load_csv_records(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_lstm(data_json: str | None, *extra_args: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(RUN_MODEL), "lstm"]
    if data_json is not None:
        command.extend(["--data", data_json])
    command.extend(extra_args)

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )


def invoke_lstm(data_json: str | None, *extra_args: str) -> dict:
    completed = run_lstm(data_json, *extra_args)
    if completed.returncode != 0:
        raise RuntimeError(
            "lstm CLI execution failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def extract_predictions(payload: dict) -> list[float]:
    return [float(item["prediction"]) for item in payload["predictions"]]


def extract_std_devs(payload: dict) -> list[float]:
    return [float(item["std_dev"]) for item in payload["predictions"]]


def assert_all_finite(values: list[float], label: str) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise AssertionError(f"{label}[{index}] is not finite: {value}")


def assert_series_close(actual: list[float], expected: list[float], tol: float = 1e-9) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"length mismatch: actual={len(actual)}, expected={len(expected)}")
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
        if not math.isclose(actual_value, expected_value, rel_tol=tol, abs_tol=tol):
            raise AssertionError(
                f"value[{index}] mismatch: actual={actual_value}, expected={expected_value}"
            )


def assert_lstm_uncertainty_contract(payload: dict) -> None:
    summary = payload["uncertainty_summary"]
    assert summary["coverage_guarantee"] is False
    assert summary["upper_error_p99_kind"] == "uncalibrated_estimate"
    for point in payload["predictions"]:
        assert point["coverage_guarantee"] is False
        assert point["upper_error_p99_kind"] == "uncalibrated_estimate"
        assert point["interval_kind"].startswith("censored_nonnegative_")
        assert 0.0 <= point["interval_lower"] <= point["prediction"] <= point["interval_upper"]


def load_pickled_state(model_path: Path) -> dict:
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    records = load_csv_records(CSV_FIXTURE)
    data_json = json.dumps(records, ensure_ascii=False)
    timestamps = [f"{int(row['年份']):04d}-{int(row['月份']):02d}" for row in records]
    parsed_auto = load_data_from_json(data_json)
    inferred_auto_types = infer_feature_types(parsed_auto)
    if inferred_auto_types != INFERRED_AUTO_FEATURE_TYPES:
        raise AssertionError(
            "unexpected auto inferred feature_types:\n"
            f"actual={json.dumps(inferred_auto_types, ensure_ascii=False, indent=2)}\n"
            f"expected={json.dumps(INFERRED_AUTO_FEATURE_TYPES, ensure_ascii=False, indent=2)}"
        )
    parsed_edge = load_data_from_json(
        json.dumps(
            [
                {"y": 1, "humidity": 60.1, "grid_load": 100.2, "promo_code": 101},
                {"y": 2, "humidity": 61.5, "grid_load": 101.7, "promo_code": 102},
                {"y": 3, "humidity": 59.2, "grid_load": 99.8, "promo_code": 103},
                {"y": 4, "humidity": 62.3, "grid_load": 102.4, "promo_code": 104},
            ],
            ensure_ascii=False,
        )
    )
    inferred_edge_types = infer_feature_types(parsed_edge)
    if inferred_edge_types != EDGE_CASE_INFERRED_TYPES:
        raise AssertionError(
            "unexpected edge-case inferred feature_types:\n"
            f"actual={json.dumps(inferred_edge_types, ensure_ascii=False, indent=2)}\n"
            f"expected={json.dumps(EDGE_CASE_INFERRED_TYPES, ensure_ascii=False, indent=2)}"
        )

    with tempfile.TemporaryDirectory(prefix="fangzhen-lstm-csv-") as temp_dir:
        temp_path = Path(temp_dir)
        backtest_model = temp_path / "lstm-backtest.pkl"
        production_model = temp_path / "lstm-production.pkl"

        train_output = invoke_lstm(
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "27",
            "--eval-start",
            "28",
            "--eval-end",
            "30",
            "--look-back",
            "4",
            "--epochs",
            "2",
            "--normalization",
            "minmax",
            "--lstm-target-feature",
            "销售数量",
            "--lstm-features",
            ",".join(FEATURE_KEYS),
            "--save-model",
            str(backtest_model),
            "--pretty",
        )

        train_predictions = extract_predictions(train_output)
        train_std_devs = extract_std_devs(train_output)
        assert train_output["mode"] == "lstm_train_eval"
        assert_method_metadata(train_output, "lstm_direct")
        assert train_output["data_points"] == len(records)
        assert train_output["target_key"] == "销售数量"
        assert train_output["feature_types"] == INFERRED_SELECTED_FEATURE_TYPES
        assert "look_back" not in train_output
        assert "epochs" not in train_output
        assert train_output["normalization"] == "minmax"
        assert train_output["forecast_horizon"] == 3
        assert train_output["training_regime"] in {"limited_time_validation", "teaching_demo"}
        assert train_output["metrics"]["evaluate_months"] == timestamps[28:31]
        assert train_output["metrics"]["forecast_horizon"] == 3
        assert len(train_output["eval_y_true"]) == 3
        assert len(train_predictions) == 3
        assert len(FEATURE_KEYS) == 9
        assert_all_finite(train_predictions, "train_predictions")
        assert_all_finite(train_std_devs, "train_std_devs")
        assert_lstm_uncertainty_contract(train_output)
        if not all(std_dev > 0 for std_dev in train_std_devs):
            raise AssertionError(f"expected positive std_dev values, got {train_std_devs}")
        backtest_state = load_pickled_state(backtest_model)
        assert backtest_state["look_back"] == backtest_state["derived_config"]["look_back"]
        assert backtest_state["derived_config"]["max_epochs"] != 2
        assert backtest_state["derived_config"]["regime"] == train_output["training_regime"]

        fit_output = invoke_lstm(
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "35",
            "--load-model",
            str(backtest_model),
            "--save-model",
            str(production_model),
            "--fit-only",
            "--pretty",
        )
        assert fit_output["mode"] == "lstm_fit_only"
        assert_method_metadata(fit_output, "lstm_direct")
        assert "look_back" not in fit_output
        assert "epochs" not in fit_output
        assert fit_output["normalization"] == "minmax"
        assert fit_output["forecast_horizon"] == 6
        assert fit_output["target_key"] == "销售数量"
        assert fit_output["feature_keys"] == FEATURE_KEYS
        assert fit_output["feature_types"] == INFERRED_SELECTED_FEATURE_TYPES
        assert fit_output["saved_model"] == str(production_model)
        production_state = load_pickled_state(production_model)
        assert production_state["forecast_strategy"] == "direct_multi_horizon"
        assert_method_metadata(production_state, "lstm_direct")
        assert production_state["forecast_horizon"] == 6
        assert production_state["production_forecast_horizon"] == 6
        assert production_state["derived_config"]["regime"] == fit_output["training_regime"]
        assert production_state["uncertainty_summary"]["coverage_guarantee"] is False
        assert production_state["uncertainty_summary"]["upper_error_p99_kind"] == "uncalibrated_estimate"
        if production_state["derived_config"]["regime"] == "limited_time_validation":
            config = production_state["derived_config"]
            assert config["validation_preprocessor_fit_points"] < 36
            assert config["selection_parameter_count"] == lstm_parameter_count(
                config["selection_units"],
                config["validation_feature_count"],
                config["forecast_horizon"],
            )

        predict_output_one = invoke_lstm(
            None,
            "--load-model",
            str(production_model),
            "--forecast-steps",
            "6",
            "--pretty",
        )
        predict_output_two = invoke_lstm(
            None,
            "--load-model",
            str(production_model),
            "--forecast-steps",
            "6",
            "--pretty",
        )

        future_predictions_one = extract_predictions(predict_output_one)
        future_predictions_two = extract_predictions(predict_output_two)
        future_std_devs_one = extract_std_devs(predict_output_one)
        future_std_devs_two = extract_std_devs(predict_output_two)
        assert predict_output_one["mode"] == "lstm_loaded_predict"
        assert_method_metadata(predict_output_one, "lstm_direct")
        assert predict_output_one["forecast_steps"] == 6
        assert predict_output_one["train_end_index"] == 35
        assert predict_output_one["target_key"] == "销售数量"
        assert predict_output_one["feature_keys"] == FEATURE_KEYS
        assert len(future_predictions_one) == 6
        assert_all_finite(future_predictions_one, "future_predictions")
        assert_all_finite(future_std_devs_one, "future_std_devs")
        assert_lstm_uncertainty_contract(predict_output_one)
        assert_series_close(future_predictions_one, future_predictions_two)
        assert_series_close(future_std_devs_one, future_std_devs_two)

        corrupt_sidecar = temp_path / "lstm-corrupt-sidecar.h5"
        corrupt_sidecar.write_bytes(b"not-a-valid-keras-model")
        corrupt_state_path = temp_path / "lstm-corrupt-state.pkl"
        corrupt_state = load_pickled_state(production_model)
        corrupt_state["keras_model_path"] = str(corrupt_sidecar)
        with corrupt_state_path.open("wb") as handle:
            pickle.dump(corrupt_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

        corrupt_predict = run_lstm(
            None,
            "--load-model",
            str(corrupt_state_path),
            "--forecast-steps",
            "1",
        )
        if corrupt_predict.returncode == 0:
            raise AssertionError("expected corrupt LSTM sidecar to fail")
        if "[MODEL] Failed to load Keras model" not in corrupt_predict.stderr:
            raise AssertionError(
                "missing stable model-artifact error tag\n"
                f"stdout:\n{corrupt_predict.stdout}\n"
                f"stderr:\n{corrupt_predict.stderr}"
            )

        small_records = [
            {"y": 1, "x": 10},
            {"y": 2, "x": 11},
            {"y": 3, "x": 12},
            {"y": 4, "x": 13},
            {"y": 5, "x": 14},
            {"y": 6, "x": 15},
            {"y": 7, "x": 16},
            {"y": 8, "x": 17},
            {"y": 9, "x": 18},
        ]
        small_data_json = json.dumps(small_records, ensure_ascii=False)
        small_backtest_model = temp_path / "lstm-small-backtest.pkl"
        small_production_model = temp_path / "lstm-small-production.pkl"

        small_train_output = invoke_lstm(
            small_data_json,
            "--train-start",
            "0",
            "--train-end",
            "6",
            "--eval-start",
            "7",
            "--eval-end",
            "8",
            "--look-back",
            "4",
            "--epochs",
            "1",
            "--normalization",
            "minmax",
            "--lstm-target-feature",
            "y",
            "--lstm-features",
            "x",
            "--save-model",
            str(small_backtest_model),
        )
        assert small_train_output["forecast_horizon"] == 2

        rejected_fit = run_lstm(
            small_data_json,
            "--train-start",
            "0",
            "--train-end",
            "7",
            "--load-model",
            str(small_backtest_model),
            "--save-model",
            str(small_production_model),
            "--fit-only",
        )
        if rejected_fit.returncode == 0:
            raise AssertionError("expected short direct LSTM production refit to fail")
        if "Training range is too short for a direct LSTM forecast" not in rejected_fit.stderr:
            raise AssertionError(
                "missing explicit direct-strategy validation error\n"
                f"stdout:\n{rejected_fit.stdout}\n"
                f"stderr:\n{rejected_fit.stderr}"
            )

        small_fit_output = invoke_lstm(
            small_data_json,
            "--train-start",
            "0",
            "--train-end",
            "7",
            "--load-model",
            str(small_backtest_model),
            "--save-model",
            str(small_production_model),
            "--fit-forecast-steps",
            "2",
            "--fit-only",
        )
        assert small_fit_output["mode"] == "lstm_fit_only"
        assert_method_metadata(small_fit_output, "lstm_direct")
        assert small_fit_output["forecast_horizon"] == 2
        small_production_state = load_pickled_state(small_production_model)
        assert small_production_state["forecast_strategy"] == "direct_multi_horizon"
        assert_method_metadata(small_production_state, "lstm_direct")
        assert small_production_state["forecast_horizon"] == 2
        assert small_production_state["production_forecast_horizon"] == 2

        small_predict_output = invoke_lstm(
            None,
            "--load-model",
            str(small_production_model),
            "--forecast-steps",
            "2",
        )
        assert small_predict_output["mode"] == "lstm_loaded_predict"
        assert_method_metadata(small_predict_output, "lstm_direct")
        assert small_predict_output["forecast_steps"] == 2
        assert len(extract_predictions(small_predict_output)) == 2

    print(
        json.dumps(
            {
                "fixture": str(CSV_FIXTURE),
                "checked_modes": [
                    "type_inference",
                    "training",
                    "fit_only",
                    "loaded_predict",
                    "corrupt_sidecar_rejection",
                    "short_refit_rejection",
                ],
                "feature_count": len(FEATURE_KEYS),
                "model": "lstm",
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
