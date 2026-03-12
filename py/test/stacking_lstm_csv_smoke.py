#!/usr/bin/env python3
import csv
import json
import math
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


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


def run_command(model_type: str, data_json: str | None, *extra_args: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(RUN_MODEL), model_type]
    if data_json is not None:
        command.extend(["--data", data_json])
    command.extend(extra_args)
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )


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


def assert_error(completed: subprocess.CompletedProcess[str], expected_substring: str) -> None:
    if completed.returncode == 0:
        raise AssertionError(f"expected failure containing {expected_substring!r}, got success")
    if expected_substring not in completed.stderr:
        raise AssertionError(
            f"missing expected error substring {expected_substring!r}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


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


def combine_stacking_predictions(state: dict, forecast_steps: int) -> tuple[list[float], dict[str, list[float]], np.ndarray]:
    meta_model = state["meta_model"]
    weights = np.asarray(meta_model["weights"], dtype=float)
    member_predictions: dict[str, list[float]] = {}
    prediction_matrix = []

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
        member_predictions[model_name] = predictions
        prediction_matrix.append(predictions)

    blended = np.column_stack(prediction_matrix) @ weights
    return blended.tolist(), member_predictions, weights


def assert_predictions_within_member_envelope(
    model_names: list[str],
    member_predictions: dict[str, list[float]],
    blended_predictions: list[float],
    tol: float = 1e-6,
) -> None:
    prediction_matrix = np.column_stack([member_predictions[name] for name in model_names])
    lower = np.min(prediction_matrix, axis=1)
    upper = np.max(prediction_matrix, axis=1)
    for index, (value, lo, hi) in enumerate(zip(blended_predictions, lower, upper)):
        if value < lo - tol or value > hi + tol:
            raise AssertionError(
                f"blended prediction[{index}] escaped member envelope: value={value}, lo={lo}, hi={hi}"
            )


def main() -> None:
    records = load_csv_records(CSV_FIXTURE)
    data_json = json.dumps(records, ensure_ascii=False)
    trimmed_feature_data_json = json.dumps(
        [{key: value for key, value in row.items() if key != "库存水平"} for row in records],
        ensure_ascii=False,
    )
    timestamps = [f"{int(row['年份']):04d}-{int(row['月份']):02d}" for row in records]
    sales = [float(row["销售数量"]) for row in records]
    train_start = 0
    train_end = 27
    eval_start = 28
    eval_end = 30
    future_steps = 6

    with tempfile.TemporaryDirectory(prefix="fangzhen-stacking-lstm-csv-") as temp_dir:
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

        stacking_model_backtest = temp_path / "stacking-lstm-backtest.pkl"
        train_output = invoke_model(
            "stacking",
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
            str(stacking_model_backtest),
            "--pretty",
        )

        train_state = load_pickled_state(stacking_model_backtest)
        actual_eval_predictions = extract_predictions(train_output)
        actual_eval_std_devs = extract_std_devs(train_output)
        expected_eval_predictions, backtest_member_predictions, backtest_weights = combine_stacking_predictions(
            train_state,
            forecast_steps=3,
        )
        assert train_output["mode"] == "stacking_train_eval"
        assert train_output["models"] == BASE_MODELS
        assert train_output["metrics"]["evaluate_months"] == timestamps[28:31]
        assert train_output["eval_y_true"] == sales[28:31]
        assert train_state["meta_model"]["kind"] == "weighted_blend"
        assert set(train_state["meta_model"]["model_names"]) == set(BASE_MODELS)
        assert_series_close([float(np.sum(backtest_weights))], [1.0])
        assert np.all(backtest_weights >= 0)
        assert_series_close(actual_eval_predictions, expected_eval_predictions)
        assert_all_finite(actual_eval_std_devs, "train_std_devs")
        assert_predictions_within_member_envelope(BASE_MODELS, backtest_member_predictions, actual_eval_predictions)

        saved_lstm_member = Path(train_state["base_model_states"]["lstm"]["model_file"])
        if not saved_lstm_member.exists():
            raise AssertionError(f"saved stacking LSTM member artifact missing: {saved_lstm_member}")
        saved_lstm_state = load_pickled_state(saved_lstm_member)
        assert saved_lstm_state["forecast_strategy"] == "direct_multi_horizon"
        assert saved_lstm_state["forecast_horizon"] == 3
        saved_lstm_keras = saved_lstm_member.with_name(f"{saved_lstm_member.stem}_keras_model.h5")
        if not saved_lstm_keras.exists():
            raise AssertionError(f"saved stacking LSTM keras artifact missing: {saved_lstm_keras}")

        stacking_model_production = temp_path / "stacking-lstm-production.pkl"
        fit_output = invoke_model(
            "stacking",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "35",
            "--load-model",
            str(stacking_model_backtest),
            "--save-model",
            str(stacking_model_production),
            "--fit-only",
            "--pretty",
        )
        assert fit_output["mode"] == "stacking_fit_only"
        assert fit_output["models"] == BASE_MODELS
        assert fit_output["saved_model"] == str(stacking_model_production)

        production_state = load_pickled_state(stacking_model_production)
        expected_future_predictions, production_member_predictions, production_weights = combine_stacking_predictions(
            production_state,
            forecast_steps=future_steps,
        )
        assert_series_close([float(np.sum(production_weights))], [1.0])
        assert np.all(production_weights >= 0)

        production_lstm_member = Path(production_state["base_model_states"]["lstm"]["model_file"])
        if not production_lstm_member.exists():
            raise AssertionError(f"production stacking LSTM member artifact missing: {production_lstm_member}")
        production_lstm_state = load_pickled_state(production_lstm_member)
        assert production_lstm_state["forecast_strategy"] == "direct_multi_horizon"
        assert production_lstm_state["forecast_horizon"] == future_steps
        assert production_lstm_state["production_forecast_horizon"] == future_steps
        production_lstm_keras = production_lstm_member.with_name(f"{production_lstm_member.stem}_keras_model.h5")
        if not production_lstm_keras.exists():
            raise AssertionError(f"production stacking LSTM keras artifact missing: {production_lstm_keras}")

        predict_output_one = invoke_model(
            "stacking",
            None,
            "--load-model",
            str(stacking_model_production),
            "--forecast-steps",
            str(future_steps),
            "--pretty",
        )
        predict_output_two = invoke_model(
            "stacking",
            None,
            "--load-model",
            str(stacking_model_production),
            "--forecast-steps",
            str(future_steps),
            "--pretty",
        )

        future_predictions_one = extract_predictions(predict_output_one)
        future_predictions_two = extract_predictions(predict_output_two)
        future_std_devs_one = extract_std_devs(predict_output_one)
        future_std_devs_two = extract_std_devs(predict_output_two)
        assert predict_output_one["mode"] == "stacking_loaded_predict"
        assert predict_output_one["forecast_steps"] == future_steps
        assert_series_close(future_predictions_one, expected_future_predictions)
        assert_series_close(future_predictions_one, future_predictions_two)
        assert_series_close(future_std_devs_one, future_std_devs_two)
        assert_all_finite(future_std_devs_one, "future_std_devs")
        assert_predictions_within_member_envelope(BASE_MODELS, production_member_predictions, future_predictions_one)

        missing_feature_result = run_command(
            "stacking",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "27",
            "--eval-start",
            "28",
            "--eval-end",
            "30",
            "--models",
            "ma,lstm",
            "--window",
            "3",
            "--look-back",
            "4",
            "--epochs",
            "2",
            "--lstm-target-feature",
            "销售数量",
            "--lstm-features",
            "价格指数,不存在字段,促销投入",
        )
        assert_error(
            missing_feature_result,
            "[VALIDATION] LSTM feature columns not found in dataset: ['不存在字段']",
        )

        oversized_look_back_result = run_command(
            "stacking",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "11",
            "--eval-start",
            "12",
            "--eval-end",
            "15",
            "--models",
            "ma,lstm",
            "--window",
            "3",
            "--look-back",
            "9",
            "--epochs",
            "2",
            "--lstm-target-feature",
            "销售数量",
            "--lstm-features",
            "价格指数,促销投入,库存水平,搜索热度",
        )
        assert_error(
            oversized_look_back_result,
            "[VALIDATION] LSTM look_back (9) is too large for the stacking internal training split (10 points) and level1 horizon (2 points).",
        )

        short_fit_result = run_command(
            "stacking",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "3",
            "--load-model",
            str(stacking_model_backtest),
            "--save-model",
            str(temp_path / "stacking-lstm-short-fit.pkl"),
            "--fit-only",
            "--pretty",
        )
        assert_error(
            short_fit_result,
            "[VALIDATION] Stacking fit-only member lstm cannot be refit: Training range must be larger than look_back.",
        )

        missing_saved_feature_fit_result = run_command(
            "stacking",
            trimmed_feature_data_json,
            "--train-start",
            "0",
            "--train-end",
            "35",
            "--load-model",
            str(stacking_model_backtest),
            "--save-model",
            str(temp_path / "stacking-lstm-missing-feature-fit.pkl"),
            "--fit-only",
            "--pretty",
        )
        assert_error(
            missing_saved_feature_fit_result,
            "[VALIDATION] Stacking fit-only member lstm is missing feature columns: ['库存水平']",
        )

        reused_lstm_source = temp_path / "lstm-reuse-mismatch.pkl"
        invoke_model(
            "lstm",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "11",
            "--eval-start",
            "12",
            "--eval-end",
            "15",
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
            "--save-model",
            str(reused_lstm_source),
            "--pretty",
        )

        reuse_param_mismatch_model = temp_path / "stacking-lstm-reuse-mismatch.pkl"
        reuse_param_mismatch_output = invoke_model(
            "stacking",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "11",
            "--eval-start",
            "12",
            "--eval-end",
            "15",
            "--models",
            "ma,lstm",
            "--window",
            "3",
            "--look-back",
            "9",
            "--epochs",
            "2",
            "--lstm-target-feature",
            "销售数量",
            "--lstm-features",
            "价格指数,不存在字段,促销投入",
            "--base-model-files",
            json.dumps({"lstm": str(reused_lstm_source)}, ensure_ascii=False),
            "--save-model",
            str(reuse_param_mismatch_model),
            "--pretty",
        )
        assert reuse_param_mismatch_output["mode"] == "stacking_train_eval"
        assert reuse_param_mismatch_output["models"] == ["ma", "lstm"]
        assert len(reuse_param_mismatch_output["predictions"]) == 4
        reuse_param_mismatch_state = load_pickled_state(reuse_param_mismatch_model)
        reused_saved_lstm_member = Path(reuse_param_mismatch_state["base_model_states"]["lstm"]["model_file"])
        if not reused_saved_lstm_member.exists():
            raise AssertionError(f"reused stacking LSTM member artifact missing: {reused_saved_lstm_member}")
        assert reuse_param_mismatch_state["base_model_states"]["lstm"]["params"]["look_back"] == 4
        assert (
            reuse_param_mismatch_state["base_model_states"]["lstm"]["params"]["lstm_features"]
            == ",".join(LSTM_FEATURE_KEYS)
        )

        short_horizon_lstm_source = temp_path / "lstm-reuse-short-horizon.pkl"
        invoke_model(
            "lstm",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "27",
            "--eval-start",
            "28",
            "--eval-end",
            "29",
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
            "--save-model",
            str(short_horizon_lstm_source),
            "--pretty",
        )

        reuse_short_horizon_model = temp_path / "stacking-lstm-reuse-short-horizon.pkl"
        reuse_short_horizon_output = invoke_model(
            "stacking",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "30",
            "--eval-start",
            "31",
            "--eval-end",
            "35",
            "--models",
            "ma,lstm",
            "--window",
            "3",
            "--look-back",
            "4",
            "--epochs",
            "2",
            "--lstm-target-feature",
            "销售数量",
            "--lstm-features",
            ",".join(LSTM_FEATURE_KEYS),
            "--base-model-files",
            json.dumps({"lstm": str(short_horizon_lstm_source)}, ensure_ascii=False),
            "--save-model",
            str(reuse_short_horizon_model),
            "--pretty",
        )
        assert reuse_short_horizon_output["mode"] == "stacking_train_eval"
        assert reuse_short_horizon_output["models"] == ["ma", "lstm"]
        assert len(reuse_short_horizon_output["predictions"]) == 5
        reuse_short_horizon_state = load_pickled_state(reuse_short_horizon_model)
        reuse_short_horizon_member = Path(reuse_short_horizon_state["base_model_states"]["lstm"]["model_file"])
        if not reuse_short_horizon_member.exists():
            raise AssertionError(f"short-horizon reused stacking LSTM member artifact missing: {reuse_short_horizon_member}")
        reuse_short_horizon_member_state = load_pickled_state(reuse_short_horizon_member)
        assert reuse_short_horizon_member_state["forecast_horizon"] >= 5

    print(
        json.dumps(
            {
                "fixture": str(CSV_FIXTURE),
                "checked_modes": ["training", "fit_only", "loaded_predict"],
                "feature_count": len(LSTM_FEATURE_KEYS),
                "model": "stacking_lstm",
                "self_trained_base_models": BASE_MODELS,
                "reused_base_models": BASE_MODELS,
                "edge_cases": [
                    "missing_lstm_feature",
                    "oversized_look_back",
                    "fit_only_short_window",
                    "fit_only_missing_saved_feature",
                    "reused_lstm_source_param_mismatch",
                    "reused_lstm_source_short_horizon",
                ],
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
