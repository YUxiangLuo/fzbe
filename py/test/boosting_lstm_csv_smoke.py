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


def load_csv_records(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_model_command(model_type: str, *args: str, data_json: str | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(RUN_MODEL), model_type]
    if data_json is not None:
        command.extend(["--data", data_json])
    command.extend(args)
    return subprocess.run(command, capture_output=True, text=True, check=False)


def run_command(*args: str, data_json: str | None = None) -> subprocess.CompletedProcess[str]:
    return run_model_command("boosting", *args, data_json=data_json)


def invoke_model(data_json: str | None, *extra_args: str) -> dict:
    completed = run_command(*extra_args, data_json=data_json)
    if completed.returncode != 0:
        raise RuntimeError(
            "boosting CLI execution failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def invoke_base_model(model_type: str, data_json: str | None, *extra_args: str) -> dict:
    completed = run_model_command(model_type, *extra_args, data_json=data_json)
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
        predictions = extract_predictions(
            subprocess_payload_to_json(
                run_member_command(model_name, model_state["model_file"], forecast_steps)
            )
        )
        weighted_predictions = [coefficient * prediction for prediction in predictions]
        if combined is None:
            combined = weighted_predictions
        else:
            combined = [left + right for left, right in zip(combined, weighted_predictions)]

    if combined is None:
        raise AssertionError("boosting state produced no stages")
    return combined


def run_member_command(model_type: str, model_file: str, forecast_steps: int) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(RUN_MODEL),
        model_type,
        "--load-model",
        model_file,
        "--forecast-steps",
        str(forecast_steps),
        "--pretty",
    ]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def subprocess_payload_to_json(completed: subprocess.CompletedProcess[str]) -> dict:
    if completed.returncode != 0:
        raise RuntimeError(
            "base model CLI execution failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


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

    with tempfile.TemporaryDirectory(prefix="fangzhen-boosting-lstm-csv-") as temp_dir:
        temp_path = Path(temp_dir)
        backtest_model = temp_path / "boosting-lstm-backtest.pkl"
        production_model = temp_path / "boosting-lstm-production.pkl"
        short_horizon_lstm_source = temp_path / "boosting-short-source-lstm.pkl"

        invoke_base_model(
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

        train_output = invoke_model(
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
            "lstm",
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
            "--max-rounds",
            "1",
            "--save-model",
            str(backtest_model),
            "--pretty",
        )

        train_state = load_pickled_state(backtest_model)
        actual_eval_predictions = extract_predictions(train_output)
        actual_eval_std_devs = extract_std_devs(train_output)
        expected_eval_predictions = combine_stage_predictions(train_state, forecast_steps=3)
        assert train_output["mode"] == "boosting_residual_learning"
        assert train_output["models"] == ["lstm"]
        assert train_output["model_chain"] == ["lstm"]
        assert train_output["stage_coefficients"] == [1.0]
        assert train_output["metrics"]["evaluate_months"] == timestamps[28:31]
        assert train_output["eval_y_true"] == sales[28:31]
        assert_series_close(actual_eval_predictions, expected_eval_predictions)
        assert_all_finite(actual_eval_predictions, "train_predictions")
        assert_all_finite(actual_eval_std_devs, "train_std_devs")

        saved_lstm_member = Path(train_state["base_model_states"]["lstm"]["models"][0]["model_file"])
        if not saved_lstm_member.exists():
            raise AssertionError(f"saved boosting LSTM member artifact missing: {saved_lstm_member}")
        saved_lstm_state = load_pickled_state(saved_lstm_member)
        assert saved_lstm_state["forecast_strategy"] == "direct_multi_horizon"
        assert saved_lstm_state["forecast_horizon"] == 3
        saved_lstm_keras = saved_lstm_member.with_name(f"{saved_lstm_member.stem}_keras_model.h5")
        if not saved_lstm_keras.exists():
            raise AssertionError(f"saved boosting LSTM keras artifact missing: {saved_lstm_keras}")

        fit_output = invoke_model(
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
        assert fit_output["mode"] == "boosting_fit_only"
        assert fit_output["model_chain"] == ["lstm"]
        assert fit_output["saved_model"] == str(production_model)

        production_state = load_pickled_state(production_model)
        production_lstm_member = Path(production_state["base_model_states"]["lstm"]["models"][0]["model_file"])
        if not production_lstm_member.exists():
            raise AssertionError(f"production boosting LSTM member artifact missing: {production_lstm_member}")
        production_lstm_state = load_pickled_state(production_lstm_member)
        assert production_lstm_state["forecast_strategy"] == "direct_multi_horizon"
        assert production_lstm_state["forecast_horizon"] == future_steps

        predict_output_one = invoke_model(
            None,
            "--load-model",
            str(production_model),
            "--forecast-steps",
            str(future_steps),
            "--pretty",
        )
        predict_output_two = invoke_model(
            None,
            "--load-model",
            str(production_model),
            "--forecast-steps",
            str(future_steps),
            "--pretty",
        )
        expected_future_predictions = combine_stage_predictions(production_state, forecast_steps=future_steps)
        future_predictions_one = extract_predictions(predict_output_one)
        future_predictions_two = extract_predictions(predict_output_two)
        future_std_devs_one = extract_std_devs(predict_output_one)
        future_std_devs_two = extract_std_devs(predict_output_two)
        assert predict_output_one["mode"] == "boosting_loaded_predict"
        assert predict_output_one["forecast_steps"] == future_steps
        assert_series_close(future_predictions_one, expected_future_predictions)
        assert_series_close(future_predictions_one, future_predictions_two)
        assert_series_close(future_std_devs_one, future_std_devs_two)
        assert_all_finite(future_predictions_one, "future_predictions")
        assert_all_finite(future_std_devs_one, "future_std_devs")

        missing_feature_result = run_command(
            "--train-start",
            str(train_start),
            "--train-end",
            str(train_end),
            "--eval-start",
            str(eval_start),
            "--eval-end",
            str(eval_end),
            "--models",
            "lstm",
            "--look-back",
            "4",
            "--epochs",
            "2",
            "--lstm-target-feature",
            "销售数量",
            "--lstm-features",
            "价格指数,不存在字段,促销投入",
            data_json=data_json,
        )
        assert_error(
            missing_feature_result,
            "[VALIDATION] LSTM feature columns not found in dataset: ['不存在字段']",
        )

        oversized_look_back_result = run_command(
            "--train-start",
            "0",
            "--train-end",
            "11",
            "--eval-start",
            "12",
            "--eval-end",
            "15",
            "--models",
            "lstm",
            "--look-back",
            "9",
            "--epochs",
            "2",
            "--lstm-target-feature",
            "销售数量",
            "--lstm-features",
            "价格指数,促销投入,库存水平,搜索热度",
            data_json=data_json,
        )
        assert_error(
            oversized_look_back_result,
            "[VALIDATION] LSTM look_back (9) is too large for the boosting internal training split (10 points) and validation horizon (2 points).",
        )

        short_fit_result = run_command(
            "--train-start",
            "0",
            "--train-end",
            "3",
            "--load-model",
            str(backtest_model),
            "--save-model",
            str(temp_path / "boosting-lstm-short-fit.pkl"),
            "--fit-only",
            "--pretty",
            data_json=data_json,
        )
        assert_error(
            short_fit_result,
            "[VALIDATION] Boosting fit-only stage 1 (lstm) cannot be refit: Training range must be larger than look_back.",
        )

        missing_feature_fit_result = run_command(
            "--train-start",
            "0",
            "--train-end",
            "35",
            "--load-model",
            str(backtest_model),
            "--save-model",
            str(temp_path / "boosting-lstm-missing-feature-fit.pkl"),
            "--fit-only",
            "--pretty",
            data_json=trimmed_feature_data_json,
        )
        assert_error(
            missing_feature_fit_result,
            "[VALIDATION] Boosting fit-only stage 1 (lstm) is missing feature columns: ['库存水平']",
        )

        reused_short_source_model = temp_path / "boosting-lstm-reuse-short-source.pkl"
        reused_short_source_output = invoke_model(
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
            "lstm",
            "--look-back",
            "9",
            "--epochs",
            "2",
            "--normalization",
            "minmax",
            "--lstm-target-feature",
            "销售数量",
            "--lstm-features",
            "价格指数",
            "--max-rounds",
            "1",
            "--base-model-files",
            json.dumps({"lstm": str(short_horizon_lstm_source)}, ensure_ascii=False),
            "--save-model",
            str(reused_short_source_model),
            "--pretty",
        )
        reused_short_source_state = load_pickled_state(reused_short_source_model)
        reused_short_source_predictions = extract_predictions(reused_short_source_output)
        assert reused_short_source_output["mode"] == "boosting_residual_learning"
        assert reused_short_source_output["model_chain"] == ["lstm"]
        assert len(reused_short_source_predictions) == 5
        assert_all_finite(reused_short_source_predictions, "reused_short_source_predictions")
        reused_short_source_member = Path(
            reused_short_source_state["base_model_states"]["lstm"]["models"][0]["model_file"]
        )
        reused_short_source_member_state = load_pickled_state(reused_short_source_member)
        assert reused_short_source_member_state["look_back"] == 4
        assert reused_short_source_member_state["forecast_horizon"] == future_steps

    print(
        json.dumps(
            {
                "fixture": str(CSV_FIXTURE),
                "checked_modes": ["training", "fit_only", "loaded_predict"],
                "model": "boosting",
                "edge_cases": [
                    "missing_lstm_feature",
                    "oversized_look_back",
                    "fit_only_short_window",
                    "fit_only_missing_saved_feature",
                    "reused_short_horizon_source",
                ],
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
