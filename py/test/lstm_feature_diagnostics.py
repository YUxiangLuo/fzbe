#!/usr/bin/env python3
import json
import math
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PY_ROOT = Path(__file__).resolve().parents[1]
RUN_MODEL = PY_ROOT / "run_model.py"

MIXED_FEATURE_KEYS = [
    "价格指数",
    "促销投入",
    "库存水平",
    "搜索热度",
    "竞争对手价格",
    "节假日强度",
    "渠道类型",
    "区域",
    "天气类型",
    "门店编号",
    "是否节假日",
]
EXPECTED_MIXED_FEATURE_TYPES = {
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
    "门店编号": "categorical",
    "是否节假日": "categorical",
}

OVERRIDE_FEATURE_KEYS = ["numeric_signal", "manual_index", "segment_code"]
OVERRIDE_FEATURE_TYPES = {
    "sales": "numeric",
    "numeric_signal": "numeric",
    "manual_index": "categorical",
    "segment_code": "categorical",
}


def build_mixed_feature_records(month_count: int = 24) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    channels = ["直营", "经销", "电商"]
    regions = ["华东", "华南"]
    weather_types = ["晴", "雨"]

    for index in range(month_count):
        year = 2024 + index // 12
        month = index % 12 + 1
        promotion = 1 if month in (6, 11, 12) else 0
        season = math.sin((month - 1) / 12 * 2 * math.pi)
        channel = "直播" if index >= 18 and index % 2 == 0 else channels[index % len(channels)]
        if index == 3:
            channel = None

        records.append(
            {
                "年份": year,
                "月份": month,
                "销售数量": round(120 + 2.0 * index + 8.0 * promotion + 3.0 * season, 2),
                "价格指数": str(round(100 + 0.2 * index, 2)),
                "促销投入": 10 + 30 * promotion,
                "库存水平": 300 + index,
                "搜索热度": round(50 + 0.5 * index + 10.0 * promotion + 4.0 * season, 2),
                "竞争对手价格": round(101 + math.cos(month), 2),
                "节假日强度": promotion,
                "渠道类型": channel,
                "区域": regions[index % len(regions)],
                "天气类型": weather_types[month % len(weather_types)],
                "门店编号": 1000 + index % 4,
                "是否节假日": bool(promotion),
            }
        )

    return records


def build_override_records(month_count: int = 24) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(month_count):
        records.append(
            {
                "sales": round(90 + 1.5 * index + [0.0, 1.0, -0.5][index % 3], 2),
                "numeric_signal": str(round(20 + 0.7 * index, 2)),
                "manual_index": 2000 + index,
                "segment_code": f"{100 + index % 3}",
            }
        )
    return records


def strict_json_loads(raw_stdout: str, label: str) -> dict:
    def reject_non_standard_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-standard JSON constant: {value}")

    try:
        payload = json.loads(raw_stdout, parse_constant=reject_non_standard_constant)
    except Exception as exc:
        raise AssertionError(f"{label} emitted invalid strict JSON\nstdout:\n{raw_stdout}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"{label} JSON payload must be an object, got {type(payload).__name__}")
    return payload


def assert_all_numbers_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise AssertionError(f"{path} is not finite: {value!r}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_all_numbers_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_all_numbers_finite(item, f"{path}.{key}")
        return


def invoke_lstm(data_json: str | None, *extra_args: str, label: str, timeout: int = 300) -> dict:
    command = [sys.executable, str(RUN_MODEL), "lstm"]
    if data_json is not None:
        command.extend(["--data", data_json])
    command.extend(extra_args)
    command.append("--pretty")

    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    payload = strict_json_loads(completed.stdout, label)
    assert_all_numbers_finite(payload)
    return payload


def invoke_lstm_expect_failure(
    data_json: str | None,
    *extra_args: str,
    expected_message: str,
    label: str,
    timeout: int = 180,
) -> None:
    command = [sys.executable, str(RUN_MODEL), "lstm"]
    if data_json is not None:
        command.extend(["--data", data_json])
    command.extend(extra_args)

    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    if completed.returncode == 0:
        raise AssertionError(f"{label} unexpectedly succeeded\nstdout:\n{completed.stdout}")
    if expected_message not in completed.stderr:
        raise AssertionError(
            f"{label} stderr missing {expected_message!r}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def extract_predictions(payload: dict) -> list[float]:
    return [float(item["prediction"]) for item in payload["predictions"]]


def extract_std_devs(payload: dict) -> list[float]:
    return [float(item["std_dev"]) for item in payload["predictions"]]


def assert_prediction_payload(payload: dict, expected_length: int, label: str) -> None:
    predictions = extract_predictions(payload)
    std_devs = extract_std_devs(payload)
    if len(predictions) != expected_length:
        raise AssertionError(f"{label} prediction length mismatch: {len(predictions)} != {expected_length}")
    if len(std_devs) != expected_length:
        raise AssertionError(f"{label} std_dev length mismatch: {len(std_devs)} != {expected_length}")
    if any(value <= 0 or not math.isfinite(value) for value in predictions):
        raise AssertionError(f"{label} expected positive finite predictions, got {predictions}")
    if any(value < 0 or not math.isfinite(value) for value in std_devs):
        raise AssertionError(f"{label} expected finite non-negative std_dev values, got {std_devs}")


def assert_series_close(actual: list[float], expected: list[float], label: str, tol: float = 1e-9) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"{label} length mismatch: actual={len(actual)}, expected={len(expected)}")
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
        if not math.isclose(actual_value, expected_value, rel_tol=tol, abs_tol=tol):
            raise AssertionError(
                f"{label}[{index}] mismatch: actual={actual_value}, expected={expected_value}"
            )


def load_pickled_state(model_path: Path) -> dict:
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def run_mixed_feature_training(temp_path: Path) -> dict:
    records = build_mixed_feature_records()
    data_json = json.dumps(records, ensure_ascii=False)
    model_path = temp_path / "mixed-lstm.pkl"

    output = invoke_lstm(
        data_json,
        "--train-start",
        "0",
        "--train-end",
        "17",
        "--eval-start",
        "18",
        "--eval-end",
        "23",
        "--look-back",
        "4",
        "--epochs",
        "2",
        "--normalization",
        "minmax",
        "--lstm-target-feature",
        "销售数量",
        "--lstm-features",
        ",".join(MIXED_FEATURE_KEYS),
        "--save-model",
        str(model_path),
        label="mixed feature LSTM training",
    )

    if output["feature_types"] != EXPECTED_MIXED_FEATURE_TYPES:
        raise AssertionError(
            "mixed feature_types mismatch:\n"
            f"actual={json.dumps(output['feature_types'], ensure_ascii=False, indent=2)}\n"
            f"expected={json.dumps(EXPECTED_MIXED_FEATURE_TYPES, ensure_ascii=False, indent=2)}"
        )
    if output["forecast_horizon"] != 6:
        raise AssertionError(f"expected direct forecast_horizon=6, got {output['forecast_horizon']}")
    assert_prediction_payload(output, 6, "mixed feature train/eval")

    state = load_pickled_state(model_path)
    if state["feature_keys"] != MIXED_FEATURE_KEYS:
        raise AssertionError(f"saved feature_keys mismatch: {state['feature_keys']}")
    if state["feature_types"] != EXPECTED_MIXED_FEATURE_TYPES:
        raise AssertionError(f"saved feature_types mismatch: {state['feature_types']}")
    if state["forecast_strategy"] != "direct_multi_horizon":
        raise AssertionError(f"unexpected forecast strategy: {state['forecast_strategy']}")
    keras_path = model_path.with_name(f"{model_path.stem}_keras_model.h5")
    if not keras_path.exists():
        raise AssertionError(f"missing Keras artifact: {keras_path}")

    predict_one = invoke_lstm(
        None,
        "--load-model",
        str(model_path),
        "--forecast-steps",
        "6",
        label="mixed feature loaded predict one",
    )
    predict_two = invoke_lstm(
        None,
        "--load-model",
        str(model_path),
        "--forecast-steps",
        "6",
        label="mixed feature loaded predict two",
    )
    assert_prediction_payload(predict_one, 6, "mixed feature loaded predict")
    assert_series_close(
        extract_predictions(predict_one),
        extract_predictions(predict_two),
        "mixed feature loaded predictions",
    )
    assert_series_close(
        extract_std_devs(predict_one),
        extract_std_devs(predict_two),
        "mixed feature loaded std_devs",
    )

    invoke_lstm_expect_failure(
        None,
        "--load-model",
        str(model_path),
        "--forecast-steps",
        "7",
        expected_message="[VALIDATION] Requested forecast_steps (7) exceeds the saved LSTM horizon (6).",
        label="direct LSTM horizon guard",
    )

    return output


def run_zscore_and_override_training() -> dict:
    records = build_override_records()
    data_json = json.dumps(records, ensure_ascii=False)

    output = invoke_lstm(
        data_json,
        "--train-start",
        "0",
        "--train-end",
        "17",
        "--eval-start",
        "18",
        "--eval-end",
        "23",
        "--look-back",
        "4",
        "--epochs",
        "2",
        "--normalization",
        "zscore",
        "--lstm-target-feature",
        "sales",
        "--lstm-features",
        ",".join(OVERRIDE_FEATURE_KEYS),
        "--feature-types",
        json.dumps(
            {
                "manual_index": "categorical",
                "segment_code": "categorical",
            },
            ensure_ascii=False,
        ),
        label="zscore override LSTM training",
    )
    if output["feature_types"] != OVERRIDE_FEATURE_TYPES:
        raise AssertionError(f"override feature_types mismatch: {output['feature_types']}")
    if output["normalization"] != "zscore":
        raise AssertionError(f"expected zscore normalization, got {output['normalization']}")
    assert_prediction_payload(output, 6, "zscore override train/eval")
    return output


def run_bad_numeric_feature_guards() -> None:
    missing_numeric_records = build_mixed_feature_records()
    missing_numeric_records[5]["价格指数"] = None
    missing_numeric_json = json.dumps(missing_numeric_records, ensure_ascii=False)
    common_args = (
        "--train-start",
        "0",
        "--train-end",
        "17",
        "--eval-start",
        "18",
        "--eval-end",
        "23",
        "--look-back",
        "4",
        "--epochs",
        "1",
        "--normalization",
        "minmax",
        "--lstm-target-feature",
        "销售数量",
        "--lstm-features",
        ",".join(MIXED_FEATURE_KEYS),
    )
    invoke_lstm_expect_failure(
        missing_numeric_json,
        *common_args,
        expected_message="[DATA] Feature column '价格指数' contains non-finite values.",
        label="missing numeric feature guard",
    )

    bad_override_records = build_override_records()
    bad_override_records[4]["numeric_signal"] = "bad-value"
    bad_override_json = json.dumps(bad_override_records, ensure_ascii=False)
    invoke_lstm_expect_failure(
        bad_override_json,
        "--train-start",
        "0",
        "--train-end",
        "17",
        "--eval-start",
        "18",
        "--eval-end",
        "23",
        "--look-back",
        "4",
        "--epochs",
        "1",
        "--normalization",
        "minmax",
        "--lstm-target-feature",
        "sales",
        "--lstm-features",
        ",".join(OVERRIDE_FEATURE_KEYS),
        "--feature-types",
        json.dumps(OVERRIDE_FEATURE_TYPES, ensure_ascii=False),
        expected_message="[DATA] LSTM numeric columns contain missing, non-numeric, or non-finite values",
        label="bad numeric override guard",
    )

    valid_override_json = json.dumps(build_override_records(), ensure_ascii=False)
    invoke_lstm_expect_failure(
        valid_override_json,
        "--train-start",
        "0",
        "--train-end",
        "17",
        "--eval-start",
        "18",
        "--eval-end",
        "23",
        "--look-back",
        "4",
        "--epochs",
        "1",
        "--normalization",
        "minmax",
        "--lstm-target-feature",
        "sales",
        "--lstm-features",
        ",".join(OVERRIDE_FEATURE_KEYS),
        "--feature-types",
        json.dumps({"manual_index": "ordinal"}, ensure_ascii=False),
        expected_message="[VALIDATION] LSTM feature_types values must be 'numeric' or 'categorical'",
        label="invalid feature type guard",
    )
    invoke_lstm_expect_failure(
        valid_override_json,
        "--train-start",
        "0",
        "--train-end",
        "17",
        "--eval-start",
        "18",
        "--eval-end",
        "23",
        "--look-back",
        "4",
        "--epochs",
        "1",
        "--normalization",
        "minmax",
        "--lstm-target-feature",
        "sales",
        "--lstm-features",
        ",".join(OVERRIDE_FEATURE_KEYS),
        "--feature-types",
        json.dumps({"sales": "categorical"}, ensure_ascii=False),
        expected_message="[VALIDATION] LSTM target feature must be numeric.",
        label="categorical target type guard",
    )


def summarize(output: dict) -> dict[str, Any]:
    return {
        "normalization": output["normalization"],
        "forecast_horizon": output["forecast_horizon"],
        "feature_types": output["feature_types"],
        "rmse": round(float(output["metrics"]["rmse"]), 6),
        "mape": round(float(output["metrics"]["mape"]), 6),
        "predictions": [round(value, 6) for value in extract_predictions(output)],
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fangzhen-lstm-feature-diagnostics-") as temp_dir:
        temp_path = Path(temp_dir)
        print("[lstm-feature-diagnostics] mixed features", file=sys.stderr, flush=True)
        mixed_output = run_mixed_feature_training(temp_path)

        print("[lstm-feature-diagnostics] zscore override", file=sys.stderr, flush=True)
        override_output = run_zscore_and_override_training()

        print("[lstm-feature-diagnostics] bad numeric guards", file=sys.stderr, flush=True)
        run_bad_numeric_feature_guards()

    print(
        json.dumps(
            {
                "scenarios": {
                    "mixed_numeric_string_categorical_bool_unseen_eval": summarize(mixed_output),
                    "zscore_manual_feature_type_override": summarize(override_output),
                    "bad_numeric_feature_guards": "ok",
                },
                "checked_paths": [
                    "auto_feature_type_inference",
                    "manual_feature_type_override",
                    "unseen_eval_categories",
                    "missing_categorical_values",
                    "minmax_training",
                    "zscore_training",
                    "save_load_repeatability",
                    "direct_horizon_validation",
                    "bad_numeric_feature_rejection",
                ],
                "model": "lstm",
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
