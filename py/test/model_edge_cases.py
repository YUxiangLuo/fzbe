#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PY_TEST_DIR = Path(__file__).resolve().parent
PY_ROOT = PY_TEST_DIR.parent
RUN_MODEL = PY_ROOT / "run_model.py"

EXISTING_FULL_SCRIPTS = [
    "metrics_smoke.py",
    "ma_csv_smoke.py",
    "es_csv_smoke.py",
    "arima_csv_smoke.py",
    "lstm_csv_smoke.py",
    "weighted_avg_csv_smoke.py",
    "weighted_avg_lstm_csv_smoke.py",
    "stacking_csv_smoke.py",
    "stacking_lstm_csv_smoke.py",
    "boosting_csv_smoke.py",
    "boosting_lstm_csv_smoke.py",
    "lstm_feature_diagnostics.py",
    "designed_model_diagnostics.py",
    "medium_model_stress.py",
]

TARGET_KEY = "销售数量"


def build_records(month_count: int = 18) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(month_count):
        month = index % 12 + 1
        promotion = 1 if month in (6, 11, 12) else 0
        records.append(
            {
                "年份": 2024 + index // 12,
                "月份": month,
                TARGET_KEY: round(120 + 3.2 * index + 11 * promotion + ((index * 7) % 5), 2),
                "价格指数": round(100 + index * 0.2, 2),
                "促销投入": 10 + 25 * promotion,
                "渠道类型": ["直营", "电商", "经销"][index % 3],
            }
        )
    return records


def strict_json_loads(raw_stdout: str, label: str) -> dict[str, Any]:
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


def run_process(command: list[str], label: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def invoke_model(
    model_type: str,
    data_json: str | None,
    *extra_args: str,
    label: str,
    timeout: int = 300,
) -> dict[str, Any]:
    command = [sys.executable, str(RUN_MODEL), model_type]
    if data_json is not None:
        command.extend(["--data", data_json])
    command.extend(extra_args)
    command.append("--pretty")
    completed = run_process(command, label, timeout=timeout)
    payload = strict_json_loads(completed.stdout, label)
    assert_all_numbers_finite(payload)
    return payload


def expect_failure(
    model_type: str,
    data_json: str | None,
    *extra_args: str,
    label: str,
    expected_tag: str,
    timeout: int = 240,
) -> dict[str, str]:
    command = [sys.executable, str(RUN_MODEL), model_type]
    if data_json is not None:
        command.extend(["--data", data_json])
    command.extend(extra_args)
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    if completed.returncode == 0:
        raise AssertionError(f"{label} unexpectedly succeeded\nstdout:\n{completed.stdout}")
    combined = f"{completed.stdout}\n{completed.stderr}"
    if expected_tag not in combined:
        raise AssertionError(
            f"{label} missing expected tag {expected_tag!r}\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return {"label": label, "tag": expected_tag}


def run_existing_script(script_name: str) -> dict[str, Any]:
    script_path = PY_TEST_DIR / script_name
    print(f"[model-edge] existing script: {script_name}", file=sys.stderr, flush=True)
    completed = run_process([sys.executable, str(script_path)], script_name, timeout=1800)
    payload = strict_json_loads(completed.stdout, script_name)
    assert_all_numbers_finite(payload)
    return {
        "script": script_name,
        "status": payload.get("status", "ok"),
    }


def assert_prediction_payload(payload: dict[str, Any], expected_length: int, label: str) -> None:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise AssertionError(f"{label} missing predictions")
    if len(predictions) != expected_length:
        raise AssertionError(f"{label} prediction length mismatch: {len(predictions)} != {expected_length}")
    for index, point in enumerate(predictions):
        if not isinstance(point, dict):
            raise AssertionError(f"{label} prediction[{index}] must be object")
        prediction = point.get("prediction")
        std_dev = point.get("std_dev")
        if not isinstance(prediction, (int, float)) or not math.isfinite(float(prediction)):
            raise AssertionError(f"{label} prediction[{index}] is not finite: {prediction!r}")
        if not isinstance(std_dev, (int, float)) or not math.isfinite(float(std_dev)) or float(std_dev) < 0:
            raise AssertionError(f"{label} std_dev[{index}] is invalid: {std_dev!r}")


def run_contract_failure_cases(temp_path: Path) -> list[dict[str, str]]:
    records = build_records()
    data_json = json.dumps(records, ensure_ascii=False)
    bad_numeric_records = build_records()
    bad_numeric_records[3]["销售数量"] = "bad-sales"
    bad_numeric_json = json.dumps(bad_numeric_records, ensure_ascii=False)
    malformed_json = "[{bad json]"

    failures = [
        expect_failure(
            "ma",
            malformed_json,
            "--train-start",
            "0",
            "--train-end",
            "8",
            "--eval-start",
            "9",
            "--eval-end",
            "11",
            "--window",
            "3",
            label="malformed JSON is rejected",
            expected_tag="[DATA]",
        ),
        expect_failure(
            "ma",
            bad_numeric_json,
            "--train-start",
            "0",
            "--train-end",
            "8",
            "--eval-start",
            "9",
            "--eval-end",
            "11",
            "--window",
            "3",
            label="non-numeric target is rejected",
            expected_tag="[DATA]",
        ),
        expect_failure(
            "ma",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "5",
            "--eval-start",
            "6",
            "--eval-end",
            "8",
            "--window",
            "20",
            label="MA oversized window is rejected",
            expected_tag="[VALIDATION]",
        ),
        expect_failure(
            "es",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "8",
            "--eval-start",
            "9",
            "--eval-end",
            "11",
            "--alpha",
            "1.5",
            label="ES alpha above one is rejected",
            expected_tag="[VALIDATION]",
        ),
        expect_failure(
            "arima",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "8",
            "--eval-start",
            "8",
            "--eval-end",
            "10",
            "--d",
            "1",
            "--max-p",
            "1",
            "--max-q",
            "1",
            label="overlapping ARIMA train/eval windows are rejected",
            expected_tag="[VALIDATION]",
        ),
        expect_failure(
            "lstm",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "11",
            "--eval-start",
            "12",
            "--eval-end",
            "14",
            "--look-back",
            "3",
            "--epochs",
            "1",
            "--normalization",
            "minmax",
            "--lstm-target-feature",
            TARGET_KEY,
            "--lstm-features",
            "不存在的特征",
            label="LSTM missing feature is rejected",
            expected_tag="[DATA]",
        ),
        expect_failure(
            "weighted_avg",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "11",
            "--eval-start",
            "12",
            "--eval-end",
            "14",
            "--models",
            "ma",
            "--window",
            "3",
            label="weighted_avg single model is rejected",
            expected_tag="[VALIDATION]",
        ),
        expect_failure(
            "stacking",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "11",
            "--eval-start",
            "12",
            "--eval-end",
            "14",
            "--models",
            "ma,unknown",
            "--window",
            "3",
            label="stacking unknown model is rejected",
            expected_tag="[VALIDATION]",
        ),
        expect_failure(
            "boosting",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "11",
            "--eval-start",
            "12",
            "--eval-end",
            "14",
            "--models",
            "ma,es",
            "--window",
            "3",
            "--alpha",
            "0.3",
            "--learning-rate",
            "0",
            label="boosting invalid learning rate is rejected",
            expected_tag="[VALIDATION]",
        ),
    ]

    ma_model_path = temp_path / "ma-edge.pkl"
    ma_output = invoke_model(
        "ma",
        data_json,
        "--train-start",
        "0",
        "--train-end",
        "11",
        "--eval-start",
        "12",
        "--eval-end",
        "14",
        "--window",
        "3",
        "--save-model",
        str(ma_model_path),
        label="MA model for prediction failure checks",
    )
    assert_prediction_payload(ma_output, 3, "MA model for prediction failure checks")
    failures.append(
        expect_failure(
            "ma",
            None,
            "--load-model",
            str(ma_model_path),
            "--forecast-steps",
            "0",
            label="loaded prediction forecast_steps zero is rejected",
            expected_tag="[VALIDATION]",
        )
    )

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full backend Python model edge-case checks.")
    parser.add_argument("--profile", choices=["full"], default="full")
    parser.add_argument("--json", action="store_true", help="Kept for symmetry with the Bun runner.")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="fangzhen-model-edge-cases-") as temp_dir:
        temp_path = Path(temp_dir)
        existing_results = [run_existing_script(script_name) for script_name in EXISTING_FULL_SCRIPTS]
        print("[model-edge] contract failures", file=sys.stderr, flush=True)
        failure_results = run_contract_failure_cases(temp_path)

    print(
        json.dumps(
            {
                "profile": args.profile,
                "existing_scripts": existing_results,
                "contract_failures": failure_results,
                "checked_paths": [
                    "existing_python_smoke_and_diagnostics",
                    "strict_json_outputs",
                    "finite_numeric_outputs",
                    "training_validation_failures",
                    "loaded_prediction_validation_failures",
                ],
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
