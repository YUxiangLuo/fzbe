#!/usr/bin/env python3
"""Regression coverage for guided/standalone parity of the four base models."""

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PY_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
RUN_MODEL = PY_ROOT / "run_model.py"
GUIDED_MODEL = PY_ROOT / "guided_model.py"

CORE_FIXTURE = TEST_ROOT / "fixtures" / "core_models_smoke.csv"
LSTM_FIXTURE = TEST_ROOT / "fixtures" / "lstm_rich_smoke.csv"
LSTM_FEATURES = [
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

STEP_SEQUENCES = {
    "ma": [
        "prepare_data",
        "build_windows",
        "forecast",
        "estimate_residuals",
        "evaluate_metrics",
        "save_artifact",
    ],
    "es": [
        "prepare_data",
        "fit_model",
        "forecast",
        "estimate_residuals",
        "evaluate_metrics",
        "save_artifact",
    ],
    "arima": [
        "prepare_data",
        "aic_search",
        "bic_search",
        "select_model",
        "forecast",
        "evaluate_metrics",
        "save_artifact",
    ],
    "lstm": [
        "prepare_data",
        "preprocess",
        "build_model",
        "train_model",
        "forecast",
        "evaluate_metrics",
        "save_artifact",
    ],
}

MODEL_CASES = {
    "ma": {
        "fixture": CORE_FIXTURE,
        "train": (0, 19),
        "evaluate": (22, 23),
        "params": ["--window", "3"],
    },
    # Deliberately omit --alpha: both paths must use the public default, 0.5.
    "es": {
        "fixture": CORE_FIXTURE,
        "train": (0, 19),
        "evaluate": (22, 23),
        "params": [],
    },
    "arima": {
        "fixture": CORE_FIXTURE,
        "train": (0, 19),
        "evaluate": (22, 23),
        "params": ["--d", "1", "--max-p", "2", "--max-q", "2"],
    },
    "lstm": {
        "fixture": LSTM_FIXTURE,
        "train": (0, 27),
        "evaluate": (30, 32),
        "params": [
            "--look-back",
            "4",
            "--epochs",
            "2",
            "--normalization",
            "minmax",
            "--lstm-target-feature",
            "销售数量",
            "--lstm-features",
            json.dumps(LSTM_FEATURES, ensure_ascii=False),
        ],
    },
}


def fixture_json(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return json.dumps(list(csv.DictReader(handle)), ensure_ascii=False)


def run_process(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(PY_ROOT),
        env=env,
        timeout=600,
        check=False,
    )


def run_json(command: list[str], label: str) -> dict[str, Any]:
    completed = run_process(command)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{label} emitted invalid JSON:\n{completed.stdout[-2000:]}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"{label} must emit a JSON object")
    return payload


def common_training_args(data: str, case: dict[str, Any], save_model: Path) -> list[str]:
    train_start, train_end = case["train"]
    eval_start, eval_end = case["evaluate"]
    return [
        "--data",
        data,
        "--train-start",
        str(train_start),
        "--train-end",
        str(train_end),
        "--eval-start",
        str(eval_start),
        "--eval-end",
        str(eval_end),
        "--save-model",
        str(save_model),
        *case["params"],
    ]


def train_standalone(model_type: str, data: str, case: dict[str, Any], save_model: Path) -> dict[str, Any]:
    return run_json(
        [
            sys.executable,
            str(RUN_MODEL),
            model_type,
            *common_training_args(data, case, save_model),
        ],
        f"standalone {model_type}",
    )


def train_guided(
    model_type: str,
    data: str,
    case: dict[str, Any],
    save_model: Path,
    temp_path: Path,
) -> dict[str, Any]:
    state_file = temp_path / f"guided-{model_type}-state.pkl"
    artifact_dir = temp_path / f"guided-{model_type}-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    common_args = [
        "--state-file",
        str(state_file),
        "--artifact-dir",
        str(artifact_dir),
        *common_training_args(data, case, save_model),
    ]

    final_result = None
    for step_id in STEP_SEQUENCES[model_type]:
        payload = run_json(
            [sys.executable, str(GUIDED_MODEL), model_type, step_id, *common_args],
            f"guided {model_type}/{step_id}",
        )
        if "final_result" in payload:
            final_result = payload["final_result"]
    if not isinstance(final_result, dict):
        raise AssertionError(f"guided {model_type} produced no final result")
    return final_result


def assert_equal(left: Any, right: Any, label: str) -> None:
    if left != right:
        raise AssertionError(f"{label} differs: standalone={left!r}, guided={right!r}")


def sequence_max_delta(left: list[Any], right: list[Any], label: str) -> float:
    if len(left) != len(right):
        raise AssertionError(f"{label} length differs: {len(left)} != {len(right)}")
    delta = 0.0
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        left_number = float(left_value)
        right_number = float(right_value)
        if not math.isfinite(left_number) or not math.isfinite(right_number):
            raise AssertionError(f"{label}[{index}] is not finite")
        delta = max(delta, abs(left_number - right_number))
    return delta


def compare_results(model_type: str, standalone: dict[str, Any], guided: dict[str, Any]) -> dict[str, float]:
    tolerance = 1e-6 if model_type == "lstm" else 1e-9
    for key in ("mode", "data_points", "method_name", "forecast_strategy", "implementation_notes"):
        assert_equal(standalone.get(key), guided.get(key), f"{model_type}.{key}")

    metric_keys = ("rmse", "mae", "mape", "r2")
    metric_delta = sequence_max_delta(
        [standalone["metrics"][key] for key in metric_keys],
        [guided["metrics"][key] for key in metric_keys],
        f"{model_type}.metrics",
    )
    truth_delta = sequence_max_delta(
        standalone["eval_y_true"], guided["eval_y_true"], f"{model_type}.eval_y_true"
    )
    prediction_delta = sequence_max_delta(
        [item["prediction"] for item in standalone["predictions"]],
        [item["prediction"] for item in guided["predictions"]],
        f"{model_type}.predictions",
    )
    uncertainty_delta = sequence_max_delta(
        [item["std_dev"] for item in standalone["predictions"]],
        [item["std_dev"] for item in guided["predictions"]],
        f"{model_type}.std_dev",
    )

    if model_type == "es":
        assert_equal(standalone.get("alpha"), 0.5, "standalone ES default alpha")
        assert_equal(guided.get("alpha"), 0.5, "guided ES default alpha")
    elif model_type == "arima":
        assert_equal(standalone["best_order"], guided["best_order"], "arima.best_order")
        for key in ("selected_model", "selection_criterion"):
            assert_equal(
                standalone["search_summary"][key],
                guided["search_summary"][key],
                f"arima.search_summary.{key}",
            )
    elif model_type == "lstm":
        for key in ("feature_types", "look_back", "epochs", "normalization", "forecast_horizon"):
            assert_equal(standalone.get(key), guided.get(key), f"lstm.{key}")

    deltas = {
        "metrics": metric_delta,
        "eval_y_true": truth_delta,
        "predictions": prediction_delta,
        "std_dev": uncertainty_delta,
    }
    for label, delta in deltas.items():
        if delta > tolerance:
            raise AssertionError(
                f"{model_type}.{label} delta {delta} exceeds tolerance {tolerance}"
            )
    return deltas


def compare_loaded_predictions(
    model_type: str, standalone_model: Path, guided_model: Path
) -> dict[str, float]:
    outputs = []
    for label, model_path in (("standalone", standalone_model), ("guided", guided_model)):
        outputs.append(
            run_json(
                [
                    sys.executable,
                    str(RUN_MODEL),
                    model_type,
                    "--load-model",
                    str(model_path),
                    "--forecast-steps",
                    "3",
                ],
                f"{label} {model_type} loaded prediction",
            )
        )
    standalone, guided = outputs
    for key in ("method_name", "forecast_strategy", "implementation_notes"):
        assert_equal(standalone.get(key), guided.get(key), f"loaded {model_type}.{key}")
    deltas = {
        "predictions": sequence_max_delta(
            [item["prediction"] for item in standalone["predictions"]],
            [item["prediction"] for item in guided["predictions"]],
            f"loaded {model_type}.predictions",
        ),
        "std_dev": sequence_max_delta(
            [item["std_dev"] for item in standalone["predictions"]],
            [item["std_dev"] for item in guided["predictions"]],
            f"loaded {model_type}.std_dev",
        ),
    }
    tolerance = 1e-6 if model_type == "lstm" else 1e-9
    for label, delta in deltas.items():
        if delta > tolerance:
            raise AssertionError(
                f"loaded {model_type}.{label} delta {delta} exceeds tolerance {tolerance}"
            )
    return deltas


def assert_arima_validation_parity(data: str, temp_path: Path) -> list[str]:
    checks = [
        (["--d", "-1", "--max-p", "2", "--max-q", "2"], "d must be >= 0, got -1"),
        (["--d", "1", "--max-p", "-1", "--max-q", "2"], "max_p and max_q must be >= 0"),
        (["--d", "1", "--max-p", "2", "--max-q", "-1"], "max_p and max_q must be >= 0"),
    ]
    completed_checks = []
    base_args = [
        "--data",
        data,
        "--train-start",
        "0",
        "--train-end",
        "19",
        "--eval-start",
        "22",
        "--eval-end",
        "23",
    ]
    for index, (params, expected_message) in enumerate(checks):
        standalone = run_process(
            [sys.executable, str(RUN_MODEL), "arima", *base_args, *params]
        )
        guided_artifact_dir = temp_path / f"invalid-arima-{index}-artifacts"
        guided = run_process(
            [
                sys.executable,
                str(GUIDED_MODEL),
                "arima",
                "prepare_data",
                "--state-file",
                str(temp_path / f"invalid-arima-{index}-state.pkl"),
                "--artifact-dir",
                str(guided_artifact_dir),
                *base_args,
                *params,
            ]
        )
        for label, completed in (("standalone", standalone), ("guided", guided)):
            if completed.returncode == 0:
                raise AssertionError(f"{label} ARIMA accepted invalid parameters: {params}")
            if expected_message not in completed.stderr:
                raise AssertionError(
                    f"{label} ARIMA error did not contain {expected_message!r}:\n"
                    f"{completed.stderr[-1500:]}"
                )
        completed_checks.append(" ".join(params))
    return completed_checks


def main() -> None:
    fixture_cache = {
        CORE_FIXTURE: fixture_json(CORE_FIXTURE),
        LSTM_FIXTURE: fixture_json(LSTM_FIXTURE),
    }
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="fangzhen-guided-parity-") as temp_dir:
        temp_path = Path(temp_dir)
        for model_type, case in MODEL_CASES.items():
            data = fixture_cache[case["fixture"]]
            standalone_model = temp_path / f"standalone-{model_type}.pkl"
            guided_model = temp_path / f"guided-{model_type}.pkl"
            standalone_result = train_standalone(model_type, data, case, standalone_model)
            guided_result = train_guided(model_type, data, case, guided_model, temp_path)
            results[model_type] = {
                "training_max_abs_delta": compare_results(
                    model_type, standalone_result, guided_result
                ),
                "loaded_max_abs_delta": compare_loaded_predictions(
                    model_type, standalone_model, guided_model
                ),
            }

        validation_checks = assert_arima_validation_parity(
            fixture_cache[CORE_FIXTURE], temp_path
        )

    print(
        json.dumps(
            {
                "models": results,
                "arima_validation_checks": validation_checks,
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
