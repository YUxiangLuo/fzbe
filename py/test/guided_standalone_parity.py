#!/usr/bin/env python3
"""End-to-end parity checks for guided and standalone base-model training."""

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
        "prepare_data", "build_windows", "forecast", "estimate_residuals",
        "evaluate_metrics", "save_artifact",
    ],
    "es": [
        "prepare_data", "fit_model", "forecast", "estimate_residuals",
        "evaluate_metrics", "save_artifact",
    ],
    "arima": [
        "prepare_data", "aic_search", "bic_search", "select_model", "forecast",
        "evaluate_metrics", "save_artifact",
    ],
    "lstm": [
        "prepare_data", "preprocess", "build_model", "train_model", "forecast",
        "evaluate_metrics", "save_artifact",
    ],
}

CASES = {
    "ma": {
        "fixture": CORE_FIXTURE,
        "train": (0, 19),
        "evaluate": (22, 23),
        "params": ["--window", "3"],
    },
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
            "--look-back", "4",
            "--epochs", "2",
            "--normalization", "minmax",
            "--lstm-target-feature", "销售数量",
            "--lstm-features", json.dumps(LSTM_FEATURES, ensure_ascii=False),
        ],
    },
}


def load_fixture(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return json.dumps(list(csv.DictReader(handle)), ensure_ascii=False)


def run_json(command: list[str], label: str) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(PY_ROOT),
        env=env,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed\nstdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}"
        )
    return json.loads(completed.stdout)


def training_args(data: str, case: dict[str, Any], save_model: Path) -> list[str]:
    train_start, train_end = case["train"]
    eval_start, eval_end = case["evaluate"]
    return [
        "--data", data,
        "--train-start", str(train_start),
        "--train-end", str(train_end),
        "--eval-start", str(eval_start),
        "--eval-end", str(eval_end),
        "--save-model", str(save_model),
        *case["params"],
    ]


def train_guided(
    model_type: str,
    data: str,
    case: dict[str, Any],
    save_model: Path,
    temp_path: Path,
) -> dict[str, Any]:
    artifact_dir = temp_path / f"guided-{model_type}-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "--state-file", str(temp_path / f"guided-{model_type}-state.pkl"),
        "--artifact-dir", str(artifact_dir),
        *training_args(data, case, save_model),
    ]
    final_result = None
    for step_id in STEP_SEQUENCES[model_type]:
        payload = run_json(
            [sys.executable, str(GUIDED_MODEL), model_type, step_id, *args],
            f"guided {model_type}/{step_id}",
        )
        final_result = payload.get("final_result", final_result)
    if not isinstance(final_result, dict):
        raise AssertionError(f"guided {model_type} produced no final result")
    return final_result


def maximum_delta(left: list[Any], right: list[Any], label: str) -> float:
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
    for key in ("mode", "method_name", "forecast_strategy", "implementation_notes", "eval_y_true"):
        if standalone.get(key) != guided.get(key):
            raise AssertionError(f"{model_type}.{key} differs")
    if standalone["metrics"].get("evaluate_indices") != guided["metrics"].get("evaluate_indices"):
        raise AssertionError(f"{model_type}.evaluate_indices differs")
    if standalone["metrics"].get("evaluate_months") != guided["metrics"].get("evaluate_months"):
        raise AssertionError(f"{model_type}.evaluate_months differs")

    deltas = {
        "metrics": maximum_delta(
            [standalone["metrics"][key] for key in ("rmse", "mae", "mape", "r2")],
            [guided["metrics"][key] for key in ("rmse", "mae", "mape", "r2")],
            f"{model_type}.metrics",
        ),
        "predictions": maximum_delta(
            [point["prediction"] for point in standalone["predictions"]],
            [point["prediction"] for point in guided["predictions"]],
            f"{model_type}.predictions",
        ),
        "std_dev": maximum_delta(
            [point["std_dev"] for point in standalone["predictions"]],
            [point["std_dev"] for point in guided["predictions"]],
            f"{model_type}.std_dev",
        ),
    }
    tolerance = 1e-6 if model_type == "lstm" else 1e-9
    for label, delta in deltas.items():
        if delta > tolerance:
            raise AssertionError(f"{model_type}.{label} delta {delta} exceeds {tolerance}")
    return deltas


def compare_loaded_predictions(model_type: str, standalone_model: Path, guided_model: Path) -> float:
    results = [
        run_json(
            [
                sys.executable, str(RUN_MODEL), model_type, "--load-model", str(model_path),
                "--forecast-steps", "3",
            ],
            f"{label} {model_type} loaded prediction",
        )
        for label, model_path in (("standalone", standalone_model), ("guided", guided_model))
    ]
    return maximum_delta(
        [point["prediction"] for point in results[0]["predictions"]],
        [point["prediction"] for point in results[1]["predictions"]],
        f"loaded {model_type}.predictions",
    )


def main() -> None:
    fixture_cache = {path: load_fixture(path) for path in (CORE_FIXTURE, LSTM_FIXTURE)}
    summary = {}
    with tempfile.TemporaryDirectory(prefix="fangzhen-guided-parity-") as temp_dir:
        temp_path = Path(temp_dir)
        for model_type, case in CASES.items():
            data = fixture_cache[case["fixture"]]
            standalone_model = temp_path / f"standalone-{model_type}.pkl"
            guided_model = temp_path / f"guided-{model_type}.pkl"
            standalone = run_json(
                [sys.executable, str(RUN_MODEL), model_type, *training_args(data, case, standalone_model)],
                f"standalone {model_type}",
            )
            guided = train_guided(model_type, data, case, guided_model, temp_path)
            deltas = compare_results(model_type, standalone, guided)
            loaded_delta = compare_loaded_predictions(model_type, standalone_model, guided_model)
            tolerance = 1e-6 if model_type == "lstm" else 1e-9
            if loaded_delta > tolerance:
                raise AssertionError(
                    f"loaded {model_type}.predictions delta {loaded_delta} exceeds {tolerance}"
                )
            summary[model_type] = {"training": deltas, "loaded_predictions": loaded_delta}

    print(json.dumps({"models": summary, "status": "ok"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
