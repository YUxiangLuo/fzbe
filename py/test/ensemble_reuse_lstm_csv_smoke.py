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
BASE_MODELS = ["ma", "es", "arima", "lstm"]
ENSEMBLE_MODELS = ["weighted_avg", "boosting", "stacking"]
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


def verify_base_model_artifacts(base_model_paths: dict[str, Path], forecast_steps: int) -> None:
    for model_name, model_path in base_model_paths.items():
        if not model_path.exists():
            raise AssertionError(f"missing base model artifact: {model_path}")

        if model_name == "lstm":
            keras_path = model_path.with_name(f"{model_path.stem}_keras_model.h5")
            if not keras_path.exists():
                raise AssertionError(f"missing base LSTM keras artifact: {keras_path}")

        predict_output = invoke_model(
            model_name,
            None,
            "--load-model",
            str(model_path),
            "--forecast-steps",
            str(forecast_steps),
            "--pretty",
        )
        predictions = extract_predictions(predict_output)
        std_devs = extract_std_devs(predict_output)
        if len(predictions) != forecast_steps:
            raise AssertionError(
                f"{model_name} produced {len(predictions)} predictions, expected {forecast_steps}"
            )
        assert_all_finite(predictions, f"{model_name}_predictions")
        assert_all_finite(std_devs, f"{model_name}_std_devs")


def verify_weighted_avg_state(model_path: Path) -> None:
    state = load_pickled_state(model_path)
    if state["models"] != BASE_MODELS:
        raise AssertionError(f"unexpected weighted_avg models: {state['models']}")
    member_model = Path(state["base_model_states"]["lstm"]["model_file"])
    keras_path = member_model.with_name(f"{member_model.stem}_keras_model.h5")
    if not member_model.exists() or not keras_path.exists():
        raise AssertionError("weighted_avg saved LSTM member artifact is incomplete")


def verify_stacking_state(model_path: Path) -> None:
    state = load_pickled_state(model_path)
    if state["models"] != BASE_MODELS:
        raise AssertionError(f"unexpected stacking models: {state['models']}")
    member_model = Path(state["base_model_states"]["lstm"]["model_file"])
    keras_path = member_model.with_name(f"{member_model.stem}_keras_model.h5")
    if not member_model.exists() or not keras_path.exists():
        raise AssertionError("stacking saved LSTM member artifact is incomplete")


def verify_boosting_state(model_path: Path) -> None:
    state = load_pickled_state(model_path)
    model_chain = state["model_chain"]
    coefficients = [float(value) for value in state["stage_coefficients"]]
    if len(coefficients) != len(model_chain):
        raise AssertionError("boosting stage coefficients do not match model_chain")
    if state["baseline"]["kind"] not in {"mean", "naive", "seasonal_naive"}:
        raise AssertionError(f"unexpected boosting baseline: {state['baseline']}")
    if not model_chain and state["base_model_states"]:
        raise AssertionError("baseline-only boosting persisted unused member states")
    for model_name, model_info in state["base_model_states"].items():
        for round_state in model_info["models"]:
            member_model = Path(round_state["model_file"])
            if not member_model.exists():
                raise AssertionError(f"missing boosting member artifact: {member_model}")
            if model_name == "lstm":
                keras_path = member_model.with_name(f"{member_model.stem}_keras_model.h5")
                if not keras_path.exists():
                    raise AssertionError(f"missing boosting LSTM keras artifact: {keras_path}")


def combine_boosting_predictions(state: dict, forecast_steps: int) -> list[float]:
    baseline = state["baseline"]
    if baseline["kind"] == "mean":
        combined = [float(baseline["training_mean"])] * forecast_steps
    elif baseline["kind"] == "naive":
        combined = [float(baseline["last_value"])] * forecast_steps
    elif baseline["kind"] == "seasonal_naive":
        tail = [float(value) for value in baseline["seasonal_tail"]]
        combined = [tail[index % len(tail)] for index in range(forecast_steps)]
    else:
        raise AssertionError(f"unknown boosting baseline: {baseline}")

    usage_counts: dict[str, int] = {}
    for stage_index, (model_name, coefficient) in enumerate(
        zip(state["model_chain"], state["stage_coefficients"])
    ):
        usage_index = usage_counts.get(model_name, 0)
        usage_counts[model_name] = usage_index + 1
        member = state["base_model_states"][model_name]["models"][usage_index]
        stage_predictions = extract_predictions(
            invoke_model(
                model_name,
                None,
                "--load-model",
                member["model_file"],
                "--forecast-steps",
                str(forecast_steps),
                "--allow-negative-predictions",
                "--pretty",
            )
        )
        combined = [
            current + float(coefficient) * stage
            for current, stage in zip(combined, stage_predictions)
        ]

    return [max(0.0, value) for value in combined]


def train_and_verify_ensemble(
    model_type: str,
    data_json: str,
    save_model: Path,
    base_model_paths: dict[str, Path],
    train_start: int,
    train_end: int,
    eval_start: int,
    eval_end: int,
    expected_eval_y_true: list[float],
    expected_eval_months: list[str],
) -> None:
    common_args = [
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
        str(save_model),
        "--pretty",
    ]
    if model_type == "boosting":
        common_args.extend(["--learning-rate", "0.3", "--max-rounds", "1"])

    train_output = invoke_model(model_type, data_json, *common_args)
    if train_output["models"] != BASE_MODELS:
        raise AssertionError(f"{model_type} models mismatch: {train_output['models']}")
    if train_output["saved_model"] != str(save_model):
        raise AssertionError(f"{model_type} saved_model mismatch: {train_output['saved_model']}")
    if train_output["metrics"]["evaluate_months"] != expected_eval_months:
        raise AssertionError(
            f"{model_type} evaluate_months mismatch: {train_output['metrics']['evaluate_months']}"
        )
    if train_output["eval_y_true"] != expected_eval_y_true:
        raise AssertionError(f"{model_type} eval_y_true mismatch: {train_output['eval_y_true']}")

    train_predictions = extract_predictions(train_output)
    train_std_devs = extract_std_devs(train_output)
    assert_all_finite(train_predictions, f"{model_type}_train_predictions")
    assert_all_finite(train_std_devs, f"{model_type}_train_std_devs")
    if model_type == "boosting":
        boosting_state = load_pickled_state(save_model)
        assert_series_close(
            train_predictions,
            combine_boosting_predictions(
                boosting_state,
                forecast_steps=len(expected_eval_y_true),
            ),
        )

    loaded_predict_output = invoke_model(
        model_type,
        None,
        "--load-model",
        str(save_model),
        "--forecast-steps",
        str(len(expected_eval_y_true)),
        "--pretty",
    )
    loaded_predictions = extract_predictions(loaded_predict_output)
    loaded_std_devs = extract_std_devs(loaded_predict_output)
    assert_series_close(loaded_predictions, train_predictions)
    assert_all_finite(loaded_std_devs, f"{model_type}_loaded_std_devs")

    if model_type == "weighted_avg":
        verify_weighted_avg_state(save_model)
    elif model_type == "stacking":
        verify_stacking_state(save_model)
    elif model_type == "boosting":
        verify_boosting_state(save_model)


def main() -> None:
    records = load_csv_records(CSV_FIXTURE)
    data_json = json.dumps(records, ensure_ascii=False)
    sales = [float(row["销售数量"]) for row in records]
    timestamps = [f"{int(row['年份']):04d}-{int(row['月份']):02d}" for row in records]
    train_start = 0
    train_end = 27
    eval_start = 28
    eval_end = 30
    forecast_steps = 3

    with tempfile.TemporaryDirectory(prefix="fangzhen-ensemble-reuse-csv-") as temp_dir:
        temp_path = Path(temp_dir)
        base_model_paths = {model_name: temp_path / f"{model_name}-base.pkl" for model_name in BASE_MODELS}

        train_base_model_artifacts(
            data_json,
            base_model_paths,
            train_start=train_start,
            train_end=train_end,
            eval_start=eval_start,
            eval_end=eval_end,
        )
        verify_base_model_artifacts(base_model_paths, forecast_steps=forecast_steps)

        ensemble_model_paths = {
            "weighted_avg": temp_path / "weighted-avg-reuse.pkl",
            "boosting": temp_path / "boosting-reuse.pkl",
            "stacking": temp_path / "stacking-reuse.pkl",
        }

        for ensemble_name in ENSEMBLE_MODELS:
            train_and_verify_ensemble(
                model_type=ensemble_name,
                data_json=data_json,
                save_model=ensemble_model_paths[ensemble_name],
                base_model_paths=base_model_paths,
                train_start=train_start,
                train_end=train_end,
                eval_start=eval_start,
                eval_end=eval_end,
                expected_eval_y_true=sales[eval_start : eval_end + 1],
                expected_eval_months=timestamps[eval_start : eval_end + 1],
            )

    print(
        json.dumps(
            {
                "fixture": str(CSV_FIXTURE),
                "base_models": BASE_MODELS,
                "ensemble_models": ENSEMBLE_MODELS,
                "checked_modes": ["base_training", "ensemble_training", "loaded_predict"],
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
