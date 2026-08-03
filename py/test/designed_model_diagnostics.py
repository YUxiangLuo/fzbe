#!/usr/bin/env python3
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from model_metadata_assertions import assert_method_metadata


PY_ROOT = Path(__file__).resolve().parents[1]
RUN_MODEL = PY_ROOT / "run_model.py"
TARGET_KEY = "销售数量"
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
BASE_MODELS = ["ma", "es", "arima", "lstm"]
ENSEMBLE_MODELS = ["weighted_avg", "stacking", "boosting"]

COMMON_LSTM_ARGS = (
    "--look-back",
    "4",
    "--epochs",
    "6",
    "--normalization",
    "minmax",
    "--lstm-target-feature",
    TARGET_KEY,
    "--lstm-features",
    ",".join(FEATURE_KEYS),
)


def build_designed_sales_records(scenario: str, month_count: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    channels = ["直营", "经销", "电商"]
    regions = ["华东", "华南"]
    weather_types = ["晴", "雨"]

    for index in range(month_count):
        year = 2022 + index // 12
        month = index % 12 + 1
        season = math.sin((month - 1) / 12 * 2 * math.pi)
        promotion = 1 if month in (6, 11, 12) else 0
        small_noise = [0.0, 1.0, -1.0, 0.0][index % 4]

        if scenario == "level_demand_24m":
            sales = 200 + [-1.0, 0.0, 1.0, 0.0][index % 4]
        elif scenario == "linear_trend_36m":
            sales = 80 + 5 * index + small_noise
        elif scenario == "seasonal_promo_36m":
            sales = 180 + 1.2 * index + 25.0 * season + 18.0 * promotion + [0.0, 2.0, -1.0, 1.0][index % 4]
        else:
            raise ValueError(f"unknown designed scenario: {scenario}")

        records.append(
            {
                "年份": year,
                "月份": month,
                TARGET_KEY: round(max(1.0, sales), 2),
                "价格指数": round(100 + 0.2 * index - 2.0 * promotion, 2),
                "促销投入": 10 + 30 * promotion,
                "库存水平": 300 + index,
                "搜索热度": round(55 + 8.0 * season + 20.0 * promotion + 0.2 * index, 2),
                "竞争对手价格": round(101 + math.cos(month), 2),
                "节假日强度": 1 if month in (1, 10, 12) else 0,
                "渠道类型": channels[index % len(channels)],
                "区域": regions[index % len(regions)],
                "天气类型": weather_types[month % len(weather_types)],
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


def invoke_model(
    model_type: str,
    data_json: str | None,
    *extra_args: str,
    label: str,
    timeout: int = 300,
) -> dict:
    command = [sys.executable, str(RUN_MODEL), model_type]
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


def model_args(model_type: str, scenario: str) -> tuple[str, ...]:
    if model_type == "ma":
        return ("--window", "3")
    if model_type == "es":
        return ("--alpha", "0.35")
    if model_type == "arima":
        differencing = "0" if scenario == "level_demand_24m" else "1"
        return ("--d", differencing, "--max-p", "2", "--max-q", "2")
    if model_type == "lstm":
        return COMMON_LSTM_ARGS
    raise ValueError(f"unsupported model type: {model_type}")


def ensemble_args(base_model_paths: dict[str, Path]) -> tuple[str, ...]:
    return (
        "--models",
        ",".join(BASE_MODELS),
        "--window",
        "3",
        "--alpha",
        "0.35",
        "--d",
        "1",
        "--max-p",
        "2",
        "--max-q",
        "2",
        *COMMON_LSTM_ARGS,
        "--base-model-files",
        json.dumps({name: str(path) for name, path in base_model_paths.items()}, ensure_ascii=False),
    )


def extract_predictions(payload: dict) -> list[float]:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise AssertionError(f"missing predictions list in payload mode={payload.get('mode')!r}")
    return [float(item["prediction"]) for item in predictions]


def extract_std_devs(payload: dict) -> list[float]:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise AssertionError(f"missing predictions list in payload mode={payload.get('mode')!r}")
    return [float(item["std_dev"]) for item in predictions]


def metrics(payload: dict, label: str) -> dict[str, float]:
    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, dict):
        raise AssertionError(f"{label} missing metrics")
    for key in ("rmse", "mae", "mape", "r2"):
        value = raw_metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AssertionError(f"{label} metric {key} is not finite: {value!r}")
    return {key: float(raw_metrics[key]) for key in ("rmse", "mae", "mape", "r2")}


def metadata_key(model_type: str) -> str:
    return "lstm_direct" if model_type == "lstm" else model_type


def assert_training_payload(payload: dict, model_type: str, expected_length: int, label: str) -> None:
    assert_method_metadata(payload, metadata_key(model_type))
    prediction_values = extract_predictions(payload)
    std_devs = extract_std_devs(payload)
    if len(prediction_values) != expected_length:
        raise AssertionError(f"{label} prediction length mismatch: {len(prediction_values)} != {expected_length}")
    if len(std_devs) != expected_length:
        raise AssertionError(f"{label} std_dev length mismatch: {len(std_devs)} != {expected_length}")
    if any(value <= 0 for value in prediction_values):
        raise AssertionError(f"{label} expected positive sales predictions, got {prediction_values}")
    if any(value < 0 or not math.isfinite(value) for value in std_devs):
        raise AssertionError(f"{label} expected finite non-negative std_dev values, got {std_devs}")
    eval_y_true = payload.get("eval_y_true")
    if not isinstance(eval_y_true, list) or len(eval_y_true) != expected_length:
        raise AssertionError(f"{label} eval_y_true length mismatch: {eval_y_true!r}")
    metrics(payload, label)


def assert_values_between(values: list[float], lower: float, upper: float, label: str) -> None:
    for index, value in enumerate(values):
        if value < lower or value > upper:
            raise AssertionError(f"{label}[{index}]={value} outside [{lower}, {upper}]")


def assert_strictly_increasing(values: list[float], label: str) -> None:
    for index, (left, right) in enumerate(zip(values, values[1:])):
        if right <= left:
            raise AssertionError(f"{label} must be strictly increasing at {index}: {values}")


def run_base_models_for_scenario(
    scenario: str,
    records: list[dict[str, Any]],
    temp_path: Path,
) -> tuple[dict[str, dict], dict[str, Path]]:
    data_json = json.dumps(records, ensure_ascii=False)
    train_end = len(records) - 7
    eval_start = len(records) - 6
    eval_end = len(records) - 1
    eval_length = eval_end - eval_start + 1
    outputs: dict[str, dict] = {}
    paths: dict[str, Path] = {}

    for model_type in BASE_MODELS:
        model_path = temp_path / f"{scenario}-{model_type}.pkl"
        output = invoke_model(
            model_type,
            data_json,
            "--train-start",
            "0",
            "--train-end",
            str(train_end),
            "--eval-start",
            str(eval_start),
            "--eval-end",
            str(eval_end),
            *model_args(model_type, scenario),
            "--save-model",
            str(model_path),
            label=f"{scenario} {model_type} train/eval",
            timeout=360 if model_type == "lstm" else 180,
        )
        assert_training_payload(output, model_type, eval_length, f"{scenario} {model_type}")
        outputs[model_type] = output
        paths[model_type] = model_path

    return outputs, paths


def assert_level_demand_behavior(outputs: dict[str, dict]) -> None:
    for model_type, output in outputs.items():
        model_metrics = metrics(output, f"level {model_type}")
        if model_metrics["mape"] >= 2.0:
            raise AssertionError(f"level {model_type} MAPE too high: {model_metrics}")
        assert_values_between(
            extract_predictions(output),
            197.0,
            203.0,
            f"level {model_type} predictions",
        )


def assert_linear_trend_behavior(outputs: dict[str, dict]) -> None:
    arima_metrics = metrics(outputs["arima"], "linear arima")
    ma_metrics = metrics(outputs["ma"], "linear ma")
    es_metrics = metrics(outputs["es"], "linear es")
    if arima_metrics["rmse"] >= 2.5 or arima_metrics["mape"] >= 1.5:
        raise AssertionError(f"linear ARIMA should fit the trend closely, got {arima_metrics}")
    if arima_metrics["rmse"] >= ma_metrics["rmse"] * 0.25:
        raise AssertionError(f"linear ARIMA should beat MA clearly: arima={arima_metrics}, ma={ma_metrics}")
    if arima_metrics["rmse"] >= es_metrics["rmse"] * 0.25:
        raise AssertionError(f"linear ARIMA should beat ES clearly: arima={arima_metrics}, es={es_metrics}")
    assert_strictly_increasing(extract_predictions(outputs["arima"]), "linear ARIMA predictions")
    assert_values_between(extract_predictions(outputs["lstm"]), 50.0, 300.0, "linear LSTM predictions")


def assert_seasonal_promo_behavior(outputs: dict[str, dict]) -> None:
    for model_type, output in outputs.items():
        model_metrics = metrics(output, f"seasonal {model_type}")
        if model_metrics["mape"] >= 25.0:
            raise AssertionError(f"seasonal {model_type} MAPE unexpectedly high: {model_metrics}")
        assert_values_between(
            extract_predictions(output),
            100.0,
            300.0,
            f"seasonal {model_type} predictions",
        )


def run_linear_ensemble_diagnostics(
    records: list[dict[str, Any]],
    base_model_paths: dict[str, Path],
) -> dict[str, dict]:
    data_json = json.dumps(records, ensure_ascii=False)
    train_end = len(records) - 7
    eval_start = len(records) - 6
    eval_end = len(records) - 1
    eval_length = eval_end - eval_start + 1
    outputs: dict[str, dict] = {}

    for model_type in ENSEMBLE_MODELS:
        extra_args: tuple[str, ...] = ()
        if model_type == "boosting":
            extra_args = ("--learning-rate", "0.3", "--max-rounds", "4")
        output = invoke_model(
            model_type,
            data_json,
            "--train-start",
            "0",
            "--train-end",
            str(train_end),
            "--eval-start",
            str(eval_start),
            "--eval-end",
            str(eval_end),
            *ensemble_args(base_model_paths),
            *extra_args,
            label=f"linear {model_type} ensemble train/eval",
            timeout=480,
        )
        assert_training_payload(output, model_type, eval_length, f"linear {model_type}")
        outputs[model_type] = output

    weighted = outputs["weighted_avg"]
    weights = weighted.get("weights")
    if not isinstance(weights, dict):
        raise AssertionError(f"weighted_avg missing weights: {weights!r}")
    arima_weight = float(weights.get("arima", 0.0))
    other_weights = [
        float(weight)
        for model_name, weight in weights.items()
        if model_name != "arima"
    ]
    if arima_weight < 0.50 or not other_weights or arima_weight <= 3.0 * max(other_weights):
        raise AssertionError(f"weighted_avg should clearly favor ARIMA on linear trend, got {weights}")
    if float(weights.get("lstm", 1.0)) > 0.20:
        raise AssertionError(f"weighted_avg should downweight LSTM on linear trend, got {weights}")
    weight_diagnostics = weighted.get("weight_diagnostics")
    if not isinstance(weight_diagnostics, dict):
        raise AssertionError(f"weighted_avg missing weight diagnostics: {weight_diagnostics!r}")
    if weight_diagnostics.get("best_member", {}).get("model_name") != "arima":
        raise AssertionError(
            f"weighted_avg selection should identify ARIMA as the best internal member: {weight_diagnostics}"
        )
    if float(weight_diagnostics["selected_weight_rmse"]) > float(
        weight_diagnostics["equal_weight_rmse"]
    ) + 1e-9:
        raise AssertionError(
            f"weighted_avg selected weights should beat equal weights internally: {weight_diagnostics}"
        )

    stacking = outputs["stacking"]
    meta_model = stacking.get("meta_model")
    if not isinstance(meta_model, dict):
        raise AssertionError(f"stacking missing meta_model: {meta_model!r}")
    condition_number = meta_model.get("condition_number")
    if condition_number is not None and not math.isfinite(float(condition_number)):
        raise AssertionError(f"stacking condition_number must be finite or null: {condition_number!r}")
    meta_model_names = meta_model.get("model_names")
    meta_coefficients = meta_model.get("coefficients")
    if not isinstance(meta_model_names, list) or not isinstance(meta_coefficients, list):
        raise AssertionError(f"stacking meta_model missing names/coefficients: {meta_model!r}")
    if any(
        not math.isfinite(float(coefficient)) or float(coefficient) < 0
        for coefficient in meta_coefficients
    ):
        raise AssertionError(f"stacking coefficients must be finite and non-negative: {meta_model!r}")
    stacking_weights = {
        str(name): float(weight)
        for name, weight in zip(meta_model_names, meta_coefficients)
    }
    if stacking_weights.get("arima", 0.0) < 0.80:
        raise AssertionError(f"stacking should favor ARIMA on linear trend, got {stacking_weights}")
    if int(meta_model.get("oof_fold_count", 0)) < 2 or int(meta_model.get("oof_row_count", 0)) < 4:
        raise AssertionError(f"stacking should persist rolling OOF evidence: {meta_model!r}")

    boosting = outputs["boosting"]
    model_chain = boosting.get("model_chain")
    if not isinstance(model_chain, list) or not model_chain:
        raise AssertionError(f"boosting expected a non-empty model_chain, got {model_chain!r}")
    if any(model_name not in BASE_MODELS for model_name in model_chain):
        raise AssertionError(f"boosting returned an unknown residual member: {model_chain}")
    stage_coefficients = boosting.get("stage_coefficients")
    if (
        not isinstance(stage_coefficients, list)
        or len(stage_coefficients) != len(model_chain)
        or any(
            not math.isfinite(float(coefficient)) or float(coefficient) <= 0
            for coefficient in stage_coefficients
        )
    ):
        raise AssertionError(
            f"boosting should persist one positive finite coefficient per stage: {stage_coefficients!r}"
        )
    search_summary = boosting.get("search_summary")
    if (
        not isinstance(search_summary, dict)
        or search_summary.get("algorithm") != "residual_boosting"
    ):
        raise AssertionError(f"boosting missing residual search metadata: {search_summary!r}")

    return outputs


def summarize_model(output: dict) -> dict[str, Any]:
    return {
        "rmse": round(float(output["metrics"]["rmse"]), 6),
        "mae": round(float(output["metrics"]["mae"]), 6),
        "mape": round(float(output["metrics"]["mape"]), 6),
        "r2": round(float(output["metrics"]["r2"]), 6),
        "predictions": [round(value, 6) for value in extract_predictions(output)],
    }


def main() -> None:
    scenarios = {
        "level_demand_24m": build_designed_sales_records("level_demand_24m", 24),
        "linear_trend_36m": build_designed_sales_records("linear_trend_36m", 36),
        "seasonal_promo_36m": build_designed_sales_records("seasonal_promo_36m", 36),
    }
    summary: dict[str, Any] = {
        "scenarios": {},
        "ensemble_diagnostics": {},
        "status": "ok",
    }

    with tempfile.TemporaryDirectory(prefix="fangzhen-designed-diagnostics-") as temp_dir:
        temp_path = Path(temp_dir)
        scenario_outputs: dict[str, dict[str, dict]] = {}
        scenario_model_paths: dict[str, dict[str, Path]] = {}

        for scenario, records in scenarios.items():
            print(f"[designed-diagnostics] base models: {scenario}", file=sys.stderr, flush=True)
            outputs, paths = run_base_models_for_scenario(scenario, records, temp_path)
            scenario_outputs[scenario] = outputs
            scenario_model_paths[scenario] = paths
            summary["scenarios"][scenario] = {
                "records": len(records),
                "models": {model_type: summarize_model(output) for model_type, output in outputs.items()},
            }

        assert_level_demand_behavior(scenario_outputs["level_demand_24m"])
        assert_linear_trend_behavior(scenario_outputs["linear_trend_36m"])
        assert_seasonal_promo_behavior(scenario_outputs["seasonal_promo_36m"])

        print("[designed-diagnostics] ensembles: linear_trend_36m", file=sys.stderr, flush=True)
        ensemble_outputs = run_linear_ensemble_diagnostics(
            scenarios["linear_trend_36m"],
            scenario_model_paths["linear_trend_36m"],
        )
        summary["ensemble_diagnostics"]["linear_trend_36m"] = {
            model_type: summarize_model(output)
            for model_type, output in ensemble_outputs.items()
        }
        summary["ensemble_diagnostics"]["linear_trend_36m"]["weighted_avg"]["weights"] = ensemble_outputs[
            "weighted_avg"
        ]["weights"]
        summary["ensemble_diagnostics"]["linear_trend_36m"]["stacking"]["meta_model"] = ensemble_outputs[
            "stacking"
        ]["meta_model"]
        summary["ensemble_diagnostics"]["linear_trend_36m"]["boosting"]["model_chain"] = ensemble_outputs[
            "boosting"
        ]["model_chain"]
        summary["ensemble_diagnostics"]["linear_trend_36m"]["boosting"]["stage_coefficients"] = ensemble_outputs[
            "boosting"
        ]["stage_coefficients"]

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
