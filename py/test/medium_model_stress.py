#!/usr/bin/env python3
import json
import math
import subprocess
import sys
import tempfile
from itertools import combinations
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
ENSEMBLE_MODEL_TYPES = ["weighted_avg", "stacking", "boosting"]


MODEL_ARGS = {
    "ma": ("--window", "6"),
    "es": ("--alpha", "0.35"),
    "arima": ("--d", "1", "--max-p", "2", "--max-q", "2"),
    "lstm": (
        "--look-back",
        "3",
        "--epochs",
        "1",
        "--normalization",
        "minmax",
        "--lstm-target-feature",
        TARGET_KEY,
        "--lstm-features",
        ",".join(FEATURE_KEYS),
    ),
}


def build_medium_sales_records(month_count: int = 18) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    channels = ["线上", "门店", "批发"]
    regions = ["华东", "华北", "华南", "西南"]
    weather_types = ["晴", "雨", "热", "冷"]

    for index in range(month_count):
        year = 2019 + index // 12
        month = index % 12 + 1
        season = math.sin((month - 1) / 12 * 2 * math.pi)
        winter_peak = 1 if month in (1, 11, 12) else 0
        summer_peak = 1 if month in (6, 7, 8) else 0
        promotion = 1 if month in (6, 11, 12) or index % 17 == 0 else 0
        holiday_strength = 2 if month in (1, 2, 10, 12) else (1 if month in (5, 6, 11) else 0)
        price_index = 98 + 0.18 * index + 2.8 * math.cos(month / 12 * 2 * math.pi)
        competitor_price = 101 + 0.12 * index + 1.7 * math.sin((month + 2) / 12 * 2 * math.pi)
        search_heat = 52 + 0.65 * index + 9.0 * season + 8.0 * promotion + 4.0 * holiday_strength
        inventory = 420 + 1.4 * index + 28.0 * summer_peak - 18.0 * winter_peak
        deterministic_noise = ((index * 37) % 17 - 8) * 1.35
        sales = (
            260
            + 2.9 * index
            + 22.0 * season
            + 30.0 * winter_peak
            + 16.0 * summer_peak
            + 34.0 * promotion
            + 7.0 * holiday_strength
            - 1.8 * (price_index - 100)
            + 0.55 * (search_heat - 60)
            - 0.7 * (competitor_price - 100)
            + deterministic_noise
        )

        records.append(
            {
                "年份": year,
                "月份": month,
                TARGET_KEY: round(max(1.0, sales), 2),
                "价格指数": round(price_index, 2),
                "促销投入": 18 + promotion * 42 + (index % 5) * 3,
                "库存水平": round(inventory, 2),
                "搜索热度": round(search_heat, 2),
                "竞争对手价格": round(competitor_price, 2),
                "节假日强度": holiday_strength,
                "渠道类型": channels[index % len(channels)],
                "区域": regions[(index // 3) % len(regions)],
                "天气类型": weather_types[(month + index // 12) % len(weather_types)],
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


def invoke_model(model_type: str, data_json: str | None, *extra_args: str, label: str) -> dict:
    command = [sys.executable, str(RUN_MODEL), model_type]
    if data_json is not None:
        command.extend(["--data", data_json])
    command.extend(extra_args)
    command.append("--pretty")

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
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


def extract_predictions(payload: dict) -> list[float]:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise AssertionError(f"missing predictions list in payload mode={payload.get('mode')!r}")
    return [float(item["prediction"]) for item in predictions]


def assert_prediction_payload(payload: dict, model_key: str, expected_length: int, label: str) -> None:
    assert_method_metadata(payload, model_key)
    predictions = extract_predictions(payload)
    if len(predictions) != expected_length:
        raise AssertionError(f"{label} prediction length mismatch: {len(predictions)} != {expected_length}")
    if any(value <= 0 for value in predictions):
        raise AssertionError(f"{label} expected positive sales predictions, got {predictions}")


def assert_repeatable_predictions(left: dict, right: dict, label: str, tolerance: float = 1e-9) -> None:
    left_predictions = extract_predictions(left)
    right_predictions = extract_predictions(right)
    if len(left_predictions) != len(right_predictions):
        raise AssertionError(f"{label} repeat length mismatch")
    for index, (left_value, right_value) in enumerate(zip(left_predictions, right_predictions)):
        if not math.isclose(left_value, right_value, rel_tol=tolerance, abs_tol=tolerance):
            raise AssertionError(
                f"{label} prediction[{index}] is not repeatable: {left_value} != {right_value}"
            )


def assert_basic_metrics(payload: dict, label: str) -> None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise AssertionError(f"{label} missing metrics")
    for key in ("rmse", "mae", "mape"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AssertionError(f"{label} metric {key} is not finite: {value!r}")


def model_metadata_key(model_type: str) -> str:
    return "lstm_direct" if model_type == "lstm" else model_type


def run_standalone_model_lifecycle(
    model_type: str,
    model_key: str,
    data_json: str,
    temp_path: Path,
    train_end: int,
    eval_start: int,
    eval_end: int,
    production_end: int,
    future_steps: int,
) -> dict:
    backtest_path = temp_path / f"{model_type}-backtest-{train_end}-{eval_end}.pkl"
    production_path = temp_path / f"{model_type}-production.pkl"
    eval_length = eval_end - eval_start + 1

    train_output = invoke_model(
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
        *MODEL_ARGS[model_type],
        "--save-model",
        str(backtest_path),
        label=f"{model_type} train/eval {train_end}-{eval_end}",
    )
    assert_prediction_payload(train_output, model_key, eval_length, f"{model_type} train/eval")
    assert_basic_metrics(train_output, f"{model_type} train/eval")
    if not backtest_path.exists():
        raise AssertionError(f"{model_type} did not save backtest model: {backtest_path}")

    loaded_backtest = invoke_model(
        model_type,
        None,
        "--load-model",
        str(backtest_path),
        "--forecast-steps",
        str(eval_length),
        label=f"{model_type} loaded backtest predict",
    )
    assert_prediction_payload(loaded_backtest, model_key, eval_length, f"{model_type} loaded backtest")

    fit_output = invoke_model(
        model_type,
        data_json,
        "--train-start",
        "0",
        "--train-end",
        str(production_end),
        "--load-model",
        str(backtest_path),
        "--save-model",
        str(production_path),
        "--fit-only",
        label=f"{model_type} fit-only production",
    )
    assert_method_metadata(fit_output, model_key)
    if fit_output.get("saved_model") != str(production_path):
        raise AssertionError(f"{model_type} fit-only saved_model mismatch")
    if not production_path.exists():
        raise AssertionError(f"{model_type} did not save production model: {production_path}")

    predict_one = invoke_model(
        model_type,
        None,
        "--load-model",
        str(production_path),
        "--forecast-steps",
        str(future_steps),
        label=f"{model_type} production predict one",
    )
    predict_two = invoke_model(
        model_type,
        None,
        "--load-model",
        str(production_path),
        "--forecast-steps",
        str(future_steps),
        label=f"{model_type} production predict two",
    )
    assert_prediction_payload(predict_one, model_key, future_steps, f"{model_type} production")
    assert_prediction_payload(predict_two, model_key, future_steps, f"{model_type} production repeat")
    assert_repeatable_predictions(predict_one, predict_two, f"{model_type} production")
    return train_output


def train_base_artifacts(
    data_json: str,
    temp_path: Path,
    train_end: int,
    eval_start: int,
    eval_end: int,
    models_to_use: list[str] | None = None,
) -> dict[str, Path]:
    model_names = models_to_use or BASE_MODELS
    paths: dict[str, Path] = {}
    for model_type in model_names:
        model_path = temp_path / f"base-{model_type}.pkl"
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
            *MODEL_ARGS[model_type],
            "--save-model",
            str(model_path),
            label=f"base {model_type} train",
        )
        assert_prediction_payload(output, model_metadata_key(model_type), eval_end - eval_start + 1, f"base {model_type}")
        paths[model_type] = model_path
    return paths


def assert_weighted_weights(payload: dict, expected_models: list[str], label: str) -> None:
    weights = payload.get("weights")
    if not isinstance(weights, dict) or set(weights) != set(expected_models):
        raise AssertionError(f"{label} missing expected weights: {weights!r}")
    total = sum(float(value) for value in weights.values())
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(f"{label} weights do not sum to 1: {weights}")
    if any(float(value) < 0 for value in weights.values()):
        raise AssertionError(f"{label} weights must be non-negative: {weights}")


def assert_weighted_output(payload: dict, expected_length: int, expected_models: list[str], label: str) -> None:
    assert_prediction_payload(payload, "weighted_avg", expected_length, label)
    assert_weighted_weights(payload, expected_models, label)


def assert_stacking_output(payload: dict, expected_length: int, expected_models: list[str], label: str) -> None:
    assert_prediction_payload(payload, "stacking", expected_length, label)
    meta_model = payload.get("meta_model")
    if not isinstance(meta_model, dict):
        raise AssertionError(f"{label} missing meta_model")
    condition_number = meta_model.get("condition_number")
    if condition_number is not None and not math.isfinite(float(condition_number)):
        raise AssertionError(f"{label} condition_number must be finite or null: {condition_number!r}")
    weights = meta_model.get("weights")
    if not isinstance(weights, list) or len(weights) != len(expected_models):
        raise AssertionError(f"{label} missing stacking weights: {weights!r}")
    model_names = meta_model.get("model_names")
    if not isinstance(model_names, list) or set(model_names) != set(expected_models):
        raise AssertionError(f"{label} stacking model_names mismatch: {model_names!r}")
    total = sum(float(value) for value in weights)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(f"{label} stacking weights do not sum to 1: {weights}")


def assert_boosting_output(payload: dict, expected_length: int, label: str) -> None:
    assert_prediction_payload(payload, "boosting", expected_length, label)
    model_chain = payload.get("model_chain")
    coefficients = payload.get("stage_coefficients")
    if not isinstance(model_chain, list) or not model_chain:
        raise AssertionError(f"{label} expected a non-empty model_chain")
    if not isinstance(coefficients, list) or len(coefficients) != len(model_chain):
        raise AssertionError(f"{label} stage_coefficients mismatch: {coefficients!r}")


def run_ensemble_lifecycle(
    model_type: str,
    models_to_use: list[str],
    data_json: str,
    base_model_paths: dict[str, Path],
    temp_path: Path,
    train_end: int,
    eval_start: int,
    eval_end: int,
    production_end: int,
    future_steps: int,
    run_fit_only: bool = False,
) -> dict:
    combo_slug = "-".join(models_to_use)
    backtest_path = temp_path / f"{model_type}-{combo_slug}-backtest.pkl"
    production_path = temp_path / f"{model_type}-{combo_slug}-production.pkl"
    eval_length = eval_end - eval_start + 1
    base_model_files = json.dumps(
        {name: str(base_model_paths[name]) for name in models_to_use},
        ensure_ascii=False,
    )
    common_args = (
        "--models",
        ",".join(models_to_use),
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
        "--look-back",
        "3",
        "--epochs",
        "1",
        "--normalization",
        "minmax",
        "--lstm-target-feature",
        TARGET_KEY,
        "--lstm-features",
        ",".join(FEATURE_KEYS),
        "--base-model-files",
        base_model_files,
    )
    if model_type == "boosting":
        common_args = (*common_args, "--learning-rate", "0.3", "--max-rounds", str(min(2, len(models_to_use))))

    train_output = invoke_model(
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
        *common_args,
        "--save-model",
        str(backtest_path),
        label=f"{model_type} {'+'.join(models_to_use)} ensemble train",
    )
    if model_type == "weighted_avg":
        assert_weighted_output(train_output, eval_length, models_to_use, f"{model_type} {combo_slug} train")
    elif model_type == "stacking":
        assert_stacking_output(train_output, eval_length, models_to_use, f"{model_type} {combo_slug} train")
    elif model_type == "boosting":
        assert_boosting_output(train_output, eval_length, f"{model_type} {combo_slug} train")
    assert_basic_metrics(train_output, f"{model_type} {combo_slug} train")

    predict_model_path = backtest_path
    if run_fit_only:
        fit_output = invoke_model(
            model_type,
            data_json,
            "--train-start",
            "0",
            "--train-end",
            str(production_end),
            "--load-model",
            str(backtest_path),
            "--save-model",
            str(production_path),
            "--fit-only",
            label=f"{model_type} {'+'.join(models_to_use)} fit-only production",
        )
        assert_method_metadata(fit_output, model_type)
        if fit_output.get("saved_model") != str(production_path):
            raise AssertionError(f"{model_type} fit-only saved_model mismatch")
        predict_model_path = production_path

    predict_one = invoke_model(
        model_type,
        None,
        "--load-model",
        str(predict_model_path),
        "--forecast-steps",
        str(future_steps),
        label=f"{model_type} {'+'.join(models_to_use)} production predict one",
    )
    assert_prediction_payload(predict_one, model_type, future_steps, f"{model_type} {combo_slug} production")
    if run_fit_only:
        predict_two = invoke_model(
            model_type,
            None,
            "--load-model",
            str(predict_model_path),
            "--forecast-steps",
            str(future_steps),
            label=f"{model_type} {'+'.join(models_to_use)} production predict two",
        )
        assert_prediction_payload(predict_two, model_type, future_steps, f"{model_type} {combo_slug} production repeat")
        assert_repeatable_predictions(predict_one, predict_two, f"{model_type} {combo_slug} production")
    return train_output


def main() -> None:
    records = build_medium_sales_records()
    data_json = json.dumps(records, ensure_ascii=False)
    train_end = 11
    eval_start = 12
    eval_end = 14
    production_end = 17
    future_steps = 3
    all_combinations = [
        list(combo)
        for combo_size in range(2, len(BASE_MODELS) + 1)
        for combo in combinations(BASE_MODELS, combo_size)
    ]

    with tempfile.TemporaryDirectory(prefix="fangzhen-medium-stress-") as temp_dir:
        temp_path = Path(temp_dir)
        base_model_paths = train_base_artifacts(
            data_json,
            temp_path,
            train_end=train_end,
            eval_start=eval_start,
            eval_end=eval_end,
        )
        ensemble_results: dict[str, list[str]] = {model_type: [] for model_type in ENSEMBLE_MODEL_TYPES}
        for models_to_use in all_combinations:
            combo_label = "+".join(models_to_use)
            for ensemble_model_type in ENSEMBLE_MODEL_TYPES:
                run_fit_only = models_to_use == BASE_MODELS
                print(
                    f"[medium-stress] {ensemble_model_type}: {combo_label}"
                    f"{' with fit-only' if run_fit_only else ''}",
                    file=sys.stderr,
                    flush=True,
                )
                run_ensemble_lifecycle(
                    ensemble_model_type,
                    models_to_use,
                    data_json,
                    base_model_paths,
                    temp_path,
                    train_end=train_end,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    production_end=production_end,
                    future_steps=future_steps,
                    run_fit_only=run_fit_only,
                )
                ensemble_results[ensemble_model_type].append(combo_label)

    print(
        json.dumps(
            {
                "dataset": {
                    "records": len(records),
                    "columns": list(records[0].keys()),
                    "target_min": min(float(row[TARGET_KEY]) for row in records),
                    "target_max": max(float(row[TARGET_KEY]) for row in records),
                },
                "base_models": BASE_MODELS,
                "ensemble_combinations": ensemble_results,
                "ensemble_combination_count_per_model": {
                    model_type: len(combos)
                    for model_type, combos in ensemble_results.items()
                },
                "checked_paths": [
                    "base_model_training",
                    "ensemble_training_all_combinations",
                    "loaded_ensemble_predict_all_combinations",
                    "fit_only_full_combination",
                    "loaded_production_predict_full_combination",
                ],
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
