#!/usr/bin/env python3
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


PY_ROOT = Path(__file__).resolve().parents[1]
RUN_MODEL = PY_ROOT / "run_model.py"
CSV_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "core_models_smoke.csv"
BASE_MODELS = ["ma", "es", "arima"]
BASE_MODEL_TRAIN_ARGS = {
    "ma": ("--window", "3"),
    "es": ("--alpha", "0.4"),
    "arima": ("--d", "1", "--max-p", "2", "--max-q", "2"),
}

sys.path.insert(0, str(PY_ROOT / "src"))

from model_metadata_assertions import assert_method_metadata  # noqa: E402
from models.stacking import _fit_meta_model  # noqa: E402


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


def compute_expected_stacking_weights(
    model_names: list[str],
    level1_predictions: dict[str, list[float]],
    meta_target: np.ndarray,
) -> np.ndarray:
    """Solve the meta-model independently of `_fit_meta_model`.

    The end-to-end run cannot be checked this way any more: Level-1 rows now
    come from internally planned rolling-origin folds, so reproducing the matrix
    outside the model would mean reimplementing the fold planner. The solver
    itself is still reproducible, and pinning it here keeps an independent
    constraint on the coefficients the end-to-end assertions consume.
    """
    meta_train = np.column_stack([level1_predictions[name] for name in model_names])
    coefficients, _ = nnls(meta_train, meta_target)
    return coefficients


def blend_predictions(model_names: list[str], member_predictions: dict[str, list[float]], weights: np.ndarray) -> list[float]:
    prediction_matrix = np.column_stack([member_predictions[name] for name in model_names])
    return (prediction_matrix @ weights).tolist()


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


def member_model_path(save_path: Path, model_name: str) -> str:
    return str(save_path.with_name(f"{save_path.stem}_base_{model_name}.pkl"))


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


def main() -> None:
    singular_meta_model = _fit_meta_model(
        np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=float,
        ),
        np.array([10.5, 11.5, 12.5, 13.5], dtype=float),
        ["left", "right"],
    )
    assert singular_meta_model["kind"] == "nnls"
    assert singular_meta_model["strategy"] == "ridge_regularized_nonnegative_least_squares"
    assert singular_meta_model["regularized"] is True
    assert singular_meta_model["coefficients"] == [0.0, 0.0]
    assert math.isfinite(float(singular_meta_model["residual_norm"]))
    json.dumps(singular_meta_model, allow_nan=False)

    # A well-conditioned matrix must take the plain NNLS path and land on the
    # same solution an independent solver finds. Without this the end-to-end
    # weight assertions below are self-referential: they read the coefficients
    # back out of the model's own output.
    well_conditioned_level1 = {
        "left": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "right": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
    }
    well_conditioned_target = np.array([1.4, 1.8, 3.4, 3.8, 5.4, 5.8], dtype=float)
    well_conditioned_meta_model = _fit_meta_model(
        np.column_stack(
            [well_conditioned_level1[name] for name in ("left", "right")]
        ),
        well_conditioned_target,
        ["left", "right"],
    )
    assert well_conditioned_meta_model["strategy"] == "nonnegative_least_squares"
    assert well_conditioned_meta_model["regularized"] is False
    assert_series_close(
        well_conditioned_meta_model["coefficients"],
        compute_expected_stacking_weights(
            ["left", "right"],
            well_conditioned_level1,
            well_conditioned_target,
        ).tolist(),
    )

    records = load_csv_records(CSV_FIXTURE)
    data_json = json.dumps(records, ensure_ascii=False)
    sales = [float(row["销售数量"]) for row in records]
    timestamps = [f"{int(row['年份']):04d}-{int(row['月份']):02d}" for row in records]
    train_start = 0
    train_end = 19
    eval_start = 20
    eval_end = 23
    future_steps = 3

    with tempfile.TemporaryDirectory(prefix="fangzhen-stacking-csv-") as temp_dir:
        temp_path = Path(temp_dir)
        base_model_paths = {
            "ma": temp_path / "ma-backtest.pkl",
            "es": temp_path / "es-backtest.pkl",
            "arima": temp_path / "arima-backtest.pkl",
        }

        train_base_model_artifacts(
            data_json,
            base_model_paths,
            train_start,
            train_end,
            eval_start,
            eval_end,
        )

        eval_predictions: dict[str, list[float]] = {}
        for model_name in BASE_MODELS:
            eval_predictions[model_name] = extract_predictions(
                invoke_model(
                    model_name,
                    None,
                    "--load-model",
                    str(base_model_paths[model_name]),
                    "--forecast-steps",
                    "4",
                    "--pretty",
                )
            )

        stacking_model_backtest = temp_path / "stacking-backtest.pkl"
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
            "--base-model-files",
            json.dumps({name: str(path) for name, path in base_model_paths.items()}, ensure_ascii=False),
            "--save-model",
            str(stacking_model_backtest),
            "--pretty",
        )

        actual_eval_predictions = extract_predictions(train_output)
        actual_eval_std_devs = extract_std_devs(train_output)
        assert train_output["mode"] == "stacking_train_eval"
        assert_method_metadata(train_output, "stacking")
        assert train_output["models"] == BASE_MODELS
        assert train_output["metrics"]["evaluate_months"] == timestamps[20:24]
        assert train_output["eval_y_true"] == sales[20:24]
        assert train_output["meta_model"]["strategy"] in {
            "nonnegative_least_squares",
            "ridge_regularized_nonnegative_least_squares",
        }
        expected_weights = np.asarray(
            train_output["meta_model"]["coefficients"],
            dtype=float,
        )
        assert len(expected_weights) == len(BASE_MODELS)
        assert np.isfinite(expected_weights).all()
        assert (expected_weights >= 0).all()
        assert train_output["meta_model"]["oof_fold_count"] >= 2
        assert train_output["meta_model"]["oof_row_count"] == train_output["split_plan"]["selection_size"]
        ranges = train_output["split_plan"]["ranges"]
        assert ranges["selection"]["end_exclusive"] <= ranges["calibration"]["start"]
        expected_eval_predictions = blend_predictions(
            BASE_MODELS,
            eval_predictions,
            expected_weights,
        )
        assert_series_close(actual_eval_predictions, expected_eval_predictions)
        assert_all_finite(actual_eval_std_devs, "train_std_devs")

        stacking_model_production = temp_path / "stacking-production.pkl"
        fit_output = invoke_model(
            "stacking",
            data_json,
            "--train-start",
            "0",
            "--train-end",
            "23",
            "--load-model",
            str(stacking_model_backtest),
            "--save-model",
            str(stacking_model_production),
            "--fit-only",
            "--pretty",
        )
        assert fit_output["mode"] == "stacking_fit_only"
        assert_method_metadata(fit_output, "stacking")
        assert fit_output["models"] == BASE_MODELS
        assert fit_output["saved_model"] == str(stacking_model_production)

        production_member_predictions = {
            model_name: extract_predictions(
                invoke_model(
                    model_name,
                    None,
                    "--load-model",
                    member_model_path(stacking_model_production, model_name),
                    "--forecast-steps",
                    str(future_steps),
                    "--pretty",
                )
            )
            for model_name in BASE_MODELS
        }
        expected_future_predictions = blend_predictions(
            BASE_MODELS,
            production_member_predictions,
            expected_weights,
        )

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
        assert_method_metadata(predict_output_one, "stacking")
        assert predict_output_one["forecast_steps"] == future_steps
        assert_series_close(future_predictions_one, expected_future_predictions)
        assert_series_close(future_predictions_one, future_predictions_two)
        assert_series_close(future_std_devs_one, future_std_devs_two)
        assert_all_finite(future_std_devs_one, "future_std_devs")

    print(
        json.dumps(
            {
                "fixture": str(CSV_FIXTURE),
                "checked_modes": ["training", "fit_only", "loaded_predict"],
                "model": "stacking",
                "self_trained_base_models": BASE_MODELS,
                "reused_base_models": BASE_MODELS,
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
