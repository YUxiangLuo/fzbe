#!/usr/bin/env python3
"""Deterministic multi-seed regression gate for LSTM forecast quality."""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PY_ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

SEEDS = (7, 17, 42, 73, 101)
EPOCHS = 20

# Main-branch reference metrics plus the review budget: median RMSE/MAE may
# regress by at most 10%, and P90 RMSE by at most 15%.
SCENARIOS = (
    {
        "dataset": "core",
        "fixture": "core_models_smoke.csv",
        "train_stop": 20,
        "eval_stop": 24,
        "max_parameters": 25_715,
        "thresholds": {
            "minmax": {"median_rmse": 13.113, "median_mae": 11.855, "p90_rmse": 16.658},
            "zscore": {"median_rmse": 13.506, "median_mae": 12.507, "p90_rmse": 14.985},
        },
    },
    {
        "dataset": "rich",
        "fixture": "lstm_rich_smoke.csv",
        "train_stop": 28,
        "eval_stop": 31,
        "max_parameters": 46_724,
        "thresholds": {
            "minmax": {"median_rmse": 29.309, "median_mae": 28.906, "p90_rmse": 34.309},
            "zscore": {"median_rmse": 24.211, "median_mae": 23.840, "p90_rmse": 26.584},
        },
    },
)


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def assert_at_most(label: str, actual: float, maximum: float) -> None:
    if not np.isfinite(actual) or actual > maximum:
        raise AssertionError(f"{label}: {actual:.6f} exceeds {maximum:.6f}")


def main() -> None:
    import tensorflow as tf

    from data_utils import infer_feature_types, load_data_from_json
    from model_defaults import resolve_lstm_look_back
    from models import lstm

    tf.get_logger().setLevel("ERROR")
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    summaries: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        dataset = str(scenario["dataset"])
        fixture = PY_ROOT / "test" / "fixtures" / str(scenario["fixture"])
        train_stop = int(scenario["train_stop"])
        eval_stop = int(scenario["eval_stop"])
        max_parameters = int(scenario["max_parameters"])
        thresholds_by_normalization = scenario["thresholds"]
        assert isinstance(thresholds_by_normalization, dict)

        with fixture.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
        parsed = load_data_from_json(json.dumps(records, ensure_ascii=False))
        feature_types = infer_feature_types(
            parsed,
            target_key=parsed.target_key,
            feature_keys=parsed.feature_keys,
        )
        look_back = resolve_lstm_look_back(train_stop)

        for normalization in ("minmax", "zscore"):
            rows: list[dict[str, float | int]] = []
            for seed in SEEDS:
                tf.keras.backend.clear_session()
                gc.collect()
                random.seed(seed)
                np.random.seed(seed)
                tf.keras.utils.set_random_seed(seed)

                started = time.perf_counter()
                metrics, model, *_ = lstm.train_and_evaluate(
                    parsed,
                    feature_types,
                    range(0, train_stop),
                    range(train_stop, eval_stop),
                    look_back,
                    EPOCHS,
                    normalization,
                    parsed.timestamps,
                )
                rows.append({
                    "seed": seed,
                    "rmse": float(metrics["rmse"]),
                    "mae": float(metrics["mae"]),
                    "duration_seconds": time.perf_counter() - started,
                    "parameters": int(model.count_params()),
                })
                del model

            observed = {
                "median_rmse": statistics.median(float(row["rmse"]) for row in rows),
                "median_mae": statistics.median(float(row["mae"]) for row in rows),
                "p90_rmse": percentile([float(row["rmse"]) for row in rows], 0.9),
                "median_duration_seconds": statistics.median(
                    float(row["duration_seconds"]) for row in rows
                ),
                "parameters": max(int(row["parameters"]) for row in rows),
            }
            thresholds = thresholds_by_normalization[normalization]
            assert isinstance(thresholds, dict)
            for metric in ("median_rmse", "median_mae", "p90_rmse"):
                assert_at_most(
                    f"{dataset}/{normalization} {metric}",
                    float(observed[metric]),
                    float(thresholds[metric]),
                )
            assert_at_most(
                f"{dataset}/{normalization} parameters",
                float(observed["parameters"]),
                float(max_parameters),
            )
            summaries.append({
                "dataset": dataset,
                "normalization": normalization,
                "look_back": look_back,
                "observed": observed,
                "thresholds": thresholds,
            })

    print(json.dumps({
        "status": "ok",
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "scenarios": summaries,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
