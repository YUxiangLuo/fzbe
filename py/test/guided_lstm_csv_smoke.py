#!/usr/bin/env python3
"""Smoke test for the guided-training LSTM path (guided_model.py).

The standalone LSTM experiment trains step-by-step via guided_model.py, which has
its own model builder. This test drives the real guided step sequence end-to-end
and guards against the 2-layer architecture regression (R-16): build_model must
report `recurrent_units` (single layer) and never `base_units`/`secondary_units`.
It also verifies the guided artifact is run_model.py-compatible (round-trip predict).
"""
import csv
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

PY_ROOT = Path(__file__).resolve().parents[1]
GUIDED_MODEL = PY_ROOT / "guided_model.py"
RUN_MODEL = PY_ROOT / "run_model.py"
CSV_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lstm_rich_smoke.csv"
FEATURE_KEYS = [
    "价格指数", "促销投入", "库存水平", "搜索热度", "竞争对手价格",
    "节假日强度", "渠道类型", "区域", "天气类型",
]
STEPS = ["prepare_data", "preprocess", "build_model", "train_model",
         "forecast", "evaluate_metrics", "save_artifact"]

sys.path.insert(0, str(PY_ROOT))


def assert_all_finite(values, label):
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise AssertionError(f"{label}[{index}] is not finite: {value}")


def run_guided_step(step, common_args):
    completed = subprocess.run(
        [sys.executable, str(GUIDED_MODEL), "lstm", step, *common_args],
        capture_output=True, text=True, cwd=str(PY_ROOT),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"guided lstm step '{step}' failed\nstderr:\n{completed.stderr[-2000:]}"
        )
    return json.loads(completed.stdout)


def assert_single_layer_architecture():
    """Directly assert the built Keras model is single-LSTM + Dropout (R-16 guard)."""
    import guided_model
    from tensorflow.keras.layers import LSTM as KerasLSTM
    from tensorflow.keras.layers import Dropout

    config = {"look_back": 4, "n_features_out": 6, "recurrent_units": 16,
              "dense_units": 8, "forecast_horizon": 3, "initial_learning_rate": 0.001}
    model = guided_model._build_lstm_model(config)
    lstm_layers = sum(1 for layer in model.layers if isinstance(layer, KerasLSTM))
    dropout_layers = sum(1 for layer in model.layers if isinstance(layer, Dropout))
    if lstm_layers != 1:
        raise AssertionError(f"expected exactly 1 LSTM layer (single-layer v2), got {lstm_layers}")
    if dropout_layers < 1:
        raise AssertionError("expected a Dropout regularization layer (v2), found none")


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    assert_single_layer_architecture()

    with CSV_FIXTURE.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    data_json = json.dumps(records, ensure_ascii=False)

    with tempfile.TemporaryDirectory(prefix="fangzhen-guided-lstm-") as temp_dir:
        temp_path = Path(temp_dir)
        state_file = temp_path / "state.pkl"
        artifact_dir = temp_path / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        save_model = temp_path / "guided-lstm.pkl"

        common_args = [
            "--state-file", str(state_file),
            "--artifact-dir", str(artifact_dir),
            "--data", data_json,
            "--train-start", "0", "--train-end", "27",
            "--eval-start", "28", "--eval-end", "30",
            "--look-back", "4", "--epochs", "2",
            "--normalization", "minmax",
            "--lstm-target-feature", "销售数量",
            "--lstm-features", ",".join(FEATURE_KEYS),
            "--save-model", str(save_model),
        ]

        final_result = None
        for step in STEPS:
            payload = run_guided_step(step, common_args)
            output = payload.get("output", {})
            if step == "build_model":
                if "recurrent_units" not in output:
                    raise AssertionError(f"build_model missing 'recurrent_units': {sorted(output)}")
                for stale_key in ("base_units", "secondary_units"):
                    if stale_key in output:
                        raise AssertionError(
                            f"build_model still reports two-layer key '{stale_key}' (R-16 regression)"
                        )
            if "final_result" in payload:
                final_result = payload["final_result"]

        if final_result is None:
            raise AssertionError("guided LSTM produced no final_result")
        metrics = final_result["metrics"]
        for key in ("rmse", "mae", "mape", "r2"):
            if not math.isfinite(float(metrics[key])):
                raise AssertionError(f"metric '{key}' is not finite: {metrics[key]}")
        eval_predictions = [float(item["prediction"]) for item in final_result["predictions"]]
        if len(eval_predictions) != 3:
            raise AssertionError(f"expected 3 eval predictions, got {len(eval_predictions)}")
        assert_all_finite(eval_predictions, "eval_predictions")

        # Guided artifact must be a valid, run_model.py-compatible LSTM model.
        with save_model.open("rb") as handle:
            saved_state = pickle.load(handle)
        if saved_state.get("model_type") != "lstm":
            raise AssertionError(f"saved model_type != lstm: {saved_state.get('model_type')}")
        if saved_state.get("forecast_strategy") != "direct_multi_horizon":
            raise AssertionError(f"unexpected forecast_strategy: {saved_state.get('forecast_strategy')}")

        # Round-trip: the guided artifact is loadable & predictable by run_model.py.
        predict_completed = subprocess.run(
            [sys.executable, str(RUN_MODEL), "lstm", "--load-model", str(save_model),
             "--forecast-steps", "3"],
            capture_output=True, text=True, cwd=str(PY_ROOT),
        )
        if predict_completed.returncode != 0:
            raise RuntimeError(
                f"round-trip predict failed\nstderr:\n{predict_completed.stderr[-2000:]}"
            )
        future_predictions = [
            float(item["prediction"])
            for item in json.loads(predict_completed.stdout)["predictions"]
        ]
        if len(future_predictions) != 3:
            raise AssertionError(f"expected 3 future predictions, got {len(future_predictions)}")
        assert_all_finite(future_predictions, "future_predictions")

    print(json.dumps({
        "fixture": str(CSV_FIXTURE),
        "checked": [
            "single_layer_architecture", "guided_steps",
            "recurrent_units_regression_guard", "final_metrics",
            "artifact_valid", "roundtrip_predict",
        ],
        "model": "guided_lstm",
        "status": "ok",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
