#!/usr/bin/env python3
"""Fast contract test for shared automatic LSTM look-back selection."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict


PY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PY_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PY_ROOT))

import guided_ensemble_model
import guided_model
from model_defaults import resolve_lstm_look_back
from models import boosting, stacking, weighted_avg


TRAIN_SIZE = 40
PHASE_TRAIN_SIZE = 32
EXPLICIT_LOOK_BACK = 9
TARGET_KEY = "销售数量"
ENSEMBLE_MODULES = {
    "weighted_avg": weighted_avg,
    "stacking": stacking,
    "boosting": boosting,
}


def _build_data_json() -> str:
    records = [
        {
            "年份": 2023 + index // 12,
            "月份": index % 12 + 1,
            TARGET_KEY: 100 + index * 2,
            "价格指数": 95 + index * 0.25,
        }
        for index in range(46)
    ]
    return json.dumps(records, ensure_ascii=False)


DATA_JSON = _build_data_json()


def _make_args(look_back: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        data=DATA_JSON,
        train_start=0,
        train_end=TRAIN_SIZE - 1,
        eval_start=TRAIN_SIZE,
        eval_end=45,
        models="ma,lstm",
        look_back=look_back,
        epochs=None,
        feature_types=None,
        lstm_features=None,
        lstm_target_feature=TARGET_KEY,
        normalization=None,
        base_model_files=None,
        learning_rate=None,
        max_rounds=None,
        load_model=None,
        save_model=None,
        artifact_dir="/tmp/fangzhen-look-back-contract",
        fit_only=False,
        pretty=False,
    )


def _fake_result(model_type: str):
    y_true = [1.0] * 6
    y_pred = [1.0] * 6
    metrics: Dict[str, float] = {}
    if model_type == "weighted_avg":
        return metrics, {}, {"ma": 0.5, "lstm": 0.5}, y_true, y_pred, 0.0
    if model_type == "stacking":
        return metrics, {}, {"type": "contract-test"}, y_true, y_pred, 0.0
    return metrics, ["ma"], [1.0], {}, y_true, y_pred, 0.0, {}


def _capture_one_shot_look_back(model_type: str, look_back: int | None) -> int:
    module = ENSEMBLE_MODULES[model_type]
    captured_params: Dict[str, Dict[str, Any]] = {}
    original_train_and_evaluate = module.train_and_evaluate

    def fake_train_and_evaluate(*call_args, **_call_kwargs):
        captured_params.update(call_args[4])
        return _fake_result(model_type)

    module.train_and_evaluate = fake_train_and_evaluate
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            module.run(_make_args(look_back))
    finally:
        module.train_and_evaluate = original_train_and_evaluate

    return int(captured_params["lstm"]["look_back"])


def _capture_guided_look_back(model_type: str, look_back: int | None) -> int:
    state: Dict[str, Any] = {}
    args = _make_args(look_back)
    args.save_model = "/tmp/fangzhen-look-back-contract.pkl"
    guided_ensemble_model._initialize_state(model_type, args, state)
    return int(state["model_params"]["lstm"]["look_back"])


def _capture_guided_lstm_look_back(look_back: int | None) -> int:
    original_lstm = guided_model.lstm
    guided_model.lstm = SimpleNamespace(
        _parse_feature_key_list=lambda _raw: [],
        _validate_lstm_feature_types=lambda *_args: None,
        _validate_lstm_numeric_columns=lambda *_args: None,
        _validate_training_args=lambda *_args: None,
    )
    try:
        state: Dict[str, Any] = {}
        result = guided_model._run_lstm("prepare_data", _make_args(look_back), state)
    finally:
        guided_model.lstm = original_lstm
    return int(result["look_back"])


def main() -> None:
    helper_boundary_cases = {
        1: 3,
        11: 3,
        12: 3,
        23: 5,
        24: 6,
        PHASE_TRAIN_SIZE: 6,
        100: 6,
    }
    for training_size, expected in helper_boundary_cases.items():
        actual = resolve_lstm_look_back(training_size)
        if actual != expected:
            raise AssertionError(
                f"helper boundary mismatch for training_size={training_size}: {actual} != {expected}"
            )
    if resolve_lstm_look_back(PHASE_TRAIN_SIZE, EXPLICIT_LOOK_BACK) != EXPLICIT_LOOK_BACK:
        raise AssertionError("explicit LSTM look_back must not be capped by the automatic default")

    checks: Dict[str, Dict[str, int]] = {}
    for model_type, module in ENSEMBLE_MODULES.items():
        actual_phase_size = module._compute_phase1_train_size(TRAIN_SIZE)
        if actual_phase_size != PHASE_TRAIN_SIZE:
            raise AssertionError(
                f"{model_type} phase split changed: {actual_phase_size} != {PHASE_TRAIN_SIZE}"
            )

        one_shot_default = _capture_one_shot_look_back(model_type, None)
        guided_default = _capture_guided_look_back(model_type, None)
        one_shot_explicit = _capture_one_shot_look_back(model_type, EXPLICIT_LOOK_BACK)
        guided_explicit = _capture_guided_look_back(model_type, EXPLICIT_LOOK_BACK)

        if one_shot_default != 6 or guided_default != 6:
            raise AssertionError(
                f"{model_type} default drift: one-shot={one_shot_default}, guided={guided_default}"
            )
        if one_shot_explicit != EXPLICIT_LOOK_BACK or guided_explicit != EXPLICIT_LOOK_BACK:
            raise AssertionError(
                f"{model_type} explicit look_back changed: "
                f"one-shot={one_shot_explicit}, guided={guided_explicit}"
            )

        checks[model_type] = {
            "phase_train_size": actual_phase_size,
            "one_shot_default": one_shot_default,
            "guided_default": guided_default,
            "explicit": one_shot_explicit,
        }

    guided_lstm_default = _capture_guided_lstm_look_back(None)
    guided_lstm_explicit = _capture_guided_lstm_look_back(EXPLICIT_LOOK_BACK)
    if guided_lstm_default != 6 or guided_lstm_explicit != EXPLICIT_LOOK_BACK:
        raise AssertionError(
            "guided standalone LSTM did not preserve the shared default/explicit contract: "
            f"default={guided_lstm_default}, explicit={guided_lstm_explicit}"
        )

    print(json.dumps({
        "train_size": TRAIN_SIZE,
        "helper_boundary_cases": helper_boundary_cases,
        "checks": checks,
        "guided_lstm": {
            "default": guided_lstm_default,
            "explicit": guided_lstm_explicit,
        },
        "status": "ok",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
