#!/usr/bin/env python3
"""Focused reproductions for the model-training review fixes."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

PY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PY_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ensemble_utils import (  # noqa: E402
    can_refit_source_artifact_for_internal_holdout,
    compute_phase1_train_size,
    estimate_residual_std,
    normalize_demand_forecast,
    parse_base_model_files_arg,
    parse_base_model_names,
    persist_member_model_artifact,
    resolve_base_model_artifact_configs,
    validate_internal_split_base_params,
)
from data_utils import load_data_from_json  # noqa: E402
from adf.core import adf_fixed_d  # noqa: E402
from error_protocol import format_known_error  # noqa: E402
from forecast_utils import (  # noqa: E402
    DEMAND_PREDICTION_DOMAIN,
    SIGNED_PREDICTION_DOMAIN,
    apply_prediction_domain,
    resolve_prediction_domain,
)
from log_safety import summarize_cli_args  # noqa: E402
from models import arima, boosting, es, lstm, ma, stacking, weighted_avg  # noqa: E402
from subprocess_utils import (  # noqa: E402
    MODEL_DEADLINE_ENV_NAME,
    _resolve_subprocess_timeout,
    extract_predictions_from_result,
    run_base_model_fit_subprocess,
    run_base_model_subprocess,
)


def sample_records(size: int = 18) -> list[dict[str, object]]:
    return [
        {
            "月份": f"{2024 + index // 12}-{index % 12 + 1:02d}",
            "销售数量": float(100 + index),
            "价格指数": float(index) / 10,
        }
        for index in range(size)
    ]


def assert_error_tag_contract() -> None:
    assert format_known_error(ValueError("[VALIDATION]abc")) == "[VALIDATION] abc"
    assert format_known_error(ValueError("[DATA]   中文内容")) == "[DATA] 中文内容"
    assert format_known_error(TimeoutError("base model expired")) == "[TIMEOUT] base model expired"
    assert format_known_error(TimeoutError("[TIMEOUT] nested")) == "[TIMEOUT] nested"


def assert_nested_timeout_respects_backend_deadline() -> None:
    with (
        patch.dict(os.environ, {MODEL_DEADLINE_ENV_NAME: "1002500"}),
        patch("subprocess_utils.time.time", return_value=1000.0),
    ):
        assert _resolve_subprocess_timeout(120) == 2.5
        assert _resolve_subprocess_timeout(1) == 1.0

    with (
        patch.dict(os.environ, {MODEL_DEADLINE_ENV_NAME: "999999"}),
        patch("subprocess_utils.time.time", return_value=1000.0),
    ):
        try:
            _resolve_subprocess_timeout(120)
        except TimeoutError as exc:
            assert "already expired" in str(exc)
        else:
            raise AssertionError("expired backend deadline was accepted")


def assert_expired_deadline_precedes_snapshot_creation() -> None:
    def fail_if_snapshot_is_created(_data: object) -> str:
        raise AssertionError("temporary snapshot was created after the deadline expired")

    runners = (
        lambda: run_base_model_subprocess("ma", [], 0, 0, 0, 0, {"window": 1}),
        lambda: run_base_model_fit_subprocess("ma", [], 0, 0, "source.pkl", "target.pkl"),
    )
    with (
        patch.dict(os.environ, {MODEL_DEADLINE_ENV_NAME: "999999"}),
        patch("subprocess_utils.time.time", return_value=1000.0),
        patch(
            "subprocess_utils._write_subprocess_data_file",
            side_effect=fail_if_snapshot_is_created,
        ),
    ):
        for runner in runners:
            try:
                runner()
            except TimeoutError as exc:
                assert "already expired" in str(exc)
            else:
                raise AssertionError("expired backend deadline was accepted")


def assert_log_redaction() -> None:
    args = argparse.Namespace(
        data='[{"secret":"REVIEW_SECRET"}]',
        data_file="/tmp/private/dataset.json",
        save_model="/srv/private/model.pkl",
        load_model="/srv/private/source.pkl",
        alpha=0.3,
    )
    summary = summarize_cli_args(args)
    rendered = repr(summary)
    assert "REVIEW_SECRET" not in rendered
    assert "/tmp/private" not in rendered
    assert "/srv/private" not in rendered
    assert summary["alpha"] == 0.3


def assert_cli_rejects_unknown_flags() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        required_guided_args = [
            "--output-state-file", os.path.join(temp_dir, "state.pkl"),
            "--receipt-file", os.path.join(temp_dir, "receipt.json"),
            "--attempt-id", "strict-cli",
            "--artifact-dir", os.path.join(temp_dir, "artifacts"),
        ]
        commands = (
            [sys.executable, str(PY_ROOT / "run_model.py"), "ma", "--widnow", "3"],
            [
                sys.executable,
                str(PY_ROOT / "guided_model.py"),
                "ma",
                "prepare_data",
                *required_guided_args,
                "--widnow", "3",
            ],
            [
                sys.executable,
                str(PY_ROOT / "guided_ensemble_model.py"),
                "weighted_avg",
                "prepare_data",
                *required_guided_args,
                "--widnow", "3",
            ],
        )
        for command in commands:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            assert completed.returncode != 0
            assert "unrecognized arguments: --widnow 3" in completed.stderr


def assert_ensemble_base_model_name_validation() -> None:
    assert parse_base_model_names(" ma,es ", "Regression") == ["ma", "es"]
    for raw_value, expected_message in (
        ("ma", "at least 2"),
        ("ma,ma", "duplicates"),
        ("ma,unknown", "unsupported"),
    ):
        try:
            parse_base_model_names(raw_value, "Regression")
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"invalid ensemble base model list was accepted: {raw_value}")

    parsed_data = load_data_from_json(json.dumps(sample_records(), ensure_ascii=False))
    params = {"ma": {"window": 3}}
    for validator in (
        weighted_avg._validate_training_args,
        stacking._validate_training_args,
        boosting._validate_training_args,
    ):
        try:
            validator(parsed_data, 0, 11, 12, 17, ["ma", "ma"], params)
        except ValueError as exc:
            assert "duplicates" in str(exc)
        else:
            raise AssertionError(f"{validator.__module__} accepted duplicate base models")


def assert_obsolete_ensemble_states_are_rejected() -> None:
    try:
        stacking._predict_with_meta_model(object(), np.ones((2, 2)))
    except ValueError as exc:
        assert "meta-model state is invalid" in str(exc)
    else:
        raise AssertionError("legacy sklearn Stacking meta-model was accepted")

    with tempfile.TemporaryDirectory() as temp_dir:
        legacy_boosting_path = os.path.join(temp_dir, "legacy-boosting.pkl")
        with open(legacy_boosting_path, "wb") as handle:
            pickle.dump({
                "model_type": "boosting",
                "model_chain": ["ma"],
                "base_model_states": {"ma": {"models": []}},
            }, handle)
        try:
            boosting._load_state(legacy_boosting_path)
        except ValueError as exc:
            assert "Invalid Boosting model file" in str(exc)
        else:
            raise AssertionError("obsolete Boosting artifact was accepted")


def assert_target_only_lstm_training() -> None:
    records_json = json.dumps(sample_records(14), ensure_ascii=False)
    completed = subprocess.run(
        [
            sys.executable,
            str(PY_ROOT / "run_model.py"),
            "lstm",
            "--data", records_json,
            "--train-start", "0",
            "--train-end", "10",
            "--eval-start", "11",
            "--eval-end", "13",
            "--look-back", "3",
            "--epochs", "1",
            "--feature-types", json.dumps({"销售数量": "numeric"}, ensure_ascii=False),
            "--lstm-features", "[]",
            "--lstm-target-feature", "销售数量",
            "--normalization", "minmax",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"target-only LSTM training failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    payload = json.loads(completed.stdout)
    assert len(payload["predictions"]) == 3
    assert payload["feature_types"] == {"销售数量": "numeric"}


def assert_internal_split_boundaries() -> None:
    assert [compute_phase1_train_size(size) for size in (8, 12, 16)] == [6, 10, 12]
    for size in (8, 12, 16):
        phase1 = compute_phase1_train_size(size)
        validate_internal_split_base_params(
            ["ma", "es"],
            {"ma": {"window": phase1}, "es": {"alpha": 0.3}},
            size,
            "regression",
        )
        try:
            validate_internal_split_base_params(
                ["ma", "es"],
                {"ma": {"window": phase1 + 1}, "es": {"alpha": 0.3}},
                size,
                "regression",
            )
        except ValueError as exc:
            assert str(exc).startswith("[VALIDATION] MA window")
        else:
            raise AssertionError("MA window above the internal split was accepted")


def assert_internal_holdout_configuration_boundaries() -> None:
    for model_name in ("ma", "es", "lstm"):
        assert can_refit_source_artifact_for_internal_holdout(model_name)
    assert not can_refit_source_artifact_for_internal_holdout("arima")
    assert not can_refit_source_artifact_for_internal_holdout("unknown")


def assert_source_artifact_configuration_resolution() -> None:
    parsed_data = load_data_from_json(json.dumps(sample_records(18), ensure_ascii=False))
    fallback_params = {
        "ma": {"window": 17},
        "es": {"alpha": 0.8},
        "arima": {"d": 0, "max_p": 4, "max_q": 5},
        "lstm": {
            "look_back": 3,
            "epochs": 20,
            "feature_types": {"销售数量": "numeric"},
            "normalization": "minmax",
            "lstm_target_feature": "销售数量",
        },
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source_states = {
            "ma": {"model_type": "ma", "window": 3},
            "es": {"model_type": "es", "alpha": 0.2},
            # Exercise the saved-order fallback used by valid ARIMA artifacts.
            "arima": {"model_type": "arima", "order": {"p": 2, "d": 1, "q": 2}},
            "lstm": {
                "model_type": "lstm",
                "look_back": 5,
                "epochs": 7,
                "target_key": "销售数量",
                "feature_keys": ["价格指数"],
                "feature_types": {"销售数量": "numeric", "价格指数": "numeric"},
                "normalization": "zscore",
            },
        }
        source_files: dict[str, str] = {}
        for model_name, source_state in source_states.items():
            source_file = temp_path / f"{model_name}.pkl"
            with source_file.open("wb") as handle:
                pickle.dump(source_state, handle)
            source_files[model_name] = str(source_file)

        resolved_params, validated_files = resolve_base_model_artifact_configs(
            parsed_data,
            ["ma", "es", "arima", "lstm"],
            fallback_params,
            source_files,
            "Regression ensemble",
        )
        assert validated_files == source_files
        assert resolved_params["ma"]["window"] == 3
        assert math.isclose(resolved_params["es"]["alpha"], 0.2)
        assert resolved_params["arima"] == {"d": 1, "max_p": 4, "max_q": 5}
        assert resolved_params["lstm"]["look_back"] == 5
        assert resolved_params["lstm"]["epochs"] == 7
        assert resolved_params["lstm"]["normalization"] == "zscore"
        assert json.loads(resolved_params["lstm"]["lstm_features"]) == ["价格指数"]
        # Resolution returns a canonical copy rather than rewriting the CLI fallback.
        assert fallback_params["ma"]["window"] == 17
        assert fallback_params["es"]["alpha"] == 0.8
        assert fallback_params["arima"]["d"] == 0

        no_artifact_params, no_artifact_files = resolve_base_model_artifact_configs(
            parsed_data,
            ["ma", "es"],
            {"ma": {"window": 3}, "es": {"alpha": 0.4}},
            {},
            "Regression ensemble",
        )
        assert no_artifact_params == {"ma": {"window": 3}, "es": {"alpha": 0.4}}
        assert no_artifact_files == {}

        invalid_cases = (
            ({"ma": str(temp_path / "missing.pkl")}, "does not exist"),
            ({"ma": source_files["es"]}, "mismatched model type"),
            ({"ma": source_files["ma"], "es": source_files["es"]}, "unselected model"),
        )
        for files, expected_message in invalid_cases:
            try:
                resolve_base_model_artifact_configs(
                    parsed_data,
                    ["ma"],
                    {"ma": {"window": 4}},
                    files,
                    "Regression ensemble",
                )
            except ValueError as exc:
                assert str(exc).startswith("[VALIDATION]")
                assert expected_message in str(exc)
            else:
                raise AssertionError(f"invalid source artifact mapping was accepted: {files}")

        corrupt_file = temp_path / "corrupt.pkl"
        corrupt_file.write_bytes(b"not a pickle")
        try:
            resolve_base_model_artifact_configs(
                parsed_data,
                ["ma"],
                {"ma": {"window": 4}},
                {"ma": str(corrupt_file)},
                "Regression ensemble",
            )
        except ValueError as exc:
            assert "cannot be read" in str(exc)
        else:
            raise AssertionError("corrupt source artifact was accepted")

    for invalid_json in ('{"ma":""}', '{"ma":3}'):
        try:
            parse_base_model_files_arg(invalid_json)
        except ValueError as exc:
            assert str(exc).startswith("[VALIDATION] --base-model-files entry")
        else:
            raise AssertionError(f"invalid base-model-files payload was accepted: {invalid_json}")


def assert_moving_average_teaching_window_boundary() -> None:
    sales = [10.0, 11.0, 12.0, 13.0, 14.0]
    for validator, args in (
        (ma._validate_fit_args, (sales, 0, 3, 1)),
        (ma._validate_training_args, (sales, 0, 2, 3, 4, 1)),
    ):
        try:
            validator(*args)
        except ValueError as exc:
            assert "Window must be >= 2" in str(exc)
        else:
            raise AssertionError("MA window=1 was accepted by a training entry point")

    try:
        ma.predict(
            "",
            1,
            state={
                "window": 1,
                "history": [10.0],
                "std_dev_residuals": 1.0,
                "prediction_domain": DEMAND_PREDICTION_DOMAIN,
            },
        )
    except ValueError as exc:
        assert "Window must be >= 2" in str(exc)
    else:
        raise AssertionError("MA window=1 was accepted by the artifact prediction path")


def assert_residual_std_small_sample_fallbacks() -> None:
    # Independent evaluation residuals win when their sample standard deviation
    # is identifiable.
    assert math.isclose(
        estimate_residual_std([1.0, 5.0], [10.0, 20.0], [100.0, 200.0]),
        math.sqrt(8.0),
    )

    # A one-point evaluation cannot identify dispersion, so ensembles use the
    # chronological internal holdout instead of reporting a misleading zero.
    assert math.isclose(
        estimate_residual_std([2.0], [1.0, 3.0], [100.0, 200.0]),
        math.sqrt(2.0),
    )

    # If both residual sources contain only one point, fall back to the training
    # series spread, matching the base-model small-data convention.
    assert math.isclose(
        estimate_residual_std([2.0], [1.0], [10.0, 14.0]),
        math.sqrt(8.0),
    )

    try:
        estimate_residual_std([float("nan")], [1.0, 2.0], [3.0, 4.0])
    except ValueError as exc:
        assert str(exc).startswith("[DATA] evaluation")
    else:
        raise AssertionError("non-finite residuals were accepted")


def assert_adf_teaching_configuration() -> None:
    fake_result = (
        -3.5,
        0.01,
        1,
        6,
        {"1%": -4.0, "5%": -3.0, "10%": -2.5},
        12.0,
    )
    with patch("adf.core.adfuller", return_value=fake_result) as mocked_adfuller:
        result = adf_fixed_d([1, 2, 3, 4, 5, 6, 7, 8], diff_orders=(0,))
    assert result[0]["stationary"] is True
    assert result[0]["p_value"] == 0.01
    assert mocked_adfuller.call_args.kwargs == {
        "regression": "c",
        "autolag": "AIC",
    }


def assert_adf_input_validation() -> None:
    fake_result = (
        -3.5,
        0.01,
        1,
        6,
        {"1%": -4.0, "5%": -3.0, "10%": -2.5},
        12.0,
    )
    with patch("adf.core.adfuller", return_value=fake_result) as mocked_adfuller:
        generated_orders = (order for order in (0, 1))
        result = adf_fixed_d(
            np.arange(10, dtype=float),
            diff_orders=generated_orders,
        )
    assert [row["diff_order"] for row in result] == [0, 1]
    assert mocked_adfuller.call_count == 2

    for invalid_values, expected_message in (
        ([[1.0, 2.0], [3.0, 4.0]], "1-D"),
        ([1.0, float("nan"), 3.0, 4.0], "NaN or inf"),
    ):
        try:
            adf_fixed_d(invalid_values, diff_orders=(0,))
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"invalid ADF values were accepted: {invalid_values}")

    for invalid_orders, expected_message in (
        ((0, 0), "duplicates"),
        ((-1,), "non-negative integers"),
        ((0.5,), "non-negative integers"),
    ):
        try:
            adf_fixed_d(np.arange(10, dtype=float), diff_orders=invalid_orders)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"invalid ADF orders were accepted: {invalid_orders}")


def assert_demand_semantics() -> None:
    assert normalize_demand_forecast([-2.0, 3.0]).tolist() == [0.0, 3.0]
    assert apply_prediction_domain([-2.0, 3.0], DEMAND_PREDICTION_DOMAIN).tolist() == [0.0, 3.0]
    assert apply_prediction_domain([-2.0, 3.0], SIGNED_PREDICTION_DOMAIN).tolist() == [-2.0, 3.0]
    assert boosting._apply_stage_output_semantics(0, [-2.0, 3.0]).tolist() == [0.0, 3.0]
    assert boosting._apply_stage_output_semantics(1, [-2.0, 3.0]).tolist() == [-2.0, 3.0]

    for corrupt_state in (
        {"prediction_domain": "typo"},
        {"model_type": "ma"},
    ):
        try:
            resolve_prediction_domain(corrupt_state)
        except ValueError as exc:
            assert "Unknown prediction domain" in str(exc)
        else:
            raise AssertionError(f"invalid prediction domain was accepted: {corrupt_state}")


def assert_boosting_ma_uses_complete_teaching_windows() -> None:
    fitted = boosting._compute_ma_in_sample_fitted(
        np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]),
        window=3,
    )
    assert fitted.tolist() == [0.0, 0.0, 0.0, 20.0, 30.0]

    try:
        boosting._normalize_fitted_values([1.0, float("inf")], 2)
    except ValueError as exc:
        assert str(exc).startswith("[DATA] Boosting stage produced infinite")
    else:
        raise AssertionError("infinite Boosting in-sample fitted values were accepted")


def assert_base_model_prediction_domains() -> None:
    ma_state = {
        "window": 2,
        "history": [-4.0, -2.0],
        "std_dev_residuals": 1.0,
        "prediction_domain": DEMAND_PREDICTION_DOMAIN,
    }
    demand_predictions, _ = ma.predict("", 1, state=ma_state)
    signed_predictions, _ = ma.predict(
        "",
        1,
        state=ma_state,
        allow_negative_predictions=True,
    )
    assert demand_predictions == [0.0]
    assert signed_predictions == [-3.0]

    class FakeEsModel:
        @staticmethod
        def forecast(_steps: int):
            return [-5.0, 2.0]

    es_state = {
        "model": FakeEsModel(),
        "alpha": 0.3,
        "std_dev_residuals": 1.0,
        "prediction_domain": DEMAND_PREDICTION_DOMAIN,
    }
    demand_predictions, _ = es.predict("", 2, state=es_state)
    signed_predictions, _ = es.predict(
        "",
        2,
        state=es_state,
        allow_negative_predictions=True,
    )
    assert demand_predictions == [0.0, 2.0]
    assert signed_predictions == [-5.0, 2.0]

    class FakeArimaModel:
        @staticmethod
        def predict(*, n_periods: int, return_conf_int: bool):
            assert n_periods == 2 and return_conf_int is True
            import numpy as np

            return np.asarray([-6.0, 4.0]), np.asarray([[-8.0, -4.0], [2.0, 6.0]])

    arima_state = {
        "model": FakeArimaModel(),
        "order": {"p": 0, "d": 0, "q": 0},
        "prediction_domain": DEMAND_PREDICTION_DOMAIN,
    }
    demand_predictions, intervals = arima.predict("", 2, state=arima_state)
    signed_predictions, _ = arima.predict(
        "",
        2,
        state=arima_state,
        allow_negative_predictions=True,
    )
    assert demand_predictions == [0.0, 4.0]
    assert signed_predictions == [-6.0, 4.0]
    assert intervals[0] == [-8.0, -4.0]

    with patch.object(lstm, "_forecast_direct_from_state", return_value=([-7.0, 5.0], 1.0)):
        demand_predictions, _ = lstm._forecast_from_state(
            {
                "forecast_strategy": lstm.DIRECT_FORECAST_STRATEGY,
                "prediction_domain": DEMAND_PREDICTION_DOMAIN,
            },
            2,
        )
        signed_predictions, _ = lstm._forecast_from_state(
            {
                "forecast_strategy": lstm.DIRECT_FORECAST_STRATEGY,
                "prediction_domain": DEMAND_PREDICTION_DOMAIN,
            },
            2,
            SIGNED_PREDICTION_DOMAIN,
        )
    assert demand_predictions == [0.0, 5.0]
    assert signed_predictions == [-7.0, 5.0]


def assert_clipped_predictions_drive_python_metrics() -> None:
    metrics, _model, y_true, y_pred, std_dev = es.train_and_evaluate(
        [-10.0, -8.0, 2.0, 4.0],
        ["1", "2", "3", "4"],
        range(0, 2),
        range(2, 4),
        0.5,
    )
    assert y_true == [2.0, 4.0]
    assert y_pred == [0.0, 0.0]
    assert math.isclose(metrics["rmse"], math.sqrt(10.0))
    assert math.isclose(std_dev, math.sqrt(2.0))


def assert_boosting_requests_signed_base_outputs() -> None:
    captured_params: dict[str, object] = {}

    def fake_train(**kwargs: object) -> dict[str, object]:
        captured_params.update(kwargs["model_params"])  # type: ignore[arg-type]
        return {"predictions": [{"prediction": -3.0}]}

    with patch.object(boosting, "run_base_model_subprocess", side_effect=fake_train):
        _model_file, predictions = boosting._train_stage_model(
            "ma",
            [{"销售数量": 1.0}, {"销售数量": -1.0}, {"销售数量": -2.0}],
            [],
            range(0, 2),
            range(2, 3),
            {"window": 2},
            10,
        )
    assert captured_params["allow_negative_predictions"] is True
    assert predictions.tolist() == [-3.0]


def assert_boosting_keeps_signed_corrections_end_to_end() -> None:
    from data_utils import load_data_from_json

    records = [
        {"月份": f"2026-{index + 1:02d}", "销售数量": 1.0}
        for index in range(10)
    ]
    parsed_data = load_data_from_json(json.dumps(records, ensure_ascii=False))
    training_calls = iter((
        ("candidate-r0-ma.pkl", np.asarray([3.0, 3.0])),
        ("candidate-r0-es.pkl", np.asarray([4.0, 4.0])),
        ("candidate-r1-ma.pkl", np.asarray([-4.0, -4.0])),
        ("candidate-r1-es.pkl", np.asarray([1.0, 1.0])),
        ("final-r0-ma.pkl", np.asarray([3.0, 3.0])),
        ("final-r1-ma.pkl", np.asarray([-4.0, -4.0])),
    ))

    def fake_train_stage(*_args: object, **_kwargs: object):
        return next(training_calls)

    def fake_fitted(
        _model_name: str,
        model_file: str,
        _parsed_data: object,
        train_range: range,
    ) -> np.ndarray:
        value = -4.0 if "r1" in model_file else (4.0 if "es" in model_file else 3.0)
        return np.full(len(train_range), value, dtype=float)

    with (
        patch.object(boosting, "_build_round_records", return_value=([], [])),
        patch.object(boosting, "_train_stage_model", side_effect=fake_train_stage),
        patch.object(boosting, "_compute_in_sample_fitted_predictions", side_effect=fake_fitted),
        patch.object(boosting, "remove_member_model_artifact"),
    ):
        (
            _metrics,
            model_chain,
            stage_coefficients,
            base_model_states,
            _y_true,
            y_pred,
            std_dev,
            _search_summary,
        ) = boosting.train_and_evaluate(
            parsed_data,
            range(0, 8),
            range(8, 10),
            ["ma", "es"],
            {"ma": {"window": 2}, "es": {"alpha": 0.3}},
            learning_rate=0.5,
            max_rounds=2,
        )

    assert model_chain == ["ma", "ma"]
    assert stage_coefficients == [1.0, 0.5]
    assert y_pred == [1.0, 1.0]

    saved_state = {
        "model_type": boosting.MODEL_TYPE,
        "boosting_variant": "residual",
        "model_chain": model_chain,
        "stage_coefficients": stage_coefficients,
        "base_model_states": base_model_states,
        "std_dev_residuals": std_dev,
    }
    prediction_calls: list[dict[str, object]] = []

    def fake_predict_stage(**kwargs: object) -> dict[str, object]:
        prediction_calls.append(kwargs)
        value = 3.0 if len(prediction_calls) == 1 else -4.0
        return {"forecasts": [value]}

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_path = os.path.join(temp_dir, "boosting.pkl")
        with open(artifact_path, "wb") as handle:
            pickle.dump(saved_state, handle)
        with patch.object(
            boosting,
            "run_base_model_predict_subprocess",
            side_effect=fake_predict_stage,
        ):
            production_prediction, _ = boosting.predict(artifact_path, 1)

    assert production_prediction == [1.0]
    assert all(call["allow_negative_predictions"] is True for call in prediction_calls)


def assert_subprocess_prediction_domains() -> None:
    records = [
        {"月份": f"2026-{index + 1:02d}", "销售数量": value}
        for index, value in enumerate([-10.0, -8.0, -6.0, -4.0, -2.0, 0.0])
    ]
    common = {
        "model_type": "ma",
        "data": records,
        "train_start": 0,
        "train_end": 3,
        "eval_start": 4,
        "eval_end": 5,
        "timeout": 30,
    }
    demand_result = run_base_model_subprocess(
        **common,
        model_params={"window": 2},
    )
    signed_result = run_base_model_subprocess(
        **common,
        model_params={"window": 2, "allow_negative_predictions": True},
    )
    assert extract_predictions_from_result(demand_result) == [0.0, 0.0]
    assert extract_predictions_from_result(signed_result) == [-5.0, -4.5]


def assert_loaded_ensemble_demand_semantics() -> None:
    base_states = {
        "ma": {"model_file": "ma.pkl"},
        "es": {"model_file": "es.pkl"},
    }

    def fake_runner(values: list[float]):
        iterator = iter(values)
        return lambda **_: {"forecasts": [next(iterator)]}

    with patch.object(
        weighted_avg,
        "run_base_model_predict_subprocess",
        side_effect=fake_runner([-10.0, 4.0]),
    ):
        predictions, _ = weighted_avg.predict("", 1, {
            "weights": {"ma": 0.5, "es": 0.5},
            "base_model_states": base_states,
            "models": ["ma", "es"],
        })
        assert predictions == [2.0]

    with patch.object(
        stacking,
        "run_base_model_predict_subprocess",
        side_effect=fake_runner([-10.0, 4.0]),
    ):
        predictions, _ = stacking.predict("", 1, {
            "meta_model": {"kind": "weighted_blend", "weights": [0.5, 0.5]},
            "base_model_states": base_states,
            "models": ["ma", "es"],
        })
        assert predictions == [2.0]

    boosting_state = {
        "model_chain": ["ma", "es"],
        "stage_coefficients": [1.0, 0.5],
        "base_model_states": {
            "ma": {"models": [{"model_file": "ma.pkl"}]},
            "es": {"models": [{"model_file": "es.pkl"}]},
        },
    }
    with patch.object(
        boosting,
        "run_base_model_predict_subprocess",
        side_effect=fake_runner([-10.0, 12.0]),
    ):
        predictions, _ = boosting.predict("", 1, boosting_state)
        assert predictions == [6.0]

    with patch.object(
        boosting,
        "run_base_model_predict_subprocess",
        side_effect=fake_runner([1.0, -10.0]),
    ):
        predictions, _ = boosting.predict("", 1, boosting_state)
        assert predictions == [0.0]


def assert_same_path_artifact_replay() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "member.pkl")
        with open(model_path, "wb") as handle:
            handle.write(b"artifact")
        persist_member_model_artifact("ma", model_path, model_path, move_source=False)
        assert Path(model_path).read_bytes() == b"artifact"


def assert_corrupt_state_errors() -> None:
    records_json = json.dumps(sample_records(8), ensure_ascii=False)
    for module, model_type, missing_name in (
        (ma, "ma", "window"),
        (es, "es", "alpha"),
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "source.pkl")
            target = os.path.join(temp_dir, "target.pkl")
            with open(source, "wb") as handle:
                pickle.dump({
                    "model_type": model_type,
                    "prediction_domain": DEMAND_PREDICTION_DOMAIN,
                }, handle)
            args = argparse.Namespace(
                data=records_json,
                train_start=0,
                train_end=7,
                save_model=target,
                load_model=source,
                fit_only=True,
                pretty=False,
                window=None,
                alpha=None,
            )
            try:
                module.run(args)
            except ValueError as exc:
                if model_type == "ma":
                    assert str(exc).startswith("[VALIDATION] Invalid MA model window")
                else:
                    assert str(exc).startswith(f"[VALIDATION] Saved {model_type.upper()} model")
                assert missing_name in str(exc)
            else:
                raise AssertionError(f"corrupt {model_type} state was accepted")


def assert_large_dataset_uses_data_file() -> None:
    records = [
        {f"feature_{column}": float(row + column) for column in range(20)}
        for row in range(10_000)
    ]
    captured_path: str | None = None

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal captured_path
        assert "--data" not in args
        data_file_index = args.index("--data-file")
        captured_path = args[data_file_index + 1]
        assert os.path.exists(captured_path)
        assert stat.S_IMODE(os.stat(captured_path).st_mode) & 0o077 == 0
        assert len(Path(captured_path).read_bytes()) > 2_000_000
        return subprocess.CompletedProcess(args, 0, '{"predictions":[]}', "")

    with patch("subprocess_utils.subprocess.run", side_effect=fake_run):
        result = run_base_model_subprocess(
            "ma",
            records,
            0,
            7999,
            8000,
            9999,
            {"window": 3},
        )
    assert result == {"predictions": []}
    assert captured_path is not None and not os.path.exists(captured_path)


def assert_guided_attempt_replay() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        candidate = temp_path / "candidate.pkl"
        receipt = temp_path / "receipt.json"
        artifact_dir = temp_path / "artifacts"
        command = [
            sys.executable,
            str(PY_ROOT / "guided_model.py"),
            "ma",
            "prepare_data",
            "--output-state-file", str(candidate),
            "--receipt-file", str(receipt),
            "--attempt-id", "attempt-replay",
            "--artifact-dir", str(artifact_dir),
            "--data", json.dumps(sample_records(), ensure_ascii=False),
            "--train-start", "0",
            "--train-end", "11",
            "--eval-start", "12",
            "--eval-end", "17",
            "--window", "4",
            "--save-model", str(artifact_dir / "ma.pkl"),
        ]
        first = subprocess.run(command, capture_output=True, text=True, check=True)
        first_payload = json.loads(first.stdout)
        checkpoint_bytes = candidate.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True, check=True)
        second_payload = json.loads(second.stdout)
        assert second_payload == first_payload
        assert candidate.read_bytes() == checkpoint_bytes
        assert second_payload["candidate_state_sha256"] == first_payload["candidate_state_sha256"]


def assert_ensemble_source_config_conflicts_end_to_end() -> None:
    records = [
        {
            "月份": f"{2024 + index // 12}-{index % 12 + 1:02d}",
            "销售数量": float(80 + 2 * index + (index % 4) * 3),
            "价格指数": float(100 + index) / 100,
        }
        for index in range(22)
    ]
    data_json = json.dumps(records, ensure_ascii=False)

    def invoke_model(model_type: str, arguments: list[object]) -> dict[str, Any]:
        command = [
            sys.executable,
            str(PY_ROOT / "run_model.py"),
            model_type,
            "--data",
            data_json,
            *[str(argument) for argument in arguments],
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"{model_type} source-config regression failed\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return json.loads(completed.stdout)

    def prediction_values(payload: dict[str, Any]) -> list[float]:
        return [float(point["prediction"]) for point in payload["predictions"]]

    expected_params: dict[str, dict[str, float | int]] = {
        "ma": {"window": 3},
        "es": {"alpha": 0.2},
        "arima": {"d": 1, "max_p": 1, "max_q": 1},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source_files: dict[str, str] = {}
        source_arguments = {
            "ma": ["--window", "3"],
            "es": ["--alpha", "0.2"],
            "arima": ["--d", "1", "--max-p", "1", "--max-q", "1"],
        }
        for model_name, model_arguments in source_arguments.items():
            source_file = temp_path / f"source-{model_name}.pkl"
            invoke_model(
                model_name,
                [
                    "--train-start", "0",
                    "--train-end", "17",
                    "--eval-start", "18",
                    "--eval-end", "21",
                    *model_arguments,
                    "--save-model", source_file,
                ],
            )
            source_files[model_name] = str(source_file)

        common_arguments: list[object] = [
            "--train-start", "0",
            "--train-end", "17",
            "--eval-start", "18",
            "--eval-end", "21",
            "--models", "ma,es,arima",
            "--max-p", "1",
            "--max-q", "1",
            "--base-model-files", json.dumps(source_files),
        ]
        canonical_arguments = ["--window", "3", "--alpha", "0.2", "--d", "1"]
        conflicting_arguments = ["--window", "7", "--alpha", "0.8", "--d", "0"]

        for ensemble_type in ("weighted_avg", "stacking", "boosting"):
            extra_arguments: list[object] = (
                ["--learning-rate", "0.3", "--max-rounds", "2"]
                if ensemble_type == "boosting"
                else []
            )
            canonical_file = temp_path / f"canonical-{ensemble_type}.pkl"
            conflicting_file = temp_path / f"conflicting-{ensemble_type}.pkl"
            canonical_output = invoke_model(
                ensemble_type,
                [
                    *common_arguments,
                    *canonical_arguments,
                    *extra_arguments,
                    "--save-model", canonical_file,
                ],
            )
            conflicting_output = invoke_model(
                ensemble_type,
                [
                    *common_arguments,
                    *conflicting_arguments,
                    *extra_arguments,
                    "--save-model", conflicting_file,
                ],
            )

            assert np.allclose(
                prediction_values(conflicting_output),
                prediction_values(canonical_output),
                rtol=1e-9,
                atol=1e-9,
            ), f"{ensemble_type} still depends on conflicting CLI base parameters"
            if ensemble_type == "weighted_avg":
                assert all(
                    math.isclose(
                        float(conflicting_output["weights"][name]),
                        float(canonical_output["weights"][name]),
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                    for name in expected_params
                )
            if ensemble_type == "boosting":
                assert conflicting_output["model_chain"] == canonical_output["model_chain"]
                assert np.allclose(
                    conflicting_output["stage_coefficients"],
                    canonical_output["stage_coefficients"],
                )

            saved_state = pickle.loads(conflicting_file.read_bytes())
            for model_name, model_state in saved_state["base_model_states"].items():
                stage_states = model_state.get("models", [model_state])
                for stage_state in stage_states:
                    assert stage_state["params"] == expected_params[model_name]

            guided_state_file = temp_path / f"guided-{ensemble_type}.pkl"
            guided_receipt_file = temp_path / f"guided-{ensemble_type}.json"
            guided_arguments: list[object] = [
                PY_ROOT / "guided_ensemble_model.py",
                ensemble_type,
                "prepare_data",
                "--output-state-file", guided_state_file,
                "--receipt-file", guided_receipt_file,
                "--attempt-id", f"artifact-config-{ensemble_type}",
                "--artifact-dir", temp_path / f"guided-artifacts-{ensemble_type}",
                "--data", data_json,
                *common_arguments[:-2],
                "--base-model-files", json.dumps(source_files),
                *conflicting_arguments,
                *extra_arguments,
                "--save-model", temp_path / f"guided-final-{ensemble_type}.pkl",
            ]
            completed = subprocess.run(
                [sys.executable, *[str(argument) for argument in guided_arguments]],
                capture_output=True,
                text=True,
                check=False,
                timeout=300,
            )
            if completed.returncode != 0:
                raise AssertionError(
                    f"guided {ensemble_type} source-config regression failed\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            guided_state = pickle.loads(guided_state_file.read_bytes())
            assert guided_state["model_params"] == expected_params

        missing_file_arguments: list[object] = [
            "--train-start", "0",
            "--train-end", "17",
            "--eval-start", "18",
            "--eval-end", "21",
            "--models", "ma,es",
            "--window", "3",
            "--alpha", "0.2",
            "--base-model-files",
            json.dumps({"ma": str(temp_path / "missing.pkl"), "es": source_files["es"]}),
        ]
        completed = subprocess.run(
            [
                sys.executable,
                str(PY_ROOT / "run_model.py"),
                "weighted_avg",
                "--data",
                data_json,
                *[str(argument) for argument in missing_file_arguments],
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert completed.returncode != 0
        assert "[VALIDATION]" in f"{completed.stdout}\n{completed.stderr}"
        assert "does not exist" in f"{completed.stdout}\n{completed.stderr}"


def main() -> None:
    assert_error_tag_contract()
    assert_nested_timeout_respects_backend_deadline()
    assert_expired_deadline_precedes_snapshot_creation()
    assert_log_redaction()
    assert_cli_rejects_unknown_flags()
    assert_ensemble_base_model_name_validation()
    assert_obsolete_ensemble_states_are_rejected()
    assert_target_only_lstm_training()
    assert_internal_split_boundaries()
    assert_internal_holdout_configuration_boundaries()
    assert_source_artifact_configuration_resolution()
    assert_moving_average_teaching_window_boundary()
    assert_residual_std_small_sample_fallbacks()
    assert_adf_teaching_configuration()
    assert_adf_input_validation()
    assert_demand_semantics()
    assert_boosting_ma_uses_complete_teaching_windows()
    assert_base_model_prediction_domains()
    assert_clipped_predictions_drive_python_metrics()
    assert_boosting_requests_signed_base_outputs()
    assert_boosting_keeps_signed_corrections_end_to_end()
    assert_subprocess_prediction_domains()
    assert_loaded_ensemble_demand_semantics()
    assert_same_path_artifact_replay()
    assert_corrupt_state_errors()
    assert_large_dataset_uses_data_file()
    assert_guided_attempt_replay()
    assert_ensemble_source_config_conflicts_end_to_end()
    print(json.dumps({"status": "PASS", "suite": "review_regressions"}))


if __name__ == "__main__":
    main()
