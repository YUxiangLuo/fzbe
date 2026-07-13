#!/usr/bin/env python3
"""Focused reproductions for the model-training review fixes."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


PY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PY_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ensemble_utils import (  # noqa: E402
    compute_phase1_train_size,
    normalize_demand_forecast,
    persist_member_model_artifact,
    validate_internal_split_base_params,
)
from error_protocol import format_known_error  # noqa: E402
from log_safety import summarize_cli_args  # noqa: E402
from models import boosting, es, ma, stacking, weighted_avg  # noqa: E402
from subprocess_utils import (  # noqa: E402
    MODEL_DEADLINE_ENV_NAME,
    _resolve_subprocess_timeout,
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


def assert_demand_semantics() -> None:
    assert normalize_demand_forecast([-2.0, 3.0]).tolist() == [0.0, 3.0]


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
            "meta_model": {"weights": [0.5, 0.5]},
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
                pickle.dump({"model_type": model_type}, handle)
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


def main() -> None:
    assert_error_tag_contract()
    assert_nested_timeout_respects_backend_deadline()
    assert_expired_deadline_precedes_snapshot_creation()
    assert_log_redaction()
    assert_internal_split_boundaries()
    assert_demand_semantics()
    assert_loaded_ensemble_demand_semantics()
    assert_same_path_artifact_replay()
    assert_corrupt_state_errors()
    assert_large_dataset_uses_data_file()
    assert_guided_attempt_replay()
    print("review regression reproductions: PASS")


if __name__ == "__main__":
    main()
