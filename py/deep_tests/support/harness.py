"""Strict JSON subprocess and artifact harness for deep model tests."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_ROOT = REPO_ROOT / "py"
DATASETS = Path(__file__).resolve().parents[1] / "datasets"
MODEL_TYPES = ("ma", "es", "arima", "lstm", "weighted_avg", "stacking", "boosting")
ENSEMBLES = {"weighted_avg", "stacking", "boosting"}
ERROR_TAGS = ("[DATA]", "[VALIDATION]", "[MODEL]", "[TIMEOUT]")
FIXED_ENV = {
    "PYTHONHASHSEED": "20250308",
    "TF_DETERMINISTIC_OPS": "1",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
}


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any] | None


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def parse_strict_json(text: str) -> dict[str, Any]:
    value = json.loads(text, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("stdout must be one JSON object")
    assert_finite_tree(value)
    return value


def assert_finite_tree(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_tree(child, f"{path}[{index}]")


def run_command(argv: Iterable[str | Path], timeout: int = 300, expected_ok: bool = True) -> CommandResult:
    command = [str(part) for part in argv]
    env = os.environ.copy()
    env.update(FIXED_ENV)
    try:
        process = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(f"[TIMEOUT] after {timeout}s: {' '.join(command)}\nstdout={str(exc.stdout)[-2000:]}\nstderr={str(exc.stderr)[-2000:]}") from exc
    payload = None
    if process.returncode == 0:
        try:
            payload = parse_strict_json(process.stdout)
        except Exception as exc:
            raise AssertionError(f"invalid strict JSON from {' '.join(command)}: {exc}\nstdout={process.stdout[-4000:]}\nstderr={process.stderr[-4000:]}") from exc
    result = CommandResult(command, process.returncode, process.stdout, process.stderr, payload)
    if expected_ok and process.returncode != 0:
        raise AssertionError(f"command failed rc={process.returncode}: {' '.join(command)}\nstdout={process.stdout[-4000:]}\nstderr={process.stderr[-4000:]}")
    if not expected_ok and process.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {' '.join(command)}")
    if not expected_ok and not any(tag in process.stderr for tag in ERROR_TAGS):
        raise AssertionError(f"failure lacks stable error tag: {process.stderr[-4000:]}")
    return result


def common_args(model: str, data_file: Path, save_model: Path | None = None, fixture: str = "seasonal_pulse_60.json") -> list[str]:
    del fixture
    args = [sys.executable, PY_ROOT / "run_model.py", model, "--data-file", data_file,
            "--train-start", "0", "--train-end", "35", "--eval-start", "40", "--eval-end", "43"]
    if save_model:
        args += ["--save-model", save_model]
    if model == "ma":
        args += ["--window", "3"]
    elif model == "es":
        args += ["--alpha", "0.4"]
    elif model == "arima":
        args += ["--d", "1"]
    elif model == "lstm":
        args += ["--lstm-target-feature", "y", "--lstm-features", "[]", "--feature-types", '{"y":"numeric"}', "--normalization", "minmax"]
    else:
        args += ["--models", "ma,es", "--window", "3", "--alpha", "0.4"]
    return [str(item) for item in args]


def run_model(model: str, data_file: Path, save_model: Path | None = None, timeout: int | None = None) -> CommandResult:
    return run_command(common_args(model, data_file, save_model), timeout=timeout or (900 if model in ENSEMBLES else 600 if model == "lstm" else 300))


def run_loaded(model: str, artifact: Path, steps: int = 4, expected_ok: bool = True, *, allow_negative: bool = False) -> CommandResult:
    argv: list[str | Path] = [sys.executable, PY_ROOT / "run_model.py", model, "--load-model", artifact, "--forecast-steps", str(steps)]
    if allow_negative:
        argv.append("--allow-negative-predictions")
    return run_command(argv, timeout=600 if model == "lstm" else 300, expected_ok=expected_ok)


def prediction_values(payload: dict[str, Any]) -> list[float]:
    return [float(point["prediction"]) for point in payload["predictions"]]


def assert_prediction_contract(payload: dict[str, Any], expected: int) -> None:
    points = payload.get("predictions")
    if not isinstance(points, list) or len(points) != expected:
        raise AssertionError(f"prediction length mismatch: expected={expected}, got={len(points) if isinstance(points, list) else type(points)}")
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise AssertionError(f"prediction {index} is not an object")
        for key in ("prediction", "std_dev", "interval_lower", "interval_upper", "upper_error_p99"):
            if key not in point or not math.isfinite(float(point[key])):
                raise AssertionError(f"prediction {index}.{key} is missing/non-finite")
        if float(point["prediction"]) < 0 or float(point["std_dev"]) < 0 or float(point["upper_error_p99"]) < 0:
            raise AssertionError(f"prediction {index} violates demand/uncertainty domain")
        if float(point["interval_lower"]) > float(point["interval_upper"]):
            raise AssertionError(f"prediction {index} interval is inverted")
    for key in ("method_name", "forecast_strategy", "implementation_notes", "algorithm_revision"):
        if not payload.get(key):
            raise AssertionError(f"missing method metadata: {key}")


def mutate_records(source: Path, destination: Path, indices: range, *, target_delta: float = 0, features: bool = False) -> None:
    records = json.loads(source.read_text(encoding="utf-8"))
    for index in indices:
        if target_delta:
            records[index]["y"] = float(records[index]["y"]) + target_delta + index
        if features:
            records[index]["promo"] = 999000 + index
            records[index]["region"] = "FUTURE_ONLY"
    destination.write_text(json.dumps(records, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def corrupt_copy(source: Path, destination: Path, mode: str = "truncate") -> None:
    shutil.copy2(source, destination)
    data = destination.read_bytes()
    destination.write_bytes(data[: max(1, len(data) // 2)] if mode == "truncate" else b"not-a-pickle")


def repository_artifacts() -> set[Path]:
    ignored = {PY_ROOT / "venv"}
    result = set()
    for suffix in ("*.pkl", "*.h5", "*.receipt.json", "*.checkpoint.json", ".*.tmp-*"):
        for path in REPO_ROOT.rglob(suffix):
            if not any(parent == ignored_path or ignored_path in parent.parents for ignored_path in ignored for parent in [path]):
                if "guided_training" not in path.parts:  # tracked application fixtures predate this suite
                    result.add(path)
    return result


class IsolatedWorkspace:
    def __enter__(self) -> Path:
        self.before = repository_artifacts()
        self.temp = tempfile.TemporaryDirectory(prefix="fz-deep-")
        return Path(self.temp.name)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.temp.cleanup()
        after = repository_artifacts()
        if after != self.before:
            created = sorted(map(str, after - self.before))
            removed = sorted(map(str, self.before - after))
            message = f"repository artifacts changed outside TemporaryDirectory: created={created}, removed={removed}"
            if exc is not None:
                raise AssertionError(message) from exc
            raise AssertionError(message)
