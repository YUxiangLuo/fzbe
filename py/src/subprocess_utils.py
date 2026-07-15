#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified subprocess utilities for running base models.
Ensures consistency between API calls and ensemble model calls.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import numpy as np


MODEL_DEADLINE_ENV_NAME = "FZ_MODEL_DEADLINE_EPOCH_MS"


def _resolve_subprocess_timeout(requested_timeout: Optional[float]) -> Optional[float]:
    """Cap a nested model timeout to the deadline inherited from the backend."""
    raw_deadline = os.environ.get(MODEL_DEADLINE_ENV_NAME, "").strip()
    if not raw_deadline:
        return requested_timeout

    try:
        deadline_epoch_ms = int(raw_deadline)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid {MODEL_DEADLINE_ENV_NAME}: {raw_deadline}"
        ) from exc

    remaining_seconds = deadline_epoch_ms / 1000.0 - time.time()
    if remaining_seconds <= 0:
        raise TimeoutError("Model execution deadline has already expired")

    if requested_timeout is None:
        return remaining_seconds
    return min(float(requested_timeout), remaining_seconds)


def get_python_executable() -> str:
    """Get the path to the Python executable in the virtual environment."""
    py_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Try venv/bin/python first (Unix-like)
    venv_python = os.path.join(py_dir, "venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python

    # Try venv/Scripts/python.exe (Windows)
    venv_python_win = os.path.join(py_dir, "venv", "Scripts", "python.exe")
    if os.path.exists(venv_python_win):
        return venv_python_win

    # Fallback to system python
    return sys.executable


def _write_subprocess_data_file(data: List[Dict[str, Any]]) -> str:
    fd, file_path = tempfile.mkstemp(prefix="fz-model-data-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.remove(file_path)
        except OSError:
            pass
        raise
    return file_path


def _parse_json_stdout(result: subprocess.CompletedProcess[str], context: str) -> Dict[str, Any]:
    output = result.stdout.strip()
    if not output:
        raise ValueError(f"No JSON output from {context}")
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse JSON from {context}. Error: {exc}; "
            f"stdout: {output[:2000]}"
        ) from exc


def _raise_tagged_subprocess_error(stderr: str) -> None:
    for line in reversed(stderr.splitlines()):
        for tag in ("[VALIDATION]", "[DATA]"):
            marker_index = line.find(tag)
            if marker_index >= 0:
                detail = line[marker_index + len(tag):].strip()
                raise ValueError(f"{tag} {detail}".rstrip())


def run_base_model_subprocess(
    model_type: str,
    data: List[Dict[str, Any]],
    train_start: int,
    train_end: int,
    eval_start: int,
    eval_end: int,
    model_params: Dict[str, Any],
    timeout: Optional[int] = 60,
) -> Dict[str, Any]:
    """
    Run a base model (ma, es, arima, lstm) training in a subprocess.

    Args:
        model_type: Type of model to run (ma, es, arima, lstm)
        data: Training data as list of dicts
        train_start: Training start index
        train_end: Training end index
        eval_start: Evaluation start index
        eval_end: Evaluation end index
        model_params: Model-specific parameters (including save_model if needed)
        timeout: Timeout in seconds (default 60s)

    Returns:
        Dict containing model results (predictions, metrics, etc.)
    """
    py_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_exec = get_python_executable()
    run_model_script = os.path.join(py_dir, "run_model.py")

    if not os.path.exists(run_model_script):
        raise FileNotFoundError(f"run_model.py not found at {run_model_script}")

    # Validate the inherited deadline before allocating a temporary snapshot so
    # an already-expired request cannot leak fz-model-data-*.json files.
    effective_timeout = _resolve_subprocess_timeout(timeout)
    data_file = _write_subprocess_data_file(data)
    args = [
        python_exec,
        run_model_script,
        model_type,
        "--data-file", data_file,
        "--train-start", str(train_start),
        "--train-end", str(train_end),
        "--eval-start", str(eval_start),
        "--eval-end", str(eval_end),
        "--pretty",
    ]

    # Add model-specific parameters
    for key, value in model_params.items():
        # Convert Python snake_case to CLI kebab-case
        cli_key = key.replace("_", "-")
        args.append(f"--{cli_key}")

        # Handle boolean flags
        if isinstance(value, bool):
            if not value:
                # Skip false boolean flags
                args.pop()  # Remove the flag we just added
        else:
            # Convert value to string
            if isinstance(value, dict):
                args.append(json.dumps(value, ensure_ascii=False))
            else:
                args.append(str(value))

    # Run subprocess
    try:
        result = subprocess.run(
            args,
            cwd=py_dir,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            env={**os.environ, "PATH": f"{os.path.join(py_dir, 'venv', 'bin')}:{os.environ.get('PATH', '')}"}
        )

        if result.returncode != 0:
            _raise_tagged_subprocess_error(result.stderr)
            raise RuntimeError(
                f"Base model {model_type} failed with exit code {result.returncode}.\n"
                f"stderr: {result.stderr}\n"
                f"stdout: {result.stdout}"
            )

        return _parse_json_stdout(result, model_type)

    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"Base model {model_type} timed out after {effective_timeout:.3f}s"
        )
    finally:
        try:
            os.remove(data_file)
        except OSError:
            pass


def run_base_model_fit_subprocess(
    model_type: str,
    data: List[Dict[str, Any]],
    train_start: int,
    train_end: int,
    source_model_file: str,
    save_model_file: str,
    fit_forecast_steps: Optional[int] = None,
    predict_steps: Optional[int] = None,
    timeout: Optional[int] = 60,
    allow_negative_predictions: bool = False,
) -> Dict[str, Any]:
    """Refit a base model from an existing artifact using fit-only mode."""
    py_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_exec = get_python_executable()
    run_model_script = os.path.join(py_dir, "run_model.py")

    if not os.path.exists(run_model_script):
        raise FileNotFoundError(f"run_model.py not found at {run_model_script}")

    if fit_forecast_steps is not None and int(fit_forecast_steps) <= 0:
        raise ValueError("fit_forecast_steps must be > 0 when provided")
    if predict_steps is not None and int(predict_steps) <= 0:
        raise ValueError("predict_steps must be > 0 when provided")

    # Keep deadline validation outside the snapshot lifecycle only while no
    # snapshot exists; every later failure is covered by the finally below.
    effective_timeout = _resolve_subprocess_timeout(timeout)
    data_file = _write_subprocess_data_file(data)
    args = [
        python_exec,
        run_model_script,
        model_type,
        "--data-file", data_file,
        "--train-start", str(train_start),
        "--train-end", str(train_end),
        "--load-model", source_model_file,
        "--save-model", save_model_file,
        "--fit-only",
        "--pretty",
    ]
    if fit_forecast_steps is not None:
        args.extend(["--fit-forecast-steps", str(int(fit_forecast_steps))])
    if predict_steps is not None:
        args.extend(["--fit-predict-steps", str(int(predict_steps))])
    if allow_negative_predictions:
        args.append("--allow-negative-predictions")

    try:
        result = subprocess.run(
            args,
            cwd=py_dir,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            env={**os.environ, "PATH": f"{os.path.join(py_dir, 'venv', 'bin')}:{os.environ.get('PATH', '')}"}
        )

        if result.returncode != 0:
            _raise_tagged_subprocess_error(result.stderr)
            raise RuntimeError(
                f"Base model {model_type} fit-only failed with exit code {result.returncode}.\n"
                f"stderr: {result.stderr}\n"
                f"stdout: {result.stdout}"
            )

        return _parse_json_stdout(result, f"{model_type} fit-only")

    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"Base model {model_type} fit-only timed out after {effective_timeout:.3f}s"
        )
    finally:
        try:
            os.remove(data_file)
        except OSError:
            pass


def run_base_model_predict_subprocess(
    model_type: str,
    model_file: str,
    forecast_steps: int,
    timeout: Optional[int] = 30,
    allow_negative_predictions: bool = False,
) -> Dict[str, Any]:
    """
    Run a base model prediction in a subprocess.

    Args:
        model_type: Type of model (ma, es, arima, lstm)
        model_file: Path to the saved model file
        forecast_steps: Number of steps to forecast
        timeout: Timeout in seconds (default 30s)

    Returns:
        Dict containing prediction results
    """
    py_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_exec = get_python_executable()
    run_model_script = os.path.join(py_dir, "run_model.py")

    args = [
        python_exec,
        run_model_script,
        model_type,
        "--load-model", model_file,
        "--forecast-steps", str(forecast_steps),
        "--pretty",
    ]
    if allow_negative_predictions:
        args.append("--allow-negative-predictions")

    effective_timeout = _resolve_subprocess_timeout(timeout)

    try:
        result = subprocess.run(
            args,
            cwd=py_dir,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            env={**os.environ, "PATH": f"{os.path.join(py_dir, 'venv', 'bin')}:{os.environ.get('PATH', '')}"}
        )

        if result.returncode != 0:
            _raise_tagged_subprocess_error(result.stderr)
            raise RuntimeError(
                f"Base model {model_type} predict failed.\n"
                f"stderr: {result.stderr}\n"
                f"stdout: {result.stdout}"
            )

        return _parse_json_stdout(result, f"{model_type} predict")

    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"Base model {model_type} predict timed out after {effective_timeout:.3f}s"
        )


def extract_predictions_from_result(result: Dict[str, Any]) -> List[float]:
    """
    Extract prediction values from a base model result.

    Handles different output formats:
    - predictions: [{"prediction": val, "std_dev": sd}, ...]
    - forecasts: [val1, val2, ...]
    """
    # Try predictions format first
    if "predictions" in result and isinstance(result["predictions"], list):
        if len(result["predictions"]) > 0:
            if isinstance(result["predictions"][0], dict) and "prediction" in result["predictions"][0]:
                return [p["prediction"] for p in result["predictions"]]
            else:
                return result["predictions"]

    # Try forecasts format
    if "forecasts" in result:
        return result["forecasts"]

    # Try eval_predictions
    if "eval_predictions" in result:
        return result["eval_predictions"]

    raise ValueError(f"Cannot extract predictions from result: {result.keys()}")


def extract_uncertainty_profile_from_result(result: Dict[str, Any]) -> List[float]:
    """Extract the per-horizon standard deviations from a model response."""
    points = result.get("predictions")
    if not isinstance(points, list) or not points:
        raise ValueError("Base model result does not contain prediction uncertainty.")
    profile: List[float] = []
    for point in points:
        if not isinstance(point, dict) or "std_dev" not in point:
            raise ValueError("Base model prediction is missing std_dev.")
        value = float(point["std_dev"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("Base model prediction std_dev must be finite and nonnegative.")
        profile.append(value)
    return profile
