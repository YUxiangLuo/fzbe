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
from typing import Any, Dict, List, Optional


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

    # Build command arguments
    data_json = json.dumps(data, ensure_ascii=False)
    args = [
        python_exec,
        run_model_script,
        model_type,
        "--data-stdin",
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
            input=data_json,
            timeout=timeout,
            env={**os.environ, "PATH": f"{os.path.join(py_dir, 'venv', 'bin')}:{os.environ.get('PATH', '')}"}
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Base model {model_type} failed with exit code {result.returncode}.\n"
                f"stderr: {result.stderr}\n"
                f"stdout: {result.stdout}"
            )

        # Parse JSON output
        output = result.stdout.strip()

        # Find the JSON part (skip any log messages before it)
        json_start = output.find('{')
        if json_start == -1:
            raise ValueError(f"No JSON output found from {model_type}. stdout: {output}")

        json_output = output[json_start:]
        return json.loads(json_output)

    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Base model {model_type} timed out after {timeout}s")
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from {model_type}.\n"
            f"Error: {e}\n"
            f"stdout: {result.stdout}"
        )


def run_base_model_fit_subprocess(
    model_type: str,
    data: List[Dict[str, Any]],
    train_start: int,
    train_end: int,
    source_model_file: str,
    save_model_file: str,
    timeout: Optional[int] = 60,
) -> Dict[str, Any]:
    """Refit a base model from an existing artifact using fit-only mode."""
    py_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_exec = get_python_executable()
    run_model_script = os.path.join(py_dir, "run_model.py")

    if not os.path.exists(run_model_script):
        raise FileNotFoundError(f"run_model.py not found at {run_model_script}")

    data_json = json.dumps(data, ensure_ascii=False)
    args = [
        python_exec,
        run_model_script,
        model_type,
        "--data-stdin",
        "--train-start", str(train_start),
        "--train-end", str(train_end),
        "--load-model", source_model_file,
        "--save-model", save_model_file,
        "--fit-only",
        "--pretty",
    ]

    try:
        result = subprocess.run(
            args,
            cwd=py_dir,
            capture_output=True,
            text=True,
            input=data_json,
            timeout=timeout,
            env={**os.environ, "PATH": f"{os.path.join(py_dir, 'venv', 'bin')}:{os.environ.get('PATH', '')}"}
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Base model {model_type} fit-only failed with exit code {result.returncode}.\n"
                f"stderr: {result.stderr}\n"
                f"stdout: {result.stdout}"
            )

        output = result.stdout.strip()
        json_start = output.find('{')
        if json_start == -1:
            raise ValueError(f"No JSON output found from {model_type} fit-only. stdout: {output}")

        json_output = output[json_start:]
        return json.loads(json_output)

    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Base model {model_type} fit-only timed out after {timeout}s")
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from {model_type} fit-only.\n"
            f"Error: {e}\n"
            f"stdout: {result.stdout}"
        )


def run_base_model_predict_subprocess(
    model_type: str,
    model_file: str,
    forecast_steps: int,
    timeout: Optional[int] = 30,
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

    try:
        result = subprocess.run(
            args,
            cwd=py_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PATH": f"{os.path.join(py_dir, 'venv', 'bin')}:{os.environ.get('PATH', '')}"}
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Base model {model_type} predict failed.\n"
                f"stderr: {result.stderr}\n"
                f"stdout: {result.stdout}"
            )

        output = result.stdout.strip()
        json_start = output.find('{')
        if json_start == -1:
            raise ValueError(f"No JSON output from {model_type} predict")

        json_output = output[json_start:]
        return json.loads(json_output)

    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Base model {model_type} predict timed out")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from {model_type} predict: {e}")


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
