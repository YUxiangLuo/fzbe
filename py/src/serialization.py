# -*- coding: utf-8 -*-
"""
Model serialization compatibility layer.

This module provides a consistent API for model serialization across all
time series models. Uses pickle exclusively to ensure 100% cross-environment
compatibility, regardless of whether joblib is installed.

Design Decision:
    We use pickle (instead of joblib) because:
    - Guaranteed compatibility across all environments
    - No dependency conflicts
    - Simpler to debug and maintain
    - joblib's optimizations are not critical for our model sizes

    While joblib can be more efficient for large numpy arrays, our models
    are small enough that the performance difference is negligible.
"""
import pickle
from typing import Any
from pathlib import Path


class SerializationError(Exception):
    """Raised when model serialization or deserialization fails."""
    pass


def dump_model(obj: Any, path: str) -> None:
    """
    Serialize an object to disk using pickle.

    Args:
        obj: The object to serialize (typically a model state dict)
        path: File path where the object will be saved

    Raises:
        SerializationError: If the object cannot be serialized
        OSError: If the file cannot be written (permissions, disk full, etc.)

    Example:
        >>> state = {"model_type": "ma", "window": 3, "history": [1, 2, 3]}
        >>> dump_model(state, "trained_models/ma_model.pkl")
    """
    try:
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PicklingError, TypeError) as e:
        raise SerializationError(f"Failed to serialize object: {e}") from e
    except OSError as e:
        raise OSError(f"Failed to write to '{path}': {e}") from e


def load_model(path: str) -> Any:
    """
    Deserialize an object from disk using pickle.

    Args:
        path: File path to load from

    Returns:
        The deserialized object (typically a model state dict)

    Raises:
        FileNotFoundError: If the file does not exist
        SerializationError: If the file is corrupted or format is invalid
        OSError: If the file cannot be read (permissions, etc.)

    Example:
        >>> state = load_model("trained_models/ma_model.pkl")
        >>> print(state["model_type"])
        'ma'
    """
    # Validate file exists
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"[MODEL] Model file not found: {path}\n"
            f"Please ensure the model has been trained and saved first."
        )

    # Load the object
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, ValueError) as e:
        raise SerializationError(
            f"[MODEL] Failed to load model from '{path}'.\n"
            f"The file may be corrupted or in an incompatible format.\n"
            f"Error: {e}"
        ) from e
    except OSError as e:
        raise OSError(f"Failed to read from '{path}': {e}") from e
