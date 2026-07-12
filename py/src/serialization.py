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
import os
import tempfile
from typing import Any, Optional
from pathlib import Path
try:
    from .durable_io import fsync_parent_directory
except ImportError:  # Script entrypoints add py/src directly to sys.path.
    from durable_io import fsync_parent_directory


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
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None

    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            dir=str(target_path.parent),
        )
        temp_path = Path(temp_name)

        with os.fdopen(fd, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(temp_path, target_path)
        fsync_parent_directory(str(target_path))
        temp_path = None
    except (pickle.PicklingError, TypeError) as e:
        raise SerializationError(f"Failed to serialize object: {e}") from e
    except OSError as e:
        raise OSError(f"Failed to write to '{path}': {e}") from e
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


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
