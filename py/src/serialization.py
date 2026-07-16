# -*- coding: utf-8 -*-
"""
Trusted model serialization layer.

Pickle is required because fitted statsmodels/scikit-learn/Keras-adjacent state
contains Python objects that do not have a shared portable wire format. Pickle
is neither safe for untrusted bytes nor cross-environment compatible. The
application therefore restricts artifact paths, verifies artifact digests before
loading, and embeds exact algorithm/schema/runtime metadata in every model state.
Artifacts must be retrained after incompatible code or dependency changes.
"""
import pickle
import os
import tempfile
from typing import Any, Optional
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
    # This function must only receive application-owned, integrity-checked
    # artifacts. Never expose it as a generic untrusted pickle loader.
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"[MODEL] Model file not found: {path}\n"
            f"Please ensure the model has been trained and saved first."
        )
    if not file_path.is_file():
        raise SerializationError(f"[MODEL] Model path is not a regular file: {path}")

    # Load the object. Any failure here means the artifact cannot be
    # reconstructed: corrupt/truncated bytes (UnpicklingError/EOFError/ValueError),
    # a pickle that references removed or renamed code (ModuleNotFoundError/
    # ImportError/AttributeError), or an unreadable file (OSError). All of these
    # are load failures, so surface them through the [MODEL] protocol uniformly
    # and let the service invalidate the artifact and require a retrain.
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        raise SerializationError(
            f"[MODEL] Failed to load model from '{path}'.\n"
            f"The file may be corrupted, unreadable, or in an incompatible format.\n"
            f"Error: {type(e).__name__}: {e}"
        ) from e
