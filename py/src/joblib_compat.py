import pickle
from typing import Any

try:
    import joblib as _joblib  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    _joblib = None


def dump_joblib(obj: Any, path: str) -> None:
    if _joblib is not None:
        _joblib.dump(obj, path)
    else:
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)


def load_joblib(path: str) -> Any:
    if _joblib is not None:
        return _joblib.load(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)
