"""Helpers for logging CLI arguments without exposing datasets or server paths."""

from typing import Any, Dict


_REDACTED_KEYS = {
    "data_file",
    "save_model",
    "load_model",
    "base_model_files",
    "input_state_file",
    "output_state_file",
    "receipt_file",
    "artifact_dir",
}


def summarize_cli_args(args: Any) -> Dict[str, Any]:
    summarized = dict(vars(args))
    data = summarized.get("data")
    if isinstance(data, str):
        summarized["data"] = f"<json payload chars={len(data)}>"
    for key in _REDACTED_KEYS:
        if summarized.get(key) is not None:
            summarized[key] = "<redacted>"
    return summarized
