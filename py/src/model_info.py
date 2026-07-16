# -*- coding: utf-8 -*-
"""Shared algorithm and runtime compatibility metadata for model artifacts."""
from copy import deepcopy
from functools import lru_cache
from importlib import metadata
import platform
import sys
from typing import Any, Dict, Iterable, List


MODEL_ALGORITHM_REVISION = 4
ARTIFACT_SCHEMA_REVISION = 2
ARTIFACT_FORMAT = "trusted_pickle_state"

_RUNTIME_DISTRIBUTIONS = {
    "numpy": ("numpy",),
    "pandas": ("pandas",),
    "pmdarima": ("pmdarima",),
    "scikit_learn": ("scikit-learn",),
    "scipy": ("scipy",),
    "statsmodels": ("statsmodels",),
    "tensorflow": ("tensorflow", "tensorflow-cpu"),
}


@lru_cache(maxsize=1)
def _cached_runtime_compatibility_manifest() -> Dict[str, Any]:
    packages: Dict[str, str | None] = {}
    for logical_name, distributions in _RUNTIME_DISTRIBUTIONS.items():
        resolved: str | None = None
        for distribution in distributions:
            try:
                resolved = metadata.version(distribution)
                break
            except metadata.PackageNotFoundError:
                continue
        packages[logical_name] = resolved
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "implementation": platform.python_implementation(),
        "packages": packages,
    }


def runtime_compatibility_manifest() -> Dict[str, Any]:
    """Return an isolated manifest so artifact code cannot mutate the cache."""
    return deepcopy(_cached_runtime_compatibility_manifest())


def method_metadata(
    method_name: str,
    forecast_strategy: str,
    implementation_notes: Iterable[str],
) -> Dict[str, Any]:
    return {
        "algorithm_revision": MODEL_ALGORITHM_REVISION,
        "artifact_schema_revision": ARTIFACT_SCHEMA_REVISION,
        "artifact_format": ARTIFACT_FORMAT,
        "runtime_compatibility": runtime_compatibility_manifest(),
        "method_name": method_name,
        "forecast_strategy": forecast_strategy,
        "implementation_notes": list(implementation_notes),
    }


def attach_method_metadata(
    payload: Dict[str, Any],
    method_name: str,
    forecast_strategy: str,
    implementation_notes: Iterable[str],
) -> Dict[str, Any]:
    payload.update(method_metadata(method_name, forecast_strategy, implementation_notes))
    return payload


def ensure_state_metadata(
    state: Dict[str, Any],
    method_name: str,
    forecast_strategy: str,
    implementation_notes: Iterable[str],
) -> Dict[str, Any]:
    if state.get("algorithm_revision") != MODEL_ALGORITHM_REVISION:
        raise ValueError(
            "[MODEL] Model artifact was created by an obsolete algorithm revision; "
            "retrain the model before reuse."
        )
    if (
        state.get("artifact_schema_revision") != ARTIFACT_SCHEMA_REVISION
        or state.get("artifact_format") != ARTIFACT_FORMAT
    ):
        raise ValueError(
            "[MODEL] Model artifact uses an obsolete serialization schema; "
            "retrain the model before reuse."
        )
    saved_runtime = state.get("runtime_compatibility")
    current_runtime = runtime_compatibility_manifest()
    if saved_runtime != current_runtime:
        raise ValueError(
            "[MODEL] Model artifact runtime dependencies differ from the current "
            "environment; retrain the model before reuse."
        )
    if (
        state.get("method_name") != method_name
        or state.get("forecast_strategy") != forecast_strategy
    ):
        raise ValueError(
            "[MODEL] Model artifact method metadata does not match the current "
            "algorithm; retrain the model before reuse."
        )
    notes: List[str] = list(implementation_notes)
    if not isinstance(state.get("implementation_notes"), list):
        state["implementation_notes"] = notes
    return state
