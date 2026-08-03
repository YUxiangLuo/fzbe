# -*- coding: utf-8 -*-
"""Shared metadata helpers for teaching-oriented model descriptions."""
from typing import Any, Dict, Iterable, List


MODEL_ALGORITHM_REVISION = 4


def method_metadata(
    method_name: str,
    forecast_strategy: str,
    implementation_notes: Iterable[str],
) -> Dict[str, Any]:
    return {
        "algorithm_revision": MODEL_ALGORITHM_REVISION,
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
    state.setdefault("method_name", method_name)
    state.setdefault("forecast_strategy", forecast_strategy)
    notes: List[str] = list(implementation_notes)
    if not isinstance(state.get("implementation_notes"), list):
        state["implementation_notes"] = notes
    return state
