"""Profile annotations consumed by the sole deep-test runner."""
from __future__ import annotations

from collections.abc import Callable


def profile(name: str):
    if name not in {"fast", "full", "nightly", "resource"}:
        raise ValueError(name)
    def decorate(function: Callable):
        setattr(function, "fz_deep_profile", name)
        return function
    return decorate
