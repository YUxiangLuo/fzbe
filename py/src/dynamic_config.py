#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic, data-sized configuration rules shared by model runners."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable


LSTM_LIMITED_TIME_VALIDATION_REGIME = "limited_time_validation"
LSTM_TEACHING_DEMO_REGIME = "teaching_demo"


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def derive_arima_search_limit(train_size: int, d: int) -> Dict[str, int | str]:
    effective_size = int(train_size) - int(d)
    if effective_size < 6:
        raise ValueError(
            "[VALIDATION] ARIMA requires at least 6 effective training points "
            f"after differencing (got train_size={train_size}, d={d}, effective={effective_size})."
        )
    if effective_size <= 12:
        limit = 1
        regime = "teaching_small_sample"
    elif effective_size <= 20:
        limit = 2
        regime = "small_sample"
    else:
        limit = min(5, effective_size // 6)
        regime = "standard"
    return {
        "effective_train_size": effective_size,
        "max_p": limit,
        "max_q": limit,
        "regime": regime,
    }


def _largest_power_of_two_not_above(value: float) -> int:
    if value < 1:
        return 1
    return 1 << int(math.floor(math.log2(value)))


def lstm_parameter_count(units: int, feature_count: int, horizon: int) -> int:
    return 4 * units * (units + feature_count + 1) + (units + 1) * horizon


def derive_lstm_config(
    train_size: int,
    forecast_horizon: int,
    feature_count: int,
) -> Dict[str, Any]:
    n = int(train_size)
    horizon = int(forecast_horizon)
    features = int(feature_count)
    if horizon < 1:
        raise ValueError("[VALIDATION] LSTM forecast horizon must be at least 1.")
    if features < 1:
        raise ValueError("[VALIDATION] LSTM requires at least the target history feature.")

    preferred_look_back = clamp(n // 4, 2, 12)
    look_back = min(preferred_look_back, n - horizon - 3)
    if look_back < 2:
        raise ValueError(
            "[VALIDATION] Training range is too short for a direct LSTM forecast "
            "with at least two historical time steps: "
            f"train_size={n}, horizon={horizon}."
        )

    windows = n - look_back - horizon + 1
    if windows < 4:
        raise ValueError(
            "[VALIDATION] Direct LSTM requires at least 4 training windows after "
            f"look_back and horizon are applied (got {windows})."
        )

    validation_windows = max(4, math.ceil(0.2 * windows))
    purge_windows = max(0, horizon - 1)
    selection_train_windows = windows - validation_windows - purge_windows
    uses_limited_time_validation = windows >= 20 and selection_train_windows >= 4
    regime = (
        LSTM_LIMITED_TIME_VALIDATION_REGIME
        if uses_limited_time_validation
        else LSTM_TEACHING_DEMO_REGIME
    )
    if not uses_limited_time_validation:
        validation_windows = 0
        purge_windows = 0
        selection_train_windows = windows

    maximum_units = 16 if uses_limited_time_validation else 4
    # The explicitly labelled teaching demo may tolerate a slightly higher
    # parameter-to-supervised-target ratio so that the smallest 2-unit network
    # remains executable on a genuinely small classroom example.  The limited
    # time-validation regime keeps the stricter budget, without presenting the
    # short reused holdout as production-grade validation.
    parameters_per_target = 4 if uses_limited_time_validation else 6
    parameter_budget = parameters_per_target * windows * horizon
    candidates = [2, 4, 8, 16]
    valid_units = [
        units
        for units in candidates
        if units <= maximum_units
        and lstm_parameter_count(units, features, horizon) <= parameter_budget
    ]
    if not valid_units:
        minimum_parameters = lstm_parameter_count(2, features, horizon)
        raise ValueError(
            "[VALIDATION] Selected LSTM features are too complex for the available "
            f"windows (minimum_parameters={minimum_parameters}, budget={parameter_budget}); "
            "reduce features or expand the training range."
        )
    units = max(valid_units)

    batch_size = clamp(
        _largest_power_of_two_not_above(math.sqrt(selection_train_windows)),
        1,
        16,
    )
    steps_per_epoch = max(1, math.ceil(selection_train_windows / batch_size))
    if uses_limited_time_validation:
        max_epochs = clamp(math.ceil(600 / steps_per_epoch), 50, 200)
        patience = clamp(math.ceil(max_epochs * 0.1), 5, 20)
    else:
        max_epochs = clamp(math.ceil(160 / steps_per_epoch), 25, 60)
        patience = 0

    return {
        "regime": regime,
        "train_size": n,
        "forecast_horizon": horizon,
        "feature_count": features,
        "preferred_look_back": preferred_look_back,
        "look_back": look_back,
        "windows": windows,
        "validation_windows": validation_windows,
        "purge_windows": purge_windows,
        "selection_train_windows": selection_train_windows,
        "units": units,
        "parameter_count": lstm_parameter_count(units, features, horizon),
        "parameter_budget": parameter_budget,
        "parameters_per_target_budget": parameters_per_target,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "max_epochs": max_epochs,
        "patience": patience,
    }


def desired_internal_validation_size(purpose: str, train_size: int, model_count: int) -> int:
    base = math.ceil(0.2 * int(train_size))
    if purpose == "weighted":
        return max(3, int(model_count) + 1, base)
    if purpose == "boosting":
        return max(3, base)
    if purpose == "stacking":
        return max(int(model_count) + 2, base)
    raise ValueError(f"[VALIDATION] Unknown ensemble split purpose: {purpose}")


def plan_internal_holdout(
    purpose: str,
    train_size: int,
    model_count: int,
    is_feasible: Callable[[int, int], bool],
) -> Dict[str, Any]:
    n = int(train_size)
    k = int(model_count)
    if n < 4:
        raise ValueError("[VALIDATION] Ensemble training requires at least 4 points.")
    desired = desired_internal_validation_size(purpose, n, k)

    if purpose == "stacking":
        # Keep enough Level-1 rows for NNLS (K coefficients plus two degrees of
        # freedom), but allow a smaller chronological holdout when the desired
        # 20% split would leave a selected base model untrainable.
        candidates: Iterable[int] = range(min(desired, n - 2), k + 1, -1)
    else:
        candidates = range(min(desired, n - 2), 1, -1)

    for validation_size in candidates:
        prefix_size = n - validation_size
        if validation_size >= 2 and prefix_size >= 2 and is_feasible(prefix_size, validation_size):
            return {
                "purpose": purpose,
                "train_size": n,
                "model_count": k,
                "prefix_size": prefix_size,
                "validation_size": validation_size,
                "desired_validation_size": desired,
                "teaching_small_sample": validation_size < desired or n < 20,
            }

    raise ValueError(
        "[VALIDATION] No chronological internal holdout can fit every selected base model "
        f"for {purpose} (train_size={n}, desired_validation={desired})."
    )


def derive_boosting_max_rounds(model_count: int, validation_size: int) -> int:
    return max(1, min(2 * int(model_count), int(validation_size) - 1))
