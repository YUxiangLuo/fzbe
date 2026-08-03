#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared causal baselines for selection, evaluation, and saved forecasts."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_utils import ParsedData

BASELINE_LABELS = {
    "mean": "训练均值",
    "naive": "上一期延续基线",
    "seasonal_naive": "同季延续基线",
}

CALENDAR_SEASONAL_ALIGNMENT = "calendar_month_slots_v1"


def _try_month_codes(timestamps: Sequence[Any]) -> Optional[np.ndarray]:
    parsed = pd.to_datetime(list(timestamps), errors="coerce")
    if len(parsed) == 0 or parsed.isna().any():
        return None
    return np.asarray(
        [value.year * 12 + value.month - 1 for value in parsed],
        dtype=int,
    )


def _require_month_codes(
    timestamps: Sequence[Any],
    expected_length: int,
    context: str,
) -> np.ndarray:
    values = list(timestamps)
    if len(values) != int(expected_length):
        raise ValueError(
            f"[VALIDATION] {context} timestamp count does not match its values."
        )
    month_codes = _try_month_codes(values)
    if month_codes is None:
        raise ValueError(f"[DATA] {context} contains an invalid timestamp.")
    if month_codes.size > 1 and np.any(np.diff(month_codes) <= 0):
        raise ValueError(
            f"[DATA] {context} timestamps must be strictly increasing by month."
        )
    return month_codes


def _seasonal_cadence_months(period: int) -> int:
    if period < 2 or 12 % period != 0:
        raise ValueError(
            "[VALIDATION] Calendar-aligned seasonal naïve requires a period "
            "that divides 12 months."
        )
    return 12 // period


def _validate_seasonal_grid(
    month_codes: np.ndarray,
    cadence_months: int,
    phase_remainder: int,
    context: str,
) -> None:
    if np.any(month_codes % cadence_months != phase_remainder):
        raise ValueError(
            f"[DATA] {context} timestamps do not stay on the selected seasonal cadence."
        )


def _calendar_seasonal_slots(
    values: np.ndarray,
    month_codes: np.ndarray,
    period: int,
    context: str,
) -> Tuple[np.ndarray, int, int]:
    cadence_months = _seasonal_cadence_months(period)
    phase_remainder = int(month_codes[0] % cadence_months)
    _validate_seasonal_grid(
        month_codes,
        cadence_months,
        phase_remainder,
        context,
    )
    slots = np.full(period, np.nan, dtype=float)
    for value, month_code in zip(values, month_codes):
        slots[int((month_code // cadence_months) % period)] = float(value)
    if not np.isfinite(slots).all():
        raise ValueError(
            "[VALIDATION] Seasonal naïve requires one observed value for every "
            "calendar position in a complete seasonal period."
        )
    return slots, cadence_months, phase_remainder


def _range_timestamps(
    parsed_data: ParsedData,
    index_range: range,
) -> Optional[List[str]]:
    timestamps = parsed_data.timestamps
    if not timestamps:
        return None
    return [timestamps[index] for index in index_range]


def _saved_finite_number(state: Dict[str, Any], field: str) -> float:
    value = state.get(field)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(
            f"[VALIDATION] Saved baseline state has an invalid {field}."
        )
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(
            f"[VALIDATION] Saved baseline state has a non-finite {field}."
        )
    return result


def _saved_seasonal_period(state: Dict[str, Any]) -> int:
    value = state.get("seasonal_period")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(
            "[VALIDATION] Saved seasonal-naïve baseline period is invalid."
        )
    period = int(value)
    if period < 2:
        raise ValueError(
            "[VALIDATION] Saved seasonal-naïve baseline period is invalid."
        )
    return period


def _saved_seasonal_tail(state: Dict[str, Any], period: int) -> np.ndarray:
    value = state.get("seasonal_tail")
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(
            "[VALIDATION] Saved seasonal-naïve baseline tail is invalid."
        )
    array = np.asarray(value, dtype=object)
    if (
        array.ndim != 1
        or array.size != period
        or any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
            for item in array
        )
    ):
        raise ValueError(
            "[VALIDATION] Saved seasonal-naïve baseline tail is invalid."
        )
    result = array.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(
            "[VALIDATION] Saved seasonal-naïve baseline tail is invalid."
        )
    return result


def infer_seasonal_period(
    parsed_data: ParsedData,
    train_range: range,
    selection_range: range,
) -> Optional[int]:
    """Infer a season from the exact training and selection windows in use."""
    timestamps = parsed_data.timestamps
    if not timestamps or len(train_range) < 4:
        return None
    train_month_codes = _try_month_codes(
        [timestamps[index] for index in train_range]
    )
    if train_month_codes is None or len(train_month_codes) < 2:
        return None

    month_steps = np.diff(train_month_codes)
    if np.all(month_steps == 1):
        period = 12 if len(train_range) >= 12 else None
    elif np.all(month_steps == 3):
        period = 4 if len(train_range) >= 4 else None
    else:
        period = None
    if period is None:
        return None

    # The training prefix establishes the cadence. The held-out selection
    # observations may omit complete periods, but they must remain on that
    # same calendar grid and follow the history. Do not inspect later rows:
    # they are outside this candidate-selection decision and may belong to a
    # separate validation/evaluation window.
    selection_month_codes = _try_month_codes(
        [timestamps[index] for index in selection_range]
    )
    required_month_codes = (
        np.concatenate((train_month_codes, selection_month_codes))
        if selection_month_codes is not None
        else None
    )
    if (
        required_month_codes is None
        or np.any(np.diff(required_month_codes) <= 0)
    ):
        return None
    cadence_months = _seasonal_cadence_months(period)
    if np.any(
        required_month_codes % cadence_months
        != train_month_codes[0] % cadence_months
    ):
        return None
    return period


def available_baseline_kinds(
    parsed_data: ParsedData,
    train_range: range,
    selection_range: range,
) -> List[Tuple[str, Optional[int]]]:
    candidates: List[Tuple[str, Optional[int]]] = [
        ("mean", None),
        ("naive", None),
    ]
    seasonal_period = infer_seasonal_period(
        parsed_data,
        train_range,
        selection_range,
    )
    if seasonal_period is not None:
        candidates.append(("seasonal_naive", seasonal_period))
    return candidates


def baseline_fitted_values(
    history: Sequence[float],
    kind: str,
    seasonal_period: Optional[int] = None,
    timestamps: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    values = np.asarray(history, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("[DATA] Baseline training history must contain finite values.")
    if kind == "mean":
        return np.full(values.size, float(np.mean(values)), dtype=float)
    if kind == "naive":
        return np.concatenate(([values[0]], values[:-1]))
    if kind == "seasonal_naive":
        period = int(seasonal_period or 0)
        if period < 2 or values.size < period:
            raise ValueError(
                "[VALIDATION] Seasonal naïve requires one complete seasonal period."
            )
        if timestamps is None:
            fitted = np.concatenate(([values[0]], values[:-1]))
            fitted[period:] = values[:-period]
            return fitted

        month_codes = _require_month_codes(
            timestamps,
            values.size,
            "Seasonal-naïve training history",
        )
        # Validate that the history covers a complete calendar cycle before
        # constructing causal fitted values. A missing month later in history
        # is fine: the latest observed value for that same calendar slot wins.
        _calendar_seasonal_slots(
            values,
            month_codes,
            period,
            "Seasonal-naïve training history",
        )
        cadence_months = _seasonal_cadence_months(period)
        fitted = np.empty(values.size, dtype=float)
        latest_by_slot: Dict[int, float] = {}
        for index, (value, month_code) in enumerate(zip(values, month_codes)):
            slot = int((month_code // cadence_months) % period)
            if slot in latest_by_slot:
                fitted[index] = latest_by_slot[slot]
            elif index == 0:
                fitted[index] = float(value)
            else:
                # Match the historical warm-up rule until every seasonal slot
                # has an earlier observation.
                fitted[index] = float(values[index - 1])
            latest_by_slot[slot] = float(value)
        return fitted
    raise ValueError(f"[VALIDATION] Unknown baseline kind: {kind}")


def baseline_forecast(
    history: Sequence[float],
    steps: int,
    kind: str,
    seasonal_period: Optional[int] = None,
    *,
    history_timestamps: Optional[Sequence[Any]] = None,
    forecast_timestamps: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    values = np.asarray(history, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("[DATA] Baseline training history must contain finite values.")
    forecast_steps = int(steps)
    if forecast_steps <= 0:
        raise ValueError("[VALIDATION] Baseline forecast steps must be positive.")
    if kind == "mean":
        return np.full(forecast_steps, float(np.mean(values)), dtype=float)
    if kind == "naive":
        return np.full(forecast_steps, float(values[-1]), dtype=float)
    if kind == "seasonal_naive":
        period = int(seasonal_period or 0)
        if period < 2 or values.size < period:
            raise ValueError(
                "[VALIDATION] Seasonal naïve requires one complete seasonal period."
            )
        has_history_timestamps = history_timestamps is not None
        has_forecast_timestamps = forecast_timestamps is not None
        if has_history_timestamps != has_forecast_timestamps:
            raise ValueError(
                "[VALIDATION] Calendar-aligned seasonal forecast requires both "
                "history and forecast timestamps."
            )
        if not has_history_timestamps:
            seasonal_tail = values[-period:].tolist()
            return np.asarray(
                [seasonal_tail[index % period] for index in range(forecast_steps)],
                dtype=float,
            )

        history_month_codes = _require_month_codes(
            history_timestamps if history_timestamps is not None else [],
            values.size,
            "Seasonal-naïve training history",
        )
        forecast_month_codes = _require_month_codes(
            forecast_timestamps if forecast_timestamps is not None else [],
            forecast_steps,
            "Seasonal-naïve forecast range",
        )
        if forecast_month_codes[0] <= history_month_codes[-1]:
            raise ValueError(
                "[VALIDATION] Seasonal-naïve forecast timestamps must follow "
                "the training history."
            )
        slots, cadence_months, phase_remainder = _calendar_seasonal_slots(
            values,
            history_month_codes,
            period,
            "Seasonal-naïve training history",
        )
        _validate_seasonal_grid(
            forecast_month_codes,
            cadence_months,
            phase_remainder,
            "Seasonal-naïve forecast range",
        )
        return np.asarray(
            [
                slots[int((month_code // cadence_months) % period)]
                for month_code in forecast_month_codes
            ],
            dtype=float,
        )
    raise ValueError(f"[VALIDATION] Unknown baseline kind: {kind}")


def fit_baseline_state(
    history: Sequence[float],
    kind: str,
    seasonal_period: Optional[int] = None,
    timestamps: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    values = np.asarray(history, dtype=float).reshape(-1)
    baseline_fitted_values(values, kind, seasonal_period, timestamps)
    state: Dict[str, Any] = {
        "kind": kind,
        "label": BASELINE_LABELS[kind],
        "training_count": int(values.size),
        "training_mean": float(np.mean(values)),
        "last_value": float(values[-1]),
        "seasonal_period": (
            int(seasonal_period) if seasonal_period is not None else None
        ),
        "seasonal_tail": [],
    }
    if kind == "seasonal_naive":
        period = int(seasonal_period or 0)
        if timestamps is None:
            state["seasonal_tail"] = values[-period:].tolist()
        else:
            month_codes = _require_month_codes(
                timestamps,
                values.size,
                "Seasonal-naïve training history",
            )
            slots, cadence_months, phase_remainder = _calendar_seasonal_slots(
                values,
                month_codes,
                period,
                "Seasonal-naïve training history",
            )
            # Preserve the existing artifact contract: seasonal_tail[0] is the
            # first consecutive period after training. Unlike a raw last-N-row
            # slice, this remains correct when observations are missing.
            state["seasonal_tail"] = [
                float(
                    slots[
                        int(
                            (
                                (month_codes[-1] + cadence_months * (step + 1))
                                // cadence_months
                            )
                            % period
                        )
                    ]
                )
                for step in range(period)
            ]
            state.update(
                {
                    "seasonal_alignment": CALENDAR_SEASONAL_ALIGNMENT,
                    "seasonal_cadence_months": cadence_months,
                    "seasonal_phase_remainder": phase_remainder,
                    "training_end_month_code": int(month_codes[-1]),
                }
            )
    return state


def forecast_from_baseline_state(
    state: Dict[str, Any],
    steps: int,
) -> np.ndarray:
    if not isinstance(state, dict):
        raise ValueError("[VALIDATION] Saved baseline state is invalid.")
    kind = state.get("kind")
    forecast_steps = int(steps)
    if forecast_steps <= 0:
        raise ValueError("[VALIDATION] Baseline forecast steps must be positive.")
    if kind == "mean":
        value = _saved_finite_number(state, "training_mean")
        return np.full(forecast_steps, value, dtype=float)
    if kind == "naive":
        value = _saved_finite_number(state, "last_value")
        return np.full(forecast_steps, value, dtype=float)
    if kind == "seasonal_naive":
        period = _saved_seasonal_period(state)
        seasonal_tail = _saved_seasonal_tail(state, period)
        alignment = state.get("seasonal_alignment")
        if alignment is not None:
            if alignment != CALENDAR_SEASONAL_ALIGNMENT:
                raise ValueError(
                    "[VALIDATION] Saved seasonal-naïve baseline alignment is invalid."
                )
            cadence = state.get("seasonal_cadence_months")
            phase = state.get("seasonal_phase_remainder")
            end_month_code = state.get("training_end_month_code")
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                for value in (cadence, phase, end_month_code)
            ):
                raise ValueError(
                    "[VALIDATION] Saved seasonal-naïve calendar metadata is invalid."
                )
            cadence = int(cadence)
            phase = int(phase)
            end_month_code = int(end_month_code)
            if (
                cadence != _seasonal_cadence_months(period)
                or phase < 0
                or phase >= cadence
                or end_month_code % cadence != phase
            ):
                raise ValueError(
                    "[VALIDATION] Saved seasonal-naïve calendar metadata is invalid."
                )
        return np.asarray(
            [seasonal_tail[index % period] for index in range(forecast_steps)],
            dtype=float,
        )
    if kind == "zero":
        return np.zeros(forecast_steps, dtype=float)
    raise ValueError("[VALIDATION] Saved baseline state has an unknown kind.")


def choose_stage_zero_baseline(
    parsed_data: ParsedData,
    train_range: range,
    selection_range: range,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Dict[str, Any]]:
    """Select the best valid zero-stage baseline on a chronological selection set."""
    history = np.asarray(parsed_data.y[train_range], dtype=float)
    y_selection = np.asarray(parsed_data.y[selection_range], dtype=float)
    candidates: Dict[str, Dict[str, Any]] = {}
    history_timestamps = _range_timestamps(parsed_data, train_range)
    selection_timestamps = _range_timestamps(parsed_data, selection_range)

    for kind, seasonal_period in available_baseline_kinds(
        parsed_data,
        train_range,
        selection_range,
    ):
        predictions = baseline_forecast(
            history,
            len(selection_range),
            kind,
            seasonal_period,
            history_timestamps=history_timestamps,
            forecast_timestamps=selection_timestamps,
        )
        candidate_rmse = float(np.sqrt(np.mean(np.square(y_selection - predictions))))
        candidates[kind] = {
            "kind": kind,
            "label": BASELINE_LABELS[kind],
            "seasonal_period": seasonal_period,
            "rmse": candidate_rmse,
            "predictions": predictions,
        }

    winner = min(candidates.values(), key=lambda value: float(value["rmse"]))
    selected_kind = str(winner["kind"])
    selected_period = winner["seasonal_period"]
    state = fit_baseline_state(
        history,
        selected_kind,
        selected_period,
        history_timestamps,
    )
    fitted = baseline_fitted_values(
        history,
        selected_kind,
        selected_period,
        history_timestamps,
    )
    predictions = np.asarray(winner["predictions"], dtype=float)
    diagnostics = {
        "strategy": "best_chronological_zero_stage_baseline",
        "selected": selected_kind,
        "selected_label": state["label"],
        "selected_rmse": float(winner["rmse"]),
        "candidates": {
            kind: {
                "label": candidate["label"],
                "seasonal_period": candidate["seasonal_period"],
                "rmse": float(candidate["rmse"]),
            }
            for kind, candidate in candidates.items()
        },
    }
    return state, fitted, predictions, diagnostics


def baseline_fitted_values_for_range(
    parsed_data: ParsedData,
    train_range: range,
    kind: str,
    seasonal_period: Optional[int] = None,
) -> np.ndarray:
    return baseline_fitted_values(
        parsed_data.y[train_range],
        kind,
        seasonal_period,
        _range_timestamps(parsed_data, train_range),
    )


def baseline_forecast_for_range(
    parsed_data: ParsedData,
    train_range: range,
    forecast_range: range,
    kind: str,
    seasonal_period: Optional[int] = None,
) -> np.ndarray:
    return baseline_forecast(
        parsed_data.y[train_range],
        len(forecast_range),
        kind,
        seasonal_period,
        history_timestamps=_range_timestamps(parsed_data, train_range),
        forecast_timestamps=_range_timestamps(parsed_data, forecast_range),
    )


def fit_baseline_state_for_range(
    parsed_data: ParsedData,
    train_range: range,
    kind: str,
    seasonal_period: Optional[int] = None,
) -> Dict[str, Any]:
    return fit_baseline_state(
        parsed_data.y[train_range],
        kind,
        seasonal_period,
        _range_timestamps(parsed_data, train_range),
    )
