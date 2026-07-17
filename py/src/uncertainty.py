#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared horizon-aware uncertainty calibration and JSON formatting.

Point forecasts are intentionally left untouched.  Prediction intervals and
the production P99 buffer are offsets calibrated from ``actual - prediction``
residuals, so a stable non-zero bias is not mistaken for zero risk.

``std_dev`` remains the centered sample standard deviation required by the
customer's safety-stock formula.  The residual mean and sample count are
reported separately as diagnostics and never alter the point forecast.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


UNCERTAINTY_MODEL = "model"
UNCERTAINTY_EMPIRICAL = "empirical"
UNCERTAINTY_FALLBACK = "fallback"
INTERVAL_LEVEL = 0.95
P99_LEVEL = 0.99
NORMAL_95 = 1.959963984540054
NORMAL_99 = 2.3263478740408408


def _finite_nonnegative(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0:
        return None
    return numeric


def _finite_residuals(values: Any) -> np.ndarray:
    try:
        residuals = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return residuals[np.isfinite(residuals)]


def _normal_horizon(
    scale: float,
    *,
    source: str,
    calibration_source: str,
    reason: str | None = None,
) -> Dict[str, Any]:
    numeric_scale = _finite_nonnegative(scale)
    if numeric_scale is None:
        raise ValueError("[MODEL] Uncertainty scale must be finite and nonnegative.")
    horizon: Dict[str, Any] = {
        "std_dev": numeric_scale,
        "interval_lower_offset": -NORMAL_95 * numeric_scale,
        "interval_upper_offset": NORMAL_95 * numeric_scale,
        "upper_error_p99": NORMAL_99 * numeric_scale,
        "source": source,
        "calibration_source": calibration_source,
        "calibration_mean_error": None,
        "calibration_count": None,
        "interval_kind": "fallback_normal_approximation" if source == UNCERTAINTY_FALLBACK else "normal_approximation",
    }
    if reason:
        horizon["reason"] = reason
    return horizon


def _empirical_horizon(
    residuals: Any,
    *,
    growth: float = 1.0,
    calibration_source: str,
) -> Dict[str, Any] | None:
    finite = _finite_residuals(residuals)
    if finite.size < 2:
        return None
    numeric_growth = float(growth)
    if not math.isfinite(numeric_growth) or numeric_growth < 0:
        return None
    scaled = finite * numeric_growth
    scale = float(np.std(scaled, ddof=1))
    lower, upper, upper_p99 = np.quantile(
        scaled,
        [0.025, 0.975, P99_LEVEL],
        method="linear",
    ).tolist()
    return {
        "std_dev": scale,
        "interval_lower_offset": float(lower),
        "interval_upper_offset": float(upper),
        "upper_error_p99": max(0.0, float(upper_p99)),
        "source": UNCERTAINTY_EMPIRICAL,
        "calibration_source": calibration_source,
        "calibration_mean_error": float(np.mean(scaled)),
        "interval_kind": "empirical_residual_quantile",
        "calibration_count": int(finite.size),
    }


def _metadata_from_horizons(
    horizons: Sequence[Mapping[str, Any]],
    *,
    method: str,
    calibration_source: str,
    reason: str | None = None,
) -> Dict[str, Any]:
    normalized = [dict(horizon) for horizon in horizons]
    fallback_horizons = [
        index + 1
        for index, horizon in enumerate(normalized)
        if horizon.get("source") == UNCERTAINTY_FALLBACK
    ]
    metadata: Dict[str, Any] = {
        "method": method,
        "fallback": bool(fallback_horizons),
        "fallback_horizons": fallback_horizons,
        "calibration_source": calibration_source,
        "horizons": normalized,
    }
    non_fallback_sources = {
        str(horizon.get("source"))
        for horizon in normalized
        if horizon.get("source") != UNCERTAINTY_FALLBACK
    }
    if len(non_fallback_sources) == 1:
        metadata["source"] = next(iter(non_fallback_sources))
    for optional_key in (
        "coverage_guarantee",
        "upper_error_p99_kind",
        "calibration_origins",
    ):
        optional_values = {
            horizon.get(optional_key)
            for horizon in normalized
            if optional_key in horizon
        }
        if len(optional_values) == 1:
            metadata[optional_key] = next(iter(optional_values))
    if reason and fallback_horizons:
        metadata["reason"] = reason
    return metadata


def fallback_profile(reference_series: Sequence[float], steps: int) -> Tuple[List[float], Dict[str, Any]]:
    values = np.asarray(reference_series, dtype=float).reshape(-1)
    differences = np.diff(values)
    if differences.size > 1 and np.isfinite(differences).all():
        centered_scale = float(np.std(differences, ddof=1))
        drift_scale = float(np.sqrt(np.mean(np.square(differences))))
        near_zero_centered_scale = centered_scale <= max(1e-12, drift_scale * 1e-9)
        if drift_scale > 0 and near_zero_centered_scale:
            # A stable trend has no centered variation, but treating its
            # non-zero step size as zero risk would make fallback intervals
            # and the production buffer misleadingly collapse to zero.
            one_step = drift_scale
            reason = "first_difference_rms_floor"
        else:
            one_step = centered_scale
            reason = "first_difference_scale"
    else:
        baseline = abs(float(np.mean(values))) if values.size else 0.0
        one_step = float(max(0.05 * baseline, 1.0))
        reason = "insufficient_residuals"
    if not math.isfinite(one_step) or one_step < 0:
        one_step = 1.0
        reason = "nonfinite_scale"
    profile = [one_step * math.sqrt(horizon) for horizon in range(1, int(steps) + 1)]
    horizons = [
        _normal_horizon(
            scale,
            source=UNCERTAINTY_FALLBACK,
            calibration_source="training_history",
            reason=reason,
        )
        for scale in profile
    ]
    return profile, _metadata_from_horizons(
        horizons,
        method="difference_sqrt_h",
        calibration_source="training_history",
        reason=reason,
    )


def _valid_saved_horizon(value: Any) -> Dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        source = str(value["source"])
        calibration_source = str(value["calibration_source"])
        interval_kind = str(value["interval_kind"])
        std_dev = float(value["std_dev"])
        lower = float(value["interval_lower_offset"])
        upper = float(value["interval_upper_offset"])
        upper_p99 = float(value["upper_error_p99"])
    except (KeyError, TypeError, ValueError):
        return None
    if source not in {UNCERTAINTY_MODEL, UNCERTAINTY_EMPIRICAL, UNCERTAINTY_FALLBACK}:
        return None
    if not calibration_source or not interval_kind:
        return None
    if not all(math.isfinite(item) for item in (std_dev, lower, upper, upper_p99)):
        return None
    if std_dev < 0 or upper_p99 < 0 or lower > upper:
        return None
    normalized = dict(value)
    if "coverage_guarantee" in value:
        coverage_guarantee = value.get("coverage_guarantee")
        if not isinstance(coverage_guarantee, bool):
            return None
        normalized["coverage_guarantee"] = coverage_guarantee
    if "upper_error_p99_kind" in value:
        upper_error_p99_kind = value.get("upper_error_p99_kind")
        if not isinstance(upper_error_p99_kind, str) or not upper_error_p99_kind:
            return None
        normalized["upper_error_p99_kind"] = upper_error_p99_kind
    if "calibration_origins" in value:
        calibration_origins = value.get("calibration_origins")
        try:
            raw_numeric_origins = float(calibration_origins)
            numeric_origins = int(raw_numeric_origins)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            not math.isfinite(raw_numeric_origins)
            or isinstance(calibration_origins, bool)
            or numeric_origins <= 0
            or raw_numeric_origins != numeric_origins
        ):
            return None
        normalized["calibration_origins"] = numeric_origins
    mean_error = value.get("calibration_mean_error")
    calibration_count = value.get("calibration_count")
    if mean_error is None and calibration_count is None:
        normalized["calibration_mean_error"] = None
        normalized["calibration_count"] = None
    elif mean_error is None or calibration_count is None:
        return None
    else:
        try:
            numeric_mean_error = float(mean_error)
            raw_numeric_count = float(calibration_count)
            numeric_count = int(raw_numeric_count)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            not math.isfinite(numeric_mean_error)
            or not math.isfinite(raw_numeric_count)
            or isinstance(calibration_count, bool)
            or numeric_count <= 0
            or raw_numeric_count != numeric_count
        ):
            return None
        normalized["calibration_mean_error"] = numeric_mean_error
        normalized["calibration_count"] = numeric_count
    return normalized


def profile_for_steps(
    saved_profile: Any,
    steps: int,
    reference_series: Sequence[float],
    saved_metadata: Mapping[str, Any] | None = None,
) -> Tuple[List[float], Dict[str, Any]]:
    requested_steps = int(steps)
    if requested_steps <= 0:
        raise ValueError("[MODEL] Uncertainty steps must be positive.")

    raw_profile = list(saved_profile) if isinstance(saved_profile, (list, tuple)) else []
    raw_horizons = (saved_metadata or {}).get("horizons")
    horizon_values = list(raw_horizons) if isinstance(raw_horizons, (list, tuple)) else []
    preserved_profile: List[float] = []
    preserved_horizons: List[Dict[str, Any]] = []
    for index in range(min(requested_steps, len(raw_profile), len(horizon_values))):
        scale = _finite_nonnegative(raw_profile[index])
        horizon = _valid_saved_horizon(horizon_values[index])
        if scale is None or horizon is None:
            break
        preserved_profile.append(scale)
        preserved_horizons.append(horizon)

    fallback, fallback_metadata = fallback_profile(reference_series, requested_steps)
    fallback_horizons = list(fallback_metadata["horizons"])
    for index in range(len(preserved_profile), requested_steps):
        preserved_profile.append(float(fallback[index]))
        preserved_horizons.append(dict(fallback_horizons[index]))

    method = str((saved_metadata or {}).get("method") or "saved_profile_with_fallback_extension")
    calibration_source = str((saved_metadata or {}).get("calibration_source") or "training_history")
    return preserved_profile, _metadata_from_horizons(
        preserved_horizons,
        method=method,
        calibration_source=calibration_source,
        reason=str(fallback_metadata.get("reason") or "insufficient_residuals"),
    )


def format_predictions(
    predictions: Sequence[float],
    profile: Sequence[float],
    metadata: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if len(predictions) != len(profile):
        raise ValueError("[MODEL] Prediction and uncertainty profile lengths differ.")
    horizon_start = metadata.get("horizon_start", 1)
    if isinstance(horizon_start, bool) or not isinstance(horizon_start, int) or horizon_start < 1:
        raise ValueError("[MODEL] horizon_start must be a positive integer.")
    raw_horizons = metadata.get("horizons")
    horizons = list(raw_horizons) if isinstance(raw_horizons, (list, tuple)) else []
    points: List[Dict[str, Any]] = []
    for index, (prediction, std_dev) in enumerate(zip(predictions, profile)):
        numeric_prediction = float(prediction)
        numeric_std_dev = _finite_nonnegative(std_dev)
        if not math.isfinite(numeric_prediction):
            raise ValueError("[MODEL] Prediction must be finite.")
        if numeric_std_dev is None:
            raise ValueError("[MODEL] Prediction uncertainty must be finite and nonnegative.")

        horizon_index = horizon_start + index - 1
        calibration = _valid_saved_horizon(
            horizons[horizon_index] if horizon_index < len(horizons) else None
        )
        if calibration is None:
            calibration = _normal_horizon(
                numeric_std_dev,
                source=UNCERTAINTY_FALLBACK,
                calibration_source="training_history",
                reason="missing_horizon_calibration",
            )
        source = str(calibration["source"])
        lower = numeric_prediction + float(calibration["interval_lower_offset"])
        upper = numeric_prediction + float(calibration["interval_upper_offset"])
        upper_error_p99 = float(calibration["upper_error_p99"])
        if not all(math.isfinite(value) for value in (lower, upper, upper_error_p99)):
            raise ValueError("[MODEL] Prediction interval calibration must be finite.")
        if lower > upper or upper_error_p99 < 0:
            raise ValueError("[MODEL] Prediction interval calibration is invalid.")

        point: Dict[str, Any] = {
            "prediction": numeric_prediction,
            "std_dev": numeric_std_dev,
            "uncertainty_source": source,
            "interval_lower": lower,
            "interval_upper": upper,
            "interval_level": INTERVAL_LEVEL,
            "interval_kind": str(calibration["interval_kind"]),
            "upper_error_p99": upper_error_p99,
            "calibration_source": str(calibration["calibration_source"]),
            "calibration_mean_error": calibration["calibration_mean_error"],
            "calibration_count": calibration["calibration_count"],
        }
        reason = calibration.get("reason")
        if reason is not None:
            if not isinstance(reason, str) or not reason:
                raise ValueError("[MODEL] Uncertainty reason must be a non-empty string.")
            point["uncertainty_reason"] = reason
        if "coverage_guarantee" in calibration:
            point["coverage_guarantee"] = calibration["coverage_guarantee"]
        if "upper_error_p99_kind" in calibration:
            point["upper_error_p99_kind"] = calibration["upper_error_p99_kind"]
        if "calibration_origins" in calibration:
            point["calibration_origins"] = calibration["calibration_origins"]
        points.append(point)
    return points


def ses_profile(
    residual_scale: float,
    alpha: float,
    steps: int,
    calibration_residuals: Sequence[float] | None = None,
) -> Tuple[List[float], Dict[str, Any]]:
    scale = _finite_nonnegative(residual_scale)
    if scale is None:
        raise ValueError("[MODEL] SES residual scale must be finite and nonnegative.")
    profile = [scale * math.sqrt(1.0 + alpha * alpha * (horizon - 1)) for horizon in range(1, steps + 1)]
    residuals = _finite_residuals(calibration_residuals)
    horizons: List[Dict[str, Any]] = []
    if residuals.size >= 2:
        for horizon in range(1, steps + 1):
            growth = math.sqrt(1.0 + alpha * alpha * (horizon - 1))
            empirical = _empirical_horizon(
                residuals,
                growth=growth,
                calibration_source="training_one_step_residuals",
            )
            assert empirical is not None
            horizons.append(empirical)
        method = "ets_ann_empirical_residual_quantiles"
    else:
        horizons = [
            _normal_horizon(
                value,
                source=UNCERTAINTY_FALLBACK,
                calibration_source="training_one_step_residuals",
                reason="insufficient_residuals",
            )
            for value in profile
        ]
        method = "ets_ann_fallback_normal"
    return profile, _metadata_from_horizons(
        horizons,
        method=method,
        calibration_source="training_one_step_residuals",
        reason="insufficient_residuals",
    )


def moving_average_profile(
    series: Sequence[float],
    window: int,
    steps: int,
) -> Tuple[List[float], Dict[str, Any]]:
    values = np.asarray(series, dtype=float).reshape(-1)
    fallback, fallback_metadata = fallback_profile(values, steps)
    fallback_horizon_metadata = list(fallback_metadata["horizons"])
    profile: List[float] = []
    horizons: List[Dict[str, Any]] = []
    for horizon in range(1, steps + 1):
        errors: List[float] = []
        for origin in range(window - 1, len(values) - horizon):
            history = list(values[origin - window + 1 : origin + 1])
            forecast = 0.0
            for _ in range(horizon):
                forecast = float(np.mean(history[-window:]))
                history.append(forecast)
            errors.append(float(values[origin + horizon] - forecast))
        empirical = _empirical_horizon(
            errors,
            calibration_source="training_rolling_origin",
        )
        if empirical is None:
            profile.append(float(fallback[horizon - 1]))
            horizons.append(dict(fallback_horizon_metadata[horizon - 1]))
        else:
            profile.append(float(empirical["std_dev"]))
            horizons.append(empirical)
    return profile, _metadata_from_horizons(
        horizons,
        method="rolling_origin_horizon_residual_quantiles",
        calibration_source="training_rolling_origin",
        reason=str(fallback_metadata.get("reason") or "insufficient_residuals"),
    )


def residual_profile_by_horizon(
    residual_matrix: Any,
    reference_series: Sequence[float],
    steps: int,
) -> Tuple[List[float], Dict[str, Any]]:
    residuals = np.asarray(residual_matrix, dtype=float)
    fallback, fallback_metadata = fallback_profile(reference_series, steps)
    fallback_horizon_metadata = list(fallback_metadata["horizons"])
    profile: List[float] = []
    horizons: List[Dict[str, Any]] = []
    for horizon in range(steps):
        finite = (
            residuals[:, horizon][np.isfinite(residuals[:, horizon])]
            if residuals.ndim == 2 and residuals.shape[1] > horizon
            else np.asarray([], dtype=float)
        )
        empirical = _empirical_horizon(
            finite,
            calibration_source="internal_time_validation",
        )
        if empirical is None:
            profile.append(float(fallback[horizon]))
            horizons.append(dict(fallback_horizon_metadata[horizon]))
        else:
            profile.append(float(empirical["std_dev"]))
            horizons.append(empirical)
    return profile, _metadata_from_horizons(
        horizons,
        method="validation_residual_quantiles_by_horizon",
        calibration_source="internal_time_validation",
        reason=str(fallback_metadata.get("reason") or "insufficient_residuals"),
    )


def ensemble_growth_profile(
    member_profiles: Mapping[str, Sequence[float]],
    coefficients: Mapping[str, float],
    calibration_residuals: Sequence[float],
    reference_series: Sequence[float],
    steps: int,
    calibration_source: str = "internal_validation",
) -> Tuple[List[float], Dict[str, Any]]:
    shares = {
        name: abs(float(value))
        for name, value in coefficients.items()
        if name in member_profiles and math.isfinite(float(value))
    }
    total = sum(shares.values())
    residuals = _finite_residuals(calibration_residuals)
    if total <= 0 or residuals.size < 2:
        return fallback_profile(reference_series, steps)
    normalized_member_growth: Dict[str, List[float]] = {}
    epsilon = float(np.finfo(float).eps)
    for name in shares:
        raw_profile = np.asarray(member_profiles[name], dtype=float).reshape(-1)
        if raw_profile.size < steps or not np.isfinite(raw_profile[:steps]).all() or np.any(raw_profile[:steps] < 0):
            return fallback_profile(reference_series, steps)
        first = float(raw_profile[0])
        if first <= epsilon:
            if np.any(raw_profile[:steps] > epsilon):
                return fallback_profile(reference_series, steps)
            normalized_member_growth[name] = [1.0] * steps
        else:
            normalized_member_growth[name] = (raw_profile[:steps] / first).tolist()
    growth: List[float] = []
    for horizon in range(steps):
        value = sum(
            (share / total) * normalized_member_growth[name][horizon]
            for name, share in shares.items()
        )
        growth.append(value)
    horizons: List[Dict[str, Any]] = []
    profile: List[float] = []
    for value in growth:
        empirical = _empirical_horizon(
            residuals,
            growth=value,
            calibration_source=calibration_source,
        )
        if empirical is None:
            return fallback_profile(reference_series, steps)
        horizons.append(empirical)
        profile.append(float(empirical["std_dev"]))
    return profile, _metadata_from_horizons(
        horizons,
        method="member_growth_residual_quantiles",
        calibration_source=calibration_source,
    )
