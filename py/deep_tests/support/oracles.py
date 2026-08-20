"""Small mathematical oracles independent of production model implementations."""
from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Mapping, Sequence


def recursive_ma(history: Sequence[float], window: int, steps: int) -> list[float]:
    if window < 2 or len(history) < window or steps < 1:
        raise ValueError("invalid recursive MA oracle arguments")
    values = deque((float(v) for v in history[-window:]), maxlen=window)
    result = []
    for _ in range(steps):
        prediction = math.fsum(values) / window
        result.append(prediction)
        values.append(prediction)
    return result


def ses(values: Sequence[float], alpha: float, initial_level: float) -> float:
    level = float(initial_level)
    for value in values:
        level = alpha * float(value) + (1.0 - alpha) * level
    return level


def metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float | None]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("metric vectors must be non-empty and equal length")
    residuals = [float(a) - float(p) for a, p in zip(actual, predicted)]
    n = len(actual)
    mse = math.fsum(value * value for value in residuals) / n
    mae = math.fsum(abs(value) for value in residuals) / n
    nonzero = [(float(a), float(p)) for a, p in zip(actual, predicted) if abs(float(a)) > 1e-12]
    mape = 100.0 * math.fsum(abs((a - p) / a) for a, p in nonzero) / len(nonzero) if nonzero else None
    mean = math.fsum(float(value) for value in actual) / n
    total = math.fsum((float(value) - mean) ** 2 for value in actual)
    r2 = 1.0 if total <= 1e-12 and mse <= 1e-12 else (0.0 if total <= 1e-12 else 1.0 - math.fsum(v * v for v in residuals) / total)
    return {"rmse": math.sqrt(mse), "mae": mae, "mape": mape, "r2": r2}


def shrunk_inverse_mse_weights(actual: Sequence[float], columns: Mapping[str, Sequence[float]]) -> dict[str, float]:
    actual_values = [float(value) for value in actual]
    prediction_values = {
        name: [float(value) for value in predictions]
        for name, predictions in columns.items()
    }
    def log_difference(left: float, right: float) -> float:
        difference = left - right
        if math.isfinite(difference):
            return math.log(abs(difference))
        scale = max(abs(left), abs(right))
        return math.log(scale) + math.log(abs(left / scale - right / scale))

    log_rms = {}
    zero_names = []
    for name, predictions in prediction_values.items():
        differences = [
            log_difference(actual_value, prediction_value)
            for actual_value, prediction_value in zip(actual_values, predictions)
            if actual_value != prediction_value
        ]
        if not differences:
            zero_names.append(name)
            continue
        largest = max(differences)
        square_sum = math.fsum(math.exp(2.0 * (value - largest)) for value in differences)
        log_rms[name] = largest + 0.5 * (
            math.log(square_sum) - math.log(len(actual_values))
        )
    count = len(prediction_values)
    if zero_names:
        raw = {
            name: 1.0 / len(zero_names) if name in zero_names else 0.0
            for name in prediction_values
        }
    else:
        minimum_log_rms = min(log_rms.values())
        scores = {
            name: math.exp(2.0 * (minimum_log_rms - value))
            for name, value in log_rms.items()
        }
        total = math.fsum(scores.values())
        raw = {name: value / total for name, value in scores.items()}
    reliability = len(actual_values) / (len(actual_values) + count + 1.0)
    shrunk = {
        name: reliability * raw[name] + (1.0 - reliability) / count
        for name in raw
    }
    total = math.fsum(shrunk.values())
    return {name: value / total for name, value in shrunk.items()}


def nonnegative_line_search(residual: Iterable[float], prediction: Iterable[float]) -> float:
    pairs = [(float(r), float(p)) for r, p in zip(residual, prediction)]
    denominator = math.fsum(p * p for _, p in pairs)
    return 0.0 if denominator <= math.ulp(1.0) else max(0.0, math.fsum(r * p for r, p in pairs) / denominator)


def baseline_fitted(history: Sequence[float], kind: str, period: int | None = None) -> list[float]:
    """In-sample fitted values for a stage-zero baseline, independently derived."""
    values = [float(v) for v in history]
    if not values:
        raise ValueError("baseline oracle needs a non-empty history")
    if kind == "mean":
        mean = math.fsum(values) / len(values)
        return [mean] * len(values)
    if kind == "naive":
        return [values[0]] + values[:-1]
    if kind == "seasonal_naive":
        span = int(period or 0)
        if span < 2 or len(values) < span:
            raise ValueError("seasonal naive oracle needs one full period")
        # Warm up with the naive value until a full season is available.
        fitted = [values[0]] + values[:-1]
        for index in range(span, len(values)):
            fitted[index] = values[index - span]
        return fitted
    raise ValueError(f"unknown baseline kind: {kind}")


def baseline_forecast(
    history: Sequence[float],
    steps: int,
    kind: str,
    period: int | None = None,
) -> list[float]:
    values = [float(v) for v in history]
    if not values or steps < 1:
        raise ValueError("baseline forecast oracle needs history and positive steps")
    if kind == "mean":
        return [math.fsum(values) / len(values)] * steps
    if kind == "naive":
        return [values[-1]] * steps
    if kind == "seasonal_naive":
        span = int(period or 0)
        if span < 2 or len(values) < span:
            raise ValueError("seasonal naive oracle needs one full period")
        tail = values[-span:]
        return [tail[index % span] for index in range(steps)]
    raise ValueError(f"unknown baseline kind: {kind}")


def _clamp(values: Iterable[float]) -> list[float]:
    """Physical demand cannot be negative."""
    return [max(0.0, float(value)) for value in values]


def weighted_blend(
    members: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
    order: Sequence[str],
) -> list[float]:
    """Members are clamped individually, combined, then clamped again."""
    clamped = {name: _clamp(members[name]) for name in order}
    steps = len(next(iter(clamped.values())))
    blended = [
        math.fsum(clamped[name][index] * float(weights[name]) for name in order)
        for index in range(steps)
    ]
    return _clamp(blended)


def stacking_blend(
    members: Mapping[str, Sequence[float]],
    coefficients: Mapping[str, float],
    order: Sequence[str],
) -> list[float]:
    """NNLS meta-model has no intercept, so this is the same shape as a blend."""
    return weighted_blend(members, coefficients, order)


def boosting_compose(
    baseline: Sequence[float],
    stages: Sequence[Sequence[float]],
    coefficients: Sequence[float],
) -> list[float]:
    """Stage outputs are signed residual corrections; only the total is clamped."""
    if len(stages) != len(coefficients):
        raise ValueError("boosting oracle needs one coefficient per stage")
    total = [float(value) for value in baseline]
    for stage, coefficient in zip(stages, coefficients):
        if len(stage) != len(total):
            raise ValueError("boosting oracle stage length mismatch")
        total = [
            current + float(coefficient) * float(correction)
            for current, correction in zip(total, stage)
        ]
    return _clamp(total)
