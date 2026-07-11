#!/usr/bin/env python3
"""Smoke test: evaluation stays aligned when it does not start right after training.

The API layer only requires the evaluation window to start after the training
window, so a gap between the two is representable. Each model must forecast from
the end of training and map predictions onto the actual evaluation indices; a
model that forecasts len(eval_range) steps and pastes them onto the evaluation
window would silently shift every prediction by the gap size.

For each of MA/ES/ARIMA this drives train_and_evaluate twice — once with a
contiguous evaluation window covering the gap, once with the gapped window — and
asserts the gapped predictions equal the tail of the contiguous ones.
"""
import math
import sys
from pathlib import Path

import numpy as np

PY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PY_ROOT / "src"))

from models import arima, es, ma  # noqa: E402

TRAIN_RANGE = range(0, 20)
GAP = 2
EVAL_LENGTH = 4
CONTIGUOUS_EVAL = range(TRAIN_RANGE.stop, TRAIN_RANGE.stop + GAP + EVAL_LENGTH)
GAPPED_EVAL = range(TRAIN_RANGE.stop + GAP, TRAIN_RANGE.stop + GAP + EVAL_LENGTH)


def build_series() -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(20260711)
    n = CONTIGUOUS_EVAL.stop
    trend = 100.0 + 1.5 * np.arange(n)
    noise = rng.normal(0.0, 3.0, n)
    series = trend + noise
    months = [f"2024-{(i % 12) + 1:02d}" for i in range(n)]
    return series, months


def assert_aligned(label: str, contiguous_pred, gapped_pred) -> None:
    contiguous_tail = np.asarray(contiguous_pred, dtype=float)[GAP:]
    gapped = np.asarray(gapped_pred, dtype=float)
    if len(gapped) != EVAL_LENGTH:
        raise AssertionError(f"{label}: expected {EVAL_LENGTH} predictions, got {len(gapped)}")
    for value in gapped:
        if not math.isfinite(float(value)):
            raise AssertionError(f"{label}: non-finite prediction {value}")
    if not np.allclose(contiguous_tail, gapped, rtol=1e-9, atol=1e-9):
        raise AssertionError(
            f"{label}: gapped evaluation is misaligned.\n"
            f"  contiguous[{GAP}:]: {contiguous_tail.tolist()}\n"
            f"  gapped:            {gapped.tolist()}"
        )
    print(f"  ✅ {label}: gapped predictions equal contiguous[{GAP}:]")


def check_ma(series: np.ndarray, months: list[str]) -> None:
    _, _, _, contiguous_pred = ma.train_and_evaluate(series, months, TRAIN_RANGE, CONTIGUOUS_EVAL, window=3)
    metrics, _, _, gapped_pred = ma.train_and_evaluate(series, months, TRAIN_RANGE, GAPPED_EVAL, window=3)
    assert_aligned("MA", contiguous_pred, gapped_pred)
    if metrics["evaluate_indices"] != list(GAPPED_EVAL):
        raise AssertionError(f"MA: evaluate_indices mismatch: {metrics['evaluate_indices']}")


def check_es(series: np.ndarray, months: list[str]) -> None:
    _, _, _, contiguous_pred, _ = es.train_and_evaluate(series, months, TRAIN_RANGE, CONTIGUOUS_EVAL, alpha=0.4)
    metrics, _, _, gapped_pred, _ = es.train_and_evaluate(series, months, TRAIN_RANGE, GAPPED_EVAL, alpha=0.4)
    assert_aligned("ES", contiguous_pred, gapped_pred)
    if metrics["evaluate_indices"] != list(GAPPED_EVAL):
        raise AssertionError(f"ES: evaluate_indices mismatch: {metrics['evaluate_indices']}")


def check_arima(series: np.ndarray, months: list[str]) -> None:
    _, _, contiguous_order, _, _, contiguous_pred, _ = arima.train_and_evaluate(
        series, months, TRAIN_RANGE, CONTIGUOUS_EVAL, d=1, max_p=2, max_q=2
    )
    metrics, _, gapped_order, _, _, gapped_pred, conf_int = arima.train_and_evaluate(
        series, months, TRAIN_RANGE, GAPPED_EVAL, d=1, max_p=2, max_q=2
    )
    if contiguous_order != gapped_order:
        raise AssertionError(f"ARIMA: order changed between runs: {contiguous_order} vs {gapped_order}")
    assert_aligned("ARIMA", contiguous_pred, gapped_pred)
    if len(conf_int) != EVAL_LENGTH:
        raise AssertionError(f"ARIMA: expected {EVAL_LENGTH} confidence intervals, got {len(conf_int)}")
    if metrics["evaluate_indices"] != list(GAPPED_EVAL):
        raise AssertionError(f"ARIMA: evaluate_indices mismatch: {metrics['evaluate_indices']}")


def main() -> None:
    series, months = build_series()
    print(f"Train {TRAIN_RANGE}, gapped eval {GAPPED_EVAL} (gap={GAP})")
    check_ma(series, months)
    check_es(series, months)
    check_arima(series, months)
    print("\n🎉 eval gap alignment smoke passed")


if __name__ == "__main__":
    main()
