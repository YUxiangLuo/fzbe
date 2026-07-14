#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared output-domain rules for forecasting models.

The standalone models forecast physical demand, whose displayed and evaluated
point forecasts cannot be negative.  The same model implementations are also
used as Boosting residual learners, where negative values are meaningful and
must remain untouched.  Keeping that distinction explicit prevents an API-only
clamp from disagreeing with Python metrics and prevents a blanket model-level
clamp from corrupting residual learning.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np


DEMAND_PREDICTION_DOMAIN = "nonnegative_demand"
SIGNED_PREDICTION_DOMAIN = "signed_residual"


def resolve_prediction_domain(
    state: Optional[Mapping[str, Any]] = None,
    *,
    allow_negative_predictions: bool = False,
) -> str:
    """Resolve and validate the output domain for training or a saved state."""
    stored_domain = None
    if state:
        stored_domain = state.get("prediction_domain")
        if stored_domain not in {
            DEMAND_PREDICTION_DOMAIN,
            SIGNED_PREDICTION_DOMAIN,
        }:
            raise ValueError(
                f"[VALIDATION] Unknown prediction domain: {stored_domain}"
            )
    if allow_negative_predictions:
        return SIGNED_PREDICTION_DOMAIN
    return stored_domain or DEMAND_PREDICTION_DOMAIN


def apply_prediction_domain(values: Any, prediction_domain: str) -> np.ndarray:
    """Apply post-processing appropriate for demand or signed residual output."""
    forecasts = np.asarray(values, dtype=float)
    if prediction_domain == SIGNED_PREDICTION_DOMAIN:
        return forecasts
    if prediction_domain != DEMAND_PREDICTION_DOMAIN:
        raise ValueError(f"[VALIDATION] Unknown prediction domain: {prediction_domain}")
    return np.maximum(forecasts, 0.0)


def normalize_demand_forecast(values: Any) -> np.ndarray:
    """Clamp physical-demand point forecasts to zero without touching residuals."""
    return apply_prediction_domain(values, DEMAND_PREDICTION_DOMAIN)
