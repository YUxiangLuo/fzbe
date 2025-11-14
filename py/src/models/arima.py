#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA model with automatic order selection for p and q.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pmdarima as pm

from data_utils import ParsedData, load_data_from_json
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model

MODEL_TYPE = "arima"


# ---------- utilities ----------
def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved ARIMA model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid ARIMA model file.")
    return state


# ---------- predict ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[List[float]]]:
    """Forecast using a saved ARIMA model."""
    state = state or _load_state(model_path)
    model = state["model"]
    
    y_pred, conf_int = model.predict(n_periods=steps, return_conf_int=True)
    
    return y_pred.tolist(), conf_int.tolist()


# ---------- train and evaluate ----------
def train_and_evaluate(
    y: np.ndarray,
    timestamps: List[str],
    train_range: range,
    eval_range: range,
    d: int,
) -> Tuple[Dict[str, Any], Any, Dict[str, int], Dict[str, Any], List[float], List[float], List[List[float]]]:
    """
    Trains an ARIMA model by searching for the best p and q using both AIC and BIC,
    and evaluates the best resulting model.
    """
    y_train = y[train_range]
    y_eval = y[eval_range]

    # --- Run auto_arima for both AIC and BIC ---
    aic_model = pm.auto_arima(
        y_train, d=d, start_p=0, start_q=0, max_p=5, max_q=5,
        seasonal=False, stepwise=True, suppress_warnings=True,
        error_action="ignore", information_criterion="aic",
    )

    bic_model = pm.auto_arima(
        y_train, d=d, start_p=0, start_q=0, max_p=5, max_q=5,
        seasonal=False, stepwise=True, suppress_warnings=True,
        error_action="ignore", information_criterion="bic",
    )

    # --- Select the best model (preferring lower AIC) ---
    selection_criterion = "aic"
    best_model = aic_model
    if bic_model.aic() < aic_model.aic():
        best_model = bic_model
        selection_criterion = "bic_tiebreak_aic"


    # --- Evaluate the best model ---
    y_pred, conf_int = best_model.predict(n_periods=len(eval_range), return_conf_int=True)

    metrics = {
        "rmse": rmse(y_eval, y_pred),
        "mae": mae(y_eval, y_pred),
        "mape": mape(y_eval, y_pred),
        "r2": r2(y_eval, y_pred),
        "aic": best_model.aic(),
        "bic": best_model.bic(),
        "selection_criterion": selection_criterion,
        "converged": best_model.arima_res_.mle_retvals["converged"],
    }
...
    out: Dict[str, Any] = {
        "mode": "arima_search_pq_fixed_d",
        "selection_criterion": metrics.pop("selection_criterion"),
        "best_order": best_order,
        "metrics": metrics,
        "eval_y_true": y_true,
...
    }
        "predictions": [
            {"prediction": p, "std_dev": (ci[1] - ci[0]) / 3.92}
            for p, ci in zip(y_pred, conf_int)
        ],
        "search_summary": search_summary,
        "diagnostics": {
            "y_pred_eval": y_pred,
            "conf_int": conf_int,
        }
    }

    if save_path:
        state = {"model_type": MODEL_TYPE, "model": model}
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
