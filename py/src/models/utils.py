# src/models/utils.py
# -*- coding: utf-8 -*-
"""
Shared utility functions for ensemble models.
"""

import math
from typing import List, Optional
from joblib_compat import load_joblib

# ---------- metrics ----------
def rmse(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def mae(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def mape(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    eps = 1e-12
    return (
        100.0
        * sum(
            abs((a - b) / (a if abs(a) > eps else eps)) for a, b in zip(y_true, y_pred)
        )
        / len(y_true)
    )


def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean_y) ** 2 for a in y_true)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


# ---------- Model Prediction Interface ----------
def get_model_predictions(model_paths: List[str], steps: int, data: Optional[str] = None) -> List[List[float]]:
    """
    Gets predictions from all models by dynamically calling their predict() functions.
    This function normalizes the output from different predict() signatures.
    """
    # Lazy import to avoid circular dependencies
    from models import arima, lstm, ma as ma_model, es as es_model

    all_predictions = []
    for path in model_paths:
        raw_preds = None
        try:
            path_lower = path.lower()

            # Determine model type and call the appropriate predict function
            if path_lower.endswith((".pt", ".pth")) or "lstm" in path_lower:
                raw_preds = lstm.predict(path, steps, data)
            else:
                # For joblib models, inspect the content to decide
                state = load_joblib(path)
                if isinstance(state, dict):
                    model_type = state.get("model_type")
                    if model_type == "ma":
                        raw_preds = ma_model.predict(path, steps, data, state=state)
                    elif model_type == "es":
                        raw_preds = es_model.predict(path, steps, data, state=state)
                
                # ARIMA models are identified by their object type
                if raw_preds is None and hasattr(state, "get_forecast"):
                    raw_preds = arima.predict(path, steps)

            if raw_preds is None:
                 raise ValueError(f"Could not determine model type for path: {path}")

            # --- Normalize the prediction output ---
            normalized_preds = []
            if isinstance(raw_preds, tuple):
                # Handles MA, ES which return (predictions, std_dev)
                normalized_preds = raw_preds[0]
            elif isinstance(raw_preds, list) and raw_preds and isinstance(raw_preds[0], dict):
                # Handles ARIMA which returns [{'prediction': p, 'std_dev': s}, ...]
                normalized_preds = [p.get("prediction") for p in raw_preds]
            elif isinstance(raw_preds, list):
                # Handles old format or LSTM which might still return a simple list
                normalized_preds = raw_preds
            else:
                raise TypeError(f"Unsupported prediction format from model {path}: {type(raw_preds)}")

            all_predictions.append(normalized_preds)

        except Exception as e:
            raise ValueError(f"Failed to get or normalize predictions from model {path}: {e}")
    
    if not all_predictions:
        return []

    if len(set(len(p) for p in all_predictions)) > 1:
        raise ValueError("All models must produce predictions of the same length")
    
    return all_predictions
