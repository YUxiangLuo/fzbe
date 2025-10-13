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
    """
    # Lazy import to avoid circular dependencies
    from models import arima, lstm, ma as ma_model, es as es_model

    predictions = []
    for path in model_paths:
        try:
            path_lower = path.lower()

            if path_lower.endswith((".pt", ".pth")) or "lstm" in path_lower:
                preds = lstm.predict(path, steps, data)
                predictions.append(preds)
                continue

            state = None
            if path_lower.endswith((".pkl", ".joblib")):
                try:
                    state = load_joblib(path)
                except Exception:
                    state = None

            if isinstance(state, dict):
                model_type = state.get("model_type")
                if model_type == "ma":
                    preds = ma_model.predict(path, steps, data, state=state)
                    predictions.append(preds)
                    continue
                if model_type == "es":
                    preds = es_model.predict(path, steps, data, state=state)
                    predictions.append(preds)
                    continue

            if state is not None and hasattr(state, "get_forecast"): # ARIMA model
                fcst = state.get_forecast(steps)
                preds = fcst.predicted_mean.tolist()
                predictions.append(preds)
                continue

            # Fallback order for models without clear type hints in the saved file
            try:
                preds = arima.predict(path, steps)
                predictions.append(preds)
                continue
            except Exception:
                pass
            try:
                preds = ma_model.predict(path, steps, data)
                predictions.append(preds)
                continue
            except Exception:
                pass
            try:
                preds = es_model.predict(path, steps, data)
                predictions.append(preds)
                continue
            except Exception:
                pass
            
            # Final fallback to LSTM if others fail
            preds = lstm.predict(path, steps, data)
            predictions.append(preds)

        except Exception as e:
            raise ValueError(f"Failed to predict with model {path}: {e}")
    
    if len(set(len(p) for p in predictions)) > 1:
        raise ValueError("All models must produce predictions of the same length")
    
    return predictions
