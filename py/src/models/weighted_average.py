# src/models/weighted_average.py
# -*- coding: utf-8 -*-
"""
Weighted Average Ensemble for combining multiple trained models.
"""

import json
import sys
import os
from typing import List, Tuple
import numpy as np
from joblib_compat import load_joblib, dump_joblib
from models.utils import rmse, mae, mape, r2_score, get_model_predictions


# ---------- Helper Functions ----------
def _extract_model_names(model_paths: List[str]) -> List[str]:
    """Extract model names from file paths for display purposes.

    Supports patterns like:
    - {studentId}_ma.pkl → ma
    - {studentId}_weighted_average_config.pkl → weighted_average
    - {prefix}_{studentId}_arima.pkl → arima
    """
    # Known model types (order matters for multi-word models)
    known_models = [
        'weighted_average', 'boosting', 'stacking',  # Multi-word first
        'ma', 'es', 'arima', 'lstm'  # Single-word after
    ]

    model_names = []
    for path in model_paths:
        # Get filename without extension
        filename = os.path.basename(path)
        name_without_ext = os.path.splitext(filename)[0]

        # Remove '_config' suffix if present
        if name_without_ext.endswith('_config'):
            name_without_ext = name_without_ext[:-7]

        # Try to match known model types
        model_type = None
        for known in known_models:
            if name_without_ext.endswith('_' + known) or name_without_ext == known:
                model_type = known
                break

        if model_type:
            model_names.append(model_type)
        else:
            # Fallback: use the last part after splitting by '_'
            parts = name_without_ext.split('_')
            model_names.append(parts[-1] if parts else name_without_ext)

    return model_names


# ---------- Weighting Strategies ----------
def _weight_equal(n_models: int) -> List[float]:
    return [1.0 / n_models] * n_models


def _weight_inverse_variance(predictions: List[List[float]], y_true: List[float]) -> List[float]:
    variances = []
    for pred in predictions:
        residuals = np.array(y_true) - np.array(pred)
        var = float(np.var(residuals))
        variances.append(var)

    if any(v == 0.0 for v in variances):
        weights = [1.0 if v == 0.0 else 0.0 for v in variances]
        if sum(weights) == 0.0:
            return _weight_equal(len(predictions))
        return weights

    inv_vars = [1.0 / v if v > 0 else 0.0 for v in variances]
    total = sum(inv_vars)
    if total <= 0.0:
        return _weight_equal(len(predictions))
    return [w / total for w in inv_vars]


def _weight_inverse_rmse(predictions: List[List[float]], y_true: List[float]) -> List[float]:
    errors = [rmse(y_true, pred) for pred in predictions]
    if any(e == 0 for e in errors):
        weights = [1.0 if e == 0 else 0.0 for e in errors]
    else:
        inv_errors = [1.0 / e for e in errors if e > 0]
        total = sum(inv_errors)
        weights = [w / total for w in inv_errors] if total > 0 else _weight_equal(len(predictions))
    return weights


def _weight_grid_search(predictions: List[List[float]], y_true: List[float], n_steps: int = 21) -> Tuple[List[float], float]:
    n_models = len(predictions)
    if n_models == 2:
        best_rmse = float("inf")
        best_weights = [0.5, 0.5]
        for i in range(n_steps):
            w1 = i / (n_steps - 1)
            weights = [w1, 1.0 - w1]
            ensemble_pred = _ensemble_predict(predictions, weights)
            curr_rmse = rmse(y_true, ensemble_pred)
            if curr_rmse < best_rmse:
                best_rmse = curr_rmse
                best_weights = weights
        return best_weights, best_rmse
    else:
        print("Warning: grid_search for >2 models falls back to inverse_rmse", file=sys.stderr)
        weights = _weight_inverse_rmse(predictions, y_true)
        ensemble_pred = _ensemble_predict(predictions, weights)
        return weights, rmse(y_true, ensemble_pred)


def _weight_linear_regression(predictions: List[List[float]], y_true: List[float]) -> List[float]:
    X = np.array(predictions).T
    y = np.array(y_true)
    try:
        w = np.linalg.lstsq(X, y, rcond=None)[0]
        w = np.maximum(w, 0)
        if w.sum() == 0:
            return _weight_equal(len(predictions))
        return (w / w.sum()).tolist()
    except Exception as e:
        print(f"Warning: linear regression failed ({e}), falling back to equal weights", file=sys.stderr)
        return _weight_equal(len(predictions))


# ---------- Ensemble Prediction ----------
def _ensemble_predict(predictions: List[List[float]], weights: List[float]) -> List[float]:
    """Weighted average ensemble"""
    return [
        sum(w * p[j] for w, p in zip(weights, predictions))
        for j in range(len(predictions[0]))
    ]


# ---------- Main Entry Point ----------
def run(args) -> None:
    pretty = bool(getattr(args, "pretty", False))
    load_ensemble_path = getattr(args, "load_ensemble", None)
    save_ensemble_path = getattr(args, "save_ensemble", None)
    
    # ========== Loading Mode ==========
    if load_ensemble_path:
        config = load_joblib(load_ensemble_path)
        model_paths = config["model_paths"]
        weights = config["weights"]
        strategy = config["strategy"]
        
        steps = None
        if hasattr(args, "eval_start") and hasattr(args, "eval_end"):
            try:
                es, ee = int(args.eval_start), int(args.eval_end)
                if ee >= es:
                    steps = ee - es + 1
            except Exception:
                pass
        if steps is None:
            fs = getattr(args, "forecast_steps", None)
            if fs is None:
                raise ValueError("When --load_ensemble, provide eval_start/eval_end or --forecast_steps.")
            steps = int(fs)
            if steps <= 0:
                raise ValueError("--forecast_steps must be > 0")
        
        data = getattr(args, "data", None)
        predictions = get_model_predictions(model_paths, steps, data)
        ensemble_pred = _ensemble_predict(predictions, weights)
        
        std_dev = config.get("std_dev_residuals", 0.0)
        predictions_with_std = [{"prediction": p, "std_dev": std_dev} for p in ensemble_pred]

        out = {
            "mode": "ensemble_loaded_predict",
            "strategy": strategy,
            "n_models": len(model_paths),
            "model_names": _extract_model_names(model_paths),
            "weights": weights,
            "forecast_steps": steps,
            "predictions": predictions_with_std,
        }
        
        if hasattr(args, "data") and hasattr(args, "eval_start") and hasattr(args, "eval_end"):
            try:
                arr = json.loads(args.data)
                sales = [float(it.get("sales", it.get("y"))) for it in arr]
                es, ee = int(args.eval_start), int(args.eval_end)
                y_true = sales[es : ee + 1]
                if len(y_true) == len(ensemble_pred):
                    out["metrics"] = {
                        "rmse": rmse(y_true, ensemble_pred),
                        "mae": mae(y_true, ensemble_pred),
                        "mape": mape(y_true, ensemble_pred),
                        "r2": r2_score(y_true, ensemble_pred),
                    }
            except Exception:
                pass
        
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return
    
    # ========== Training Mode ==========
    if not hasattr(args, "models") or not args.models:
        raise ValueError("Must provide --models <path1> <path2> ... for ensemble training")
    
    model_paths = args.models if isinstance(args.models, list) else [args.models]
    if len(model_paths) < 2:
        raise ValueError("Ensemble requires at least 2 models")
    
    try:
        arr = json.loads(args.data)
        sales = [float(it.get("sales", it.get("y"))) for it in arr]
    except Exception as e:
        raise ValueError(f"Failed to parse --data: {e}")
    
    n = len(sales)
    es, ee = int(args.eval_start), int(args.eval_end)
    
    if not (0 <= es <= ee < n):
        raise ValueError(f"Invalid eval range: [{es},{ee}] within [0,{n-1}]")
    
    strategy = getattr(args, "weight_strategy", "inverse_variance").lower()
    if strategy not in ("equal", "inverse_variance", "inverse_rmse", "grid_search", "linear_regression"):
        raise ValueError("--weight_strategy must be one of: equal, inverse_variance, inverse_rmse, grid_search, linear_regression")
    
    eval_steps = ee - es + 1
    eval_predictions = get_model_predictions(model_paths, eval_steps, args.data)
    y_true = sales[es : ee + 1]
    
    weights = []
    if strategy == "equal":
        weights = _weight_equal(len(model_paths))
    elif strategy == "inverse_variance":
        weights = _weight_inverse_variance(eval_predictions, y_true)
    elif strategy == "inverse_rmse":
        weights = _weight_inverse_rmse(eval_predictions, y_true)
    elif strategy == "grid_search":
        weights, _ = _weight_grid_search(eval_predictions, y_true)
    elif strategy == "linear_regression":
        weights = _weight_linear_regression(eval_predictions, y_true)
    
    ensemble_pred = _ensemble_predict(eval_predictions, weights)
    
    residuals = np.array(y_true) - np.array(ensemble_pred)
    std_dev_residuals = float(np.std(residuals))

    out = {
        "mode": "ensemble_train_eval",
        "strategy": strategy,
        "n_models": len(model_paths),
        "model_names": _extract_model_names(model_paths),
        "weights": [float(w) for w in weights],
        "eval_range": {"start": es, "end": ee},
        "metrics": {
            "rmse": rmse(y_true, ensemble_pred),
            "mae": mae(y_true, ensemble_pred),
            "mape": mape(y_true, ensemble_pred),
            "r2": r2_score(y_true, ensemble_pred),
        },
        "eval_y_true": [float(v) for v in y_true],
        "eval_predictions": [float(v) for v in ensemble_pred],
        "notes": [f"Weights determined on eval set [{es},{ee}] using strategy: {strategy}"],
    }
    
    if save_ensemble_path:
        config = {
            "model_paths": model_paths, 
            "weights": weights, 
            "strategy": strategy,
            "std_dev_residuals": std_dev_residuals,
        }
        dump_joblib(config, save_ensemble_path)
        out["saved_ensemble"] = save_ensemble_path
    
    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
