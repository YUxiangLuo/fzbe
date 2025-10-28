# src/models/stacking.py
# -*- coding: utf-8 -*-
"""
Stacking Ensemble: Train a meta-model to combine predictions from base models.

This implementation uses a Linear Regression model (trained with least squares)
as the meta-model.

Usage:
1) Training Mode (to train the meta-model):
   --method stacking
   --data <JSON>
   --eval_start, --eval_end (to get base predictions and train the meta-model)
   --models <model1_path> <model2_path> ...
   --save_ensemble <path_to_save_stacking_config>

2) Prediction Mode:
   --method stacking
   --load_ensemble <path>
   --forecast_steps <int>
"""

import json
import sys
from typing import List, Dict, Any
import numpy as np
from joblib_compat import load_joblib, dump_joblib
from models.utils import rmse, mae, mape, r2_score, get_model_predictions


# ---------- Meta-Model Training ----------
def _train_meta_model(
    predictions: List[List[float]],
    y_true: List[float],
    l2_strength: float,
) -> Dict[str, Any]:
    """
    Trains a linear regression meta-model.
    Returns a dictionary containing the learned coefficients and intercept.
    """
    # The predictions from base models are our features.
    # Shape: (n_models, n_samples) -> transpose to (n_samples, n_models)
    X = np.array(predictions).T
    y = np.array(y_true)

    # Add a column of ones to X to account for the intercept
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    lam = max(float(l2_strength), 0.0)
    XtX = X_b.T @ X_b
    reg = np.eye(XtX.shape[0], dtype=float)
    reg[0, 0] = 0.0  # do not regularise intercept

    try:
        theta = np.linalg.solve(XtX + lam * reg, X_b.T @ y)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(X_b, y, rcond=None)[0]

    intercept = float(theta[0])
    coefficients = [float(c) for c in theta[1:]]

    return {"intercept": intercept, "coefficients": coefficients, "l2": lam}


# ---------- Stacking Prediction ----------
def _stacking_predict(predictions: List[List[float]], meta_model: Dict[str, Any]) -> List[float]:
    """
    Applies the trained meta-model to a new set of base predictions.
    """
    X = np.array(predictions).T
    intercept = meta_model["intercept"]
    coefficients = np.array(meta_model["coefficients"], dtype=float)

    if X.shape[1] != len(coefficients):
        raise ValueError(f"Number of base model predictions ({X.shape[1]}) does not match number of meta-model coefficients ({len(coefficients)}).")

    # Final prediction = intercept + (prediction_1 * coeff_1) + (prediction_2 * coeff_2) + ...
    final_predictions = X @ coefficients + intercept
    return final_predictions.tolist()


# ---------- Main Entry Point ----------
def run(args) -> None:
    pretty = bool(getattr(args, "pretty", False))
    load_ensemble_path = getattr(args, "load_ensemble", None)
    save_ensemble_path = getattr(args, "save_ensemble", None)
    l2_strength = float(getattr(args, "meta_l2", 0.1))
    if l2_strength < 0:
        raise ValueError("--meta_l2 must be >= 0")

    # ========== Loading/Prediction Mode ==========
    if load_ensemble_path:
        config = load_joblib(load_ensemble_path)
        model_paths = config["base_model_paths"]
        meta_model = config["meta_model"]
        if "meta_l2" in config and "l2" not in meta_model:
            meta_model = {**meta_model, "l2": config["meta_l2"]}

        fs = getattr(args, "forecast_steps", None)
        if fs is None:
            raise ValueError("When --load_ensemble, you must provide --forecast_steps.")
        steps = int(fs)
        if steps <= 0:
            raise ValueError("--forecast_steps must be > 0.")

        # Get future predictions from all base models
        base_predictions = get_model_predictions(model_paths, steps, data=None)

        # Combine them using the meta-model
        final_predictions = _stacking_predict(base_predictions, meta_model)

        std_dev = config.get("std_dev_residuals", 0.0)
        predictions_with_std = [{"prediction": p, "std_dev": std_dev} for p in final_predictions]

        out = {
            "mode": "stacking_loaded_predict",
            "n_base_models": len(model_paths),
            "meta_model": meta_model,
            "forecast_steps": steps,
            "predictions": predictions_with_std,
        }

        es = getattr(args, "eval_start", None)
        ee = getattr(args, "eval_end", None)
        if getattr(args, "data", None) and es is not None and ee is not None:
            try:
                es_i, ee_i = int(es), int(ee)
                arr = json.loads(args.data)
                sales = [float(it.get("sales", it.get("y"))) for it in arr]
                y_true = sales[es_i : ee_i + 1]
                if len(y_true) == len(final_predictions):
                    out["metrics"] = {
                        "rmse": rmse(y_true, final_predictions),
                        "mae": mae(y_true, final_predictions),
                        "mape": mape(y_true, final_predictions),
                        "r2": r2_score(y_true, final_predictions),
                    }
            except Exception:
                pass

        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # ========== Training Mode ==========
    if not hasattr(args, "models") or not args.models:
        raise ValueError("Must provide --models <path1> <path2> ... for stacking training.")
    
    model_paths = args.models if isinstance(args.models, list) else [args.models]
    if len(model_paths) < 2:
        raise ValueError("Stacking requires at least 2 base models.")

    try:
        arr = json.loads(args.data)
        sales = [float(it.get("sales", it.get("y"))) for it in arr]
    except Exception as e:
        raise ValueError(f"Failed to parse --data: {e}")

    n = len(sales)
    es = int(args.eval_start)
    ee = int(args.eval_end)

    if not (0 <= es <= ee < n):
        raise ValueError(f"Invalid eval range: [{es},{ee}] within [0,{n-1}]")

    # 1. Get predictions from base models on the evaluation set
    eval_steps = ee - es + 1
    # Pass the full data, as base models might need it to construct their prediction windows
    base_predictions_on_eval = get_model_predictions(model_paths, eval_steps, args.data)
    y_true_on_eval = sales[es : ee + 1]

    if not base_predictions_on_eval:
        raise ValueError("Base models produced no predictions; check model paths and data.")

    # 2. Train the meta-model
    meta_model = _train_meta_model(base_predictions_on_eval, y_true_on_eval, l2_strength)

    # 3. (Optional) Evaluate the trained stacking model on the same eval set
    final_predictions_on_eval = _stacking_predict(base_predictions_on_eval, meta_model)

    residuals = np.array(y_true_on_eval) - np.array(final_predictions_on_eval)
    std_dev_residuals = float(np.std(residuals))

    out = {
        "mode": "stacking_train_eval",
        "n_base_models": len(model_paths),
        "meta_model": meta_model,
        "eval_range": {"start": es, "end": ee},
        "metrics": {
            "rmse": rmse(y_true_on_eval, final_predictions_on_eval),
            "mae": mae(y_true_on_eval, final_predictions_on_eval),
            "mape": mape(y_true_on_eval, final_predictions_on_eval),
            "r2": r2_score(y_true_on_eval, final_predictions_on_eval),
        },
        "notes": [
            "Meta-model (Linear Regression) was trained on the predictions of base models over the evaluation set.",
            f"L2 regularisation strength meta_l2={l2_strength}",
        ],
    }

    if save_ensemble_path:
        config = {
            "base_model_paths": model_paths,
            "meta_model": meta_model,
            "strategy": "stacking",
            "std_dev_residuals": std_dev_residuals,
            "meta_l2": l2_strength,
        }
        dump_joblib(config, save_ensemble_path)
        out["saved_ensemble"] = save_ensemble_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
