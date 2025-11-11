# py/src/models/boosting.py
# -*- coding: utf-8 -*-
"""
AdaBoost.R2 Ensemble: A regression-specific variant of AdaBoost.
"""

import json
import sys
import os
import tempfile
from copy import deepcopy
from typing import List
import numpy as np
from joblib_compat import load_joblib, dump_joblib
from models.utils import rmse, mae, mape, r2_score, get_model_predictions


# ---------- Ensemble Prediction ----------
def _ensemble_predict_adaboost_r2(predictions: List[List[float]], model_betas: List[float]) -> List[float]:
    """AdaBoost.R2 ensemble (weighted average)"""
    if not predictions or not model_betas:
        return []
    
    # Weights are log(1/beta), which is proportional to model performance.
    model_weights = np.array([np.log(1.0 / beta) for beta in model_betas if beta > 0 and beta < 1])
    
    if len(model_weights) != len(predictions):
        raise ValueError("Number of model weights must match number of prediction sets.")

    # Use weighted average for the final prediction.
    # predictions shape: (n_models, n_steps)
    # weights shape: (n_models,)
    # np.average with axis=0 computes the weighted average across models for each step.
    ensemble_pred = np.average(predictions, axis=0, weights=model_weights)
    return ensemble_pred.tolist()


# ---------- Main Entry Point ----------
def run(args) -> None:
    pretty = bool(getattr(args, "pretty", False))
    load_ensemble_path = getattr(args, "load_ensemble", None)
    save_ensemble_path = getattr(args, "save_ensemble", None)
    learning_rate = float(getattr(args, "learning_rate", 1.0))
    if learning_rate <= 0.0:
        raise ValueError("--learning_rate must be > 0")
    
    # ========== Loading Mode ==========
    if load_ensemble_path:
        config = load_joblib(load_ensemble_path)
        model_paths = config["model_paths"]
        strategy = config.get("strategy", "adaboost_r2")
        model_betas = config.get("model_betas")

        if model_betas is None:
            raise ValueError("Loaded ensemble is not a valid AdaBoost.R2 model (missing model_betas).")

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
        ensemble_pred = _ensemble_predict_adaboost_r2(predictions, model_betas)
        
        std_dev = config.get("std_dev_residuals", 0.0)
        predictions_with_std = [{"prediction": p, "std_dev": std_dev} for p in ensemble_pred]

        out = {
            "mode": "adaboost_r2_loaded_predict",
            "strategy": strategy,
            "n_models": len(model_paths),
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
        raise ValueError("Must provide --models <path1> <path2> ... for AdaBoost.R2 training")
    
    model_paths = args.models if isinstance(args.models, list) else [args.models]
    
    try:
        arr = json.loads(args.data)
        sales = [float(it.get("sales", it.get("y"))) for it in arr]
    except Exception as e:
        raise ValueError(f"Failed to parse --data: {e}")
    
    n = len(sales)
    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)
    
    if not (0 <= ts <= te < es <= ee < n):
        raise ValueError(f"Invalid ranges: train [{ts},{te}], eval [{es},{ee}] within [0,{n-1}]")

    from models import arima, lstm, ma as ma_model, es as es_model

    temp_dir = tempfile.mkdtemp(prefix="adaboost_r2_ensemble_")
    print(f"AdaBoost.R2: Storing base models in {temp_dir}", file=sys.stderr)

    train_arr = arr[ts : te + 1]
    y_train = np.array([float(it.get("sales", it.get("y"))) for it in train_arr])
    y_eval_true = sales[es : ee + 1]
    
    sample_weights = np.ones(len(y_train)) / len(y_train)
    trained_models = []
    model_betas = []
    discarded_models_info = []

    for i, model_path in enumerate(model_paths):
        print(f"AdaBoost.R2 Step {i+1}/{len(model_paths)}: Training with base model {model_path}", file=sys.stderr)

        # Resample training data based on current sample weights.
        # NOTE: This is a key part of AdaBoost. For time series models, this shuffling
        # may disrupt temporal dependencies. This implementation proceeds as it is the
        # standard way to implement AdaBoost with learners that don't support sample weights.
        resample_indices = np.random.choice(len(train_arr), size=len(train_arr), p=sample_weights, replace=True)
        resampled_train_arr = [train_arr[idx] for idx in resample_indices]

        temp_args = deepcopy(args)
        temp_args.data = json.dumps(resampled_train_arr)

        # Set up a valid train/eval split for the sub-model training on resampled data.
        resampled_len = len(resampled_train_arr)
        model_type = "unknown"
        path_lower = model_path.lower()
        if "arima" in path_lower: model_type = "arima"
        elif "lstm" in path_lower: model_type = "lstm"
        elif "es" in path_lower: model_type = "es"
        elif "ma" in path_lower: model_type = "ma"

        if model_type == "lstm":
            lookback = int(getattr(args, "lookback", 4))
            min_eval_len = lookback + 1
            if resampled_len < min_eval_len + 1:
                print(f"Warning: Resampled data length ({resampled_len}) is too short for LSTM with lookback {lookback}. Stopping.", file=sys.stderr)
                break
            temp_args.train_end = resampled_len - min_eval_len - 1
            temp_args.eval_start = resampled_len - min_eval_len
            temp_args.eval_end = resampled_len - 1
        else:
            if resampled_len < 2:
                print(f"Warning: Resampled data length ({resampled_len}) is too short for {model_path}. Stopping.", file=sys.stderr)
                break
            temp_args.train_end = resampled_len - 2
            temp_args.eval_start = resampled_len - 1
            temp_args.eval_end = resampled_len - 1
        temp_args.train_start = 0

        model_filename = f"adaboost_model_{i+1}_{os.path.basename(model_path)}"
        new_model_path = os.path.join(temp_dir, model_filename)
        temp_args.save_model = new_model_path

        original_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            if model_type == "arima": arima.run(temp_args)
            elif model_type == "lstm": lstm.run(temp_args)
            elif model_type == "es": es_model.run(temp_args)
            elif model_type == "ma": ma_model.run(temp_args)
            else: raise ValueError(f"Cannot determine model type for boosting from path: {model_path}")
        finally:
            sys.stdout = original_stdout
        
        trained_models.append(new_model_path)

        # Get predictions on the *original* training set to calculate errors.
        train_preds = get_model_predictions([new_model_path], len(y_train), json.dumps(train_arr))[0]
        
        errors = np.abs(y_train - np.array(train_preds))
        max_error = np.max(errors)
        if max_error == 0:
            print("Model is perfect, no errors to boost. Stopping.", file=sys.stderr)
            model_betas.append(1e-10) # Add a very small beta
            break
        
        normalized_errors = errors / max_error
        model_error = np.sum(sample_weights * normalized_errors)

        if model_error >= 0.5:
            print(f"Model error is {model_error:.4f} >= 0.5, stopping training.", file=sys.stderr)
            # Remove the last trained model as it's too weak.
            trained_models.pop()
            discarded_models_info.append({
                "path": model_path,
                "reason": f"Model error ({model_error:.4f}) was >= 0.5"
            })
            break
        
        beta = model_error / (1.0 - model_error)
        model_betas.append(beta)

        # Update sample weights
        # The learning_rate parameter is used here as shrinkage.
        update_exponents = learning_rate * (1.0 - normalized_errors)
        sample_weights *= np.power(beta, update_exponents)
        sample_weights /= np.sum(sample_weights)

    if not trained_models:
        raise RuntimeError("AdaBoost.R2 training failed to produce any valid models.")

    # Get predictions on the original evaluation set from all trained models.
    all_eval_preds = get_model_predictions(trained_models, len(y_eval_true), data=None)
    ensemble_pred = _ensemble_predict_adaboost_r2(all_eval_preds, model_betas)

    residuals = np.array(y_eval_true) - np.array(ensemble_pred)
    std_dev_residuals = float(np.std(residuals))

    notes = [
        f"AdaBoost.R2 ensemble trained. Base models stored in {temp_dir}",
        f"Learning rate (shrinkage) applied during weight update: {learning_rate}",
    ]
    for discarded in discarded_models_info:
        notes.append(f"Discarded base model '{os.path.basename(discarded['path'])}': {discarded['reason']}")

    out = {
        "mode": "adaboost_r2_train_eval",
        "strategy": "adaboost_r2",
        "n_models": len(trained_models),
        "trained_base_models": [os.path.basename(p) for p in trained_models],
        "eval_range": {"start": es, "end": ee},
        "learning_rate": learning_rate,
        "metrics": {
            "rmse": rmse(y_eval_true, ensemble_pred),
            "mae": mae(y_eval_true, ensemble_pred),
            "mape": mape(y_eval_true, ensemble_pred),
            "r2": r2_score(y_eval_true, ensemble_pred),
        },
        "eval_y_true": [float(v) for v in y_eval_true],
        "eval_predictions": [float(v) for v in ensemble_pred],
        "notes": notes,
    }
    
    if save_ensemble_path:
        config = {
            "model_paths": trained_models, 
            "model_betas": model_betas,
            "strategy": "adaboost_r2",
            "std_dev_residuals": std_dev_residuals,
            "learning_rate": learning_rate,
        }
        dump_joblib(config, save_ensemble_path)
        out["saved_ensemble"] = save_ensemble_path
    
    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))