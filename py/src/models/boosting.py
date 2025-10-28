# src/models/boosting.py
# -*- coding: utf-8 -*-
"""
Boosting Ensemble: Sequentially train models on the residuals of predecessors.
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
def _ensemble_predict_boosting(predictions: List[List[float]], learning_rate: float = 1.0) -> List[float]:
    """Boosting ensemble (summation)"""
    if not predictions:
        return []
    lr = float(learning_rate)
    ensemble_pred = [0.0] * len(predictions[0])
    for stage in predictions:
        for j, val in enumerate(stage):
            ensemble_pred[j] += lr * float(val)
    return ensemble_pred


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
        strategy = config.get("strategy", "boosting")
        learning_rate_cfg = float(config.get("learning_rate", learning_rate))

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
        ensemble_pred = _ensemble_predict_boosting(predictions, learning_rate_cfg)
        
        std_dev = config.get("std_dev_residuals", 0.0)
        predictions_with_std = [{"prediction": p, "std_dev": std_dev} for p in ensemble_pred]

        out = {
            "mode": "boosting_loaded_predict",
            "strategy": strategy,
            "n_models": len(model_paths),
            "forecast_steps": steps,
            "predictions": predictions_with_std,
            "learning_rate": learning_rate_cfg,
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
        raise ValueError("Must provide --models <path1> <path2> ... for boosting training")
    
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

    temp_dir = tempfile.mkdtemp(prefix="boosting_ensemble_")
    print(f"Boosting: Storing residual models in {temp_dir}", file=sys.stderr)

    y_train = sales[ts : te + 1]
    y_eval_true = sales[es : ee + 1]
    
    current_residuals_train = list(y_train)
    final_eval_predictions = np.zeros(len(y_eval_true))
    trained_residual_models = []

    for i, model_path in enumerate(model_paths):
        print(f"Boosting Step {i+1}/{len(model_paths)}: Training on residuals with base model {model_path}", file=sys.stderr)

        temp_args = deepcopy(args)

        model_type = "unknown"
        path_lower = model_path.lower()
        if "arima" in path_lower: model_type = "arima"
        elif "lstm" in path_lower: model_type = "lstm"
        elif "es" in path_lower: model_type = "es"
        elif "ma" in path_lower: model_type = "ma"

        if model_type == "lstm":
            residual_data = [{"ts": f"res_{i}", "y": val, "x": {"year": 0, "month_of_year": 0, "day_of_week": 0, "is_holiday": 0}} for i, val in enumerate(current_residuals_train)]
            temp_args.data = json.dumps(residual_data)
        else:
            temp_args.data = json.dumps([{"month": f"res_{i}", "sales": val} for i, val in enumerate(current_residuals_train)])

        # Set up a valid train/eval split for the sub-model training,
        # respecting the specific needs of the LSTM model.
        residual_len = len(current_residuals_train)

        if model_type == "lstm":
            lookback = int(getattr(args, "lookback", 4))
            min_eval_len = lookback + 1
            if residual_len < min_eval_len + 1:
                print(f"Warning: Residual data length ({residual_len}) is too short for LSTM with lookback {lookback}. Stopping boosting.", file=sys.stderr)
                break
            temp_args.train_end = residual_len - min_eval_len - 1
            temp_args.eval_start = residual_len - min_eval_len
            temp_args.eval_end = residual_len - 1
        else:
            if residual_len < 2:
                print(f"Warning: Residual data length ({residual_len}) is too short to train model {model_path}. Stopping boosting.", file=sys.stderr)
                break
            temp_args.train_end = residual_len - 2
            temp_args.eval_start = residual_len - 1
            temp_args.eval_end = residual_len - 1

        temp_args.train_start = 0

        model_filename = f"residual_model_{i+1}_{os.path.basename(model_path)}"
        new_model_path = os.path.join(temp_dir, model_filename)
        temp_args.save_model = new_model_path

        # Suppress stdout from sub-model training to avoid polluting boosting output
        original_stdout = sys.stdout
        sys.stdout = sys.stderr

        try:
            if model_type == "arima":
                if not hasattr(args, "d"):
                    raise ValueError("ARIMA model for boosting requires --d")
                arima.run(temp_args)
            elif model_type == "lstm":
                if not hasattr(args, "feature_types"):
                    raise ValueError("LSTM model for boosting requires --feature_types. Please provide feature types for the CSV data.")
                lstm.run(temp_args)
            elif model_type == "es":
                if not hasattr(args, "alpha"):
                    raise ValueError("ES model for boosting requires --alpha")
                es_model.run(temp_args)
            elif model_type == "ma":
                if not hasattr(args, "window"):
                    raise ValueError("MA model for boosting requires --window")
                ma_model.run(temp_args)
            else:
                raise ValueError(f"Cannot determine model type for boosting from path: {model_path}")
        finally:
            sys.stdout = original_stdout

        trained_residual_models.append(new_model_path)

        train_preds = get_model_predictions([new_model_path], len(current_residuals_train), temp_args.data)[0]
        # Get predictions on the original evaluation set.
        # CRITICAL: Pass `data=None` to force pure recursive forecasting,
        # ensuring consistency between training evaluation and final prediction.
        eval_preds = get_model_predictions([new_model_path], len(y_eval_true), data=None)[0]

        current_residuals_train = [
            true - learning_rate * pred for true, pred in zip(current_residuals_train, train_preds)
        ]
        final_eval_predictions += learning_rate * np.array(eval_preds, dtype=float)

    ensemble_pred = final_eval_predictions.tolist()

    residuals = np.array(y_eval_true) - np.array(ensemble_pred)
    std_dev_residuals = float(np.std(residuals))

    out = {
        "mode": "boosting_train_eval",
        "strategy": "boosting",
        "n_models": len(trained_residual_models),
        "eval_range": {"start": es, "end": ee},
        "learning_rate": learning_rate,
        "metrics": {
            "rmse": rmse(y_eval_true, ensemble_pred),
            "mae": mae(y_eval_true, ensemble_pred),
            "mape": mape(y_eval_true, ensemble_pred),
            "r2": r2_score(y_eval_true, ensemble_pred),
        },
        "notes": [
            f"Boosting ensemble trained. Residual models stored in {temp_dir}",
            f"Learning rate (shrinkage) applied at each stage: {learning_rate}",
        ],
    }
    
    if save_ensemble_path:
        config = {
            "model_paths": trained_residual_models, 
            "weights": None, 
            "strategy": "boosting",
            "std_dev_residuals": std_dev_residuals,
            "learning_rate": learning_rate,
        }
        dump_joblib(config, save_ensemble_path)
        out["saved_ensemble"] = save_ensemble_path
    
    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
