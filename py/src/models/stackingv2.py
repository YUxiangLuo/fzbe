# py/src/models/stackingv2.py
import json
import sys
import os
import sys
from typing import List, Dict, Any
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from concurrent.futures import ThreadPoolExecutor
import subprocess

# Add src to path to be able to import other models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics import rmse, mae, mape, r2

def _run_base_model(model_name: str, data_records: List[Dict[str, Any]], train_indices: np.ndarray, test_indices: np.ndarray, model_args: Dict[str, Any]) -> np.ndarray:
    """
    A helper function to run a single base model as a separate process for a given train/test split.
    """
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main.py'))
    python_executable = sys.executable  # Use the same python interpreter

    # Create a subset of data for this specific fold
    fold_records = [data_records[i] for i in np.concatenate([train_indices, test_indices])]
    fold_data_str = json.dumps(fold_records)

    # Arguments for the subprocess
    cmd = [
        python_executable,
        script_path,
        "--method", model_name,
        "--data", fold_data_str,
        "--train_start", "0",
        "--train_end", str(len(train_indices) - 1),
        "--eval_start", str(len(train_indices)),
        "--eval_end", str(len(train_indices) + len(test_indices) - 1),
    ]

    # Add model-specific arguments
    for key, value in model_args.items():
        cmd.extend([f"--{key}", str(value)])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = json.loads(result.stdout)

    # Base models might return predictions under different keys
    if "eval_predictions" in output:
        return np.array(output["eval_predictions"])
    elif "predictions" in output:
        # ARIMA's training output has a different structure
        return np.array([p["prediction"] for p in output["predictions"]])
    else:
        raise ValueError(f"Base model '{model_name}' output does not contain 'predictions' or 'eval_predictions'")

def run(args) -> None:
    """
    Main function for the Stacking v2 model.
    """
    pretty = bool(getattr(args, "pretty", False))
    
    # 1. Parse arguments
    try:
        records = json.loads(args.data)
        base_models = args.base_models
        eval_start = int(args.eval_start)
        eval_end = int(args.eval_end)
        meta_l2 = float(getattr(args, "meta_l2", 1.0))
    except Exception as e:
        raise ValueError(f"[VALIDATION] Invalid arguments: {e}")

    # For simplicity, we'll assume the entire dataset before the evaluation start is for training.
    train_records = records[:eval_start]
    eval_records = records[eval_start : eval_end + 1]
    y_true_eval = np.array([r["sales"] for r in eval_records])

    n_splits = 5 # Standard for k-fold
    kf = KFold(n_splits=n_splits, shuffle=False)

    # 2. First stage: Generate out-of-fold predictions from base models
    oof_preds = np.zeros((len(train_records), len(base_models)))
    
    y_true_train = np.array([r["sales"] for r in train_records])

    with ThreadPoolExecutor() as executor:
        futures = []
        for model_idx, model_name in enumerate(base_models):
            for train_indices, test_indices in kf.split(train_records):
                # For stacking, we need to pass the specific args for each model
                model_args = {}
                if model_name == "ma":
                    # Ensure window is not larger than the training set
                    model_args["window"] = min(10, len(train_indices))
                elif model_name == "es":
                    model_args["alpha"] = 0.5 # A reasonable default
                elif model_name == "arima":
                    model_args["d"] = 1 # A reasonable default
                elif model_name == "lstm":
                    # LSTM requires feature_types. For simplicity, we'll assume no extra features for now.
                    model_args["feature_types"] = json.dumps({})

                future = executor.submit(_run_base_model, model_name, train_records, train_indices, test_indices, model_args)
                futures.append((future, model_idx, test_indices))

        for future, model_idx, test_indices in futures:
            oof_preds[test_indices, model_idx] = future.result()
    
    # 3. Second stage: Train the meta-learner
    meta_learner = Ridge(alpha=meta_l2)
    meta_learner.fit(oof_preds, y_true_train)

    # 4. Final prediction
    # Retrain base models on the full training data and predict on the evaluation set.
    final_preds_base = np.zeros((len(eval_records), len(base_models)))
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for model_idx, model_name in enumerate(base_models):
            model_args = {}
            if model_name == "ma":
                model_args["window"] = 10
            elif model_name == "es":
                model_args["alpha"] = 0.5
            elif model_name == "arima":
                model_args["d"] = 1
            elif model_name == "lstm":
                model_args["feature_types"] = json.dumps({})

            future = executor.submit(_run_base_model, model_name, records, np.arange(len(train_records)), np.arange(eval_start, eval_end + 1), model_args)
            futures.append((future, model_idx))

        for future, model_idx in futures:
            final_preds_base[:, model_idx] = future.result()

    final_predictions = meta_learner.predict(final_preds_base)

    # 5. Assemble and print output
    y_pred = final_predictions.tolist()
    out = {
        "mode": "stacking_v2",
        "metrics": {
            "rmse": rmse(y_true_eval.tolist(), y_pred),
            "mae": mae(y_true_eval.tolist(), y_pred),
            "mape": mape(y_true_eval.tolist(), y_pred),
            "r2": r2(y_true_eval.tolist(), y_pred),
        },
        "eval_y_true": y_true_eval.tolist(),
        "eval_predictions": y_pred,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))

if __name__ == "__main__":
    # This part is for standalone testing if needed.
    pass
