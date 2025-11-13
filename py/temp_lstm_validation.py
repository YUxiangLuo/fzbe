# py/temp_lstm_validation.py
import json
import numpy as np
import subprocess
import os
import sys
import tempfile

def run_lstm_script(args):
    """Helper to run the lstm.py script via subprocess."""
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'main.py'))
    python_executable = os.path.abspath(os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'python'))
    
    cmd = [python_executable, script_path, "--method", "lstm"] + args
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)

def main():
    print("Starting LSTM cross-validation...")
    temp_dir = tempfile.TemporaryDirectory()
    model_path = os.path.join(temp_dir.name, "lstm_model.pt")

    # 1. Prepare Data with mixed feature types
    n = 100
    records = [
        {
            "y": 100 + 20 * np.sin(i * 0.5) + np.random.randn() * 5,
            "x": {
                "numeric_feat": i,
                "binary_feat": i % 2,
                "categorical_feat": ["A", "B", "C"][i % 3],
            },
        }
        for i in range(n)
    ]
    data_str = json.dumps(records)
    feature_types_str = json.dumps({
        "numeric_feat": "numeric",
        "binary_feat": "binary",
        "categorical_feat": "categorical",
    })

    train_start, train_end = 0, 79
    eval_start, eval_end = 80, 99

    print(f"Dataset created. Size={n}, Train=[{train_start},{train_end}], Eval=[{eval_start},{eval_end}]")

    # 2. Run lstm.py script for training
    print("\nRunning lstm.py script for training...")
    try:
        train_args = [
            "--data", data_str,
            "--feature_types", feature_types_str,
            "--train_start", str(train_start),
            "--train_end", str(train_end),
            "--eval_start", str(eval_start),
            "--eval_end", str(eval_end),
            "--epochs", "10", # Keep it fast
            "--save_model", model_path,
        ]
        train_output = run_lstm_script(train_args)
        print("Training finished. Metrics:", train_output["metrics"])
    except subprocess.CalledProcessError as e:
        print("ERROR: lstm.py training script failed!")
        print("Stderr:", e.stderr)
        sys.exit(1)

    # 3. Run lstm.py script for prediction
    print("\nRunning lstm.py script for prediction...")
    try:
        predict_args = [
            "--load_model", model_path,
            "--forecast_steps", "10",
        ]
        predict_output = run_lstm_script(predict_args)
        predictions = [p["prediction"] for p in predict_output["predictions"]]
        print("Prediction finished. Predictions (first 5):", predictions[:5])
    except subprocess.CalledProcessError as e:
        print("ERROR: lstm.py prediction script failed!")
        print("Stderr:", e.stderr)
        sys.exit(1)

    # 4. Validate results
    print("\nValidating results...")
    if len(predictions) != 10:
        print(f"❌ FAILURE: Expected 10 predictions, but got {len(predictions)}.")
        sys.exit(1)

    # Check if predictions are reasonable (not NaN or Inf)
    if not all(np.isfinite(p) for p in predictions):
        print("❌ FAILURE: Predictions contain NaN or Inf values.")
        sys.exit(1)
        
    # A simple sanity check on the prediction range
    y_values = [r['y'] for r in records]
    y_mean = np.mean(y_values)
    y_std = np.std(y_values)
    
    # Check if predictions are within a reasonable range of the original data's distribution
    reasonable_min = y_mean - 5 * y_std
    reasonable_max = y_mean + 5 * y_std
    
    if not all(reasonable_min <= p <= reasonable_max for p in predictions):
        print(f"⚠️ WARNING: Some predictions are outside the expected range [{reasonable_min:.2f}, {reasonable_max:.2f}]. This is not necessarily an error but might indicate a problem.")

    print("\n✅ SUCCESS: LSTM validation script completed without critical errors.")
    temp_dir.cleanup()

if __name__ == "__main__":
    main()
