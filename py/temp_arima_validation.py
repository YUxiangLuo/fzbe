# py/temp_arima_validation.py
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import subprocess
import os
import sys

def run_arima_script(data_str, train_start, train_end, eval_start, eval_end, d):
    """Helper to run the arima.py script via subprocess."""
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'main.py'))
    python_executable = os.path.abspath(os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'python'))
    
    cmd = [
        python_executable,
        script_path,
        "--method", "arima",
        "--data", data_str,
        "--train_start", str(train_start),
        "--train_end", str(train_end),
        "--eval_start", str(eval_start),
        "--eval_end", str(eval_end),
        "--d", str(d),
        "--p_max", "2", # Keep it small for validation
        "--q_max", "2",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)

def main():
    print("Starting ARIMA cross-validation...")

    # 1. Prepare Data
    # Using a well-known dataset for reproducibility
    try:
        # Johnson & Johnson quarterly earnings per share
        jj = pd.read_csv("https://raw.githubusercontent.com/statsmodels/statsmodels/main/statsmodels/datasets/jj/jj.csv")
        series = jj['data'].to_numpy(dtype=np.float64)
    except Exception as e:
        print(f"Failed to download dataset. Using a simple sine wave instead. Error: {e}")
        t = np.linspace(0, 50, 100)
        series = 100 + 20 * np.sin(t * 0.5) + np.random.randn(100) * 5

    n = len(series)
    train_start, train_end = 0, n - 11
    eval_start, eval_end = n - 10, n - 1
    d = 1 # Seasonal data, differencing is appropriate

    data_list = [{"sales": float(s)} for s in series]
    data_str = json.dumps(data_list)

    print(f"Dataset loaded. Size={n}, Train=[{train_start},{train_end}], Eval=[{eval_start},{eval_end}], d={d}")

    # 2. Run arima.py script
    print("\nRunning arima.py script...")
    try:
        script_output = run_arima_script(data_str, train_start, train_end, eval_start, eval_end, d)
        script_p = script_output["best_order"]["p"]
        script_q = script_output["best_order"]["q"]
        script_preds = [p["prediction"] for p in script_output["predictions"]]
        print(f"Script finished. Best order (p,d,q): ({script_p}, {d}, {script_q})")
    except subprocess.CalledProcessError as e:
        print("ERROR: arima.py script failed!")
        print("Stderr:", e.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while running the script: {e}")
        sys.exit(1)

    # 3. Run direct statsmodels for comparison
    print("\nRunning direct statsmodels calculation...")
    y_train = series[train_start : train_end + 1]
    
    model = SARIMAX(
        y_train,
        order=(script_p, d, script_q),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    res = model.fit(disp=False)
    
    steps = eval_end - eval_start + 1
    forecast = res.get_forecast(steps=steps)
    direct_preds = forecast.predicted_mean

    print("Direct statsmodels calculation finished.")

    # 4. Compare results
    print("\nComparing results...")
    script_preds = np.array(script_preds)
    direct_preds = np.array(direct_preds)

    abs_diff = np.abs(script_preds - direct_preds)
    
    print(f"Script predictions (first 5): {script_preds[:5]}")
    print(f"Direct predictions (first 5): {direct_preds[:5]}")
    print(f"Max absolute difference: {np.max(abs_diff):.6f}")
    print(f"Mean absolute difference: {np.mean(abs_diff):.6f}")

    # We expect the results to be very close. A small tolerance is acceptable due to
    # potential minor differences in floating point arithmetic or environment.
    if np.allclose(script_preds, direct_preds, atol=1e-5):
        print("\n✅ SUCCESS: Predictions are consistent.")
    else:
        print("\n❌ FAILURE: Predictions are significantly different.")
        sys.exit(1)

if __name__ == "__main__":
    main()
