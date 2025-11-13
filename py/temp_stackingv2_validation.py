# py/temp_stackingv2_validation.py
import json
import numpy as np
import subprocess
import os
import sys

def run_stacking_script(args):
    """Helper to run the stackingv2.py script via subprocess."""
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'main.py'))
    python_executable = os.path.abspath(os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'python'))
    
    cmd = [python_executable, script_path, "--method", "stackingv2"] + args
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)

def main():
    print("Starting Stacking v2 cross-validation...")

    # 1. Prepare Data
    n = 100
    records = [{"sales": 100 + 20 * np.sin(i * 0.5) + np.random.randn() * 5} for i in range(n)]
    data_str = json.dumps(records)

    eval_start, eval_end = 80, 99

    print(f"Dataset created. Size={n}, Eval=[{eval_start},{eval_end}]")

    # 2. Run stackingv2.py script
    print("\nRunning stackingv2.py script...")
    try:
        args = [
            "--data", data_str,
            "--eval_start", str(eval_start),
            "--eval_end", str(eval_end),
            "--base_models", "ma", "es", "arima",
        ]
        output = run_stacking_script(args)
        print("Stacking finished. Metrics:", output["metrics"])
    except subprocess.CalledProcessError as e:
        print("ERROR: stackingv2.py script failed!")
        print("Stderr:", e.stderr)
        sys.exit(1)

    # 3. Validate results
    print("\nValidating results...")
    if len(output["eval_predictions"]) != (eval_end - eval_start + 1):
        print(f"❌ FAILURE: Expected {eval_end - eval_start + 1} predictions, but got {len(output['eval_predictions'])}.")
        sys.exit(1)

    if not all(np.isfinite(p) for p in output["eval_predictions"]):
        print("❌ FAILURE: Predictions contain NaN or Inf values.")
        sys.exit(1)

    print("\n✅ SUCCESS: Stacking v2 validation script completed without critical errors.")

if __name__ == "__main__":
    main()
