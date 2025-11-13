# py/temp_ma_validation.py
import numpy as np
import pandas as pd
import sys
import os

# Add src to path to be able to import ma from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from models import ma

def validate_ma_implementation():
    """
    Cross-validates our custom MA implementation against the pandas library.
    """
    # 1. Define sample data and parameters
    series = np.array([10, 12, 15, 11, 14, 16, 18, 20, 22, 25], dtype=np.float64)
    window = 3
    # We want to predict for indices 3, 4, and 5
    eval_range = range(3, 6)

    print("--- Input Data ---")
    print(f"Series: {series.tolist()}")
    print(f"Window: {window}")
    print(f"Evaluation Indices: {list(eval_range)}")
    print("-" * 20)

    # 2. Run our custom implementation
    try:
        our_idxs, our_preds = ma.moving_average_forecast(
            series.tolist(), eval_range, window, min_index=0
        )
        print("--- Our Implementation Results ---")
        print(f"Evaluation Predictions: {[f'{p:.4f}' for p in our_preds]}")
        print(f"Effective Indices: {our_idxs}")
    except Exception as e:
        print(f"Our implementation failed: {e}")
        our_preds = None

    print("-" * 20)

    # 3. Run pandas implementation
    try:
        # pandas rolling mean is a backward-looking window, same as ours.
        # .shift(1) is crucial because forecast for time `t` uses data up to `t-1`.
        pd_rolling_mean = pd.Series(series).rolling(window=window).mean().shift(1)
        
        # Extract the predictions for the evaluation range
        pd_preds = pd_rolling_mean[eval_range].tolist()

        print("--- Pandas Implementation Results ---")
        print(f"Evaluation Predictions: {[f'{p:.4f}' for p in pd_preds]}")
    except Exception as e:
        print(f"Pandas implementation failed: {e}")
        pd_preds = None

    print("-" * 20)

    # 4. Compare results
    print("--- Comparison ---")
    if our_preds is not None and pd_preds is not None:
        preds_diff = np.abs(np.array(our_preds) - np.array(pd_preds)).sum()
        print(f"Sum of absolute differences in predictions: {preds_diff:.6f}")
        if np.allclose(our_preds, pd_preds):
            print("✅ Predictions are consistent.")
        else:
            print("❌ Predictions are NOT consistent.")

if __name__ == "__main__":
    validate_ma_implementation()
