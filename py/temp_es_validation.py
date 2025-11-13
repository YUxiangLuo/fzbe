# py/temp_es_validation.py
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
import sys
import os

# Add src to path to be able to import es from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from models import es

def validate_es_implementation():
    """
    Cross-validates our custom ES implementation against the statsmodels library.
    """
    # 1. Define sample data and parameters
    series = np.array([10, 12, 15, 11, 14, 16, 18, 20, 22, 25], dtype=np.float64)
    months = [f"M{i+1}" for i in range(len(series))]
    alpha = 0.5
    train_range = range(0, 8)  # Train on first 8 points (indices 0-7)
    eval_range = range(8, 10) # Evaluate on last 2 points (indices 8-9)

    print("--- Input Data ---")
    print(f"Series: {series.tolist()}")
    print(f"Alpha: {alpha}")
    print(f"Training Indices: {list(train_range)}")
    print(f"Evaluation Indices: {list(eval_range)}")
    print("-" * 20)

    # 2. Run our custom implementation
    try:
        our_metrics, our_std_dev, our_final_level, our_y_true, our_preds = es.train_and_evaluate(
            series.tolist(), months, train_range, eval_range, alpha
        )
        print("--- Our Implementation Results ---")
        print(f"Final Level (at train_end): {our_final_level:.4f}")
        print(f"Evaluation Predictions: {[f'{p:.4f}' for p in our_preds]}")
        print(f"Evaluation True Values: {our_y_true}")
    except Exception as e:
        print(f"Our implementation failed: {e}")
        our_final_level, our_preds = None, None

    print("-" * 20)

    # 3. Run statsmodels implementation
    try:
        train_data = series[train_range.start:train_range.stop]
        
        # Fit the model
        # `initialization_method="known"` and `initial_level=train_data[0]`
        # mimics our implementation's s_0 = y_0.
        fit = SimpleExpSmoothing(
            train_data, initialization_method="known", initial_level=train_data[0]
        ).fit(smoothing_level=alpha, optimized=False)

        # Get the level at the end of the training period
        sm_final_level = fit.level[-1]
        
        # Forecast for the evaluation period
        sm_preds = fit.forecast(len(eval_range))

        print("--- Statsmodels Implementation Results ---")
        print(f"Final Level (at train_end): {sm_final_level:.4f}")
        print(f"Evaluation Predictions: {[f'{p:.4f}' for p in sm_preds]}")
    except Exception as e:
        print(f"Statsmodels implementation failed: {e}")
        sm_final_level, sm_preds = None, None

    print("-" * 20)

    # 4. Compare results
    print("--- Comparison ---")
    if our_final_level is not None and sm_final_level is not None:
        level_diff = abs(our_final_level - sm_final_level)
        print(f"Absolute difference in final level: {level_diff:.6f}")
        if np.isclose(our_final_level, sm_final_level):
            print("✅ Final levels are consistent.")
        else:
            print("❌ Final levels are NOT consistent.")

    if our_preds is not None and sm_preds is not None:
        preds_diff = np.abs(np.array(our_preds) - np.array(sm_preds)).sum()
        print(f"Sum of absolute differences in predictions: {preds_diff:.6f}")
        if np.allclose(our_preds, sm_preds):
            print("✅ Predictions are consistent.")
        else:
            print("❌ Predictions are NOT consistent.")

if __name__ == "__main__":
    validate_es_implementation()
