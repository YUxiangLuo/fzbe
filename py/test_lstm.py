import io
import json
import os
import sys
import tempfile
from argparse import Namespace
from contextlib import redirect_stdout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import numpy as np
from models.lstm import run

def create_test_data(n=100):
    """Create simple sine-wave based test data for LSTM."""
    data = []
    for i in range(n):
        value = 100 + 20 * np.sin(i * 0.2) + 5 * np.cos(i * 0.5) + np.random.randn() * 2
        data.append({"month": f"2024-{(i % 12) + 1:02d}", "sales": value})
    return data

def test_index_based_split():
    """
    Test the primary functionality: training and evaluating the LSTM model
    using explicit train_start, train_end, eval_start, eval_end, and eval_end indices.
    """
    print("\n" + "=" * 60)
    print("Test: LSTM with Index-Based Data Splitting")
    print("=" * 60)

    # 1. Generate Data
    n_total = 60
    test_data = create_test_data(n_total)
    data_json = json.dumps(test_data)

    # 2. Define ranges
    ts, te = 0, 39  # 40 points for training
    es, ee = 40, 59 # 20 points for evaluation
    lookback = 5
    
    train_len = te - ts + 1
    eval_len = ee - es + 1

    print(f"Total data points: {n_total}")
    print(f"Training range: [{ts}, {te}] (Length: {train_len})")
    print(f"Evaluation range: [{es}, {ee}] (Length: {eval_len})")
    print(f"Lookback: {lookback}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 3. Set up arguments
        model_path = os.path.join(tmpdir, "lstm_test_model.pkl")
        
        args = Namespace(
            data=data_json,
            train_start=ts,
            train_end=te,
            eval_start=es,
            eval_end=ee,
            lookback=lookback,
            epochs=5,
            save_model=model_path,
            pretty=True,
            # Add other optional args with defaults
            lstm_normalization="zscore",
            lstm_target_feature="sales",
            lstm_features=[],
            batch_size=6,
            lr=0.01,
            load_model=None,
            forecast_steps=None
        )

        # 4. Run the function and capture output
        print("\nRunning LSTM script...")
        output_io = io.StringIO()
        with redirect_stdout(output_io):
            run(args)
        
        print("Script executed successfully.")
        output_str = output_io.getvalue()
        
        assert output_str, "Script produced no output."
        output = json.loads(output_str)

        # 5. Assertions
        assert output["mode"] == "lstm_trained"
        print("✓ Mode is correct ('lstm_trained')")

        assert output["train_range"]["start_index"] == ts
        assert output["train_range"]["end_index"] == te
        print(f"✓ Train range in output matches: [{ts}, {te}]")

        assert output["evaluate_range"]["start_index"] == es
        assert output["evaluate_range"]["end_index"] == ee
        print(f"✓ Eval range in output matches: [{es}, {ee}]")

        # The number of predictions should be eval_len - lookback
        expected_preds_len = eval_len - lookback
        assert len(output["eval_predictions"]) == expected_preds_len
        assert len(output["eval_y_true"]) == expected_preds_len
        print(f"✓ Correct number of predictions generated: {len(output['eval_predictions'])}")

        assert "rmse" in output["metrics"] and output["metrics"]["rmse"] > 0
        print(f"✓ Metrics calculated (RMSE: {output['metrics']['rmse']:.4f})")

        assert os.path.exists(model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # 6. Test loading the model and predicting
        print("\nTesting model loading and prediction...")
        
        load_args = Namespace(
            load_model=model_path,
            forecast_steps=10,
            data=None, # Ensure data is None for pure forecasting
            pretty=True
        )
        
        load_output_io = io.StringIO()
        with redirect_stdout(load_output_io):
            run(load_args)
            
        load_output_str = load_output_io.getvalue()
        assert load_output_str, "Load script produced no output."
        load_output = json.loads(load_output_str)
        
        assert load_output["mode"] == "lstm_loaded_predict"
        assert len(load_output["predictions"]) == 10
        print("✓ Model loaded and predicted 10 steps successfully.")

    print("\n" + "=" * 60)
    print("✓ Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_index_based_split()
