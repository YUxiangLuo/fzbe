#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for ARIMA model to verify basic functionality
"""
import json
import os
import sys
import tempfile
from argparse import Namespace

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from models.arima import run, predict

def create_test_data():
    """Create simple test data with trend"""
    # Simple linear trend with some noise: y = 10 + 2*t + noise
    data = []
    for i in range(30):
        month = f"2024-{(i % 12) + 1:02d}"
        value = 10 + 2 * i + (i % 3) - 1  # Simple pattern
        data.append({"month": month, "y": value})
    return data

def test_training_mode():
    """Test ARIMA training mode"""
    print("=" * 60)
    print("Test 1: ARIMA Training Mode")
    print("=" * 60)

    test_data = create_test_data()
    data_json = json.dumps(test_data)

    # Create args for training
    args = Namespace(
        data=data_json,
        train_start=0,
        train_end=19,      # Train on first 20 points
        eval_start=20,
        eval_end=24,       # Evaluate on next 5 points
        d=1,               # First difference
        p_max=2,           # Max AR order
        q_max=2,           # Max MA order
        save_model=None,
        pretty=True
    )

    print("\nTest Data Summary:")
    print(f"  Total points: {len(test_data)}")
    print(f"  Train range: [{args.train_start}, {args.train_end}] = 20 points")
    print(f"  Eval range: [{args.eval_start}, {args.eval_end}] = 5 points")
    print(f"  Parameters: d={args.d}, p_max={args.p_max}, q_max={args.q_max}")
    print(f"\nFirst 5 data points: {test_data[:5]}")
    print(f"Last 5 data points: {test_data[-5:]}")

    print("\n--- Running ARIMA Training ---\n")
    try:
        run(args)
        print("\n✓ Training mode completed successfully")
    except Exception as e:
        print(f"\n✗ Training mode failed: {e}")
        import traceback
        traceback.print_exc()

def test_training_with_save_and_predict():
    """Test ARIMA training with model save, then prediction"""
    print("\n" + "=" * 60)
    print("Test 2: ARIMA Training + Save + Predict")
    print("=" * 60)

    test_data = create_test_data()
    data_json = json.dumps(test_data)

    # Create temporary file for model
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pkl', delete=False) as f:
        model_path = f.name

    try:
        # Step 1: Train and save model
        print("\n--- Step 1: Training and saving model ---")
        args_train = Namespace(
            data=data_json,
            train_start=0,
            train_end=19,
            eval_start=20,
            eval_end=24,
            d=1,
            p_max=2,
            q_max=2,
            save_model=model_path,
            pretty=True
        )

        print(f"Model will be saved to: {model_path}")
        run(args_train)
        print(f"\n✓ Model saved successfully")

        # Step 2: Load and predict
        print("\n--- Step 2: Loading model and predicting ---")
        args_predict = Namespace(
            load_model=model_path,
            forecast_steps=5,
            ci_level=0.95,
            pretty=True
        )

        print(f"Predicting {args_predict.forecast_steps} steps ahead")
        run(args_predict)
        print(f"\n✓ Prediction completed successfully")

        # Step 3: Test prediction with eval window
        print("\n--- Step 3: Prediction with eval window (in-sample) ---")
        args_predict_eval = Namespace(
            load_model=model_path,
            data=data_json,
            eval_start=10,
            eval_end=15,
            ci_level=0.95,
            pretty=True
        )

        print(f"Predicting for eval window [{args_predict_eval.eval_start}, {args_predict_eval.eval_end}]")
        run(args_predict_eval)
        print(f"\n✓ Eval window prediction completed successfully")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"\n✓ Cleaned up temporary model file: {model_path}")

def test_predict_function():
    """Test the predict() function directly"""
    print("\n" + "=" * 60)
    print("Test 3: Direct predict() function call")
    print("=" * 60)

    test_data = create_test_data()
    data_json = json.dumps(test_data)

    # Create and train model first
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pkl', delete=False) as f:
        model_path = f.name

    try:
        # Train and save
        print("\n--- Training model first ---")
        args_train = Namespace(
            data=data_json,
            train_start=0,
            train_end=19,
            eval_start=20,
            eval_end=24,
            d=1,
            p_max=1,
            q_max=1,
            save_model=model_path,
            pretty=False
        )
        run(args_train)

        # Test predict function
        print("\n--- Testing predict() function ---")
        steps = 3
        print(f"Calling predict(model_path='{model_path}', steps={steps})")

        result = predict(model_path, steps)

        print(f"\nResult type: {type(result)}")
        print(f"Number of predictions: {len(result)}")
        print("\nPredictions:")
        print(json.dumps(result, indent=2))

        # Verify format
        if result and len(result) > 0:
            first = result[0]
            has_prediction = 'prediction' in first
            has_std_dev = 'std_dev' in first
            print(f"\n✓ Format check:")
            print(f"  - Has 'prediction' field: {has_prediction}")
            print(f"  - Has 'std_dev' field: {has_std_dev}")
            if has_prediction and has_std_dev:
                print(f"  ✓ Format is correct")
            else:
                print(f"  ✗ Format is missing fields")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)

def test_output_format_consistency():
    """Test output format consistency between training and prediction modes"""
    print("\n" + "=" * 60)
    print("Test 4: Output Format Consistency Check")
    print("=" * 60)

    test_data = create_test_data()
    data_json = json.dumps(test_data)

    print("\n--- Checking predictions field format ---")
    print("This test verifies the issue found in review:")
    print("Training mode: predictions = [{'prediction': p}]")
    print("Prediction mode: predictions = [{'prediction': p, 'std_dev': s}]")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pkl', delete=False) as f:
        model_path = f.name

    try:
        # Capture training output
        print("\n1. Training mode output:")
        import io
        from contextlib import redirect_stdout

        args_train = Namespace(
            data=data_json,
            train_start=0,
            train_end=19,
            eval_start=20,
            eval_end=22,
            d=1,
            p_max=1,
            q_max=1,
            save_model=model_path,
            pretty=True
        )

        train_output = io.StringIO()
        with redirect_stdout(train_output):
            run(args_train)

        train_json = json.loads(train_output.getvalue())
        train_preds = train_json.get('predictions', [])

        if train_preds:
            print(f"   Sample: {train_preds[0]}")
            print(f"   Fields: {list(train_preds[0].keys())}")

        # Capture prediction output
        print("\n2. Prediction mode output:")
        args_pred = Namespace(
            load_model=model_path,
            forecast_steps=3,
            ci_level=0.95,
            pretty=True
        )

        pred_output = io.StringIO()
        with redirect_stdout(pred_output):
            run(args_pred)

        pred_json = json.loads(pred_output.getvalue())
        pred_preds = pred_json.get('predictions', [])

        if pred_preds:
            print(f"   Sample: {pred_preds[0]}")
            print(f"   Fields: {list(pred_preds[0].keys())}")

        # Compare
        print("\n--- Consistency Check ---")
        if train_preds and pred_preds:
            train_keys = set(train_preds[0].keys())
            pred_keys = set(pred_preds[0].keys())

            if train_keys == pred_keys:
                print("✓ Formats are consistent")
            else:
                print("✗ Formats are INCONSISTENT")
                print(f"  Training has: {train_keys}")
                print(f"  Prediction has: {pred_keys}")
                print(f"  Missing in training: {pred_keys - train_keys}")
                print(f"  Missing in prediction: {train_keys - pred_keys}")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ARIMA Model Simple Test Suite")
    print("=" * 60)

    test_training_mode()
    test_training_with_save_and_predict()
    test_predict_function()
    test_output_format_consistency()

    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60)
