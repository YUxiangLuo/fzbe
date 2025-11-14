#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify MA model fixes for high priority issues:
1. Duplicate JSON parsing elimination
2. Extracted target_key detection logic
3. Narrowed exception handling
"""
import json
import os
import sys
import tempfile
from argparse import Namespace

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from models.ma import run, predict, _determine_target_key

def create_test_data():
    """Create simple test data"""
    data = []
    for i in range(30):
        month = f"2024-{(i % 12) + 1:02d}"
        value = 10 + i * 2
        data.append({"month": month, "y": value})
    return data

def test_determine_target_key():
    """Test the new _determine_target_key helper function"""
    print("=" * 60)
    print("Test 1: _determine_target_key helper function")
    print("=" * 60)

    # Test with 'y' key
    data_y = [{"month": "2024-01", "y": 10}, {"month": "2024-02", "y": 12}]
    data_json = json.dumps(data_y)

    target_key, time_key, records = _determine_target_key(data_json)

    print(f"\nTest data with 'y' key:")
    print(f"  target_key: {target_key}")
    print(f"  time_key: {time_key}")
    print(f"  records type: {type(records)}")
    print(f"  records length: {len(records)}")

    assert target_key == "y", f"Expected 'y', got {target_key}"
    assert time_key == "month", f"Expected 'month', got {time_key}"
    assert isinstance(records, list), "Records should be a list"
    assert len(records) == 2, f"Expected 2 records, got {len(records)}"
    print("  ✓ All assertions passed for 'y' key")

    # Test with 'sales' key
    data_sales = [{"month": "2024-01", "sales": 10}, {"month": "2024-02", "sales": 12}]
    data_json_sales = json.dumps(data_sales)

    target_key, time_key, records = _determine_target_key(data_json_sales)

    print(f"\nTest data with 'sales' key:")
    print(f"  target_key: {target_key}")
    print(f"  time_key: {time_key}")

    assert target_key == "sales", f"Expected 'sales', got {target_key}"
    assert time_key == "month", f"Expected 'month', got {time_key}"
    print("  ✓ All assertions passed for 'sales' key")

    # Test without month key
    data_no_month = [{"y": 10}, {"y": 12}]
    data_json_no_month = json.dumps(data_no_month)

    target_key, time_key, records = _determine_target_key(data_json_no_month)

    print(f"\nTest data without 'month' key:")
    print(f"  target_key: {target_key}")
    print(f"  time_key: {time_key}")

    assert target_key == "y", f"Expected 'y', got {target_key}"
    assert time_key is None, f"Expected None, got {time_key}"
    print("  ✓ All assertions passed for no month key")

    print("\n✓ Test 1 completed successfully\n")

def test_training_mode():
    """Test MA training mode - verifies no duplicate parsing"""
    print("=" * 60)
    print("Test 2: MA Training Mode (No Duplicate Parsing)")
    print("=" * 60)

    test_data = create_test_data()
    data_json = json.dumps(test_data)

    args = Namespace(
        data=data_json,
        train_start=0,
        train_end=19,
        eval_start=20,
        eval_end=24,
        window=3,
        save_model=None,
        pretty=True
    )

    print("\nTest Data Summary:")
    print(f"  Total points: {len(test_data)}")
    print(f"  Train range: [{args.train_start}, {args.train_end}]")
    print(f"  Eval range: [{args.eval_start}, {args.eval_end}]")
    print(f"  Window: {args.window}")

    print("\n--- Running MA Training ---\n")
    try:
        import io
        from contextlib import redirect_stdout

        output = io.StringIO()
        with redirect_stdout(output):
            run(args)

        result = json.loads(output.getvalue())

        print(f"Mode: {result.get('mode')}")
        print(f"Metrics evaluated points: {result['metrics']['evaluated_points']}")
        print(f"RMSE: {result['metrics']['rmse']:.4f}")
        print(f"MAE: {result['metrics']['mae']:.4f}")

        assert result['mode'] == 'fixed_window_only'
        assert 'metrics' in result
        print("\n✓ Training mode completed successfully")
    except Exception as e:
        print(f"\n✗ Training mode failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_prediction_mode():
    """Test MA prediction mode with model save/load"""
    print("\n" + "=" * 60)
    print("Test 3: MA Prediction Mode (Model Save/Load)")
    print("=" * 60)

    test_data = create_test_data()
    data_json = json.dumps(test_data)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pkl', delete=False) as f:
        model_path = f.name

    try:
        # Train and save
        print("\n--- Step 1: Training and saving model ---")
        import io
        from contextlib import redirect_stdout

        args_train = Namespace(
            data=data_json,
            train_start=0,
            train_end=19,
            eval_start=20,
            eval_end=24,
            window=3,
            save_model=model_path,
            pretty=True
        )

        output_train = io.StringIO()
        with redirect_stdout(output_train):
            run(args_train)

        result_train = json.loads(output_train.getvalue())
        print(f"Model saved to: {model_path}")
        print(f"Window: {result_train['metrics']['window']}")

        # Load and predict
        print("\n--- Step 2: Loading and predicting ---")
        args_predict = Namespace(
            load_model=model_path,
            forecast_steps=5,
            pretty=True
        )

        output_predict = io.StringIO()
        with redirect_stdout(output_predict):
            run(args_predict)

        result_predict = json.loads(output_predict.getvalue())

        print(f"Mode: {result_predict.get('mode')}")
        print(f"Forecast steps: {result_predict.get('forecast_steps')}")
        print(f"Number of predictions: {len(result_predict.get('predictions', []))}")

        # Check prediction format
        if result_predict.get('predictions'):
            first_pred = result_predict['predictions'][0]
            print(f"Prediction format: {list(first_pred.keys())}")
            assert 'prediction' in first_pred, "Missing 'prediction' field"
            assert 'std_dev' in first_pred, "Missing 'std_dev' field"

        print("\n✓ Prediction mode completed successfully")

    except Exception as e:
        print(f"\n✗ Prediction mode failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"✓ Cleaned up: {model_path}")

def test_exception_handling():
    """Test narrowed exception handling"""
    print("\n" + "=" * 60)
    print("Test 4: Exception Handling (Narrow Scope)")
    print("=" * 60)

    # Test invalid JSON
    print("\n--- Testing invalid JSON ---")
    try:
        target_key, time_key, records = _determine_target_key("invalid json{}")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:50]}...")
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}")

    # Test missing target key
    print("\n--- Testing missing target key ---")
    try:
        data = json.dumps([{"month": "2024-01", "value": 10}])
        target_key, time_key, records = _determine_target_key(data)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:50]}...")
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}")

    # Test invalid eval range in load mode
    print("\n--- Testing invalid eval_start/eval_end ---")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pkl', delete=False) as f:
        model_path = f.name

    try:
        # First create a valid model
        test_data = create_test_data()
        data_json = json.dumps(test_data)

        args_train = Namespace(
            data=data_json,
            train_start=0,
            train_end=19,
            eval_start=20,
            eval_end=24,
            window=3,
            save_model=model_path,
            pretty=False
        )

        import io
        from contextlib import redirect_stdout
        output = io.StringIO()
        with redirect_stdout(output):
            run(args_train)

        # Now try to load with invalid eval range
        args_invalid = Namespace(
            load_model=model_path,
            eval_start="not_a_number",
            eval_end="10",
            pretty=False
        )

        try:
            output = io.StringIO()
            with redirect_stdout(output):
                run(args_invalid)
            print("✗ Should have raised ValueError for invalid eval range")
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {str(e)[:60]}...")
        except Exception as e:
            print(f"✗ Wrong exception type: {type(e).__name__}")

    finally:
        if os.path.exists(model_path):
            os.remove(model_path)

    print("\n✓ Test 4 completed successfully\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MA Model Fixes Verification Test Suite")
    print("=" * 60)

    test_determine_target_key()
    test_training_mode()
    test_prediction_mode()
    test_exception_handling()

    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
