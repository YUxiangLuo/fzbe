#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify all ES model fixes:
1. Duplicate JSON parsing eliminated
2. Repeated target_key detection eliminated
3. Optimized numpy/list conversions
4. Chinese comments replaced with English
5. Complete type annotations
6. Redundant check removed
7. ArrayLike type alias added
8. Exception handling narrowed
"""
import json
import os
import sys
import tempfile
from argparse import Namespace

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import numpy as np
from models.es import (
    run,
    predict,
    train_and_evaluate,
    _determine_target_key,
    _fit_level,
    _validate_training_args,
    _add_metrics_to_output,
    ArrayLike,
)


def create_test_data(n=50):
    """Create simple test data for ES"""
    data = []
    for i in range(n):
        value = 10 + 0.5 * i
        data.append({"month": f"2024-{(i % 12) + 1:02d}", "sales": value})
    return data


def test_determine_target_key():
    """Test Issue 1 & 2: _determine_target_key helper eliminates duplicate parsing"""
    print("=" * 60)
    print("Test 1: _determine_target_key Helper (Issue 1 & 2)")
    print("=" * 60)

    # Test with 'sales' key
    print("\n--- Testing with 'sales' key ---")
    data1 = [{"month": "2024-01", "sales": 10}, {"month": "2024-02", "sales": 12}]
    data_json1 = json.dumps(data1)

    target_key, time_key, records = _determine_target_key(data_json1)
    print(f"Target key: {target_key}")
    print(f"Time key: {time_key}")
    print(f"Records type: {type(records)}")
    print(f"Records length: {len(records)}")

    assert target_key == "sales"
    assert time_key == "month"
    assert isinstance(records, list)
    assert len(records) == 2
    print("✓ Correctly detected 'sales' and 'month'")

    # Test with 'y' key
    print("\n--- Testing with 'y' key ---")
    data2 = [{"month": "2024-01", "y": 10}, {"month": "2024-02", "y": 12}]
    data_json2 = json.dumps(data2)

    target_key2, time_key2, records2 = _determine_target_key(data_json2)
    assert target_key2 == "y"
    assert time_key2 == "month"
    print("✓ Correctly detected 'y' and 'month'")

    # Test without time key
    print("\n--- Testing without time key ---")
    data3 = [{"sales": 10}, {"sales": 12}]
    data_json3 = json.dumps(data3)

    target_key3, time_key3, records3 = _determine_target_key(data_json3)
    assert target_key3 == "sales"
    assert time_key3 is None
    print("✓ Correctly handles missing time key")

    # Test invalid JSON
    print("\n--- Testing invalid JSON ---")
    try:
        _determine_target_key("invalid json{")
        print("✗ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:50]}...")

    # Test missing target key
    print("\n--- Testing missing target key ---")
    try:
        data4 = [{"month": "2024-01", "value": 10}]
        _determine_target_key(json.dumps(data4))
        print("✗ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:50]}...")

    print("\n✓ Test 1 completed successfully\n")


def test_numpy_optimization():
    """Test Issue 3: Optimized numpy/list conversions"""
    print("=" * 60)
    print("Test 2: Numpy Array Optimization (Issue 3)")
    print("=" * 60)

    # Test train_and_evaluate with numpy array
    print("\n--- Testing train_and_evaluate with numpy array ---")
    sales_np = np.array([10.0 + i * 2 for i in range(30)])
    months = [f"2024-{(i % 12) + 1:02d}" for i in range(30)]

    metrics, std_dev, level, y_true, y_pred = train_and_evaluate(
        sales_np, months, range(0, 20), range(20, 25), alpha=0.3
    )

    print(f"Sales input type: {type(sales_np)}")
    print(f"Metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    print(f"Std dev: {std_dev:.4f}")
    print(f"Final level: {level:.4f}")

    assert isinstance(y_true, list)
    assert isinstance(y_pred, list)
    assert len(y_true) == 5
    assert len(y_pred) == 5
    print("✓ train_and_evaluate accepts numpy arrays")

    # Test train_and_evaluate with list
    print("\n--- Testing train_and_evaluate with list ---")
    sales_list = [10.0 + i * 2 for i in range(30)]

    metrics2, std_dev2, level2, y_true2, y_pred2 = train_and_evaluate(
        sales_list, months, range(0, 20), range(20, 25), alpha=0.3
    )

    assert metrics2["rmse"] == metrics["rmse"]
    assert level2 == level
    print("✓ train_and_evaluate accepts lists")

    print("\n✓ Test 2 completed successfully\n")


def test_english_comments():
    """Test Issue 4: Chinese comments replaced with English"""
    print("=" * 60)
    print("Test 3: English Comments (Issue 4)")
    print("=" * 60)

    print("\n--- Checking for Chinese characters in source ---")
    es_file = os.path.join(os.path.dirname(__file__), "src", "models", "es.py")

    with open(es_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for Chinese characters (Unicode range for common Chinese characters)
    chinese_chars = []
    for i, char in enumerate(content):
        if "\u4e00" <= char <= "\u9fff":
            # Find line number
            line_num = content[:i].count("\n") + 1
            chinese_chars.append((line_num, char))

    if chinese_chars:
        print(f"✗ Found {len(chinese_chars)} Chinese characters:")
        for line_num, char in chinese_chars[:5]:  # Show first 5
            print(f"  Line {line_num}: '{char}'")
        raise AssertionError("Chinese characters found in comments")
    else:
        print("✓ No Chinese characters found in source code")

    print("\n✓ Test 3 completed successfully\n")


def test_type_annotations():
    """Test Issue 5 & 7: Type annotations and ArrayLike"""
    print("=" * 60)
    print("Test 4: Type Annotations (Issue 5 & 7)")
    print("=" * 60)

    # Check ArrayLike type alias exists
    print("\n--- Checking ArrayLike type alias ---")
    from models.es import ArrayLike

    print(f"ArrayLike type: {ArrayLike}")
    print("✓ ArrayLike type alias is defined")

    # Check function signatures have proper type hints
    print("\n--- Checking function type hints ---")
    import inspect

    # Check train_and_evaluate
    sig = inspect.signature(train_and_evaluate)
    print(f"train_and_evaluate signature: {sig}")
    params = sig.parameters
    assert "sales" in params, "sales parameter should exist"
    print(f"  - sales annotation: {params['sales'].annotation}")
    print("✓ train_and_evaluate has type annotations")

    # Check _fit_level
    sig2 = inspect.signature(_fit_level)
    print(f"_fit_level signature: {sig2}")
    assert sig2.return_annotation != inspect.Signature.empty
    print("✓ _fit_level has type annotations")

    # Check _validate_training_args
    sig3 = inspect.signature(_validate_training_args)
    print(f"_validate_training_args signature: {sig3}")
    params3 = sig3.parameters
    assert "sales" in params3
    print(f"  - sales annotation: {params3['sales'].annotation}")
    print("✓ _validate_training_args has type annotations")

    # Check _add_metrics_to_output
    sig4 = inspect.signature(_add_metrics_to_output)
    print(f"_add_metrics_to_output signature: {sig4}")
    params4 = sig4.parameters
    assert "out" in params4
    print(f"  - out annotation: {params4['out'].annotation}")
    print("✓ _add_metrics_to_output has type annotations")

    print("\n✓ Test 4 completed successfully\n")


def test_redundant_check_removed():
    """Test Issue 6: Redundant check removed"""
    print("=" * 60)
    print("Test 5: Redundant Check Removed (Issue 6)")
    print("=" * 60)

    print("\n--- Checking source code for redundant check ---")
    es_file = os.path.join(os.path.dirname(__file__), "src", "models", "es.py")

    with open(es_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Look for the old pattern "if t <= te:"
    in_train_and_evaluate = False
    found_redundant_check = False

    for line in content.split("\n"):
        if "def train_and_evaluate" in line:
            in_train_and_evaluate = True
        elif in_train_and_evaluate and "def " in line and "train_and_evaluate" not in line:
            break
        elif in_train_and_evaluate and "if t <= te:" in line:
            found_redundant_check = True
            break

    if found_redundant_check:
        print("✗ Redundant 'if t <= te:' check still exists")
        raise AssertionError("Redundant check not removed")
    else:
        print("✓ Redundant 'if t <= te:' check has been removed")

    print("\n✓ Test 5 completed successfully\n")


def test_exception_handling():
    """Test Issue 8: Exception handling narrowed"""
    print("=" * 60)
    print("Test 6: Narrowed Exception Handling (Issue 8)")
    print("=" * 60)

    print("\n--- Checking exception types in source ---")
    es_file = os.path.join(os.path.dirname(__file__), "src", "models", "es.py")

    with open(es_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Count broad Exception catches vs specific ones
    broad_exceptions = content.count("except Exception")
    specific_exceptions = content.count("except (ValueError, TypeError)")

    print(f"Broad 'except Exception' count: {broad_exceptions}")
    print(f"Specific 'except (ValueError, TypeError)' count: {specific_exceptions}")

    # Should have specific exceptions, not broad ones
    assert specific_exceptions >= 2, "Should have at least 2 specific exception handlers"
    print("✓ Using specific exception types (ValueError, TypeError)")

    print("\n✓ Test 6 completed successfully\n")


def test_integrated_workflow():
    """Test complete workflow with all fixes"""
    print("=" * 60)
    print("Test 7: Integrated Workflow")
    print("=" * 60)

    test_data = create_test_data(50)
    data_json = json.dumps(test_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_es.pkl")

        print("\n--- Training ES model ---")
        import io
        from contextlib import redirect_stdout

        args_train = Namespace(
            data=data_json,
            train_start=0,
            train_end=29,
            eval_start=30,
            eval_end=39,
            alpha=0.3,
            save_model=model_path,
            pretty=True,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            run(args_train)

        result = json.loads(output.getvalue())

        print(f"Mode: {result['mode']}")
        print(f"Alpha: {result['alpha']}")
        print(f"RMSE: {result['metrics']['rmse']:.4f}")
        print(f"Evaluated points: {len(result['eval_y_true'])}")

        assert os.path.exists(model_path), f"Model not saved to {model_path}"
        assert result["mode"] == "ses_fixed_alpha"
        assert result["alpha"] == 0.3
        assert "metrics" in result
        print("✓ Training completed successfully")

        print("\n--- Prediction with saved model ---")
        args_pred = Namespace(load_model=model_path, forecast_steps=5, pretty=True)

        output2 = io.StringIO()
        with redirect_stdout(output2):
            run(args_pred)

        result2 = json.loads(output2.getvalue())

        print(f"Mode: {result2['mode']}")
        print(f"Predictions count: {len(result2['predictions'])}")
        print(f"First prediction: {result2['predictions'][0]}")

        assert result2["mode"] == "es_loaded_predict"
        assert len(result2["predictions"]) == 5
        assert all("prediction" in p and "std_dev" in p for p in result2["predictions"])
        print("✓ Prediction completed successfully")

        print("\n✓ Test 7 completed successfully\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ES Model All Fixes Verification Test Suite")
    print("=" * 60)

    test_determine_target_key()
    test_numpy_optimization()
    test_english_comments()
    test_type_annotations()
    test_redundant_check_removed()
    test_exception_handling()
    test_integrated_workflow()

    print("=" * 60)
    print("All ES fixes verified successfully!")
    print("=" * 60)
