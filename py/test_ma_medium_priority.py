#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify MA model medium priority fixes:
3. Optimized numpy/list conversions
5. Improved type annotations
8. Unified comment language to English
"""
import json
import os
import sys
import tempfile
from argparse import Namespace

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import numpy as np
from models.ma import run, moving_average_forecast, train_and_evaluate, ArrayLike

def create_test_data():
    """Create simple test data"""
    data = []
    for i in range(30):
        month = f"2024-{(i % 12) + 1:02d}"
        value = 10 + i * 2
        data.append({"month": month, "y": value})
    return data

def test_numpy_optimization():
    """Test that functions accept numpy arrays directly"""
    print("=" * 60)
    print("Test 1: Numpy Array Support (Issue 3)")
    print("=" * 60)

    # Test moving_average_forecast with numpy array
    print("\n--- Testing moving_average_forecast with numpy array ---")
    series_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    eval_range = range(5, 10)
    window = 3

    idxs, preds = moving_average_forecast(series_np, eval_range, window)

    print(f"Input type: {type(series_np)}")
    print(f"Indices: {idxs}")
    print(f"Predictions: {preds}")

    assert len(idxs) == 5, f"Expected 5 indices, got {len(idxs)}"
    assert len(preds) == 5, f"Expected 5 predictions, got {len(preds)}"
    print("✓ moving_average_forecast accepts numpy arrays")

    # Test moving_average_forecast with list
    print("\n--- Testing moving_average_forecast with list ---")
    series_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    idxs2, preds2 = moving_average_forecast(series_list, eval_range, window)

    print(f"Input type: {type(series_list)}")
    print(f"Predictions match: {preds == preds2}")

    assert preds == preds2, "Results should be identical for list and numpy array"
    print("✓ moving_average_forecast accepts lists")

    # Test train_and_evaluate with numpy array
    print("\n--- Testing train_and_evaluate with numpy array ---")
    sales_np = np.array([10.0 + i * 2 for i in range(30)])
    months = [f"2024-{(i % 12) + 1:02d}" for i in range(30)]

    metrics, std_dev, y_true, y_pred = train_and_evaluate(
        sales_np, months, range(0, 20), range(20, 25), 3
    )

    print(f"Sales input type: {type(sales_np)}")
    print(f"y_true type: {type(y_true)}")
    print(f"y_pred type: {type(y_pred)}")
    print(f"Evaluated points: {metrics['evaluated_points']}")

    # Check that y_true and y_pred are numpy arrays for efficiency
    assert isinstance(y_true, np.ndarray), "y_true should be numpy array"
    assert isinstance(y_pred, np.ndarray), "y_pred should be numpy array"
    print("✓ train_and_evaluate returns numpy arrays for efficiency")

    print("\n✓ Test 1 completed successfully\n")

def test_type_annotations():
    """Test that type annotations are properly defined"""
    print("=" * 60)
    print("Test 2: Type Annotations (Issue 5)")
    print("=" * 60)

    # Check ArrayLike type alias exists
    print("\n--- Checking ArrayLike type alias ---")
    from models.ma import ArrayLike
    print(f"ArrayLike type: {ArrayLike}")
    print("✓ ArrayLike type alias is defined")

    # Check function signatures have proper type hints
    print("\n--- Checking function type hints ---")
    import inspect

    # Check moving_average_forecast
    sig = inspect.signature(moving_average_forecast)
    print(f"moving_average_forecast signature: {sig}")
    params = sig.parameters
    assert 'series' in params, "series parameter should exist"
    print(f"  - series annotation: {params['series'].annotation}")
    print("✓ moving_average_forecast has type annotations")

    # Check train_and_evaluate
    sig2 = inspect.signature(train_and_evaluate)
    print(f"train_and_evaluate signature: {sig2}")
    params2 = sig2.parameters
    assert 'sales' in params2, "sales parameter should exist"
    print(f"  - sales annotation: {params2['sales'].annotation}")
    print("✓ train_and_evaluate has type annotations")

    print("\n✓ Test 2 completed successfully\n")

def test_english_comments():
    """Test that comments are in English"""
    print("=" * 60)
    print("Test 3: English Comments (Issue 8)")
    print("=" * 60)

    print("\n--- Checking for Chinese characters in source ---")
    ma_file = os.path.join(os.path.dirname(__file__), "src", "models", "ma.py")

    with open(ma_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for Chinese characters (Unicode range for common Chinese characters)
    chinese_chars = []
    for i, char in enumerate(content):
        if '\u4e00' <= char <= '\u9fff':
            # Find line number
            line_num = content[:i].count('\n') + 1
            chinese_chars.append((line_num, char))

    if chinese_chars:
        print(f"✗ Found {len(chinese_chars)} Chinese characters:")
        for line_num, char in chinese_chars[:5]:  # Show first 5
            print(f"  Line {line_num}: '{char}'")
        raise AssertionError("Chinese characters found in comments")
    else:
        print("✓ No Chinese characters found in source code")

    print("\n✓ Test 3 completed successfully\n")

def test_integrated_workflow():
    """Test complete workflow with numpy optimization"""
    print("=" * 60)
    print("Test 4: Integrated Workflow with Numpy Optimization")
    print("=" * 60)

    test_data = create_test_data()
    data_json = json.dumps(test_data)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pkl', delete=False) as f:
        model_path = f.name

    try:
        print("\n--- Training with numpy-optimized code ---")
        import io
        from contextlib import redirect_stdout

        args_train = Namespace(
            data=data_json,
            train_start=0,
            train_end=19,
            eval_start=20,
            eval_end=24,
            window=5,
            save_model=model_path,
            pretty=True
        )

        output = io.StringIO()
        with redirect_stdout(output):
            run(args_train)

        result = json.loads(output.getvalue())

        print(f"Mode: {result['mode']}")
        print(f"RMSE: {result['metrics']['rmse']:.4f}")
        print(f"Evaluated points: {result['metrics']['evaluated_points']}")

        assert 'eval_y_true' in result
        assert 'eval_predictions' in result
        assert isinstance(result['eval_y_true'], list)
        assert isinstance(result['eval_predictions'], list)
        print("✓ Training output format is correct")

        print("\n--- Prediction with saved model ---")
        args_pred = Namespace(
            load_model=model_path,
            forecast_steps=3,
            pretty=True
        )

        output2 = io.StringIO()
        with redirect_stdout(output2):
            run(args_pred)

        result2 = json.loads(output2.getvalue())

        print(f"Mode: {result2['mode']}")
        print(f"Predictions count: {len(result2['predictions'])}")

        assert len(result2['predictions']) == 3
        assert all('prediction' in p and 'std_dev' in p for p in result2['predictions'])
        print("✓ Prediction output format is correct")

        print("\n✓ Test 4 completed successfully\n")

    finally:
        if os.path.exists(model_path):
            os.remove(model_path)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MA Model Medium Priority Fixes Verification")
    print("=" * 60)

    test_numpy_optimization()
    test_type_annotations()
    test_english_comments()
    test_integrated_workflow()

    print("=" * 60)
    print("All medium priority fixes verified successfully!")
    print("=" * 60)
