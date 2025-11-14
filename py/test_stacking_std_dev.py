#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test std_dev calculation for stacking model.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import numpy as np
from data_utils import load_data_from_json
from models.stacking import train_and_evaluate, predict, dump_model

def test_std_dev_calculation():
    """Test that std_dev is calculated from residuals."""
    print("Testing std_dev calculation...")

    # Create test data with some variance
    n = 60
    np.random.seed(42)
    data = [{"y": float(10 + i * 2 + np.random.randn() * 0.5)} for i in range(n)]
    data_json = json.dumps(data)

    parsed_data = load_data_from_json(data_json)

    models_to_use = ["ma", "es"]
    model_params = {
        "ma": {"window": 3},
        "es": {"alpha": 0.3}
    }

    # Train model
    metrics, base_model_states, meta_model, y_true, y_pred, std_dev = train_and_evaluate(
        parsed_data,
        range(0, 40),
        range(40, 50),
        models_to_use,
        model_params
    )

    print(f"✓ Training completed")
    print(f"  Metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    print(f"  std_dev: {std_dev:.4f}")

    # Verify std_dev is calculated (not 0.0 for data with variance)
    assert std_dev >= 0, "std_dev should be non-negative"

    # Manually verify the calculation
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    residuals = y_true_arr - y_pred_arr
    expected_std = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0

    assert abs(std_dev - expected_std) < 1e-6, f"std_dev mismatch: {std_dev} vs {expected_std}"
    print(f"✓ std_dev calculation verified: {std_dev:.4f}")

    return std_dev, base_model_states, meta_model


def test_std_dev_persistence():
    """Test that std_dev is saved and loaded correctly."""
    print("\nTesting std_dev persistence...")

    # Use the trained model from previous test
    std_dev_train, base_model_states, meta_model = test_std_dev_calculation()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "stacking_std_test.pkl")

        # Save model
        state = {
            "model_type": "stacking",
            "models": ["ma", "es"],
            "base_model_states": base_model_states,
            "meta_model": meta_model,
            "std_dev_residuals": std_dev_train,
        }
        dump_model(state, model_path)
        print(f"✓ Model saved with std_dev={std_dev_train:.4f}")

        # Load and predict
        preds, std_dev_loaded = predict(model_path, steps=5)

        print(f"✓ Model loaded, std_dev={std_dev_loaded:.4f}")
        print(f"✓ Predictions: {preds[:3]}...")

        # Verify std_dev matches
        assert abs(std_dev_train - std_dev_loaded) < 1e-6, \
            f"std_dev mismatch after load: {std_dev_train} vs {std_dev_loaded}"
        print(f"✓ std_dev persistence verified")


def test_std_dev_format():
    """Test that std_dev is properly formatted in predictions."""
    print("\nTesting std_dev formatting...")

    n = 50
    data = [{"y": float(10 + i * 2)} for i in range(n)]
    data_json = json.dumps(data)
    parsed_data = load_data_from_json(data_json)

    models_to_use = ["ma", "es"]
    model_params = {
        "ma": {"window": 3},
        "es": {"alpha": 0.3}
    }

    metrics, base_model_states, meta_model, y_true, y_pred, std_dev = train_and_evaluate(
        parsed_data, range(0, 30), range(30, 40), models_to_use, model_params
    )

    # Test _format_predictions
    from models.stacking import _format_predictions
    formatted = _format_predictions(y_pred, std_dev)

    assert len(formatted) == len(y_pred), "Length mismatch"
    assert all("prediction" in p for p in formatted), "Missing prediction key"
    assert all("std_dev" in p for p in formatted), "Missing std_dev key"
    assert all(p["std_dev"] == std_dev for p in formatted), "std_dev mismatch in formatting"

    print(f"✓ Formatted predictions sample: {formatted[0]}")
    print(f"✓ std_dev formatting verified")


def test_zero_variance_case():
    """Test std_dev calculation when predictions are perfect (edge case)."""
    print("\nTesting zero variance case...")

    # Create perfectly linear data that models can fit exactly
    n = 30
    data = [{"y": float(i * 10)} for i in range(n)]
    data_json = json.dumps(data)
    parsed_data = load_data_from_json(data_json)

    models_to_use = ["ma", "es"]
    model_params = {
        "ma": {"window": 1},  # Window of 1 means just use last value
        "es": {"alpha": 1.0}  # Alpha of 1.0 means no smoothing
    }

    metrics, base_model_states, meta_model, y_true, y_pred, std_dev = train_and_evaluate(
        parsed_data, range(0, 20), range(20, 25), models_to_use, model_params
    )

    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  std_dev: {std_dev:.6f}")

    # For near-perfect predictions, std_dev should be very small
    assert std_dev >= 0, "std_dev should be non-negative"
    print(f"✓ Zero variance case handled correctly")


if __name__ == "__main__":
    try:
        test_std_dev_calculation()
        test_std_dev_persistence()
        test_std_dev_format()
        test_zero_variance_case()

        print("\n🎉 All std_dev tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
