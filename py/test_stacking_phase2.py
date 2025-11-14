#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Phase 2 fixes for stacking model:
- Parameter validation
- Exception handling
- _format_predictions helper
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data_utils import load_data_from_json
from models.stacking import (
    _validate_training_args,
    _format_predictions,
    train_and_evaluate,
    predict,
    dump_model
)

def test_format_predictions():
    """Test _format_predictions helper function."""
    print("Testing _format_predictions...")

    preds = [10.0, 20.0, 30.0]
    result = _format_predictions(preds)

    assert len(result) == 3
    assert result[0] == {"prediction": 10.0, "std_dev": 0.0}
    assert result[1] == {"prediction": 20.0, "std_dev": 0.0}

    # With custom std_dev
    result2 = _format_predictions(preds, 2.5)
    assert result2[0] == {"prediction": 10.0, "std_dev": 2.5}

    print("✓ _format_predictions working correctly")


def test_validate_empty_models():
    """Test validation catches empty models list."""
    print("\nTesting empty models validation...")

    n = 50
    data = [{"y": float(10 + i * 2)} for i in range(n)]
    data_json = json.dumps(data)
    parsed_data = load_data_from_json(data_json)

    try:
        _validate_training_args(
            parsed_data, 0, 30, 31, 40,
            [],  # Empty models list
            {}
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "models_to_use cannot be empty" in str(e)
        print(f"✓ Caught empty models: {e}")


def test_validate_single_model():
    """Test validation requires at least 2 models for stacking."""
    print("\nTesting single model validation...")

    n = 50
    data = [{"y": float(10 + i * 2)} for i in range(n)]
    data_json = json.dumps(data)
    parsed_data = load_data_from_json(data_json)

    try:
        _validate_training_args(
            parsed_data, 0, 30, 31, 40,
            ["ma"],  # Only one model
            {"ma": {"window": 3}}
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "at least 2 base models" in str(e)
        print(f"✓ Caught single model: {e}")


def test_validate_unknown_model():
    """Test validation catches unknown model names."""
    print("\nTesting unknown model validation...")

    n = 50
    data = [{"y": float(10 + i * 2)} for i in range(n)]
    data_json = json.dumps(data)
    parsed_data = load_data_from_json(data_json)

    try:
        _validate_training_args(
            parsed_data, 0, 30, 31, 40,
            ["ma", "unknown_model"],
            {"ma": {"window": 3}, "unknown_model": {}}
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown model" in str(e)
        print(f"✓ Caught unknown model: {e}")


def test_validate_invalid_ranges():
    """Test validation of train/eval ranges."""
    print("\nTesting range validation...")

    n = 50
    data = [{"y": float(10 + i * 2)} for i in range(n)]
    data_json = json.dumps(data)
    parsed_data = load_data_from_json(data_json)

    models_to_use = ["ma", "es"]
    model_params = {
        "ma": {"window": 3},
        "es": {"alpha": 0.3}
    }

    # Test invalid train range
    try:
        _validate_training_args(parsed_data, 40, 30, 31, 40, models_to_use, model_params)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid train range" in str(e)
        print(f"✓ Caught invalid train range: {e}")

    # Test invalid eval range
    try:
        _validate_training_args(parsed_data, 0, 30, 40, 35, models_to_use, model_params)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid eval range" in str(e)
        print(f"✓ Caught invalid eval range: {e}")

    # Test eval before train
    try:
        _validate_training_args(parsed_data, 0, 30, 30, 35, models_to_use, model_params)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "eval_start must be > train_end" in str(e)
        print(f"✓ Caught eval before train: {e}")


def test_validate_model_params():
    """Test validation of model-specific parameters."""
    print("\nTesting model parameter validation...")

    n = 50
    data = [{"y": float(10 + i * 2)} for i in range(n)]
    data_json = json.dumps(data)
    parsed_data = load_data_from_json(data_json)

    # Test missing ARIMA d
    try:
        _validate_training_args(
            parsed_data, 0, 30, 31, 40,
            ["ma", "arima"],
            {"ma": {"window": 3}, "arima": {}}  # Missing 'd'
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "ARIMA requires 'd'" in str(e)
        print(f"✓ Caught missing ARIMA d: {e}")

    # Test invalid ES alpha
    try:
        _validate_training_args(
            parsed_data, 0, 30, 31, 40,
            ["ma", "es"],
            {"ma": {"window": 3}, "es": {"alpha": 1.5}}  # alpha > 1
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "alpha must be in (0, 1]" in str(e)
        print(f"✓ Caught invalid ES alpha: {e}")

    # Test invalid MA window
    try:
        _validate_training_args(
            parsed_data, 0, 30, 31, 40,
            ["ma", "es"],
            {"ma": {"window": 0}, "es": {"alpha": 0.3}}  # window < 1
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "MA window must be >= 1" in str(e)
        print(f"✓ Caught invalid MA window: {e}")


def test_predict_invalid_steps():
    """Test predict validation for invalid steps."""
    print("\nTesting predict steps validation...")

    # Create a simple model
    n = 50
    data = [{"y": float(10 + i * 2)} for i in range(n)]
    data_json = json.dumps(data)
    parsed_data = load_data_from_json(data_json)

    models_to_use = ["ma", "es"]
    model_params = {
        "ma": {"window": 3},
        "es": {"alpha": 0.3}
    }

    metrics, base_model_states, meta_model, y_true, y_pred = train_and_evaluate(
        parsed_data, range(0, 30), range(30, 40), models_to_use, model_params
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test.pkl")
        state = {
            "model_type": "stacking",
            "models": models_to_use,
            "base_model_states": base_model_states,
            "meta_model": meta_model,
        }
        dump_model(state, model_path)

        # Test steps <= 0
        try:
            predict(model_path, steps=0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "steps must be > 0" in str(e)
            print(f"✓ Caught invalid steps: {e}")


def test_successful_validation():
    """Test that valid parameters pass validation."""
    print("\nTesting successful validation...")

    n = 50
    data = [{"y": float(10 + i * 2)} for i in range(n)]
    data_json = json.dumps(data)
    parsed_data = load_data_from_json(data_json)

    models_to_use = ["ma", "es"]
    model_params = {
        "ma": {"window": 3},
        "es": {"alpha": 0.3}
    }

    # Should not raise any exception
    _validate_training_args(parsed_data, 0, 30, 31, 40, models_to_use, model_params)
    print("✓ Valid parameters passed validation")


if __name__ == "__main__":
    try:
        test_format_predictions()
        test_validate_empty_models()
        test_validate_single_model()
        test_validate_unknown_model()
        test_validate_invalid_ranges()
        test_validate_model_params()
        test_predict_invalid_steps()
        test_successful_validation()

        print("\n🎉 All Phase 2 tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
