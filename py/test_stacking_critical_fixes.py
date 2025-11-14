#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test to verify critical stacking fixes.
This is NOT a comprehensive test suite - just a smoke test.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data_utils import load_data_from_json
from models.stacking import train_and_evaluate, predict, dump_model

def test_critical_fixes():
    """Test that critical fixes work correctly."""
    print("Testing critical stacking fixes...")

    # Create simple test data
    n = 50
    data = [{"y": float(10 + i * 2)} for i in range(n)]
    data_json = json.dumps(data)

    parsed_data = load_data_from_json(data_json)

    # Test with MA + ES (simplest models, fastest)
    models_to_use = ["ma", "es"]
    model_params = {
        "ma": {"window": 3},
        "es": {"alpha": 0.3}
    }

    print(f"✓ Training with models: {models_to_use}")
    metrics, base_model_states, meta_model, y_true, y_pred = train_and_evaluate(
        parsed_data,
        range(0, 30),
        range(30, 40),
        models_to_use,
        model_params
    )

    print(f"✓ Training completed. RMSE: {metrics['rmse']:.2f}")

    # Verify base model states
    assert "ma" in base_model_states, "MA state missing"
    assert "es" in base_model_states, "ES state missing"
    assert "window" in base_model_states["ma"], "MA window missing"
    assert "level" in base_model_states["es"], "ES level missing"
    print("✓ Base model states validated")

    # Test save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "stacking_test.pkl")

        state = {
            "model_type": "stacking",
            "models": models_to_use,
            "base_model_states": base_model_states,
            "meta_model": meta_model,
        }
        dump_model(state, model_path)
        print(f"✓ Model saved to {model_path}")

        # Test prediction
        preds, std_dev = predict(model_path, steps=5)
        assert len(preds) == 5, f"Expected 5 predictions, got {len(preds)}"
        print(f"✓ Prediction succeeded: {preds[:3]}...")

    print("\n✅ All critical fixes validated!")

def test_arima_fix():
    """Test ARIMA-specific fixes (return value unpacking and state saving)."""
    print("\nTesting ARIMA critical fixes...")

    # Create simple test data
    n = 30
    data = [{"y": float(10 + i * 2)} for i in range(n)]
    data_json = json.dumps(data)

    parsed_data = load_data_from_json(data_json)

    # Test with ARIMA only
    models_to_use = ["arima"]
    model_params = {
        "arima": {
            "d": 1,
            "max_p": 2,
            "max_q": 2
        }
    }

    print(f"✓ Training ARIMA with d=1, max_p=2, max_q=2")
    metrics, base_model_states, meta_model, y_true, y_pred = train_and_evaluate(
        parsed_data,
        range(0, 20),
        range(20, 25),
        models_to_use,
        model_params
    )

    print(f"✓ ARIMA training completed")

    # Verify ARIMA state has all required fields
    arima_state = base_model_states["arima"]
    assert "model" in arima_state, "ARIMA model missing in state"
    assert "order" in arima_state, "ARIMA order missing in state"
    assert "d" in arima_state, "ARIMA d missing in state"
    assert arima_state["d"] == 1, f"Expected d=1, got {arima_state['d']}"

    print(f"✓ ARIMA state validated: order={arima_state['order']}, d={arima_state['d']}")

    # Test save and predict
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "arima_stacking_test.pkl")

        state = {
            "model_type": "stacking",
            "models": models_to_use,
            "base_model_states": base_model_states,
            "meta_model": meta_model,
        }
        dump_model(state, model_path)

        preds, std_dev = predict(model_path, steps=3)
        assert len(preds) == 3, f"Expected 3 predictions, got {len(preds)}"
        print(f"✓ ARIMA prediction succeeded: {preds}")

    print("\n✅ ARIMA fixes validated!")

if __name__ == "__main__":
    try:
        test_critical_fixes()
        test_arima_fix()
        print("\n🎉 All critical fixes working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
