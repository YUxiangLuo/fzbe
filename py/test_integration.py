#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test to verify all refactored models work correctly.
Tests that all models can import shared utilities and run basic operations.
"""
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from serialization import dump_model, load_model
from metrics import rmse, mae, mape, r2
from models import ma, es, arima


def test_utilities():
    """Test shared utility modules."""
    print("Testing shared utilities...")

    # Test metrics
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.1, 2.1, 2.9, 3.8, 5.2]

    assert isinstance(rmse(y_true, y_pred), float)
    assert isinstance(mae(y_true, y_pred), float)
    assert isinstance(mape(y_true, y_pred), float)
    assert isinstance(r2(y_true, y_pred), float)
    print("  ✓ Metrics working")

    # Test joblib_compat
    temp_dir = tempfile.mkdtemp()
    try:
        test_data = {"test": "data", "value": 123}
        test_path = os.path.join(temp_dir, "test.pkl")

        dump_model(test_data, test_path)
        loaded = load_model(test_path)
        assert loaded == test_data
        print("  ✓ joblib_compat working")
    finally:
        shutil.rmtree(temp_dir)


def test_model_imports():
    """Test that all models import correctly."""
    print("\nTesting model imports...")

    # Check MA model
    assert hasattr(ma, 'run')
    assert hasattr(ma, 'predict')
    assert hasattr(ma, 'MODEL_TYPE')
    assert ma.MODEL_TYPE == "ma"
    print("  ✓ MA model imports")

    # Check ES model
    assert hasattr(es, 'run')
    assert hasattr(es, 'predict')
    assert hasattr(es, 'MODEL_TYPE')
    assert es.MODEL_TYPE == "es"
    print("  ✓ ES model imports")

    # Check ARIMA model
    assert hasattr(arima, 'run')
    assert hasattr(arima, 'predict')
    print("  ✓ ARIMA model imports")


def test_no_code_duplication():
    """Verify no duplicate code in models."""
    print("\nChecking for code duplication...")

    # Read model files
    models_dir = os.path.join(os.path.dirname(__file__), 'src', 'models')

    for model_file in ['ma.py', 'es.py', 'arima.py']:
        path = os.path.join(models_dir, model_file)
        with open(path, 'r') as f:
            content = f.read()

        # Check no duplicate pickle functions
        assert 'def save_pickle(' not in content, f"{model_file} still has save_pickle"
        assert 'def load_pickle(' not in content, f"{model_file} still has load_pickle"

        # Check no duplicate metrics (except imports)
        lines = content.split('\n')
        metric_defs = [l for l in lines if l.strip().startswith('def rmse(') or
                       l.strip().startswith('def mae(') or
                       l.strip().startswith('def mape(') or
                       l.strip().startswith('def r2(')]
        assert len(metric_defs) == 0, f"{model_file} still has metric definitions"

        # Check proper imports
        assert 'from serialization import' in content, f"{model_file} missing serialization import"
        assert 'from metrics import' in content, f"{model_file} missing metrics import"

        print(f"  ✓ {model_file} has no duplicates")


def main():
    print("=" * 60)
    print("Integration Test: Refactored Models")
    print("=" * 60)

    try:
        test_utilities()
        test_model_imports()
        test_no_code_duplication()

        print("\n" + "=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print("  • Shared utilities (metrics, joblib_compat) work correctly")
        print("  • All models (MA, ES, ARIMA) import successfully")
        print("  • No code duplication detected")
        print("  • All models use shared utility modules")
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
