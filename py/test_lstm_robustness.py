#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive robustness test suite for LSTM (Long Short-Term Memory) model.
Tests multiple realistic scenarios with various look_back values, epochs, and normalization methods.
"""
import json
import subprocess
import sys
import os
import tempfile
from typing import Dict, List, Any, Optional

# Get the Python executable from venv
py_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(py_dir, "venv", "bin", "python")
run_model_script = os.path.join(py_dir, "run_model.py")


def generate_linear_data(start: float, step: float, count: int) -> List[Dict[str, float]]:
    """Generate linear trend data for testing."""
    return [{"y": start + i * step} for i in range(count)]


def generate_exponential_data(base: float, growth_rate: float, count: int) -> List[Dict[str, float]]:
    """Generate exponential growth data for testing."""
    return [{"y": base * (1 + growth_rate) ** i} for i in range(count)]


def generate_multi_feature_data(count: int) -> List[Dict[str, Any]]:
    """Generate multi-feature data for testing."""
    import random
    random.seed(42)
    data = []
    for i in range(count):
        base_value = 100 + i * 5
        x1 = random.uniform(10, 50)
        x2 = random.uniform(20, 80)
        # y is correlated with x1 and x2
        y = base_value + 0.5 * x1 + 0.3 * x2 + random.uniform(-5, 5)
        data.append({"y": y, "x1": x1, "x2": x2})
    return data


class TestCase:
    def __init__(self, name: str, description: str, data: List[Dict], train_start: int, train_end: int,
                 eval_start: int, eval_end: int, look_back: Optional[int] = None,
                 epochs: Optional[int] = None, normalization: Optional[str] = None,
                 lstm_target_feature: Optional[str] = None, lstm_features: Optional[str] = None,
                 feature_types: Optional[Dict[str, str]] = None,
                 should_succeed: bool = True, expected_logs: Optional[List[str]] = None,
                 forecast_steps: Optional[int] = None):
        self.name = name
        self.description = description
        self.data = data
        self.train_start = train_start
        self.train_end = train_end
        self.eval_start = eval_start
        self.eval_end = eval_end
        self.look_back = look_back
        self.epochs = epochs
        self.normalization = normalization
        self.lstm_target_feature = lstm_target_feature
        self.lstm_features = lstm_features
        self.feature_types = feature_types
        self.should_succeed = should_succeed
        self.expected_logs = expected_logs or []
        self.forecast_steps = forecast_steps
        self.model_file = None  # For save/load testing


def run_lstm_test(test_case: TestCase) -> Dict[str, Any]:
    """Run a single LSTM model test case using run_model.py."""
    print(f"\n{'='*80}")
    print(f"Test: {test_case.name}")
    print(f"{'='*80}")
    print(f"Description: {test_case.description}")
    print(f"Data points: {len(test_case.data)}")
    print(f"Train: [{test_case.train_start}, {test_case.train_end}]")
    print(f"Eval: [{test_case.eval_start}, {test_case.eval_end}]")
    if test_case.look_back is not None:
        print(f"Look-back: {test_case.look_back}")
    else:
        print(f"Look-back: default (auto)")
    if test_case.epochs is not None:
        print(f"Epochs: {test_case.epochs}")
    else:
        print(f"Epochs: default (auto)")
    print(f"Normalization: {test_case.normalization or 'minmax (default)'}")

    data_json = json.dumps(test_case.data, ensure_ascii=False)

    # Build command
    cmd = [
        venv_python, run_model_script, "lstm",
        "--data", data_json,
        "--train-start", str(test_case.train_start),
        "--train-end", str(test_case.train_end),
        "--eval-start", str(test_case.eval_start),
        "--eval-end", str(test_case.eval_end),
        "--pretty"
    ]

    if test_case.look_back is not None:
        cmd.extend(["--look-back", str(test_case.look_back)])

    if test_case.epochs is not None:
        cmd.extend(["--epochs", str(test_case.epochs)])

    if test_case.normalization is not None:
        cmd.extend(["--normalization", test_case.normalization])

    if test_case.lstm_target_feature is not None:
        cmd.extend(["--lstm-target-feature", test_case.lstm_target_feature])

    if test_case.lstm_features is not None:
        cmd.extend(["--lstm-features", test_case.lstm_features])

    if test_case.feature_types is not None:
        cmd.extend(["--feature-types", json.dumps(test_case.feature_types)])

    if test_case.model_file:
        cmd.extend(["--save-model", test_case.model_file])

    try:
        result = subprocess.run(
            cmd,
            cwd=py_dir,
            capture_output=True,
            text=True,
            timeout=60  # LSTM may take longer
        )

        # Parse output
        if result.returncode != 0:
            if test_case.should_succeed:
                print(f"\n❌ FAILED: Unexpected error")
                print(f"stderr: {result.stderr}")
                return {"success": False, "error": "Non-zero exit code", "stderr": result.stderr}
            else:
                print(f"\n✅ PASSED: Failed as expected")
                # Check if error message is in stderr
                for expected_log in test_case.expected_logs:
                    if expected_log in result.stderr:
                        print(f"  ✓ Found expected error: '{expected_log}'")
                return {"success": True, "expected_failure": True}

        stdout = result.stdout.strip()
        json_start = stdout.find('{')
        if json_start == -1:
            print(f"\n❌ FAILED: No JSON output")
            return {"success": False, "error": "No JSON output"}

        json_output = stdout[json_start:]
        output = json.loads(json_output)

        # Extract logs
        logs_str = output.get("logs", "")
        log_lines = logs_str.split('\n') if isinstance(logs_str, str) else []

        # Validate expected logs
        print(f"\n📝 Log Validation:")
        all_logs_found = True
        for expected_log in test_case.expected_logs:
            found = any(expected_log in line for line in log_lines)
            if found:
                print(f"  ✓ Found: '{expected_log}'")
            else:
                print(f"  ✗ Missing: '{expected_log}'")
                all_logs_found = False

        # Display key logs
        print(f"\n📊 Key Logs:")
        for line in log_lines:
            if any(keyword in line for keyword in ["Validation passed", "Training LSTM model", "Model training completed", "metrics computed", "Calculated residuals"]):
                print(f"  {line.strip()}")

        # Display metrics
        if "metrics" in output:
            metrics = output["metrics"]
            print(f"\n📈 Metrics:")
            print(f"  Look-back: {metrics.get('look_back', 'N/A')}")
            print(f"  Epochs: {metrics.get('epochs', 'N/A')}")
            print(f"  Evaluated points: {metrics.get('evaluated_points', 'N/A')}")
            print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            print(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
            print(f"  MAPE: {metrics.get('mape', 'N/A'):.4f}%")
            print(f"  R²: {metrics.get('r2', 'N/A'):.4f}")

        # Display predictions sample
        if "predictions" in output and output["predictions"]:
            print(f"\n🔮 Predictions (first 3):")
            for i, pred in enumerate(output["predictions"][:3]):
                true_val = output.get("eval_y_true", [])[i] if i < len(output.get("eval_y_true", [])) else "N/A"
                print(f"  [{i}] pred={pred['prediction']:.4f}, std_dev={pred['std_dev']:.4f}, true={true_val}")

        # Final verdict
        if test_case.should_succeed:
            if all_logs_found and "metrics" in output:
                print(f"\n✅ PASSED: All validations passed")
                return {"success": True, "output": output}
            else:
                print(f"\n⚠️  PARTIAL: Completed but some validations failed")
                return {"success": False, "output": output, "error": "Missing expected logs"}
        else:
            print(f"\n❌ FAILED: Should have failed but succeeded")
            return {"success": False, "error": "Expected failure but got success"}

    except subprocess.TimeoutExpired:
        print(f"\n❌ FAILED: Timeout")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print(f"\n❌ FAILED: Exception: {e}")
        return {"success": False, "error": str(e)}


def run_lstm_predict_test(model_file: str, forecast_steps: int, expected_logs: List[str]) -> Dict[str, Any]:
    """Test LSTM prediction with a saved model."""
    print(f"\n{'='*80}")
    print(f"Test: LSTM Prediction")
    print(f"{'='*80}")
    print(f"Model file: {model_file}")
    print(f"Forecast steps: {forecast_steps}")

    cmd = [
        venv_python, run_model_script, "lstm",
        "--load-model", model_file,
        "--forecast-steps", str(forecast_steps),
        "--pretty"
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=py_dir,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"\n❌ FAILED: Non-zero exit code")
            print(f"stderr: {result.stderr}")
            return {"success": False, "error": result.stderr}

        stdout = result.stdout.strip()
        json_start = stdout.find('{')
        json_output = stdout[json_start:]
        output = json.loads(json_output)

        # Validate logs
        logs_str = output.get("logs", "")
        log_lines = logs_str.split('\n') if isinstance(logs_str, str) else []

        print(f"\n📝 Log Validation:")
        all_logs_found = True
        for expected_log in expected_logs:
            found = any(expected_log in line for line in log_lines)
            if found:
                print(f"  ✓ Found: '{expected_log}'")
            else:
                print(f"  ✗ Missing: '{expected_log}'")
                all_logs_found = False

        # Display predictions
        if "predictions" in output:
            print(f"\n🔮 Predictions ({len(output['predictions'])} steps):")
            for i, pred in enumerate(output["predictions"]):
                print(f"  Step {i+1}: {pred['prediction']:.4f} ± {pred['std_dev']:.4f}")

        if all_logs_found:
            print(f"\n✅ PASSED")
            return {"success": True, "output": output}
        else:
            print(f"\n⚠️  PARTIAL")
            return {"success": False, "error": "Missing expected logs"}

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return {"success": False, "error": str(e)}


def main():
    """Run comprehensive LSTM model tests."""
    print("🧪 LSTM Model Robustness Test Suite")
    print("="*80)

    # Create temporary directory for model files
    temp_dir = tempfile.mkdtemp(prefix="lstm_test_")
    print(f"Using temp directory: {temp_dir}")

    test_cases = [
        # Test 1: Small dataset with auto-selected look_back and epochs
        TestCase(
            name="Small Dataset (Auto Params)",
            description="6 training points, 2 eval points, auto look_back and epochs",
            data=generate_linear_data(100, 5, 8),
            train_start=0, train_end=5,
            eval_start=6, eval_end=7,
            look_back=None,  # Test auto-selection
            epochs=None,  # Test auto-selection
            expected_logs=[
                "Validation passed",
                "Auto-selected look_back=2 based on train_size=6",
                "Using default epochs=20",
                "Training LSTM model for 20 epochs",
                "Model training completed",
                "metrics computed"
            ]
        ),

        # Test 2: Medium dataset with specified parameters
        TestCase(
            name="Medium Dataset (Specified Params)",
            description="15 training points, 5 eval points, look_back=3, epochs=30",
            data=generate_linear_data(50, 2, 20),
            train_start=0, train_end=14,
            eval_start=15, eval_end=19,
            look_back=3,
            epochs=30,
            expected_logs=[
                "Validation passed: train_size=15, eval_size=5, look_back=3",
                "Training LSTM model for 30 epochs",
                "Evaluation metrics computed"
            ]
        ),

        # Test 3: Z-score normalization
        TestCase(
            name="Z-Score Normalization",
            description="15 training points, 5 eval points, zscore normalization",
            data=generate_exponential_data(base=100, growth_rate=0.05, count=20),
            train_start=0, train_end=14,
            eval_start=15, eval_end=19,
            look_back=3,
            epochs=20,
            normalization="zscore",
            expected_logs=[
                "Validation passed",
                "Using StandardScaler (z-score normalization)",
                "Training LSTM model"
            ]
        ),

        # Test 4: MinMax normalization (explicit)
        TestCase(
            name="MinMax Normalization",
            description="15 training points, 5 eval points, minmax normalization",
            data=generate_linear_data(50, 3, 20),
            train_start=0, train_end=14,
            eval_start=15, eval_end=19,
            look_back=2,
            epochs=25,
            normalization="minmax",
            expected_logs=[
                "Validation passed",
                "Using MinMaxScaler (0-1 normalization)",
                "Training LSTM model"
            ]
        ),

        # Test 5: Multi-feature LSTM
        TestCase(
            name="Multi-Feature LSTM",
            description="Multi-feature data with x1, x2 predicting y",
            data=generate_multi_feature_data(20),
            train_start=0, train_end=14,
            eval_start=15, eval_end=19,
            look_back=2,
            epochs=20,
            lstm_target_feature="y",
            lstm_features="y,x1,x2",
            feature_types={"y": "numeric", "x1": "numeric", "x2": "numeric"},
            expected_logs=[
                "Validation passed",
                "Numeric features:",
                "Training LSTM model"
            ]
        ),

        # Test 6: Model saving
        TestCase(
            name="Model Save Test",
            description="Save LSTM model with look_back=2, epochs=20",
            data=generate_linear_data(100, 5, 12),
            train_start=0, train_end=8,
            eval_start=9, eval_end=11,
            look_back=2,
            epochs=20,
            expected_logs=[
                "Validation passed",
                "Saving model to",
                "Model saved successfully"
            ]
        ),
    ]

    # Set model file for save test
    test_cases[5].model_file = os.path.join(temp_dir, "lstm_test_model.pkl")

    # Run all test cases
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Test {i}/{len(test_cases)}")
        print(f"{'#'*80}")
        result = run_lstm_test(test_case)
        results.append((test_case.name, result["success"]))

    # Test 7: Prediction with saved model
    if test_cases[5].model_file and os.path.exists(test_cases[5].model_file):
        print(f"\n\n{'#'*80}")
        print(f"# Test {len(test_cases)+1}/{len(test_cases)+1}")
        print(f"{'#'*80}")
        pred_result = run_lstm_predict_test(
            model_file=test_cases[5].model_file,
            forecast_steps=3,
            expected_logs=["Prediction finished successfully", "Generated 3 predictions"]
        )
        results.append(("LSTM Prediction", pred_result["success"]))

    # Summary
    print(f"\n\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n{'='*80}")
    print(f"Total: {passed}/{total} tests passed ({100*passed//total}%)")
    print(f"{'='*80}")

    # Cleanup
    print(f"\nCleaning up temp directory: {temp_dir}")
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
