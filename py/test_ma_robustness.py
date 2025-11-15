#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive robustness test suite for MA (Moving Average) model.
Tests multiple realistic scenarios and validates detailed logging.
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


def generate_seasonal_data(base: float, amplitude: float, period: int, count: int) -> List[Dict[str, float]]:
    """Generate seasonal data for testing."""
    import math
    return [{"y": base + amplitude * math.sin(2 * math.pi * i / period)} for i in range(count)]


class TestCase:
    def __init__(self, name: str, description: str, data: List[Dict], train_start: int, train_end: int,
                 eval_start: int, eval_end: int, window: Optional[int] = None,
                 should_succeed: bool = True, expected_logs: Optional[List[str]] = None,
                 forecast_steps: Optional[int] = None):
        self.name = name
        self.description = description
        self.data = data
        self.train_start = train_start
        self.train_end = train_end
        self.eval_start = eval_start
        self.eval_end = eval_end
        self.window = window
        self.should_succeed = should_succeed
        self.expected_logs = expected_logs or []
        self.forecast_steps = forecast_steps
        self.model_file = None  # For save/load testing


def run_ma_test(test_case: TestCase) -> Dict[str, Any]:
    """Run a single MA model test case using run_model.py."""
    print(f"\n{'='*80}")
    print(f"Test: {test_case.name}")
    print(f"{'='*80}")
    print(f"Description: {test_case.description}")
    print(f"Data points: {len(test_case.data)}")
    print(f"Train: [{test_case.train_start}, {test_case.train_end}]")
    print(f"Eval: [{test_case.eval_start}, {test_case.eval_end}]")
    if test_case.window:
        print(f"Window: {test_case.window}")
    else:
        print(f"Window: auto-select")

    data_json = json.dumps(test_case.data, ensure_ascii=False)

    # Build command
    cmd = [
        venv_python, run_model_script, "ma",
        "--data", data_json,
        "--train-start", str(test_case.train_start),
        "--train-end", str(test_case.train_end),
        "--eval-start", str(test_case.eval_start),
        "--eval-end", str(test_case.eval_end),
        "--pretty"
    ]

    if test_case.window is not None:
        cmd.extend(["--window", str(test_case.window)])

    if test_case.model_file:
        cmd.extend(["--save-model", test_case.model_file])

    try:
        result = subprocess.run(
            cmd,
            cwd=py_dir,
            capture_output=True,
            text=True,
            timeout=30
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
            if any(keyword in line for keyword in ["Auto-selected", "Validation passed", "Generated", "metrics computed", "Calculated residuals"]):
                print(f"  {line.strip()}")

        # Display metrics
        if "metrics" in output:
            metrics = output["metrics"]
            print(f"\n📈 Metrics:")
            print(f"  Window: {metrics.get('window', 'N/A')}")
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


def run_ma_predict_test(model_file: str, forecast_steps: int, expected_logs: List[str]) -> Dict[str, Any]:
    """Test MA prediction with a saved model."""
    print(f"\n{'='*80}")
    print(f"Test: MA Prediction")
    print(f"{'='*80}")
    print(f"Model file: {model_file}")
    print(f"Forecast steps: {forecast_steps}")

    cmd = [
        venv_python, run_model_script, "ma",
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
            timeout=30
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
    """Run comprehensive MA model tests."""
    print("🧪 MA Model Robustness Test Suite")
    print("="*80)

    # Create temporary directory for model files
    temp_dir = tempfile.mkdtemp(prefix="ma_test_")
    print(f"Using temp directory: {temp_dir}")

    test_cases = [
        # Test 1: Small dataset with auto-selected window
        TestCase(
            name="Small Dataset (Auto Window)",
            description="6 training points, 2 eval points, auto-select window",
            data=generate_linear_data(100, 5, 8),
            train_start=0, train_end=5,
            eval_start=6, eval_end=7,
            window=None,
            expected_logs=[
                "Auto-selected window=2",
                "Validation passed",
                "Generated 2 predictions",
                "metrics computed"
            ]
        ),

        # Test 2: Medium dataset with specified window
        TestCase(
            name="Medium Dataset (Specified Window)",
            description="15 training points, 5 eval points, window=3",
            data=generate_linear_data(50, 2, 20),
            train_start=0, train_end=14,
            eval_start=15, eval_end=19,
            window=3,
            expected_logs=[
                "Validation passed: train_size=15",
                "Generated 5 predictions",
                "Evaluation metrics computed"
            ]
        ),

        # Test 3: Large dataset with seasonal pattern
        TestCase(
            name="Large Dataset (Seasonal)",
            description="30 training points, 10 eval points, auto window",
            data=generate_seasonal_data(base=100, amplitude=20, period=12, count=40),
            train_start=0, train_end=29,
            eval_start=30, eval_end=39,
            window=None,
            expected_logs=[
                "Auto-selected window=3",
                "Generated 10 predictions"
            ]
        ),

        # Test 4: Minimum viable dataset
        TestCase(
            name="Minimum Dataset",
            description="3 training points, 1 eval point, window=2",
            data=generate_linear_data(10, 1, 4),
            train_start=0, train_end=2,
            eval_start=3, eval_end=3,
            window=2,
            expected_logs=[
                "Validation passed: train_size=3",
                "Generated 1 predictions"
            ]
        ),

        # Test 5: Model saving
        TestCase(
            name="Model Save Test",
            description="Save model to file",
            data=generate_linear_data(100, 5, 12),
            train_start=0, train_end=8,
            eval_start=9, eval_end=11,
            window=3,
            expected_logs=[
                "Validation passed",
                "Saving model to"
            ]
        ),
    ]

    # Set model file for save test
    test_cases[4].model_file = os.path.join(temp_dir, "ma_test_model.pkl")

    # Run all test cases
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Test {i}/{len(test_cases)}")
        print(f"{'#'*80}")
        result = run_ma_test(test_case)
        results.append((test_case.name, result["success"]))

    # Test 6: Prediction with saved model
    if test_cases[4].model_file and os.path.exists(test_cases[4].model_file):
        print(f"\n\n{'#'*80}")
        print(f"# Test {len(test_cases)+1}/{len(test_cases)+1}")
        print(f"{'#'*80}")
        pred_result = run_ma_predict_test(
            model_file=test_cases[4].model_file,
            forecast_steps=5,
            expected_logs=["Prediction finished successfully"]
        )
        results.append(("MA Prediction", pred_result["success"]))

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
