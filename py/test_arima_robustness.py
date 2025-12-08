#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive robustness test suite for ARIMA model.
Tests auto-search functionality, different d values, and model selection logic.
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
    """Generate linear trend data (non-stationary, d=1 works well)."""
    return [{"y": start + i * step} for i in range(count)]


def generate_random_walk(start: float, noise_level: float, count: int) -> List[Dict[str, float]]:
    """Generate random walk data (non-stationary, d=1 makes it stationary)."""
    import random
    random.seed(42)
    data = [{"y": start}]
    for i in range(1, count):
        data.append({"y": data[-1]["y"] + random.uniform(-noise_level, noise_level)})
    return data


def generate_stationary_data(mean: float, noise_level: float, count: int) -> List[Dict[str, float]]:
    """Generate stationary data (d=0 is appropriate)."""
    import random
    random.seed(42)
    return [{"y": mean + random.uniform(-noise_level, noise_level)} for i in range(count)]


def generate_ar_data(count: int) -> List[Dict[str, float]]:
    """Generate AR(1) process data."""
    import random
    random.seed(42)
    phi = 0.7  # AR coefficient
    data = [{"y": random.gauss(0, 1)}]
    for i in range(1, count):
        data.append({"y": phi * data[-1]["y"] + random.gauss(0, 1)})
    return data


class TestCase:
    def __init__(self, name: str, description: str, data: List[Dict], train_start: int, train_end: int,
                 eval_start: int, eval_end: int, d: int, max_p: Optional[int] = None, max_q: Optional[int] = None,
                 should_succeed: bool = True, expected_logs: Optional[List[str]] = None,
                 forecast_steps: Optional[int] = None):
        self.name = name
        self.description = description
        self.data = data
        self.train_start = train_start
        self.train_end = train_end
        self.eval_start = eval_start
        self.eval_end = eval_end
        self.d = d
        self.max_p = max_p
        self.max_q = max_q
        self.should_succeed = should_succeed
        self.expected_logs = expected_logs or []
        self.forecast_steps = forecast_steps
        self.model_file = None


def run_arima_test(test_case: TestCase) -> Dict[str, Any]:
    """Run a single ARIMA model test case using run_model.py."""
    print(f"\n{'='*80}")
    print(f"Test: {test_case.name}")
    print(f"{'='*80}")
    print(f"Description: {test_case.description}")
    print(f"Data points: {len(test_case.data)}")
    print(f"Train: [{test_case.train_start}, {test_case.train_end}]")
    print(f"Eval: [{test_case.eval_start}, {test_case.eval_end}]")
    print(f"d (differencing): {test_case.d}")
    if test_case.max_p is not None:
        print(f"max_p: {test_case.max_p}")
    else:
        print(f"max_p: auto-select")
    if test_case.max_q is not None:
        print(f"max_q: {test_case.max_q}")
    else:
        print(f"max_q: auto-select")

    data_json = json.dumps(test_case.data, ensure_ascii=False)

    # Build command
    cmd = [
        venv_python, run_model_script, "arima",
        "--data", data_json,
        "--train-start", str(test_case.train_start),
        "--train-end", str(test_case.train_end),
        "--eval-start", str(test_case.eval_start),
        "--eval-end", str(test_case.eval_end),
        "--d", str(test_case.d),
        "--pretty"
    ]

    if test_case.max_p is not None:
        cmd.extend(["--max-p", str(test_case.max_p)])
    if test_case.max_q is not None:
        cmd.extend(["--max-q", str(test_case.max_q)])

    if test_case.model_file:
        cmd.extend(["--save-model", test_case.model_file])

    try:
        result = subprocess.run(
            cmd,
            cwd=py_dir,
            capture_output=True,
            text=True,
            timeout=60  # ARIMA can take longer
        )

        # Parse output
        if result.returncode != 0:
            if test_case.should_succeed:
                print(f"\n❌ FAILED: Unexpected error")
                print(f"stderr: {result.stderr[:500]}")
                return {"success": False, "error": "Non-zero exit code", "stderr": result.stderr}
            else:
                print(f"\n✅ PASSED: Failed as expected")
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
            if any(keyword in line for keyword in [
                "Auto-selected", "Validation passed", "Running auto_arima",
                "Selected", "Final selected order", "Model converged",
                "Evaluation metrics computed"
            ]):
                print(f"  {line.strip()}")

        # Display model selection details
        if "best_order" in output:
            order = output["best_order"]
            print(f"\n🎯 Selected ARIMA Order: (p={order['p']}, d={order['d']}, q={order['q']})")

        if "search_summary" in output:
            summary = output["search_summary"]
            print(f"\n🔍 Search Summary:")
            aic_model = summary.get("aic_optimized_model", {})
            bic_model = summary.get("bic_optimized_model", {})
            print(f"  AIC-optimized: order={aic_model.get('order')}, AIC={aic_model.get('aic', 0):.2f}, BIC={aic_model.get('bic', 0):.2f}")
            print(f"  BIC-optimized: order={bic_model.get('order')}, AIC={bic_model.get('aic', 0):.2f}, BIC={bic_model.get('bic', 0):.2f}")
            print(f"  Selected: {summary.get('selected_model')}")

        # Display metrics
        if "metrics" in output:
            metrics = output["metrics"]
            print(f"\n📈 Metrics:")
            print(f"  Evaluated points: {metrics.get('evaluated_points', 'N/A')}")
            print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            print(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
            print(f"  MAPE: {metrics.get('mape', 'N/A'):.4f}%")
            print(f"  R²: {metrics.get('r2', 'N/A'):.4f}")
            print(f"  AIC: {metrics.get('aic', 'N/A'):.4f}")
            print(f"  BIC: {metrics.get('bic', 'N/A'):.4f}")
            print(f"  Converged: {metrics.get('converged', 'N/A')}")

        # Display predictions sample
        if "predictions" in output and output["predictions"]:
            print(f"\n🔮 Predictions (first 3):")
            for i, pred in enumerate(output["predictions"][:3]):
                true_val = output.get("eval_y_true", [])[i] if i < len(output.get("eval_y_true", [])) else "N/A"
                print(f"  [{i}] pred={pred['prediction']:.4f}, CI=[{pred['std_dev']:.4f}], true={true_val}")

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
        print(f"\n❌ FAILED: Timeout after 60s")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print(f"\n❌ FAILED: Exception: {e}")
        return {"success": False, "error": str(e)}


def run_arima_predict_test(model_file: str, forecast_steps: int, expected_logs: List[str]) -> Dict[str, Any]:
    """Test ARIMA prediction with a saved model."""
    print(f"\n{'='*80}")
    print(f"Test: ARIMA Prediction")
    print(f"{'='*80}")
    print(f"Model file: {model_file}")
    print(f"Forecast steps: {forecast_steps}")

    cmd = [
        venv_python, run_model_script, "arima",
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
            for i, pred in enumerate(output["predictions"][:5]):  # Show first 5
                print(f"  Step {i+1}: {pred['prediction']:.4f} (CI width: {pred['std_dev']:.4f})")

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
    """Run comprehensive ARIMA model tests."""
    print("🧪 ARIMA Model Robustness Test Suite")
    print("="*80)

    # Create temporary directory for model files
    temp_dir = tempfile.mkdtemp(prefix="arima_test_")
    print(f"Using temp directory: {temp_dir}")

    test_cases = [
        # Test 1: Small dataset with auto-selected search space
        TestCase(
            name="Small Dataset (Auto Search Space)",
            description="6 training points, 2 eval points, d=0, auto max_p/max_q",
            data=generate_stationary_data(mean=100, noise_level=5, count=8),
            train_start=0, train_end=5,
            eval_start=6, eval_end=7,
            d=0,
            expected_logs=[
                "Auto-selected max_p=1, max_q=1",
                "Validation passed",
                "Running auto_arima with AIC",
                "Running auto_arima with BIC",
                "Final selected order",
                "Evaluation metrics computed"
            ]
        ),

        # Test 2: Linear trend (non-stationary) with d=1
        TestCase(
            name="Linear Trend (d=1 Differencing)",
            description="20 training points, 5 eval points, d=1 to handle trend",
            data=generate_linear_data(start=50, step=2, count=25),
            train_start=0, train_end=19,
            eval_start=20, eval_end=24,
            d=1,
            max_p=3, max_q=3,
            expected_logs=[
                "Validation passed",
                "Running auto_arima",
                "Model converged successfully"
            ]
        ),

        # Test 3: Random walk with d=1
        TestCase(
            name="Random Walk (d=1)",
            description="30 training points, 10 eval points, random walk pattern",
            data=generate_random_walk(start=100, noise_level=3, count=40),
            train_start=0, train_end=29,
            eval_start=30, eval_end=39,
            d=1,
            max_p=2, max_q=2,
            expected_logs=[
                "Validation passed",
                "Final selected order"
            ]
        ),

        # Test 4: AR process (stationary) with d=0
        TestCase(
            name="AR Process (d=0)",
            description="25 training points, 5 eval points, AR(1) process",
            data=generate_ar_data(count=30),
            train_start=0, train_end=24,
            eval_start=25, eval_end=29,
            d=0,
            max_p=3, max_q=1,
            expected_logs=[
                "Validation passed",
                "Running auto_arima",
                "Selected"
            ]
        ),

        # Test 5: Specified large search space
        TestCase(
            name="Large Search Space",
            description="20 training points, 5 eval points, large max_p/max_q",
            data=generate_stationary_data(mean=100, noise_level=10, count=25),
            train_start=0, train_end=19,
            eval_start=20, eval_end=24,
            d=0,
            max_p=5, max_q=5,
            expected_logs=[
                "Validation passed",
                "Running auto_arima with AIC criterion: searching p in [0, 5], q in [0, 5]"
            ]
        ),

        # Test 6: Model saving
        TestCase(
            name="Model Save Test",
            description="Save ARIMA model with order selection",
            data=generate_linear_data(100, 5, 12),
            train_start=0, train_end=8,
            eval_start=9, eval_end=11,
            d=1,
            max_p=2, max_q=2,
            expected_logs=[
                "Validation passed",
                "Saving model to",
                "Model saved successfully"
            ]
        ),

        # Test 7: Invalid d (negative)
        TestCase(
            name="Invalid d (Negative)",
            description="d=-1 should fail validation",
            data=generate_linear_data(100, 5, 20),
            train_start=0, train_end=14,
            eval_start=15, eval_end=19,
            d=-1,
            should_succeed=False,
            expected_logs=["d must be >= 0"]
        ),
    ]

    # Set model file for save test
    test_cases[5].model_file = os.path.join(temp_dir, "arima_test_model.pkl")

    # Run all test cases
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Test {i}/{len(test_cases)}")
        print(f"{'#'*80}")
        result = run_arima_test(test_case)
        results.append((test_case.name, result["success"]))

    # Test 7: Prediction with saved model
    if test_cases[5].model_file and os.path.exists(test_cases[5].model_file):
        print(f"\n\n{'#'*80}")
        print(f"# Test {len(test_cases)+1}/{len(test_cases)+1}")
        print(f"{'#'*80}")
        pred_result = run_arima_predict_test(
            model_file=test_cases[5].model_file,
            forecast_steps=5,
            expected_logs=["Prediction finished successfully", "Generated 5 predictions"]
        )
        results.append(("ARIMA Prediction", pred_result["success"]))

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
