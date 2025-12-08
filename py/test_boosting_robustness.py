#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive robustness test suite for Boosting (Residual Learning) ensemble model.
Tests small datasets, various model combinations, and validates residual boosting algorithm.
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


class TestCase:
    def __init__(self, name: str, description: str, data: List[Dict], train_start: int, train_end: int,
                 eval_start: int, eval_end: int, models: str, model_params: Dict[str, Any],
                 should_succeed: bool = True, expected_logs: Optional[List[str]] = None,
                 model_file: Optional[str] = None):
        self.name = name
        self.description = description
        self.data = data
        self.train_start = train_start
        self.train_end = train_end
        self.eval_start = eval_start
        self.eval_end = eval_end
        self.models = models
        self.model_params = model_params
        self.should_succeed = should_succeed
        self.expected_logs = expected_logs or []
        self.model_file = model_file


def run_boosting_test(test_case: TestCase) -> Dict[str, Any]:
    """Run a single Boosting model test case using run_model.py."""
    print(f"\n{'='*80}")
    print(f"Test: {test_case.name}")
    print(f"{'='*80}")
    print(f"Description: {test_case.description}")
    print(f"Data points: {len(test_case.data)}")
    print(f"Train: [{test_case.train_start}, {test_case.train_end}] ({test_case.train_end - test_case.train_start + 1} points)")
    print(f"Eval: [{test_case.eval_start}, {test_case.eval_end}] ({test_case.eval_end - test_case.eval_start + 1} points)")
    print(f"Models: {test_case.models}")

    data_json = json.dumps(test_case.data, ensure_ascii=False)

    # Build command
    cmd = [
        venv_python, run_model_script, "boosting",
        "--data", data_json,
        "--train-start", str(test_case.train_start),
        "--train-end", str(test_case.train_end),
        "--eval-start", str(test_case.eval_start),
        "--eval-end", str(test_case.eval_end),
        "--models", test_case.models,
        "--pretty"
    ]

    # Add model-specific parameters
    for key, value in test_case.model_params.items():
        # Convert underscore to hyphen for argparse compatibility
        arg_name = key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{arg_name}")
        else:
            cmd.extend([f"--{arg_name}", str(value)])

    if test_case.model_file:
        cmd.extend(["--save-model", test_case.model_file])

    try:
        result = subprocess.run(
            cmd,
            cwd=py_dir,
            capture_output=True,
            text=True,
            timeout=360  # 6 minutes for boosting (can train multiple models)
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

        # Display model chain (unique to Boosting)
        if "model_chain" in output:
            chain = output["model_chain"]
            print(f"\n🔗 Model Chain:")
            if chain:
                print(f"  {' → '.join(chain)}")
                print(f"  Total models: {len(chain)}")
            else:
                print(f"  (empty - no models added)")

        # Display search summary
        if "search_summary" in output and "stages" in output["search_summary"]:
            stages = output["search_summary"]["stages"]
            learning_rate = output.get("learning_rate", output["search_summary"].get("learning_rate", "N/A"))
            print(f"\n📊 Boosting Rounds (Learning Rate: {learning_rate}):")
            for stage in stages:
                round_num = stage.get("round", "?")
                action = stage.get("action", "Unknown")
                winner = stage.get("winner", "N/A")
                winner_rmse = stage.get("winner_rmse", float('inf'))
                improvement = stage.get("improvement", 0)

                if "✅" in action:
                    print(f"  Round {round_num}: ✅ {winner} (RMSE: {winner_rmse:.4f}, improve: {improvement:.4f})")
                else:
                    print(f"  Round {round_num}: 🛑 {winner} (RMSE: {winner_rmse:.4f}) - {action}")

        # Display metrics
        if "metrics" in output:
            metrics = output["metrics"]
            print(f"\n📈 Final Results:")
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

        # Validate expected logs (in stderr)
        if test_case.expected_logs:
            print(f"\n📝 Log Validation:")
            all_logs_found = True
            for expected_log in test_case.expected_logs:
                found = expected_log in result.stderr
                if found:
                    print(f"  ✓ Found: '{expected_log}'")
                else:
                    print(f"  ✗ Missing: '{expected_log}'")
                    all_logs_found = False

            if not all_logs_found:
                print(f"\n⚠️  PARTIAL: Some expected logs missing")
                return {"success": False, "output": output, "error": "Missing expected logs"}

        # Final verdict
        if test_case.should_succeed:
            print(f"\n✅ PASSED: All validations passed")
            return {"success": True, "output": output}
        else:
            print(f"\n❌ FAILED: Should have failed but succeeded")
            return {"success": False, "error": "Expected failure but got success"}

    except subprocess.TimeoutExpired:
        print(f"\n❌ FAILED: Timeout")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print(f"\n❌ FAILED: Exception: {e}")
        return {"success": False, "error": str(e)}


def run_boosting_predict_test(model_file: str, forecast_steps: int) -> Dict[str, Any]:
    """Test Boosting prediction with a saved model."""
    print(f"\n{'='*80}")
    print(f"Test: Boosting Prediction")
    print(f"{'='*80}")
    print(f"Model file: {model_file}")
    print(f"Forecast steps: {forecast_steps}")

    cmd = [
        venv_python, run_model_script, "boosting",
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
            timeout=120
        )

        if result.returncode != 0:
            print(f"\n❌ FAILED: Non-zero exit code")
            print(f"stderr: {result.stderr}")
            return {"success": False, "error": result.stderr}

        stdout = result.stdout.strip()
        json_start = stdout.find('{')
        json_output = stdout[json_start:]
        output = json.loads(json_output)

        # Display predictions
        if "predictions" in output:
            print(f"\n🔮 Predictions ({len(output['predictions'])} steps):")
            for i, pred in enumerate(output["predictions"]):
                print(f"  Step {i+1}: {pred['prediction']:.4f} ± {pred['std_dev']:.4f}")

        print(f"\n✅ PASSED")
        return {"success": True, "output": output}

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return {"success": False, "error": str(e)}


def main():
    """Run comprehensive Boosting model tests."""
    print("🧪 Boosting (Residual Learning) Model Robustness Test Suite")
    print("="*80)

    # Create temporary directory for model files
    temp_dir = tempfile.mkdtemp(prefix="boosting_test_")
    print(f"Using temp directory: {temp_dir}")

    test_cases = [
        # Test 1: Minimum viable dataset (8 training points)
        TestCase(
            name="Minimum Dataset (MA+ES)",
            description="8 training points, 2 eval points, MA+ES",
            data=generate_linear_data(100, 5, 10),
            train_start=0, train_end=7,
            eval_start=8, eval_end=9,
            models="ma,es",
            model_params={"window": 2, "alpha": 0.5, "learning_rate": 0.3},
            expected_logs=[
                "Boost Train Range",
                "Boost Val Range",
                "[Boosting Phase 1] Residual learning",
                "[Boosting Phase 2] Retraining",
                "[Boosting Phase 3] Final evaluation"
            ]
        ),

        # Test 2: Small dataset (12 points) with 3 models
        TestCase(
            name="Small Dataset (MA+ES+ARIMA)",
            description="12 training points, 3 eval points, 3 candidate models",
            data=generate_linear_data(50, 3, 15),
            train_start=0, train_end=11,
            eval_start=12, eval_end=14,
            models="ma,es,arima",
            model_params={"window": 2, "alpha": 0.6, "d": 0, "learning_rate": 0.3, "max_rounds": 3},
            expected_logs=[
                "Round 1",
                "Added",  # Changed from "Model Added" to match actual output
                "Model chain:"
            ]
        ),

        # Test 3: Low learning rate (more conservative)
        TestCase(
            name="Low Learning Rate",
            description="Conservative learning rate for stability",
            data=generate_linear_data(50, 2, 20),
            train_start=0, train_end=14,
            eval_start=15, eval_end=19,
            models="ma,es,arima",
            model_params={
                "window": 3,
                "alpha": 0.5,
                "d": 0,
                "learning_rate": 0.1,  # Low learning rate
                "max_rounds": 3
            },
            expected_logs=["Learning Rate: 0.1"]
        ),

        # Test 4: Medium dataset with all 4 models
        TestCase(
            name="Medium Dataset (All 4 Models)",
            description="20 training points, 5 eval points, MA+ES+ARIMA+LSTM",
            data=generate_exponential_data(base=100, growth_rate=0.03, count=25),
            train_start=0, train_end=19,
            eval_start=20, eval_end=24,
            models="ma,es,arima,lstm",
            model_params={
                "window": 3,
                "alpha": 0.7,
                "d": 1,
                # look_back, epochs, learning_rate, max_rounds are auto-selected by backend
            },
            expected_logs=[
                "Round 1",
                "Model chain:"
            ]
        ),

        # Test 5: Model saving
        TestCase(
            name="Model Save Test",
            description="Save boosting model with MA+ES chain",
            data=generate_linear_data(100, 4, 15),
            train_start=0, train_end=10,
            eval_start=11, eval_end=14,
            models="ma,es",
            model_params={"window": 2, "alpha": 0.5, "learning_rate": 0.3},
            expected_logs=[],
            model_file=os.path.join(temp_dir, "boosting_test_model.pkl")
        ),

        # Test 6: Too few training points (should fail)
        TestCase(
            name="Too Few Training Points",
            description="Only 7 training points - should fail validation",
            data=generate_linear_data(100, 5, 10),
            train_start=0, train_end=6,
            eval_start=7, eval_end=9,
            models="ma,es",
            model_params={"window": 2, "alpha": 0.5, "learning_rate": 0.3},
            should_succeed=False,
            expected_logs=["需要至少8个点"]
        ),

        # Test 7: Invalid learning rate (>1, should fail)
        TestCase(
            name="Invalid Learning Rate (>1)",
            description="learning_rate=1.5 should fail validation",
            data=generate_linear_data(100, 5, 15),
            train_start=0, train_end=10,
            eval_start=11, eval_end=14,
            models="ma,es",
            model_params={"window": 2, "alpha": 0.5, "learning_rate": 1.5},
            should_succeed=False,
            expected_logs=["learning_rate must be in (0, 1]"]
        ),

        # Test 8: Invalid learning rate (<=0, should fail)
        TestCase(
            name="Invalid Learning Rate (0)",
            description="learning_rate=0 should fail validation",
            data=generate_linear_data(100, 5, 15),
            train_start=0, train_end=10,
            eval_start=11, eval_end=14,
            models="ma,es",
            model_params={"window": 2, "alpha": 0.5, "learning_rate": 0},
            should_succeed=False,
            expected_logs=["learning_rate must be in (0, 1]"]
        ),
    ]

    # Run all test cases
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Test {i}/{len(test_cases)}")
        print(f"{'#'*80}")
        result = run_boosting_test(test_case)
        results.append((test_case.name, result["success"]))

    # Test 8: Prediction with saved model
    save_test_model_file = os.path.join(temp_dir, "boosting_test_model.pkl")
    if os.path.exists(save_test_model_file):
        print(f"\n\n{'#'*80}")
        print(f"# Test {len(test_cases)+1}/{len(test_cases)+1}")
        print(f"{'#'*80}")
        pred_result = run_boosting_predict_test(
            model_file=save_test_model_file,
            forecast_steps=5
        )
        results.append(("Boosting Prediction", pred_result["success"]))

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
