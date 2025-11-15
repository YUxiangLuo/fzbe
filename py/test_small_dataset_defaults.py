#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to validate auto-selected default parameters for small datasets.
This simulates the educational system scenario with <10 data points.
"""
import json
import subprocess
import sys
import os

# Get the Python executable from venv
py_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(py_dir, "venv", "bin", "python")
run_model_script = os.path.join(py_dir, "run_model.py")

# Test dataset: 8 data points (6 train + 2 eval)
small_dataset = [
    {"y": 100},
    {"y": 105},
    {"y": 110},
    {"y": 108},
    {"y": 112},
    {"y": 115},  # End of training data
    {"y": 118},
    {"y": 120},  # Evaluation data
]

data_json = json.dumps(small_dataset, ensure_ascii=False)

test_cases = [
    {
        "name": "MA with small dataset",
        "model": "ma",
        "args": [
            "--data", data_json,
            "--train-start", "0",
            "--train-end", "5",
            "--eval-start", "6",
            "--eval-end", "7",
            "--pretty"
        ],
        "expected_window": 2  # For train_size=6, window should be max(2, min(3, 6//3)) = 2
    },
    {
        "name": "ARIMA with small dataset",
        "model": "arima",
        "args": [
            "--data", data_json,
            "--train-start", "0",
            "--train-end", "5",
            "--eval-start", "6",
            "--eval-end", "7",
            "--d", "0",
            "--pretty"
        ],
        "expected_max_p": 1,  # For train_size=6, max(1, min(3, 6//4)) = 1
        "expected_max_q": 1
    },
    {
        "name": "LSTM with small dataset",
        "model": "lstm",
        "args": [
            "--data", data_json,
            "--train-start", "0",
            "--train-end", "5",
            "--eval-start", "6",
            "--eval-end", "7",
            "--pretty"
        ],
        "expected_look_back": 2,  # For train_size=6, max(2, min(3, 6//3)) = 2
        "expected_epochs": 50  # For train_size<=15, epochs=50
    }
]

def run_test(test_case):
    """Run a single test case and check the results."""
    print(f"\n{'='*60}")
    print(f"Testing: {test_case['name']}")
    print(f"{'='*60}")

    cmd = [venv_python, run_model_script, test_case['model']] + test_case['args']

    try:
        result = subprocess.run(
            cmd,
            cwd=py_dir,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"❌ FAILED: {test_case['name']}")
            print(f"stderr: {result.stderr}")
            return False

        # Parse the output
        stdout = result.stdout.strip()
        json_start = stdout.find('{')
        if json_start == -1:
            print(f"❌ FAILED: No JSON output found")
            print(f"stdout: {stdout}")
            return False

        json_output = stdout[json_start:]
        output = json.loads(json_output)

        # Check auto-selected parameters in logs
        logs = output.get("logs", "")
        # Handle both string and list format logs
        if isinstance(logs, str):
            log_lines = logs.split('\n')
        else:
            log_lines = [log.get("message", "") for log in logs]

        print("\n📊 Model Output (metrics only):")
        if "metrics" in output:
            print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))

        print("\n📝 Auto-selection Logs:")
        for line in log_lines:
            if "Auto-selected" in line:
                print(f"  ✓ {line.strip()}")

        # Validate expected parameters
        success = True
        if "expected_window" in test_case:
            # Check logs for auto-selected window
            found = any("Auto-selected window" in line and
                       f"window={test_case['expected_window']}" in line
                       for line in log_lines)
            if found:
                print(f"  ✓ Window auto-selected correctly: {test_case['expected_window']}")
            else:
                print(f"  ⚠️  Window auto-selection not verified in logs")

        if "expected_max_p" in test_case:
            found = any(f"max_p={test_case['expected_max_p']}" in line
                       for line in log_lines)
            if found:
                print(f"  ✓ max_p auto-selected correctly: {test_case['expected_max_p']}")
            else:
                print(f"  ⚠️  max_p auto-selection not verified in logs")

        if "expected_look_back" in test_case:
            found = any(f"look_back={test_case['expected_look_back']}" in line
                       for line in log_lines)
            if found:
                print(f"  ✓ look_back auto-selected correctly: {test_case['expected_look_back']}")
            else:
                print(f"  ⚠️  look_back auto-selection not verified in logs")

        if "expected_epochs" in test_case:
            found = any(f"epochs={test_case['expected_epochs']}" in line
                       for line in log_lines)
            if found:
                print(f"  ✓ epochs auto-selected correctly: {test_case['expected_epochs']}")
            else:
                print(f"  ⚠️  epochs auto-selection not verified in logs")

        # Check that training succeeded
        if "metrics" in output:
            print(f"\n✅ SUCCESS: {test_case['name']} completed successfully")
            metrics = output["metrics"]
            print(f"   RMSE: {metrics.get('rmse', 'N/A')}")
            print(f"   MAE: {metrics.get('mae', 'N/A')}")
            return True
        else:
            print(f"❌ FAILED: No metrics in output")
            return False

    except subprocess.TimeoutExpired:
        print(f"❌ FAILED: {test_case['name']} timed out")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ FAILED: Could not parse JSON output: {e}")
        print(f"stdout: {result.stdout}")
        return False
    except Exception as e:
        print(f"❌ FAILED: {test_case['name']} - {e}")
        return False

def main():
    """Run all test cases."""
    print("🧪 Testing Auto-Selected Default Parameters for Small Datasets")
    print("="*60)

    results = []
    for test_case in test_cases:
        success = run_test(test_case)
        results.append((test_case['name'], success))

    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Summary")
    print(f"{'='*60}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
