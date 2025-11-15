#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test to verify the complete parameter flow from TypeScript to Python.
This script simulates the parameter transformation and validates the mapping.
"""
import json
import sys

# TypeScript Zod Schema Parameters (from modelParamsSchema)
ZOD_SCHEMA_PARAMS = {
    "moving_average_window",
    "exponential_smoothing_alpha",
    "arimaD",
    "lstmFeatures",
    "lstmTargetFeature",
    "lstmNormalization",
    "models",
}

# TypeScript allPossiblePythonParams mapping
ALL_POSSIBLE_PYTHON_PARAMS = {
    # Common
    "data": "data",
    "trainStart": "train-start",
    "trainEnd": "train-end",
    "evalStart": "eval-start",
    "evalEnd": "eval-end",
    "saveModel": "save-model",
    "loadModel": "load-model",
    "forecastSteps": "forecast-steps",
    "pretty": "pretty",
    # MA
    "moving_average_window": "window",
    # ES
    "exponential_smoothing_alpha": "alpha",
    # ARIMA
    "arimaD": "d",
    "arimaMaxP": "max-p",
    "arimaMaxQ": "max-q",
    # LSTM
    "lstmLookBack": "look-back",
    "lstmEpochs": "epochs",
    "lstmFeatures": "lstm-features",
    "lstmTargetFeature": "lstm-target-feature",
    "lstmFeatureTypes": "feature-types",
    "lstmNormalization": "normalization",
    # Ensemble Models
    "models": "models",
}

# Python run_model.py expected parameters
PYTHON_RUN_MODEL_PARAMS = {
    # Common
    "data",
    "train-start",
    "train-end",
    "eval-start",
    "eval-end",
    "save-model",
    "load-model",
    "forecast-steps",
    "pretty",
    # MA
    "window",
    # ES
    "alpha",
    # ARIMA
    "d",
    "max-p",
    "max-q",
    # LSTM
    "look-back",
    "epochs",
    "feature-types",
    "lstm-features",
    "lstm-target-feature",
    "normalization",
    # Ensemble Models
    "models",
}


def simulate_typescript_flow(body_params):
    """
    Simulate the TypeScript parameter transformation flow.

    This mimics:
    1. Manual pythonParams creation (lines 322-330)
    2. Loop through allPossiblePythonParams (lines 332-337)
    3. runPythonModel transformation (lines 178-186)
    """
    print("\n" + "="*80)
    print("Simulating TypeScript Parameter Flow")
    print("="*80)

    # Step 1: Manual pythonParams creation (lines 322-330)
    pythonParams = {
        "data": json.dumps([{"y": 100}]),
        "trainStart": body_params.get("data_window_train_start_index", 0),
        "trainEnd": body_params.get("data_window_train_end_index", 5),
        "evalStart": body_params.get("data_window_evaluate_start_index", 6),
        "evalEnd": body_params.get("data_window_evaluate_end_index", 7),
        "saveModel": "/tmp/model.pkl",
        "pretty": True,
    }

    print("\nStep 1: Manual pythonParams creation")
    print(f"  pythonParams = {json.dumps(pythonParams, indent=4)}")

    # Step 2: Loop through allPossiblePythonParams (lines 332-337)
    print("\nStep 2: Loop through allPossiblePythonParams")
    for bodyKey, scriptKey in ALL_POSSIBLE_PYTHON_PARAMS.items():
        if bodyKey in body_params:
            pythonParams[scriptKey] = body_params[bodyKey]
            print(f"  Found {bodyKey} in body -> pythonParams['{scriptKey}'] = {body_params[bodyKey]}")

    # Step 3: runPythonModel transformation (lines 178-186)
    print("\nStep 3: runPythonModel transformation")
    cli_args = []
    for key, value in pythonParams.items():
        scriptKey = ALL_POSSIBLE_PYTHON_PARAMS.get(key, key)
        cliArg = f"--{scriptKey}"
        if isinstance(value, bool):
            if value:
                cli_args.append(cliArg)
                print(f"  pythonParams['{key}'] -> {cliArg}")
        else:
            cli_args.extend([cliArg, str(value)])
            print(f"  pythonParams['{key}'] -> {cliArg} {value}")

    print(f"\nFinal CLI arguments:")
    print(f"  python run_model.py <model_type> {' '.join(cli_args)}")

    # Verify all generated args are in PYTHON_RUN_MODEL_PARAMS
    print("\n" + "="*80)
    print("Validation")
    print("="*80)

    generated_param_names = set()
    for arg in cli_args:
        if arg.startswith("--"):
            param_name = arg[2:]  # Remove --
            generated_param_names.add(param_name)

    unknown_params = generated_param_names - PYTHON_RUN_MODEL_PARAMS
    if unknown_params:
        print(f"\n❌ ERROR: Unknown parameters generated: {unknown_params}")
        return False
    else:
        print(f"\n✅ All generated parameters are valid Python run_model.py parameters")
        return True


def check_zod_coverage():
    """Check if all Zod schema parameters are mapped in allPossiblePythonParams."""
    print("\n" + "="*80)
    print("Checking Zod Schema Coverage")
    print("="*80)

    unmapped_zod_params = ZOD_SCHEMA_PARAMS - set(ALL_POSSIBLE_PYTHON_PARAMS.keys())

    if unmapped_zod_params:
        print(f"\n❌ ERROR: Zod schema parameters not in allPossiblePythonParams:")
        for param in unmapped_zod_params:
            print(f"  - {param}")
        return False
    else:
        print(f"\n✅ All Zod schema parameters are mapped in allPossiblePythonParams")

        print(f"\nMapping details:")
        for param in ZOD_SCHEMA_PARAMS:
            python_param = ALL_POSSIBLE_PYTHON_PARAMS[param]
            print(f"  {param:35} -> --{python_param}")
        return True


def check_python_params_validity():
    """Check if all mapped Python parameters are accepted by run_model.py."""
    print("\n" + "="*80)
    print("Checking Python Parameter Validity")
    print("="*80)

    mapped_python_params = set(ALL_POSSIBLE_PYTHON_PARAMS.values())
    invalid_params = mapped_python_params - PYTHON_RUN_MODEL_PARAMS

    if invalid_params:
        print(f"\n❌ ERROR: Mapped parameters not accepted by run_model.py:")
        for param in invalid_params:
            print(f"  --{param}")
        return False
    else:
        print(f"\n✅ All mapped parameters are accepted by run_model.py")
        return True


def main():
    """Run all validation tests."""
    print("🧪 Parameter Flow Validation Test Suite")
    print("="*80)

    results = []

    # Test 1: Check Zod coverage
    results.append(("Zod Schema Coverage", check_zod_coverage()))

    # Test 2: Check Python parameter validity
    results.append(("Python Parameter Validity", check_python_params_validity()))

    # Test 3: Simulate MA model training
    print("\n" + "="*80)
    print("Test Case 1: MA Model Training")
    print("="*80)
    ma_body = {
        "data_window_train_start_index": 0,
        "data_window_train_end_index": 5,
        "data_window_evaluate_start_index": 6,
        "data_window_evaluate_end_index": 7,
        "moving_average_window": 3,
    }
    results.append(("MA Model Simulation", simulate_typescript_flow(ma_body)))

    # Test 4: Simulate LSTM model training
    print("\n" + "="*80)
    print("Test Case 2: LSTM Model Training")
    print("="*80)
    lstm_body = {
        "data_window_train_start_index": 0,
        "data_window_train_end_index": 14,
        "data_window_evaluate_start_index": 15,
        "data_window_evaluate_end_index": 19,
        "lstmFeatures": "y,x1,x2",
        "lstmTargetFeature": "y",
        "lstmNormalization": "minmax",
    }
    results.append(("LSTM Model Simulation", simulate_typescript_flow(lstm_body)))

    # Test 5: Simulate ARIMA model training
    print("\n" + "="*80)
    print("Test Case 3: ARIMA Model Training")
    print("="*80)
    arima_body = {
        "data_window_train_start_index": 0,
        "data_window_train_end_index": 14,
        "data_window_evaluate_start_index": 15,
        "data_window_evaluate_end_index": 19,
        "arimaD": 1,
    }
    results.append(("ARIMA Model Simulation", simulate_typescript_flow(arima_body)))

    # Summary
    print("\n\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n{'='*80}")
    print(f"Total: {passed}/{total} tests passed ({100*passed//total}%)")
    print(f"{'='*80}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
