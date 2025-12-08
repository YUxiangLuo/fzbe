#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门测试 LSTM 多特征场景的测试套件
"""
import json
import subprocess
import sys
import os
import tempfile
from typing import Dict, List, Any

# Get the Python executable from venv
py_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(py_dir, "venv", "bin", "python")
run_model_script = os.path.join(py_dir, "run_model.py")


def generate_scenario_1_data(count: int) -> List[Dict[str, Any]]:
    """
    场景1: 销售预测
    - y: 销售额（目标）
    - price: 价格（特征）
    - advertising: 广告费用（特征）
    """
    import random
    random.seed(42)
    data = []
    for i in range(count):
        price = 100 + i * 2 + random.uniform(-5, 5)
        advertising = 50 + i * 1 + random.uniform(-3, 3)
        # 销售额与价格负相关，与广告正相关
        sales = 1000 - 2 * price + 5 * advertising + random.uniform(-20, 20)
        data.append({
            "sales": sales,
            "price": price,
            "advertising": advertising
        })
    return data


def generate_scenario_2_data(count: int) -> List[Dict[str, Any]]:
    """
    场景2: 库存预测
    - inventory: 库存量（目标）
    - demand: 需求量（特征）
    - lead_time: 交货时间（特征）
    - season: 季节因子（特征）
    """
    import random
    random.seed(123)
    data = []
    for i in range(count):
        demand = 200 + 50 * (i % 4) + random.uniform(-10, 10)  # 季节性需求
        lead_time = 3 + random.uniform(-0.5, 0.5)
        season = (i % 4) / 4.0  # 0, 0.25, 0.5, 0.75
        # 库存与需求、交货时间正相关
        inventory = 500 + 1.2 * demand + 50 * lead_time + 100 * season + random.uniform(-30, 30)
        data.append({
            "inventory": inventory,
            "demand": demand,
            "lead_time": lead_time,
            "season": season
        })
    return data


def generate_scenario_3_data(count: int) -> List[Dict[str, Any]]:
    """
    场景3: 只有目标变量（单变量时间序列）
    用于测试 LSTM 在没有额外特征时是否正常工作
    """
    data = []
    for i in range(count):
        value = 100 + i * 3 + 10 * (i % 5)
        data.append({"value": value})
    return data


def run_lstm_test(
    name: str,
    description: str,
    data: List[Dict[str, Any]],
    train_start: int,
    train_end: int,
    eval_start: int,
    eval_end: int,
    lstm_target_feature: str,
    lstm_features: str,
    look_back: int = None,
    epochs: int = None,
    save_model: str = None
) -> Dict[str, Any]:
    """运行单个 LSTM 测试"""
    print(f"\n{'='*80}")
    print(f"Test: {name}")
    print(f"{'='*80}")
    print(f"Description: {description}")
    print(f"Data points: {len(data)}")
    print(f"Train: [{train_start}, {train_end}] ({train_end - train_start + 1} points)")
    print(f"Eval: [{eval_start}, {eval_end}] ({eval_end - eval_start + 1} points)")
    print(f"Target: {lstm_target_feature}")
    print(f"Features: {lstm_features}")
    print(f"Look back: {look_back if look_back else 'auto'}")
    print(f"Epochs: {epochs if epochs else 'auto'}")

    data_json = json.dumps(data, ensure_ascii=False)

    # Build command
    cmd = [
        venv_python, run_model_script, "lstm",
        "--data", data_json,
        "--train-start", str(train_start),
        "--train-end", str(train_end),
        "--eval-start", str(eval_start),
        "--eval-end", str(eval_end),
        "--lstm-target-feature", lstm_target_feature,
        "--lstm-features", lstm_features,
        "--pretty"
    ]

    if look_back is not None:
        cmd.extend(["--look-back", str(look_back)])
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    if save_model:
        cmd.extend(["--save-model", save_model])

    try:
        result = subprocess.run(
            cmd,
            cwd=py_dir,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            print(f"\n❌ FAILED: Unexpected error")
            print(f"stderr: {result.stderr}")
            return {"success": False, "error": result.stderr}

        stdout = result.stdout.strip()
        json_start = stdout.find('{')
        if json_start == -1:
            print(f"\n❌ FAILED: No JSON output")
            return {"success": False, "error": "No JSON output"}

        json_output = stdout[json_start:]
        output = json.loads(json_output)

        # Display feature types
        if "feature_types" in output:
            print(f"\n🔧 Feature Types:")
            for feat, ftype in output["feature_types"].items():
                print(f"  {feat}: {ftype}")

        # Display metrics
        if "metrics" in output:
            metrics = output["metrics"]
            print(f"\n📊 Results:")
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

        print(f"\n✅ PASSED")
        return {"success": True, "output": output}

    except subprocess.TimeoutExpired:
        print(f"\n❌ FAILED: Timeout")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print(f"\n❌ FAILED: Exception: {e}")
        return {"success": False, "error": str(e)}


def run_prediction_test(model_file: str, forecast_steps: int, expected_features: List[str]) -> Dict[str, Any]:
    """测试多特征模型的预测功能"""
    print(f"\n{'='*80}")
    print(f"Test: Multi-Feature Prediction")
    print(f"{'='*80}")
    print(f"Model file: {model_file}")
    print(f"Forecast steps: {forecast_steps}")
    print(f"Expected features: {', '.join(expected_features)}")

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
    """运行多特征 LSTM 测试套件"""
    print("🧪 LSTM Multi-Feature Comprehensive Test Suite")
    print("="*80)

    # Create temporary directory for model files
    temp_dir = tempfile.mkdtemp(prefix="lstm_multifeature_test_")
    print(f"Using temp directory: {temp_dir}")

    results = []

    # Test 1: 场景1 - 销售预测（2个特征）
    print(f"\n\n{'#'*80}")
    print(f"# Test 1: Sales Prediction (price + advertising → sales)")
    print(f"{'#'*80}")
    result = run_lstm_test(
        name="Sales Prediction",
        description="Predict sales from price and advertising",
        data=generate_scenario_1_data(30),
        train_start=0,
        train_end=24,
        eval_start=25,
        eval_end=29,
        lstm_target_feature="sales",
        lstm_features="sales,price,advertising",
        look_back=3,
        epochs=20
    )
    results.append(("Sales Prediction (2 features)", result["success"]))

    # Test 2: 场景2 - 库存预测（3个特征）
    print(f"\n\n{'#'*80}")
    print(f"# Test 2: Inventory Prediction (demand + lead_time + season → inventory)")
    print(f"{'#'*80}")
    model_file = os.path.join(temp_dir, "inventory_model.pkl")
    result = run_lstm_test(
        name="Inventory Prediction",
        description="Predict inventory from demand, lead_time, and season",
        data=generate_scenario_2_data(32),
        train_start=0,
        train_end=26,
        eval_start=27,
        eval_end=31,
        lstm_target_feature="inventory",
        lstm_features="inventory,demand,lead_time,season",
        look_back=4,
        epochs=20,
        save_model=model_file
    )
    results.append(("Inventory Prediction (3 features)", result["success"]))

    # Test 3: 场景3 - 单变量时间序列（0个额外特征）
    print(f"\n\n{'#'*80}")
    print(f"# Test 3: Univariate Time Series (no additional features)")
    print(f"{'#'*80}")
    result = run_lstm_test(
        name="Univariate LSTM",
        description="LSTM with only target variable (no extra features)",
        data=generate_scenario_3_data(25),
        train_start=0,
        train_end=19,
        eval_start=20,
        eval_end=24,
        lstm_target_feature="value",
        lstm_features="value",
        look_back=3,
        epochs=15
    )
    results.append(("Univariate Time Series", result["success"]))

    # Test 4: 使用自动参数选择的多特征场景
    print(f"\n\n{'#'*80}")
    print(f"# Test 4: Multi-Feature with Auto Parameters")
    print(f"{'#'*80}")
    result = run_lstm_test(
        name="Auto Params Multi-Feature",
        description="Use auto look_back and epochs with multiple features",
        data=generate_scenario_1_data(40),
        train_start=0,
        train_end=34,
        eval_start=35,
        eval_end=39,
        lstm_target_feature="sales",
        lstm_features="sales,price,advertising",
        # look_back and epochs will be auto-selected
    )
    results.append(("Auto Params Multi-Feature", result["success"]))

    # Test 5: 预测测试（使用保存的多特征模型）
    if os.path.exists(model_file):
        print(f"\n\n{'#'*80}")
        print(f"# Test 5: Prediction with Saved Multi-Feature Model")
        print(f"{'#'*80}")
        pred_result = run_prediction_test(
            model_file=model_file,
            forecast_steps=5,
            expected_features=["inventory", "demand", "lead_time", "season"]
        )
        results.append(("Multi-Feature Prediction", pred_result["success"]))

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
