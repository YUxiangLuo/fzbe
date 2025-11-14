# test_stacking_simple.py
# Simple test for stacking ensemble model

import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.stacking import train_and_evaluate


def main():
    # Create simple data
    sales = list(range(1, 21))  # 1 to 20
    months = [f"2023-{i:02d}" for i in range(1, 21)]

    # Test stacking with MA and ES
    base_models = ["ma", "es"]
    base_params = {"ma": {"window": 3}, "es": {"alpha": 0.5}}
    train_range = range(0, 15)  # 0-14
    eval_range = range(15, 20)  # 15-19

    print("Testing stacking with MA and ES...")
    metrics, predictions = train_and_evaluate(
        sales,
        months,
        train_range,
        eval_range,
        base_models,
        meta_model="average",
        base_params=base_params,
    )

    print(f"Metrics: {metrics}")
    print(f"Predictions: {predictions}")
    print("Stacking test complete!")


if __name__ == "__main__":
    main()
