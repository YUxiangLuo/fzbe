#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to invoke the LSTM model with multi-feature data.
"""
import json
import os
import sys
import tempfile
import io
from argparse import Namespace
from contextlib import redirect_stdout

# Add the src directory to the Python path for direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import numpy as np
from models.lstm import run

def create_multi_feature_data(n=100):
    """Create sine-wave based test data with multiple features."""
    data = []
    for i in range(n):
        # Base sales on a sine wave
        sales = 100 + 20 * np.sin(i * 0.2) + np.random.randn() * 2
        
        # Marketing spend is somewhat correlated with sales, with some noise
        marketing_spend = 10 + (sales - 100) * 0.5 + np.random.randn() * 3
        
        # Competitor price has its own pattern
        competitor_price = 50 - 10 * np.cos(i * 0.1) + np.random.randn() * 1.5
        
        data.append({
            "month": f"2024-{(i % 12) + 1:02d}", 
            "sales": round(sales, 2),
            "marketing_spend": round(marketing_spend, 2),
            "competitor_price": round(competitor_price, 2)
        })
    return data

def run_multi_feature_test():
    """
    Tests the LSTM model with a dataset containing multiple features.
    """
    print("=" * 60)
    print("Running LSTM with Multi-Feature Data")
    print("=" * 60)

    # 1. Generate Data
    n_total = 70
    test_data = create_multi_feature_data(n_total)
    data_json = json.dumps(test_data)
    print(f"Generated {n_total} data points with 3 features.")
    print("Sample data point:", json.dumps(test_data[0], indent=2))


    # 2. Define ranges and features
    ts, te = 0, 49
    es, ee = 50, 69
    target_feature = "sales"
    features = ["marketing_spend", "competitor_price"]
    
    print(f"Target Feature: '{target_feature}'")
    print(f"Additional Features: {features}")
    print(f"Training Range: [{ts}, {te}]")
    print(f"Evaluation Range: [{es}, {ee}]")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 3. Set up arguments
        model_path = os.path.join(tmpdir, "lstm_multi_feature_model.pkl")
        
        args = Namespace(
            data=data_json,
            train_start=ts,
            train_end=te,
            eval_start=es,
            eval_end=ee,
            lookback=10,
            epochs=10, # More epochs for slightly more complex data
            save_model=model_path,
            pretty=True,
            lstm_target_feature=target_feature,
            lstm_features=features,
            # Add other optional args with defaults
            lstm_normalization="zscore",
            batch_size=6,
            lr=0.01,
            load_model=None,
            forecast_steps=None
        )

        # 4. Run the function and capture output
        print("\nInvoking lstm.py...")
        output_io = io.StringIO()
        with redirect_stdout(output_io):
            run(args)
        
        output_str = output_io.getvalue()
        
        # 5. Print the results
        print("lstm.py executed successfully.")
        print("\n" + "-" * 25 + " JSON Output " + "-" * 24)
        print(output_str)
        print("-" * 60)

        # Basic verification
        output = json.loads(output_str)
        assert output["mode"] == "lstm_trained"
        assert output["target_feature"] == target_feature
        assert output["features"] == features
        print("\nVerification successful: Output contains correct mode and feature names.")


if __name__ == "__main__":
    run_multi_feature_test()
