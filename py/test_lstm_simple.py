# test_lstm_simple.py
# Simple test for reimplemented LSTM model
# test_lstm_simple.py
# Simple test for reimplemented LSTM model

import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.lstm import run


def test_single_feature():
    """Test single feature with zscore."""
    data = json.dumps([{"y": i + 1, "month": f"{i + 1:02d}"} for i in range(20)])
    args = type(
        "Args",
        (),
        {
            "data": data,
            "lstm_normalization": "zscore",
            "lstm_target_feature": "y",
            "lstm_features": [],
            "lookback": 3,
            "epochs": 10,
            "save_model": "/tmp/lstm_single.pkl",
            "pretty": False,
        },
    )()
    print("Testing single feature with zscore...")
    run(args)
    print("Single feature test complete!")


def test_multi_feature():
    """Test multiple features."""
    data = json.dumps(
        [
            {"y": i + 1, "promo": 1 if i % 5 == 0 else 0, "season": (i % 12) / 11.0}
            for i in range(20)
        ]
    )
    args = type(
        "Args",
        (),
        {
            "data": data,
            "lstm_normalization": "zscore",
            "lstm_target_feature": "y",
            "lstm_features": ["promo", "season"],
            "lookback": 3,
            "epochs": 10,
            "save_model": "/tmp/lstm_multi.pkl",
            "pretty": False,
        },
    )()
    print("Testing multi-feature with zscore...")
    run(args)
    print("Multi-feature test complete!")


def test_minmax_normalization():
    """Test minmax normalization."""
    data = json.dumps([{"sales": i + 1} for i in range(20)])
    args = type(
        "Args",
        (),
        {
            "data": data,
            "lstm_normalization": "minmax",
            "lstm_target_feature": "sales",
            "lstm_features": [],
            "lookback": 3,
            "epochs": 10,
            "save_model": "/tmp/lstm_minmax.pkl",
            "pretty": False,
        },
    )()
    print("Testing minmax normalization...")
    run(args)
    print("Minmax normalization test complete!")


def test_prediction():
    """Test prediction mode."""
    from models.lstm import predict

    print("Testing prediction...")
    preds = predict("/tmp/lstm_single.pkl", 3)
    print(f"Predictions: {preds}")


def main():
    test_single_feature()
    test_multi_feature()
    test_minmax_normalization()
    test_prediction()
    print("All tests complete!")


if __name__ == "__main__":
    main()
