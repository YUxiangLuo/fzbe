# test_lstm_simple.py
# Comprehensive test for LSTM model including edge cases

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


def test_edge_data_too_short():
    """Test data too short for lookback."""
    data = json.dumps([{"y": 1}, {"y": 2}])
    args = type(
        "Args",
        (),
        {
            "data": data,
            "lstm_normalization": "zscore",
            "lstm_target_feature": "y",
            "lstm_features": [],
            "lookback": 5,
            "epochs": 1,
        },
    )()
    print("Testing data too short...")
    try:
        run(args)
        print("ERROR: Should have failed")
    except ValueError:
        print("Correctly failed for short data")


def test_edge_invalid_normalization():
    """Test invalid normalization."""
    data = json.dumps([{"y": 1}, {"y": 2}, {"y": 3}])
    args = type(
        "Args",
        (),
        {
            "data": data,
            "lstm_normalization": "invalid",
            "lstm_target_feature": "y",
            "lstm_features": [],
            "lookback": 2,
            "epochs": 1,
        },
    )()
    print("Testing invalid normalization...")
    try:
        run(args)
        print("ERROR: Should have failed")
    except ValueError:
        print("Correctly failed for invalid normalization")


def test_edge_missing_feature():
    """Test missing feature in data."""
    data = json.dumps([{"y": 1, "promo": 0}, {"y": 2}])
    args = type(
        "Args",
        (),
        {
            "data": data,
            "lstm_normalization": "zscore",
            "lstm_target_feature": "y",
            "lstm_features": ["promo"],
            "lookback": 1,
            "epochs": 1,
        },
    )()
    print("Testing missing feature...")
    try:
        run(args)
        print("ERROR: Should have failed")
    except ValueError:
        print("Correctly failed for missing feature")


def test_edge_large_epochs():
    """Test with large epochs."""
    data = json.dumps([{"y": i + 1} for i in range(30)])
    args = type(
        "Args",
        (),
        {
            "data": data,
            "lstm_normalization": "zscore",
            "lstm_target_feature": "y",
            "lstm_features": [],
            "lookback": 5,
            "epochs": 100,  # Large epochs
            "save_model": "/tmp/lstm_large_epochs.pkl",
            "pretty": False,
        },
    )()
    print("Testing large epochs...")
    run(args)
    print("Large epochs test complete!")


def test_edge_no_features():
    """Test with no additional features."""
    data = json.dumps([{"sales": i + 1} for i in range(30)])
    args = type(
        "Args",
        (),
        {
            "data": data,
            "lstm_normalization": "minmax",
            "lstm_target_feature": "sales",
            "lstm_features": [],
            "lookback": 5,
            "epochs": 10,
            "save_model": "/tmp/lstm_no_features.pkl",
        },
    )()
    print("Testing no additional features...")
    run(args)
    print("No features test complete!")


def main():
    test_single_feature()
    test_multi_feature()
    test_minmax_normalization()
    test_prediction()
    test_edge_data_too_short()
    test_edge_invalid_normalization()
    test_edge_missing_feature()
    test_edge_large_epochs()
    test_edge_no_features()
    print("All tests complete!")


if __name__ == "__main__":
    main()
