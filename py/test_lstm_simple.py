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


def main():
    # Create simple data: 20 data points
    data = json.dumps([{"y": i + 1, "month": f"{i + 1:02d}"} for i in range(20)])

    # Training args
    args = type(
        "Args",
        (),
        {
            "data": data,
            "lookback": 3,
            "epochs": 10,
            "save_model": "/tmp/lstm_model.pkl",
            "pretty": True,
        },
    )()

    print("Testing LSTM training...")
    run(args)
    print("Training complete!")


if __name__ == "__main__":
    main()
