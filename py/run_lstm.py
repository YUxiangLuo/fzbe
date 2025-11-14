#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command-line runner for the LSTM model with multi-feature support."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from models.lstm import run

def main():
    parser = argparse.ArgumentParser(
        description="LSTM Model Runner - Supports multiple features (numeric and categorical)",
        epilog="""
Examples:
  # Train a model with single feature (sales)
  %(prog)s --data '[{"y":10,"month":"M1"},{"y":20,"month":"M2"}]' \\
           --look_back 3 --epochs 50 \\
           --train_start 0 --train_end 10 --eval_start 11 --eval_end 15 \\
           --save_model lstm_model.pkl --pretty

  # Train with multiple features
  %(prog)s --data '[{"sales":100,"marketing":50,"price":10},...]' \\
           --lstm_target_feature sales \\
           --lstm_features '["marketing","price"]' \\
           --feature_types '{"sales":"numeric","marketing":"numeric","price":"numeric"}' \\
           --look_back 5 --epochs 100 \\
           --train_start 0 --train_end 50 --eval_start 51 --eval_end 60 \\
           --save_model model.pkl --pretty

  # Load and predict
  %(prog)s --load_model lstm_model.pkl --forecast_steps 10 --pretty
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data', type=str, help='JSON string of the time series data.')
    parser.add_argument('--save_model', type=str, help='Path to save the trained model.')
    parser.add_argument('--load_model', type=str, help='Path to load a pre-trained model.')
    parser.add_argument('--forecast_steps', type=int, help='Number of steps to forecast (required for prediction mode).')
    parser.add_argument('--train_start', type=int, help='Training data start index (required for training mode).')
    parser.add_argument('--train_end', type=int, help='Training data end index (required for training mode).')
    parser.add_argument('--eval_start', type=int, help='Evaluation data start index (required for training mode).')
    parser.add_argument('--eval_end', type=int, help='Evaluation data end index (required for training mode).')
    parser.add_argument('--look_back', type=int, help='LSTM lookback period (required for training mode).')
    parser.add_argument('--epochs', type=int, help='LSTM training epochs (required for training mode).')
    parser.add_argument('--lstm_target_feature', type=str, help='Target feature name (optional, auto-detected if not provided).')
    parser.add_argument('--lstm_features', type=str, help='JSON array of feature names to include, e.g. \'["marketing","price"]\' (optional).')
    parser.add_argument('--feature_types', type=str, default='{}',
                        help='JSON dict of feature types, e.g. \'{"sales":"numeric","category":"categorical"}\' (optional, auto-inferred if not provided).')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print the JSON output.')

    args = parser.parse_args()

    try:
        run(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
