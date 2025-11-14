#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command-line runner for the ARIMA model."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from models.arima import run

def main():
    parser = argparse.ArgumentParser(
        description="ARIMA Model Runner - Auto-searches for optimal p and q given d",
        epilog="""
Examples:
  # Train a model (automatically finds best p and q for d=1)
  %(prog)s --data '[{"y":10},{"y":20},{"y":30},{"y":40},{"y":50}]' --d 1 \\
           --train_start 0 --train_end 3 --eval_start 4 --eval_end 4 \\
           --save_model arima_model.pkl --pretty

  # Load and predict
  %(prog)s --load_model arima_model.pkl --forecast_steps 3 --pretty

  # Train with custom max_p and max_q
  %(prog)s --data '[{"y":10},{"y":20}]' --d 1 --max_p 3 --max_q 3 \\
           --train_start 0 --train_end 5 --eval_start 6 --eval_end 9 --pretty
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
    parser.add_argument('--d', type=int, help='ARIMA differencing term d (required for training mode).')
    parser.add_argument('--max_p', type=int, default=5, help='Maximum AR order to search (default: 5).')
    parser.add_argument('--max_q', type=int, default=5, help='Maximum MA order to search (default: 5).')
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
