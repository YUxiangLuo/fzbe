#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command-line runner for the Exponential Smoothing model."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from models.es import run

def main():
    parser = argparse.ArgumentParser(
        description="Exponential Smoothing Model Runner",
        epilog="""
Examples:
  # Train a model
  %(prog)s --data '{"values":[1,2,3,4,5,6,7,8,9,10]}' --alpha 0.3 \\
           --train_start 0 --train_end 6 --eval_start 7 --eval_end 9 \\
           --save_model model.pkl --pretty

  # Load and predict
  %(prog)s --load_model model.pkl --forecast_steps 5 --pretty
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
    parser.add_argument('--alpha', type=float, help='Smoothing factor alpha in (0,1] (required for training mode).')
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
