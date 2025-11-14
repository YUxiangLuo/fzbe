#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A command-line runner for the Stacking ensemble model.
This script handles path setup and command-line argument parsing.
"""
import argparse
import json
import os
import sys

# Add the src directory to the Python path for direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from models.stacking import run

def main():
    parser = argparse.ArgumentParser(
        description="Stacking Ensemble Model Runner - Combines predictions from multiple base models",
        epilog="""
Examples:
  # Train stacking model with MA + ES
  %(prog)s --data '[{"y":10},{"y":20},...,{"y":100}]' \\
           --models ma,es \\
           --train_start 0 --train_end 60 --eval_start 61 --eval_end 80 \\
           --ma_window 5 --es_alpha 0.3 \\
           --save_model stacking_model.pkl --pretty

  # Train with MA + ES + ARIMA
  %(prog)s --data '[{"y":10},{"y":20},...,{"y":100}]' \\
           --models ma,es,arima \\
           --train_start 0 --train_end 60 --eval_start 61 --eval_end 80 \\
           --ma_window 5 --es_alpha 0.3 \\
           --arima_d 1 --arima_max_p 3 --arima_max_q 3 \\
           --save_model model.pkl --pretty

  # Train with all models including LSTM
  %(prog)s --data '[{"sales":100,"price":50},...}]' \\
           --models ma,es,arima,lstm \\
           --train_start 0 --train_end 40 --eval_start 41 --eval_end 50 \\
           --ma_window 3 --es_alpha 0.5 --arima_d 1 \\
           --lstm_look_back 5 --lstm_epochs 20 \\
           --lstm_feature_types '{"sales":"numeric","price":"numeric"}' \\
           --save_model model.pkl --pretty

  # Load and predict
  %(prog)s --load_model stacking_model.pkl --forecast_steps 10 --pretty
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Common arguments
    parser.add_argument('--data', type=str, help='JSON string of the time series data (required for training).')
    parser.add_argument('--save_model', type=str, help='Path to save the trained model.')
    parser.add_argument('--load_model', type=str, help='Path to load a pre-trained model.')
    parser.add_argument('--forecast_steps', type=int, help='Number of steps to forecast.')
    parser.add_argument('--train_start', type=int, help='Training data start index.')
    parser.add_argument('--train_end', type=int, help='Training data end index.')
    parser.add_argument('--eval_start', type=int, help='Evaluation data start index.')
    parser.add_argument('--eval_end', type=int, help='Evaluation data end index.')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print the JSON output.')

    # --- Stacking specific arguments ---
    parser.add_argument('--models', type=str, help='Comma-separated list of models to include in the stacking ensemble (e.g., "arima,lstm,ma").')

    # --- Base model parameters with prefixes ---
    parser.add_argument('--arima_d', type=int, default=1, help='ARIMA differencing order d (auto-searches for optimal p and q).')
    parser.add_argument('--arima_max_p', type=int, default=5, help='ARIMA maximum AR order to search (default: 5).')
    parser.add_argument('--arima_max_q', type=int, default=5, help='ARIMA maximum MA order to search (default: 5).')
    parser.add_argument('--es_alpha', type=float, default=0.5, help='Exponential Smoothing alpha.')
    parser.add_argument('--ma_window', type=int, default=3, help='Moving Average window size.')
    parser.add_argument('--lstm_look_back', type=int, default=12, help='LSTM look_back period.')
    parser.add_argument('--lstm_epochs', type=int, default=10, help='LSTM training epochs.')
    parser.add_argument('--lstm_feature_types', type=str, default='{}', help='JSON string for LSTM feature types.')

    args = parser.parse_args()

    # --- Validation ---
    if args.load_model:
        if not args.forecast_steps:
            parser.error("--load_model requires --forecast_steps.")
    else: # Training mode
        if not args.data:
            parser.error("Training mode requires --data.")
        if not args.models:
            parser.error("Training mode requires --models.")
        if args.train_start is None or args.train_end is None or args.eval_start is None or args.eval_end is None:
            parser.error("Training mode requires --train_start, --train_end, --eval_start, and --eval_end.")

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
