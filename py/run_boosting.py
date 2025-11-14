#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A command-line runner for the Boosting ensemble model.
"""
import argparse
import json
import os
import sys

# Add the src directory to the Python path for direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from models.boosting import run

def main():
    parser = argparse.ArgumentParser(description="Boosting Ensemble Model Runner")
    
    # --- Core arguments ---
    parser.add_argument('--data', type=str, help='JSON string of the time series data (required for training).')
    parser.add_argument('--save_model', type=str, help='Path to save the trained model.')
    parser.add_argument('--load_model', type=str, help='Path to load a pre-trained model.')
    parser.add_argument('--forecast_steps', type=int, help='Number of steps to forecast.')
    parser.add_argument('--train_start', type=int, help='Training data start index.')
    parser.add_argument('--train_end', type=int, help='Training data end index.')
    parser.add_argument('--eval_start', type=int, help='Evaluation data start index.')
    parser.add_argument('--eval_end', type=int, help='Evaluation data end index.')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print the JSON output.')

    # --- Boosting specific arguments ---
    parser.add_argument('--model-chain', type=str, help='Comma-separated, ordered list of models to use in the boosting chain (e.g., "ma,arima,lstm").')

    # --- Base model parameters with prefixes ---
    parser.add_argument('--arima_order', type=str, default='1,1,1', help='ARIMA order p,d,q.')
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
        if not args.model_chain:
            parser.error("Training mode requires --model_chain.")
        if args.train_start is None or args.train_end is None or args.eval_start is None or args.eval_end is None:
            parser.error("Training mode requires --train_start, --train_end, --eval_start, and --eval_end.")

    run(args)

if __name__ == "__main__":
    main()
