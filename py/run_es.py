#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command-line runner for the Exponential Smoothing model."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from models.es import run

def main():
    parser = argparse.ArgumentParser(description="Exponential Smoothing Model Runner")
    parser.add_argument('--data', type=str, help='JSON string of the time series data.')
    parser.add_argument('--save_model', type=str, help='Path to save the trained model.')
    parser.add_argument('--load_model', type=str, help='Path to load a pre-trained model.')
    parser.add_argument('--forecast_steps', type=int, help='Number of steps to forecast.')
    parser.add_argument('--train_start', type=int, help='Training data start index.')
    parser.add_argument('--train_end', type=int, help='Training data end index.')
    parser.add_argument('--eval_start', type=int, help='Evaluation data start index.')
    parser.add_argument('--eval_end', type=int, help='Evaluation data end index.')
    parser.add_argument('--alpha', type=float, required=True, help='Smoothing factor alpha.')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print the JSON output.')
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
