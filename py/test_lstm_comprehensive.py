#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive unit test suite for the refactored LSTM model.
"""
import json
import os
import sys
import tempfile
import unittest
import io
from argparse import Namespace
from contextlib import redirect_stdout

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from models.lstm import run
import numpy as np

def create_test_data(n=60, keys=["sales", "feature1"]):
    """Helper to create test data."""
    data = []
    for i in range(n):
        record = {
            keys[0]: 150 + i*3 + 8*np.sin(i/5) + np.random.randn()
        }
        if len(keys) > 1:
            record[keys[1]] = 50 + i*0.5 + 3*np.cos(i/3) + np.random.randn()
        data.append(record)
    return data

class TestLSTMComprehensiveRefactored(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def capture_run_output(self, args):
        """Run the script and capture its JSON output."""
        output_io = io.StringIO()
        try:
            with redirect_stdout(output_io):
                run(args)
            output_str = output_io.getvalue()
            return json.loads(output_str) if output_str else None
        except (ValueError, FileNotFoundError):
            raise

    # 1. Core Functionality
    def test_01_train_save_predict_univariate(self):
        """Test end-to-end: train, save, load, and predict for a univariate case."""
        data = create_test_data(60, keys=["sales"])
        model_path = os.path.join(self.tmpdir.name, "lstm_univariate.pkl")
        
        train_args = Namespace(
            data=json.dumps(data),
            train_start=0, train_end=44, eval_start=45, eval_end=59,
            lookback=10, epochs=2, save_model=model_path, load_model=None,
            lstm_target_feature="sales", lstm_features=[],
            batch_size=4, lr=0.01, lstm_normalization="zscore", pretty=False
        )
        train_output = self.capture_run_output(train_args)
        self.assertEqual(train_output["mode"], "lstm_trained")
        self.assertTrue(os.path.exists(model_path))
        self.assertIn("rmse", train_output["metrics"])

        predict_args = Namespace(
            load_model=model_path, forecast_steps=10, data=None, pretty=False
        )
        predict_output = self.capture_run_output(predict_args)
        self.assertEqual(predict_output["mode"], "lstm_loaded_predict")
        self.assertEqual(len(predict_output["predictions"]), 10)

    def test_02_train_save_predict_multivariate(self):
        """Test end-to-end for a multivariate case with an extra feature."""
        data = create_test_data(70, keys=["y", "feature1"])
        model_path = os.path.join(self.tmpdir.name, "lstm_multivariate.pkl")
        
        train_args = Namespace(
            data=json.dumps(data),
            train_start=0, train_end=54, eval_start=55, eval_end=69,
            lookback=12, epochs=2, save_model=model_path, load_model=None,
            lstm_target_feature="y", lstm_features=["feature1"],
            batch_size=4, lr=0.01, lstm_normalization="minmax", pretty=False
        )
        train_output = self.capture_run_output(train_args)
        self.assertEqual(train_output["mode"], "lstm_trained")
        self.assertTrue(os.path.exists(model_path))

        predict_args = Namespace(
            load_model=model_path, forecast_steps=8, data=None, pretty=False
        )
        predict_output = self.capture_run_output(predict_args)
        self.assertEqual(predict_output["mode"], "lstm_loaded_predict")
        self.assertEqual(len(predict_output["predictions"]), 8)

    # 2. Validation and Error Handling
    def test_10_invalid_range_overlap(self):
        """Test that overlapping train/eval ranges raise an error."""
        data = json.dumps(create_test_data(40))
        args = Namespace(data=data, train_start=0, train_end=25, eval_start=20, eval_end=39, lookback=5, epochs=1,
                         lstm_target_feature="sales", lstm_features=["feature1"])
        with self.assertRaisesRegex(ValueError, "eval_start must be > train_end"):
            run(args)

    def test_11_data_too_short_for_lookback(self):
        """Test that an error is raised if data is shorter than lookback."""
        data = json.dumps(create_test_data(10))
        args = Namespace(data=data, train_start=0, train_end=4, eval_start=5, eval_end=9, lookback=12, epochs=1,
                         lstm_target_feature="sales", lstm_features=["feature1"])
        with self.assertRaisesRegex(ValueError, "Data length .* must be > lookback"):
            run(args)

    def test_12_missing_target_key(self):
        """Test that an error is raised if the target key is missing."""
        data = json.dumps(create_test_data(20, keys=["value", "feature1"]))
        args = Namespace(data=data, train_start=0, train_end=9, eval_start=10, eval_end=19, lookback=5, epochs=1,
                         lstm_target_feature="sales", lstm_features=["feature1"])
        with self.assertRaisesRegex(ValueError, "Provided or auto-detected target key 'sales' not in data"):
            run(args)

    def test_13_missing_feature_key(self):
        """Test that an error is raised if a feature key is missing."""
        data = json.dumps(create_test_data(20, keys=["sales", "other_feature"]))
        args = Namespace(data=data, train_start=0, train_end=9, eval_start=10, eval_end=19, lookback=5, epochs=1,
                         lstm_target_feature="sales", lstm_features=["feature1"]) # Expects 'feature1'
        with self.assertRaisesRegex(ValueError, "Error processing record"):
            run(args)

    def test_14_load_nonexistent_model(self):
        """Test that loading a non-existent model raises FileNotFoundError."""
        args = Namespace(load_model="/tmp/nonexistent.pkl", forecast_steps=5)
        with self.assertRaises(FileNotFoundError):
            run(args)

if __name__ == '__main__':
    unittest.main(verbosity=2)