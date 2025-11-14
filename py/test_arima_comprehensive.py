#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive unit test suite for the refactored ARIMA model.
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

from models.arima import run
import numpy as np

def create_test_data(n=60, key="y"):
    """Helper to create test data."""
    return [{key: 200 + i*10 + 5*np.sin(i/4) + np.random.randn()*2} for i in range(n)]

class TestARIMAComprehensiveRefactored(unittest.TestCase):

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
    def test_01_train_search_save_predict(self):
        """Test end-to-end: train, search, save, load, and predict."""
        data = create_test_data(50)
        model_path = os.path.join(self.tmpdir.name, "arima_model.pkl")
        
        # Train and search
        train_args = Namespace(
            data=json.dumps(data),
            train_start=0, train_end=39, eval_start=40, eval_end=49,
            d=1, p_max=2, q_max=2,
            save_model=model_path, load_model=None, pretty=False
        )
        train_output = self.capture_run_output(train_args)
        self.assertEqual(train_output["mode"], "arima_search_pq_fixed_d")
        self.assertTrue(os.path.exists(model_path))
        self.assertIn("rmse", train_output["metrics"])
        self.assertEqual(len(train_output["eval_predictions"]), 10)

        # Predict
        predict_args = Namespace(
            load_model=model_path, forecast_steps=15, data=None, pretty=False
        )
        predict_output = self.capture_run_output(predict_args)
        self.assertEqual(predict_output["mode"], "arima_loaded_predict")
        self.assertEqual(len(predict_output["predictions"]), 15)

    def test_02_data_with_sales_key(self):
        """Test that the model works correctly when the target key is 'sales'."""
        data = create_test_data(40, key="sales")
        args = Namespace(
            data=json.dumps(data),
            train_start=0, train_end=29, eval_start=30, eval_end=39,
            d=1, p_max=1, q_max=1,
            save_model=None, load_model=None, pretty=False
        )
        output = self.capture_run_output(args)
        self.assertEqual(output["mode"], "arima_search_pq_fixed_d")
        self.assertIn("metrics", output)

    def test_03_predict_with_evaluation_data(self):
        """Test prediction with new data provided for metrics calculation."""
        train_data = create_test_data(40)
        model_path = os.path.join(self.tmpdir.name, "arima_eval.pkl")
        
        train_args = Namespace(
            data=json.dumps(train_data),
            train_start=0, train_end=29, eval_start=30, eval_end=39,
            d=1, p_max=1, q_max=1,
            save_model=model_path, load_model=None, pretty=False
        )
        self.capture_run_output(train_args)

        eval_data = create_test_data(10)
        predict_args = Namespace(
            load_model=model_path, forecast_steps=10,
            data=json.dumps(eval_data), pretty=False
        )
        output = self.capture_run_output(predict_args)
        self.assertEqual(output["mode"], "arima_loaded_predict")
        self.assertIn("metrics", output)
        self.assertNotIn("metrics_error", output)

    # 2. Validation and Error Handling
    def test_10_invalid_range_overlap(self):
        """Test that overlapping train/eval ranges raise an error."""
        data = json.dumps(create_test_data(30))
        args = Namespace(data=data, train_start=0, train_end=20, eval_start=15, eval_end=29, d=1)
        with self.assertRaisesRegex(ValueError, "eval_start must be > train_end"):
            run(args)

    def test_11_invalid_range_out_of_bounds(self):
        """Test that out-of-bounds ranges raise an error."""
        data = json.dumps(create_test_data(30))
        args = Namespace(data=data, train_start=0, train_end=19, eval_start=20, eval_end=35, d=1)
        with self.assertRaisesRegex(ValueError, "eval range invalid"):
            run(args)

    def test_12_missing_target_key(self):
        """Test that data missing 'y' or 'sales' raises an error."""
        data = json.dumps([{"value": 100}])
        args = Namespace(data=data, train_start=0, train_end=0, eval_start=1, eval_end=0, d=1)
        with self.assertRaisesRegex(ValueError, "Cannot determine target key"):
            run(args)

    def test_13_invalid_json(self):
        """Test that malformed JSON raises an error."""
        data = '{"sales": 100}' # Not an array
        args = Namespace(data=data, train_start=0, train_end=0, eval_start=1, eval_end=0, d=1)
        with self.assertRaisesRegex(ValueError, "JSON data must be a non-empty array"):
            run(args)

    def test_14_load_nonexistent_model(self):
        """Test that loading a non-existent model raises FileNotFoundError."""
        args = Namespace(load_model="/tmp/nonexistent.pkl", forecast_steps=5)
        with self.assertRaises(FileNotFoundError):
            run(args)
            
    def test_15_all_fits_fail(self):
        """Test that an error is raised if all ARIMA fits fail."""
        # Data that is too short will cause all fits to fail
        data = json.dumps(create_test_data(5))
        args = Namespace(data=data, train_start=0, train_end=1, eval_start=2, eval_end=4, d=1, p_max=2, q_max=2)
        with self.assertRaisesRegex(ValueError, "All \(p,q\) fits failed"):
            run(args)

if __name__ == '__main__':
    unittest.main(verbosity=2)
