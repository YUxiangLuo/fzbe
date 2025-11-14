#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive unit test suite for the refactored Exponential Smoothing (ES) model.
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

from models.es import run
import numpy as np

def create_test_data(n=50, key="sales"):
    """Helper to create test data with a specific key."""
    return [{key: 100 + i*5 + np.random.randn()} for i in range(n)]

class TestESComprehensiveRefactored(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def capture_run_output(self, args):
        """Run the script and capture its JSON output, handles empty output for errors."""
        output_io = io.StringIO()
        try:
            with redirect_stdout(output_io):
                run(args)
            output_str = output_io.getvalue()
            return json.loads(output_str) if output_str else None
        except ValueError:
            # Re-raise the ValueError to be caught by assertRaises
            raise

    # 1. Core Functionality
    def test_01_train_save_predict(self):
        """Test end-to-end: train, save, load, and predict."""
        data = create_test_data(40)
        model_path = os.path.join(self.tmpdir.name, "es_model.pkl")
        
        # Train
        train_args = Namespace(
            data=json.dumps(data),
            train_start=0, train_end=29, eval_start=30, eval_end=39,
            alpha=0.4, save_model=model_path, load_model=None, pretty=False
        )
        train_output = self.capture_run_output(train_args)
        self.assertEqual(train_output["mode"], "ses_fixed_alpha")
        self.assertTrue(os.path.exists(model_path))
        self.assertEqual(len(train_output["predictions"]), 10)

        # Predict
        predict_args = Namespace(
            load_model=model_path, forecast_steps=15, data=None,
            eval_start=None, eval_end=None, pretty=False
        )
        predict_output = self.capture_run_output(predict_args)
        self.assertEqual(predict_output["mode"], "es_loaded_predict")
        self.assertEqual(len(predict_output["predictions"]), 15)

    def test_02_training_with_empty_eval_range(self):
        """Test that training works correctly with an empty evaluation range."""
        data = create_test_data(30)
        model_path = os.path.join(self.tmpdir.name, "es_train_only.pkl")
        
        train_args = Namespace(
            data=json.dumps(data),
            train_start=0, train_end=29, eval_start=30, eval_end=29, # Empty range
            alpha=0.5, save_model=model_path, load_model=None, pretty=False
        )
        train_output = self.capture_run_output(train_args)
        self.assertEqual(train_output["mode"], "ses_fixed_alpha")
        self.assertTrue(os.path.exists(model_path))
        self.assertEqual(len(train_output["predictions"]), 0)
        self.assertEqual(train_output["metrics"], {})

    def test_03_predict_with_future_data_updates(self):
        """Test that the model level updates correctly when future data is provided."""
        train_data = [{"sales": 100}, {"sales": 110}]
        model_path = os.path.join(self.tmpdir.name, "es_future.pkl")
        
        train_args = Namespace(
            data=json.dumps(train_data),
            train_start=0, train_end=1, eval_start=2, eval_end=1,
            alpha=0.5, save_model=model_path, load_model=None, pretty=False
        )
        self.capture_run_output(train_args)

        future_data = [{"sales": 120}, {"sales": 130}]
        predict_args = Namespace(
            load_model=model_path, forecast_steps=2, data=json.dumps(future_data),
            eval_start=0, eval_end=1, pretty=False
        )
        output = self.capture_run_output(predict_args)
        
        # s0=100. s1 (saved level) = 0.5*110 + 0.5*100 = 105.
        # 1st prediction is 105.
        self.assertAlmostEqual(output["predictions"][0]["prediction"], 105.0)
        # Level updates with future_data[0]: s2 = 0.5*120 + 0.5*105 = 112.5
        # 2nd prediction is 112.5.
        self.assertAlmostEqual(output["predictions"][1]["prediction"], 112.5)

    def test_04_data_with_y_key(self):
        """Test that the model works correctly when the target key is 'y'."""
        data = create_test_data(20, key="y")
        args = Namespace(
            data=json.dumps(data),
            train_start=0, train_end=9, eval_start=10, eval_end=19,
            alpha=0.5, save_model=None, load_model=None, pretty=False
        )
        output = self.capture_run_output(args)
        self.assertEqual(output["mode"], "ses_fixed_alpha")
        self.assertIn("metrics", output)

    # 2. Validation and Error Handling
    def test_10_invalid_alpha_value(self):
        """Test that alpha values outside (0, 1] raise an error."""
        data = json.dumps(create_test_data(20))
        with self.assertRaisesRegex(ValueError, "alpha must be in"):
            run(Namespace(data=data, train_start=0, train_end=9, eval_start=10, eval_end=19, alpha=1.1))
        with self.assertRaisesRegex(ValueError, "alpha must be in"):
            run(Namespace(data=data, train_start=0, train_end=9, eval_start=10, eval_end=19, alpha=0))
        with self.assertRaisesRegex(ValueError, "alpha must be in"):
            run(Namespace(data=data, train_start=0, train_end=9, eval_start=10, eval_end=19, alpha=-0.5))

    def test_11_invalid_range_overlap(self):
        """Test that overlapping train/eval ranges raise an error."""
        data = json.dumps(create_test_data(20))
        args = Namespace(data=data, train_start=0, train_end=15, eval_start=10, eval_end=19, alpha=0.5)
        with self.assertRaisesRegex(ValueError, "eval_start must be > train_end"):
            run(args)

    def test_12_invalid_range_out_of_bounds(self):
        """Test that out-of-bounds ranges raise an error."""
        data = json.dumps(create_test_data(20))
        args = Namespace(data=data, train_start=0, train_end=15, eval_start=16, eval_end=25, alpha=0.5)
        with self.assertRaisesRegex(ValueError, "Invalid eval range"):
            run(args)

    def test_13_missing_target_key(self):
        """Test that data missing 'y' or 'sales' raises an error."""
        data = json.dumps([{"value": 100}])
        args = Namespace(data=data, train_start=0, train_end=0, eval_start=1, eval_end=0, alpha=0.5)
        with self.assertRaisesRegex(ValueError, "Cannot determine target key"):
            run(args)

    def test_14_invalid_json(self):
        """Test that malformed JSON raises an error."""
        data = '{"sales": 100}' # Not an array
        args = Namespace(data=data, train_start=0, train_end=0, eval_start=1, eval_end=0, alpha=0.5)
        with self.assertRaisesRegex(ValueError, "Failed to parse --data"):
            run(args)

    def test_15_load_nonexistent_model(self):
        """Test that loading a non-existent model raises FileNotFoundError."""
        args = Namespace(load_model="/tmp/nonexistent.pkl", forecast_steps=5)
        with self.assertRaises(FileNotFoundError):
            run(args)

if __name__ == '__main__':
    unittest.main(verbosity=2)
