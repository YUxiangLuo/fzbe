#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive unit test suite for the refactored Moving Average (MA) model.
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

from models.ma import run
import numpy as np

def create_test_data(n=50, key="sales"):
    """Helper to create test data."""
    return [{key: 100 + i*2 + np.random.randn()} for i in range(n)]

class TestMAComprehensiveRefactored(unittest.TestCase):

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
    def test_01_train_save_predict(self):
        """Test end-to-end: train, save, load, and predict."""
        data = create_test_data(40)
        model_path = os.path.join(self.tmpdir.name, "ma_model.pkl")
        
        train_args = Namespace(
            data=json.dumps(data),
            train_start=0, train_end=29, eval_start=30, eval_end=39,
            window=5, save_model=model_path, load_model=None, pretty=False
        )
        train_output = self.capture_run_output(train_args)
        self.assertEqual(train_output["mode"], "fixed_window_only")
        self.assertTrue(os.path.exists(model_path))
        self.assertEqual(len(train_output["predictions"]), 10)

        predict_args = Namespace(
            load_model=model_path, forecast_steps=15, data=None, pretty=False
        )
        predict_output = self.capture_run_output(predict_args)
        self.assertEqual(predict_output["mode"], "ma_loaded_predict")
        self.assertEqual(len(predict_output["predictions"]), 15)

    def test_02_training_with_empty_eval_range(self):
        """Test that training with an empty evaluation range works correctly."""
        data = create_test_data(30)
        model_path = os.path.join(self.tmpdir.name, "ma_train_only.pkl")
        
        train_args = Namespace(
            data=json.dumps(data),
            train_start=0, train_end=29, eval_start=30, eval_end=29, # Empty range
            window=5, save_model=model_path, load_model=None, pretty=False
        )
        train_output = self.capture_run_output(train_args)
        self.assertEqual(train_output["mode"], "fixed_window_only")
        self.assertTrue(os.path.exists(model_path))
        self.assertEqual(len(train_output["predictions"]), 0)
        self.assertEqual(train_output["metrics"], {})

    def test_03_data_with_y_key(self):
        """Test that the model works correctly when the target key is 'y'."""
        data = create_test_data(20, key="y")
        args = Namespace(
            data=json.dumps(data),
            train_start=0, train_end=9, eval_start=10, eval_end=19,
            window=3, save_model=None, load_model=None, pretty=False
        )
        output = self.capture_run_output(args)
        self.assertEqual(output["mode"], "fixed_window_only")
        self.assertIn("metrics", output)

    # 2. Validation and Error Handling
    def test_10_invalid_window_size(self):
        """Test that a window size <= 0 raises an error."""
        data = json.dumps(create_test_data(20))
        with self.assertRaisesRegex(ValueError, "--window must be >= 1"):
            run(Namespace(data=data, train_start=0, train_end=9, eval_start=10, eval_end=19, window=0))

    def test_11_invalid_range_overlap(self):
        """Test that overlapping train/eval ranges raise an error."""
        data = json.dumps(create_test_data(20))
        args = Namespace(data=data, train_start=0, train_end=15, eval_start=10, eval_end=19, window=5)
        with self.assertRaisesRegex(ValueError, "Evaluation start must be after training end"):
            run(args)

    def test_12_insufficient_history_for_eval(self):
        """Test that an error is raised if no evaluation points can be generated."""
        data = json.dumps(create_test_data(20))
        # train_start=0, window=7. First possible eval point is index 7.
        # eval range is [5, 6], which is entirely before index 7.
        args = Namespace(data=data, train_start=0, train_end=4, eval_start=5, eval_end=6, window=7)
        with self.assertRaisesRegex(ValueError, "Insufficient history for all evaluation points"):
            run(args)

    def test_13_missing_target_key(self):
        """Test that data missing 'y' or 'sales' raises an error."""
        data = json.dumps([{"value": 100}])
        args = Namespace(data=data, train_start=0, train_end=0, eval_start=1, eval_end=0, window=1)
        with self.assertRaisesRegex(ValueError, "Cannot determine target key"):
            run(args)

    def test_14_invalid_json(self):
        """Test that malformed JSON raises an error."""
        data = '{"sales": 100}' # Not an array
        args = Namespace(data=data, train_start=0, train_end=0, eval_start=1, eval_end=0, window=1)
        with self.assertRaisesRegex(ValueError, "Failed to parse --data"):
            run(args)

    def test_15_load_nonexistent_model(self):
        """Test that loading a non-existent model raises FileNotFoundError."""
        args = Namespace(load_model="/tmp/nonexistent.pkl", forecast_steps=5)
        with self.assertRaises(FileNotFoundError):
            run(args)

if __name__ == '__main__':
    unittest.main(verbosity=2)
