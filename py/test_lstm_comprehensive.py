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

def create_test_data(n=60, keys=["sales", "feature1"], categorical=False):
    """Helper to create test data."""
    data = []
    for i in range(n):
        record = {
            keys[0]: 150 + i*3 + 8*np.sin(i/5) + np.random.randn()
        }
        if len(keys) > 1:
            record[keys[1]] = 50 + i*0.5 + 3*np.cos(i/3) + np.random.randn()
        if categorical:
            record["day_of_week"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i % 7]
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
            # Filter out potential non-JSON warning lines from TF
            json_lines = [line for line in output_str.strip().split('\n') if line.startswith('{')]
            if not json_lines:
                # Handle cases where output might be just warnings or errors
                if "ValueError" in output_str or "FileNotFoundError" in output_str:
                    raise Exception(output_str) # Propagate error for assertion
                return None
            # In case of multiple JSON outputs, parse the last one
            return json.loads(json_lines[-1])
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error capturing output: {e}\nOutput was:\n{output_io.getvalue()}")
            raise

    # 1. Core Functionality
    def test_01_train_save_predict_univariate(self):
        """Test end-to-end: train, save, load, and predict for a univariate case."""
        data = create_test_data(60, keys=["sales"])
        model_path = os.path.join(self.tmpdir.name, "lstm_univariate.pkl")
        
        train_args = Namespace(
            data=json.dumps(data),
            train_start=0, train_end=44, eval_start=45, eval_end=59,
            look_back=10, epochs=2, save_model=model_path, load_model=None,
            pretty=False, feature_types='{"sales": "numeric"}'
        )
        train_output = self.capture_run_output(train_args)
        self.assertEqual(train_output["mode"], "lstm_train_eval")
        self.assertTrue(os.path.exists(model_path))
        self.assertIn("rmse", train_output["metrics"])
        self.assertEqual(len(train_output["predictions"]), 15)

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
            look_back=12, epochs=2, save_model=model_path, load_model=None,
            pretty=False, lstm_target_feature="y",
            feature_types='{"y": "numeric", "feature1": "numeric"}'
        )
        train_output = self.capture_run_output(train_args)
        self.assertEqual(train_output["mode"], "lstm_train_eval")
        self.assertTrue(os.path.exists(model_path))

        predict_args = Namespace(
            load_model=model_path, forecast_steps=8, data=None, pretty=False
        )
        predict_output = self.capture_run_output(predict_args)
        self.assertEqual(predict_output["mode"], "lstm_loaded_predict")
        self.assertEqual(len(predict_output["predictions"]), 8)

    def test_03_train_with_categorical_feature(self):
        """Test end-to-end with a mix of numeric and categorical features."""
        data = create_test_data(60, keys=["sales", "promo_spend"], categorical=True)
        model_path = os.path.join(self.tmpdir.name, "lstm_categorical.pkl")
        
        feature_types = json.dumps({
            "sales": "numeric",
            "promo_spend": "numeric",
            "day_of_week": "categorical"
        })

        train_args = Namespace(
            data=json.dumps(data),
            train_start=0, train_end=44, eval_start=45, eval_end=59,
            look_back=5, epochs=2, save_model=model_path, load_model=None,
            pretty=False, feature_types=feature_types
        )
        train_output = self.capture_run_output(train_args)
        self.assertEqual(train_output["mode"], "lstm_train_eval")
        self.assertIn("day_of_week", train_output["feature_types"])
        self.assertEqual(train_output["feature_types"]["day_of_week"], "categorical")

        predict_args = Namespace(
            load_model=model_path, forecast_steps=10, data=None, pretty=False
        )
        predict_output = self.capture_run_output(predict_args)
        self.assertEqual(predict_output["mode"], "lstm_loaded_predict")
        self.assertEqual(len(predict_output["predictions"]), 10)


    # 2. Validation and Error Handling
    def test_10_invalid_range_overlap(self):
        """Test that overlapping train/eval ranges raise an error."""
        data = json.dumps(create_test_data(40))
        args = Namespace(data=data, train_start=0, train_end=25, eval_start=20, eval_end=39, look_back=5, epochs=1, feature_types='{}')
        with self.assertRaisesRegex(ValueError, "eval_start must be > train_end"):
            run(args)

    def test_11_data_too_short_for_lookback(self):
        """Test that an error is raised if data is shorter than lookback."""
        data = json.dumps(create_test_data(10))
        args = Namespace(data=data, train_start=0, train_end=4, eval_start=5, eval_end=9, look_back=12, epochs=1, feature_types='{}')
        with self.assertRaisesRegex(ValueError, "Training range must be larger than look_back"):
            run(args)

    def test_12_missing_target_key(self):
        """Test that an error is raised if the target key is missing."""
        data = json.dumps(create_test_data(20, keys=["value", "feature1"]))
        args = Namespace(data=data, train_start=0, train_end=9, eval_start=10, eval_end=19, look_back=5, epochs=1, feature_types='{}')
        with self.assertRaisesRegex(ValueError, "Cannot determine target key"):
            run(args)

    def test_13_missing_feature_key(self):
        """Test that an error is raised if a feature key is missing in one record."""
        data = create_test_data(20, keys=["sales", "other_feature"])
        # Make the key missing in just one record
        del data[5]["other_feature"]
        
        args = Namespace(data=json.dumps(data), train_start=0, train_end=9, eval_start=10, eval_end=19, look_back=5, epochs=1, feature_types='{}')
        with self.assertRaisesRegex(ValueError, r"\[DATA\] Failed to parse --data: Error processing record.*(missing key|Key mismatch)"):
            run(args)

    def test_14_load_nonexistent_model(self):
        """Test that loading a non-existent model raises FileNotFoundError."""
        args = Namespace(load_model="/tmp/nonexistent.pkl", forecast_steps=5)
        with self.assertRaises(FileNotFoundError):
            run(args)

if __name__ == '__main__':
    unittest.main(verbosity=2)