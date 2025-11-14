#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal unit test for the Stacking ensemble model.
"""
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
import io

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from models.stacking import run
import numpy as np

def create_test_data(n=60):
    """Helper to create test data."""
    data = []
    for i in range(n):
        record = {
            "sales": 150 + i * 3 + 8 * np.sin(i / 5) + np.random.randn(),
            "feature1": 50 + i * 0.5 + 3 * np.cos(i / 3) + np.random.randn(),
            "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i % 7]
        }
        data.append(record)
    return data

class TestStackingMinimal(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        # Suppress TensorFlow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def capture_run_output(self, args):
        """Run the script and capture its JSON output."""
        output_io = io.StringIO()
        try:
            with redirect_stdout(output_io):
                run(args)
            output_str = output_io.getvalue()
            json_lines = [line for line in output_str.strip().split('\n') if line.startswith('{')]
            if not json_lines:
                return None
            return json.loads(json_lines[-1])
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error capturing output: {e}\nOutput was:\n{output_io.getvalue()}")
            raise

    def test_01_stacking_end_to_end(self):
        """Test the full train -> save -> load -> predict workflow for the Stacking model."""
        data = create_test_data(60)
        model_path = os.path.join(self.tmpdir.name, "stacking_model.pkl")
        
        lstm_feature_types = json.dumps({
            "sales": "numeric",
            "feature1": "numeric",
            "day_of_week": "categorical"
        })

        train_args = Namespace(
            data=json.dumps(data),
            train_start=0, train_end=44, eval_start=45, eval_end=59,
            save_model=model_path,
            load_model=None,
            pretty=False,
            # Stacking specific
            models="ma,lstm",
            # Base model params
            arima_order="1,1,1", # Still provide all, though only used if in models list
            es_alpha="0.5",
            ma_window="5",
            lstm_look_back="5",
            lstm_epochs="1",
            lstm_feature_types=lstm_feature_types
        )
        
        train_output = self.capture_run_output(train_args)
        self.assertIsNotNone(train_output)
        self.assertEqual(train_output["mode"], "stacking_train_eval")
        self.assertEqual(train_output["models"], ["ma", "lstm"])
        self.assertTrue(os.path.exists(model_path))
        self.assertIn("rmse", train_output["metrics"])
        self.assertEqual(len(train_output["predictions"]), 15)

        predict_args = Namespace(
            load_model=model_path,
            forecast_steps=10,
            data=None,
            pretty=False
        )
        
        predict_output = self.capture_run_output(predict_args)
        self.assertIsNotNone(predict_output)
        self.assertEqual(predict_output["mode"], "stacking_loaded_predict")
        self.assertEqual(len(predict_output["predictions"]), 10)

if __name__ == '__main__':
    unittest.main(verbosity=2)
