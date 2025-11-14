#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal unit test for the refactored LSTM model.
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

class TestLSTMMinimal(unittest.TestCase):

    def test_end_to_end(self):
        """A minimal end-to-end test for the refactored lstm.py."""
        # 1. Create data
        data = [{"sales": 150 + i*3 + 8*np.sin(i/5) + np.random.randn()} for i in range(60)]
        data_json = json.dumps(data)

        # 2. Define args for training
        ts, te = 0, 44
        es, ee = 45, 59
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "lstm_minimal.pkl")
            
            train_args = Namespace(
                data=data_json,
                train_start=ts, train_end=te, eval_start=es, eval_end=ee,
                lookback=10, epochs=2, # Use minimal epochs for speed
                save_model=model_path, pretty=False,
                load_model=None,
                lstm_target_feature="sales", lstm_features=[],
                batch_size=4, lr=0.01, lstm_normalization="zscore"
            )

            # 3. Run training and capture output
            output_io = io.StringIO()
            with redirect_stdout(output_io):
                run(train_args)
            output_str = output_io.getvalue()
            
            self.assertTrue(output_str, "Training run produced no output.")
            train_output = json.loads(output_str)

            # 4. Assert training results
            self.assertEqual(train_output["mode"], "lstm_trained")
            self.assertTrue(os.path.exists(model_path))
            self.assertIn("rmse", train_output["metrics"])

            # 5. Define args for prediction
            predict_args = Namespace(
                load_model=model_path,
                forecast_steps=10,
                data=None, pretty=False
            )

            # 6. Run prediction and capture output
            predict_io = io.StringIO()
            with redirect_stdout(predict_io):
                run(predict_args)
            predict_str = predict_io.getvalue()

            self.assertTrue(predict_str, "Prediction run produced no output.")
            predict_output = json.loads(predict_str)

            # 7. Assert prediction results
            self.assertEqual(predict_output["mode"], "lstm_loaded_predict")
            self.assertEqual(len(predict_output["predictions"]), 10)

if __name__ == '__main__':
    unittest.main(verbosity=2)
