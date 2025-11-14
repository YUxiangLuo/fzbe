#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal unit test for the refactored Moving Average (MA) model.
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

class TestMAMinimal(unittest.TestCase):

    def test_end_to_end(self):
        """A minimal end-to-end test for the refactored ma.py."""
        # 1. Create data
        data = [{"sales": 100 + i*2 + np.random.randn()} for i in range(30)]
        data_json = json.dumps(data)

        # 2. Define args for training
        ts, te = 0, 19
        es, ee = 20, 29
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "ma_minimal.pkl")
            
            train_args = Namespace(
                data=data_json,
                train_start=ts, train_end=te, eval_start=es, eval_end=ee,
                window=5,
                save_model=model_path, pretty=False,
                load_model=None
            )

            # 3. Run training and capture output
            output_io = io.StringIO()
            with redirect_stdout(output_io):
                run(train_args)
            output_str = output_io.getvalue()
            
            self.assertTrue(output_str, "Training run produced no output.")
            train_output = json.loads(output_str)

            # 4. Assert training results
            self.assertEqual(train_output["mode"], "fixed_window_only")
            self.assertTrue(os.path.exists(model_path))
            self.assertIn("rmse", train_output["metrics"])

            # 5. Define args for prediction
            predict_args = Namespace(
                load_model=model_path,
                forecast_steps=8,
                data=None, pretty=False,
                eval_start=None, eval_end=None
            )

            # 6. Run prediction and capture output
            predict_io = io.StringIO()
            with redirect_stdout(predict_io):
                run(predict_args)
            predict_str = predict_io.getvalue()

            self.assertTrue(predict_str, "Prediction run produced no output.")
            predict_output = json.loads(predict_str)

            # 7. Assert prediction results
            self.assertEqual(predict_output["mode"], "ma_loaded_predict")
            self.assertEqual(len(predict_output["predictions"]), 8)

if __name__ == '__main__':
    unittest.main(verbosity=2)
