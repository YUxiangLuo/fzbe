#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal unit test for the refactored ARIMA model.
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

class TestARIMAMinimal(unittest.TestCase):

    def test_end_to_end(self):
        """A minimal end-to-end test for the refactored arima.py."""
        # 1. Create data
        data = [{"y": 200 + i*10 + 5*np.sin(i/4) + np.random.randn()*2} for i in range(50)]
        data_json = json.dumps(data)

        # 2. Define args for training
        ts, te = 0, 39
        es, ee = 40, 49
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "arima_minimal.pkl")
            
            train_args = Namespace(
                data=data_json,
                train_start=ts, train_end=te, eval_start=es, eval_end=ee,
                d=1, p_max=1, q_max=1,
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
            self.assertEqual(train_output["mode"], "arima_search_pq_fixed_d")
            self.assertTrue(os.path.exists(model_path))
            self.assertIn("rmse", train_output["metrics"])

            # 5. Define args for prediction
            predict_args = Namespace(
                load_model=model_path,
                forecast_steps=12,
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
            self.assertEqual(predict_output["mode"], "arima_loaded_predict")
            self.assertEqual(len(predict_output["predictions"]), 12)

if __name__ == '__main__':
    unittest.main(verbosity=2)
