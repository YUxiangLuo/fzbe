# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for es.py (Exponential Smoothing model)
Tests cover: core functions, prediction, training, error handling, and integration scenarios
"""

import json
import os
import sys
import tempfile
import unittest
from typing import Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import numpy as np

# Import from es module
from src.models.es import (
    MODEL_TYPE,
    _add_metrics_to_output,
    _fit_level,
    _load_state,
    _validate_training_args,
    predict,
    run,
    train_and_evaluate,
)


class TestFitLevel(unittest.TestCase):
    """Test _fit_level function."""

    def test_normal_fit(self):
        series = np.array([1, 2, 3, 4, 5])
        alpha = 0.5
        level = _fit_level(series, range(0, 5), alpha)
        expected_level = 1.0  # Manually calculate: after updates
        # s0 = 1
        # s1 = 0.5*2 + 0.5*1 = 1.5
        # s2 = 0.5*3 + 0.5*1.5 = 2.25
        # s3 = 0.5*4 + 0.5*2.25 = 3.125
        # s4 = 0.5*5 + 0.5*3.125 = 4.0625
        self.assertAlmostEqual(level, 4.0625)

    def test_empty_train_range_raises_error(self):
        series = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            _fit_level(series, range(0, 0), 0.5)


class TestPredictFunction(unittest.TestCase):
    """Test predict function."""

    def setUp(self):
        from serialization import dump_model

        self.state = {
            "model_type": MODEL_TYPE,
            "alpha": 0.5,
            "train_end_index": 4,
            "level": 4.0625,
            "value_key": "sales",
            "std_dev_residuals": 0.5,
        }
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        dump_model(self.state, self.temp_file.name)

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_predict_without_data(self):
        preds, std_dev = predict(self.temp_file.name, 2)
        expected_preds = [4.0625, 4.0625]  # Same level for all steps
        self.assertEqual(preds, expected_preds)
        self.assertEqual(std_dev, 0.5)

    def test_predict_with_future_data(self):
        data = json.dumps([{"sales": 6}, {"sales": 7}])
        preds, _ = predict(self.temp_file.name, 2, data=data)
        # First prediction: level=4.0625, then update with 6: new_level = 0.5*6 + 0.5*4.0625 = 5.03125
        self.assertEqual(preds, [4.0625, 5.03125])


class TestTrainAndEvaluate(unittest.TestCase):
    """Test train_and_evaluate function."""

    def test_normal_training(self):
        sales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        months = [str(i) for i in range(10)]
        train_range = range(0, 5)
        eval_range = range(5, 10)
        metrics, std_dev, level, y_true, y_pred = train_and_evaluate(
            sales, months, train_range, eval_range, 0.5
        )
        self.assertIn("rmse", metrics)
        self.assertGreater(std_dev, 0)
        self.assertEqual(len(y_true), len(y_pred))
        self.assertEqual(level, 4.0625)


class TestRunFunction(unittest.TestCase):
    """Test run function."""

    def setUp(self):
        self.temp_save_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_save_file.close()

    def tearDown(self):
        os.unlink(self.temp_save_file.name)

    def _capture_run_output(self, args):
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            run(args)
        return json.loads(f.getvalue())

    def test_training_mode_basic(self):
        data = json.dumps([{"sales": i, "month": f"2023-{i:02d}"} for i in range(10)])
        args = type(
            "Args",
            (),
            {
                "data": data,
                "train_start": 0,
                "train_end": 4,
                "eval_start": 5,
                "eval_end": 9,
                "alpha": 0.5,
                "save_model": None,
                "pretty": False,
            },
        )()
        out = self._capture_run_output(args)
        self.assertEqual(out["mode"], "ses_fixed_alpha")
        self.assertIn("metrics", out)


class TestValidation(unittest.TestCase):
    """Test validation functions."""

    def test_validate_training_args_normal(self):
        sales = [1, 2, 3, 4, 5]
        _validate_training_args(sales, 0, 2, 3, 4)  # Should not raise

    def test_validate_invalid_train_range(self):
        sales = [1, 2, 3]
        with self.assertRaises(ValueError):
            _validate_training_args(sales, 0, 5, 1, 2)

    def test_alpha_zero_raises_error(self):
        from src.models.es import run

        data = json.dumps([{"sales": 1}, {"sales": 2}])
        args = type(
            "Args",
            (),
            {
                "data": data,
                "train_start": 0,
                "train_end": 0,
                "eval_start": 1,
                "eval_end": 1,
                "alpha": 0.0,
                "pretty": False,
            },
        )()
        with self.assertRaises(ValueError):
            run(args)


if __name__ == "__main__":
    unittest.main()
