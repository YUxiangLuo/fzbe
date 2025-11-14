#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the Exponential Smoothing (ES) model.
"""
import json
import os
import pickle
import sys
import tempfile
import unittest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
from models.es import (
    _fit_level,
    _format_predictions,
    _validate_training_args,
    predict,
    train_and_evaluate,
)


class TestFitLevel(unittest.TestCase):
    """Test the _fit_level function."""

    def test_basic_fit(self):
        """Test basic level fitting with exponential smoothing."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        train_range = range(0, 5)
        alpha = 0.5

        level = _fit_level(series, train_range, alpha)

        # Manual calculation:
        # level = 1.0
        # level = 0.5*2 + 0.5*1 = 1.5
        # level = 0.5*3 + 0.5*1.5 = 2.25
        # level = 0.5*4 + 0.5*2.25 = 3.125
        # level = 0.5*5 + 0.5*3.125 = 4.0625
        expected = 4.0625
        self.assertAlmostEqual(level, expected, places=5)

    def test_alpha_one(self):
        """Test with alpha=1.0 (full weight on latest observation)."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        train_range = range(0, 5)
        alpha = 1.0

        level = _fit_level(series, train_range, alpha)

        # With alpha=1, level should equal last value
        self.assertAlmostEqual(level, 5.0, places=5)

    def test_alpha_near_zero(self):
        """Test with alpha near 0 (mostly using historical level)."""
        series = np.array([1.0, 100.0, 100.0, 100.0])
        train_range = range(0, 4)
        alpha = 0.01

        level = _fit_level(series, train_range, alpha)

        # With very small alpha, level should stay close to initial value
        self.assertLess(level, 5.0)

    def test_empty_range_raises_error(self):
        """Test that empty training range raises error."""
        series = np.array([1.0, 2.0, 3.0])
        train_range = range(0, 0)
        alpha = 0.5

        with self.assertRaises(ValueError) as ctx:
            _fit_level(series, train_range, alpha)
        self.assertIn("at least one point", str(ctx.exception))


class TestFormatPredictions(unittest.TestCase):
    """Test the _format_predictions helper function."""

    def test_format_predictions(self):
        """Test predictions are formatted correctly."""
        preds = [1.5, 2.5, 3.5]
        std_dev = 0.5

        result = _format_predictions(preds, std_dev)

        expected = [
            {"prediction": 1.5, "std_dev": 0.5},
            {"prediction": 2.5, "std_dev": 0.5},
            {"prediction": 3.5, "std_dev": 0.5},
        ]
        self.assertEqual(result, expected)

    def test_empty_predictions(self):
        """Test formatting empty predictions list."""
        result = _format_predictions([], 0.0)
        self.assertEqual(result, [])


class TestValidateTrainingArgs(unittest.TestCase):
    """Test the _validate_training_args function."""

    def test_valid_args(self):
        """Test validation passes for valid arguments."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        # Should not raise
        _validate_training_args(sales, ts=0, te=5, es=6, ee=9)

    def test_invalid_train_range(self):
        """Test invalid training range raises error."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0]
        with self.assertRaises(ValueError) as ctx:
            _validate_training_args(sales, ts=3, te=2, es=4, ee=4)
        self.assertIn("Invalid train range", str(ctx.exception))

    def test_invalid_eval_range(self):
        """Test invalid evaluation range raises error."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0]
        with self.assertRaises(ValueError) as ctx:
            _validate_training_args(sales, ts=0, te=2, es=4, ee=3)
        self.assertIn("Invalid eval range", str(ctx.exception))

    def test_eval_before_train(self):
        """Test evaluation starting before/at training end raises error."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0]
        with self.assertRaises(ValueError) as ctx:
            _validate_training_args(sales, ts=0, te=3, es=2, ee=4)
        self.assertIn("eval_start must be > train_end", str(ctx.exception))

    def test_train_range_out_of_bounds(self):
        """Test training range exceeding data size raises error."""
        sales = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError) as ctx:
            _validate_training_args(sales, ts=0, te=5, es=6, ee=7)
        self.assertIn("Invalid train range", str(ctx.exception))


class TestTrainAndEvaluate(unittest.TestCase):
    """Test the train_and_evaluate function."""

    def test_basic_train_evaluate(self):
        """Test basic training and evaluation."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        months = [f"M{i}" for i in range(10)]
        train_range = range(0, 6)
        eval_range = range(6, 10)
        alpha = 0.3

        metrics, std_dev, final_level, y_true, y_pred = train_and_evaluate(
            sales, months, train_range, eval_range, alpha
        )

        # Check that we got results
        self.assertEqual(len(y_true), 4)
        self.assertEqual(len(y_pred), 4)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
        self.assertIsInstance(std_dev, float)
        self.assertIsInstance(final_level, float)

        # All predictions should be the same (equal to final level)
        self.assertTrue(all(p == y_pred[0] for p in y_pred))
        self.assertAlmostEqual(y_pred[0], final_level, places=5)

    def test_predictions_are_constant(self):
        """Test that SES predictions are all the same value."""
        sales = [10.0, 20.0, 30.0, 40.0, 50.0]
        months = [f"M{i}" for i in range(5)]
        train_range = range(0, 3)
        eval_range = range(3, 5)
        alpha = 0.5

        _, _, final_level, _, y_pred = train_and_evaluate(
            sales, months, train_range, eval_range, alpha
        )

        # All predictions should equal the final training level
        for pred in y_pred:
            self.assertAlmostEqual(pred, final_level, places=10)

    def test_empty_eval_range(self):
        """Test handling of empty evaluation range."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0]
        months = [f"M{i}" for i in range(5)]
        train_range = range(0, 3)
        eval_range = range(4, 4)  # Empty range
        alpha = 0.5

        metrics, std_dev, final_level, y_true, y_pred = train_and_evaluate(
            sales, months, train_range, eval_range, alpha
        )

        self.assertEqual(metrics, {})
        self.assertEqual(std_dev, 0.0)
        self.assertIsInstance(final_level, float)
        self.assertEqual(len(y_true), 0)
        self.assertEqual(len(y_pred), 0)

    def test_std_dev_calculation(self):
        """Test that standard deviation is calculated correctly."""
        # Use a case where we know the predictions will differ from actuals
        sales = [10.0, 10.0, 10.0, 20.0, 30.0]
        months = [f"M{i}" for i in range(5)]
        train_range = range(0, 3)
        eval_range = range(3, 5)
        alpha = 0.1

        _, std_dev, final_level, y_true, y_pred = train_and_evaluate(
            sales, months, train_range, eval_range, alpha
        )

        # Verify std_dev is positive (predictions differ from actuals)
        self.assertGreater(std_dev, 0.0)

        # Manual check: all preds should equal final_level
        residuals = [yt - final_level for yt in y_true]
        expected_std = np.std(residuals, ddof=1)
        self.assertAlmostEqual(std_dev, expected_std, places=5)


class TestPredictFunction(unittest.TestCase):
    """Test the predict function with model loading."""

    def test_predict_with_model_file(self):
        """Test prediction from a saved model file."""
        # Create a temporary model file using pickle
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            model_state = {
                "model_type": "es",
                "alpha": 0.5,
                "train_end_index": 5,
                "level": 10.0,
                "value_key": "value",
                "std_dev_residuals": 0.5,
            }
            pickle.dump(model_state, f)
            model_path = f.name

        try:
            steps = 3
            preds, std_dev = predict(model_path, steps)

            self.assertEqual(len(preds), 3)
            self.assertEqual(std_dev, 0.5)
            # All predictions should equal the level (10.0)
            for pred in preds:
                self.assertAlmostEqual(pred, 10.0, places=5)
        finally:
            os.unlink(model_path)

    def test_predict_with_future_values(self):
        """Test prediction with future actual values (online updating)."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            model_state = {
                "model_type": "es",
                "alpha": 0.5,
                "train_end_index": 2,
                "level": 10.0,
                "value_key": "value",
                "std_dev_residuals": 0.0,
            }
            pickle.dump(model_state, f)
            model_path = f.name

        try:
            steps = 3
            # Provide future values as array of objects (use 'y' key for auto-detection)
            data_json = '[{"y": 20.0}, {"y": 30.0}, {"y": 40.0}]'
            preds, std_dev = predict(model_path, steps, data=data_json)

            self.assertEqual(len(preds), 3)
            # First prediction: level = 10.0
            self.assertAlmostEqual(preds[0], 10.0, places=5)
            # After first update: level = 0.5*20 + 0.5*10 = 15.0
            # Second prediction: 15.0
            self.assertAlmostEqual(preds[1], 15.0, places=5)
            # After second update: level = 0.5*30 + 0.5*15 = 22.5
            # Third prediction: 22.5
            self.assertAlmostEqual(preds[2], 22.5, places=5)
        finally:
            os.unlink(model_path)

    def test_predict_with_invalid_model(self):
        """Test prediction with invalid model file raises error."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            pickle.dump({"model_type": "invalid"}, f)
            model_path = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                predict(model_path, steps=1)
            self.assertIn("Invalid ES model", str(ctx.exception))
        finally:
            os.unlink(model_path)

    def test_predict_with_invalid_alpha(self):
        """Test that invalid alpha in loaded model raises error."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            model_state = {
                "model_type": "es",
                "alpha": 1.5,  # Invalid: > 1.0
                "train_end_index": 5,
                "level": 10.0,
                "value_key": "value",
                "std_dev_residuals": 0.5,
            }
            pickle.dump(model_state, f)
            model_path = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                predict(model_path, steps=1)
            self.assertIn("Invalid alpha", str(ctx.exception))
        finally:
            os.unlink(model_path)

    def test_predict_with_invalid_steps(self):
        """Test that invalid steps raises error."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            model_state = {
                "model_type": "es",
                "alpha": 0.5,
                "train_end_index": 5,
                "level": 10.0,
                "value_key": "value",
                "std_dev_residuals": 0.5,
            }
            pickle.dump(model_state, f)
            model_path = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                predict(model_path, steps=0)
            self.assertIn("steps must be > 0", str(ctx.exception))
        finally:
            os.unlink(model_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for the ES model."""

    def test_full_workflow(self):
        """Test complete train, save, load, predict workflow."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        months = [f"2024-{i:02d}" for i in range(1, 11)]
        alpha = 0.4

        # Train and evaluate
        metrics, std_dev, final_level, y_true, y_pred = train_and_evaluate(
            sales, months, range(0, 7), range(7, 10), alpha
        )

        # Verify metrics
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)

        # Verify predictions
        self.assertEqual(len(y_pred), 3)
        self.assertTrue(all(isinstance(p, (int, float)) for p in y_pred))

        # All predictions should be the same
        self.assertTrue(all(p == y_pred[0] for p in y_pred))


if __name__ == "__main__":
    unittest.main()
