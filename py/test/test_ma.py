#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the Moving Average (MA) model.
"""
import json
import os
import pickle
import sys
import tempfile
import unittest
from collections import deque

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
from models.ma import (
    _forecast_future,
    _format_predictions,
    _validate_training_args,
    moving_average_forecast,
    predict,
    train_and_evaluate,
)


class TestMovingAverageForecast(unittest.TestCase):
    """Test the core moving average forecast function."""

    def test_basic_forecast(self):
        """Test basic moving average forecasting."""
        series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        eval_range = range(3, 7)
        window = 3
        min_index = 0

        idxs, preds = moving_average_forecast(series, eval_range, window, min_index)

        self.assertEqual(idxs, [3, 4, 5, 6])
        # Expected: avg([1,2,3])=2, avg([2,3,4])=3, avg([3,4,5])=4, avg([4,5,6])=5
        expected_preds = [2.0, 3.0, 4.0, 5.0]
        np.testing.assert_array_almost_equal(preds, expected_preds)

    def test_window_validation(self):
        """Test that invalid window raises error."""
        series = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError) as ctx:
            moving_average_forecast(series, range(2, 3), window=0, min_index=0)
        self.assertIn("greater than 0", str(ctx.exception))

    def test_insufficient_history(self):
        """Test forecast with insufficient history."""
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        eval_range = range(1, 3)  # Not enough history for window=3
        window = 3
        min_index = 0

        idxs, preds = moving_average_forecast(series, eval_range, window, min_index)

        # Should skip indices without enough history
        self.assertEqual(len(idxs), 0)
        self.assertEqual(len(preds), 0)


class TestForecastFuture(unittest.TestCase):
    """Test the _forecast_future helper function."""

    def test_forecast_with_actual_values(self):
        """Test forecasting when future values are provided."""
        history = [1.0, 2.0, 3.0]
        window = 3
        steps = 2
        future_values = [4.0, 5.0]

        preds = _forecast_future(history, window, steps, future_values)

        # First prediction: avg([1,2,3]) = 2.0
        # Second prediction: avg([2,3,4]) = 3.0
        expected = [2.0, 3.0]
        np.testing.assert_array_almost_equal(preds, expected)

    def test_forecast_with_predictions(self):
        """Test forecasting using predicted values when no future values provided."""
        history = [1.0, 2.0, 3.0]
        window = 3
        steps = 3

        preds = _forecast_future(history, window, steps, future_values=None)

        # First: avg([1,2,3]) = 2.0
        # Second: avg([2,3,2]) = 2.333...
        # Third: avg([3,2,2.333...]) = 2.444...
        self.assertEqual(len(preds), 3)
        self.assertAlmostEqual(preds[0], 2.0, places=5)

    def test_invalid_steps(self):
        """Test that invalid steps raises error."""
        with self.assertRaises(ValueError) as ctx:
            _forecast_future([1.0, 2.0], window=2, steps=0)
        self.assertIn("steps must be > 0", str(ctx.exception))

    def test_empty_history(self):
        """Test that empty history raises error."""
        with self.assertRaises(ValueError) as ctx:
            _forecast_future([], window=2, steps=1)
        self.assertIn("History is empty", str(ctx.exception))

    def test_uses_deque(self):
        """Test that deque is used for efficient operations."""
        # This is a structural test to ensure deque is being used
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        window = 3
        steps = 10

        # Should not raise error and should complete efficiently
        preds = _forecast_future(history, window, steps)
        self.assertEqual(len(preds), 10)


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
        _validate_training_args(sales, ts=0, te=5, es=6, ee=9, w=3)

    def test_invalid_train_range(self):
        """Test invalid training range raises error."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0]
        with self.assertRaises(ValueError) as ctx:
            _validate_training_args(sales, ts=3, te=2, es=4, ee=4, w=2)
        self.assertIn("Invalid training range", str(ctx.exception))

    def test_invalid_eval_range(self):
        """Test invalid evaluation range raises error."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0]
        with self.assertRaises(ValueError) as ctx:
            _validate_training_args(sales, ts=0, te=2, es=4, ee=3, w=2)
        self.assertIn("Invalid evaluation range", str(ctx.exception))

    def test_eval_before_train(self):
        """Test evaluation starting before training end raises error."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0]
        with self.assertRaises(ValueError) as ctx:
            _validate_training_args(sales, ts=0, te=3, es=2, ee=4, w=2)
        self.assertIn("after training end", str(ctx.exception))

    def test_invalid_window(self):
        """Test invalid window size raises error."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0]
        with self.assertRaises(ValueError) as ctx:
            _validate_training_args(sales, ts=0, te=2, es=3, ee=4, w=0)
        self.assertIn("window must be >= 1", str(ctx.exception))


class TestTrainAndEvaluate(unittest.TestCase):
    """Test the train_and_evaluate function."""

    def test_basic_train_evaluate(self):
        """Test basic training and evaluation."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        months = [f"M{i}" for i in range(10)]
        train_range = range(0, 6)
        eval_range = range(6, 10)
        window = 3

        metrics, std_dev, y_true, y_pred = train_and_evaluate(
            sales, months, train_range, eval_range, window
        )

        self.assertEqual(len(y_true), 4)
        self.assertEqual(len(y_pred), 4)
        self.assertIn("window", metrics)
        self.assertEqual(metrics["window"], 3)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIsInstance(std_dev, float)

    def test_insufficient_history_for_eval(self):
        """Test that insufficient history for evaluation raises error."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0]
        months = [f"M{i}" for i in range(5)]
        train_range = range(0, 2)
        eval_range = range(2, 4)  # Not enough history for window=5
        window = 5

        with self.assertRaises(ValueError) as ctx:
            train_and_evaluate(sales, months, train_range, eval_range, window)
        self.assertIn("Insufficient history", str(ctx.exception))

    def test_empty_eval_range(self):
        """Test handling of empty evaluation range."""
        sales = [1.0, 2.0, 3.0, 4.0, 5.0]
        months = [f"M{i}" for i in range(5)]
        train_range = range(0, 3)
        eval_range = range(4, 4)  # Empty range
        window = 2

        metrics, std_dev, y_true, y_pred = train_and_evaluate(
            sales, months, train_range, eval_range, window
        )

        self.assertEqual(metrics, {})
        self.assertEqual(std_dev, 0.0)
        self.assertEqual(len(y_true), 0)
        self.assertEqual(len(y_pred), 0)


class TestPredictFunction(unittest.TestCase):
    """Test the predict function with model loading."""

    def test_predict_with_model_file(self):
        """Test prediction from a saved model file."""
        # Create a temporary model file using pickle
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            model_state = {
                "model_type": "ma",
                "window": 3,
                "train_end_index": 5,
                "history": [4.0, 5.0, 6.0],
                "value_key": "value",
                "std_dev_residuals": 0.5,
            }
            pickle.dump(model_state, f)
            model_path = f.name

        try:
            steps = 2
            preds, std_dev = predict(model_path, steps)

            self.assertEqual(len(preds), 2)
            self.assertEqual(std_dev, 0.5)
            # First prediction should be avg([4,5,6]) = 5.0
            self.assertAlmostEqual(preds[0], 5.0, places=5)
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
            self.assertIn("Invalid MA model", str(ctx.exception))
        finally:
            os.unlink(model_path)


    def test_predict_with_future_values(self):
        """Test prediction with future actual values (online updating)."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            model_state = {
                "model_type": "ma",
                "window": 3,
                "train_end_index": 5,  # Training ended at index 5
                "history": [4.0, 5.0, 6.0],
                "value_key": "y",
                "std_dev_residuals": 0.5,
            }
            pickle.dump(model_state, f)
            model_path = f.name

        try:
            steps = 3
            # Provide COMPLETE time series including future values
            # Indices 0-5 are historical, 6-8 are future
            data_json = '[{"y": 1.0}, {"y": 2.0}, {"y": 3.0}, {"y": 4.0}, {"y": 5.0}, {"y": 6.0}, {"y": 7.0}, {"y": 8.0}, {"y": 9.0}]'
            preds, std_dev = predict(model_path, steps, data=data_json)

            self.assertEqual(len(preds), 3)
            self.assertEqual(std_dev, 0.5)

            # First prediction: avg([4,5,6]) = 5.0
            self.assertAlmostEqual(preds[0], 5.0, places=5)
            # Second prediction: avg([5,6,7]) = 6.0
            self.assertAlmostEqual(preds[1], 6.0, places=5)
            # Third prediction: avg([6,7,8]) = 7.0
            self.assertAlmostEqual(preds[2], 7.0, places=5)
        finally:
            os.unlink(model_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for the MA model."""

    def test_full_workflow(self):
        """Test complete train, save, load, predict workflow."""
        sales = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        months = [f"2024-{i:02d}" for i in range(1, 11)]

        # Train and evaluate
        metrics, std_dev, y_true, y_pred = train_and_evaluate(
            sales, months, range(0, 7), range(7, 10), window=3
        )

        # Verify metrics
        self.assertGreater(metrics["evaluated_points"], 0)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)

        # Verify predictions
        self.assertEqual(len(y_pred), 3)
        self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in y_pred))

    def test_save_and_load_model(self):
        """Test saving and loading a trained model."""
        sales = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        months = [f"M{i}" for i in range(7)]
        window = 3

        # Train
        metrics, std_dev, y_true, y_pred = train_and_evaluate(
            sales, months, range(0, 5), range(5, 7), window
        )

        # Save model - should save last window points from training
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            from serialization import dump_model
            train_end = 4  # range(0, 5) means train up to index 4
            # Save last 3 points: sales[2], sales[3], sales[4] = 30, 40, 50
            state = {
                "model_type": "ma",
                "window": window,
                "train_end_index": train_end,
                "history": sales[train_end - window + 1 : train_end + 1],
                "value_key": "y",
                "std_dev_residuals": std_dev,
            }
            dump_model(state, f.name)
            model_path = f.name

        try:
            # Load and predict
            preds, loaded_std_dev = predict(model_path, steps=2)

            self.assertEqual(len(preds), 2)
            self.assertEqual(loaded_std_dev, std_dev)

            # Without future_values, MA uses recursive prediction:
            # 1st pred: avg([30, 40, 50]) = 40.0
            # 2nd pred: avg([40, 50, 40]) = 43.333... (using 1st prediction)
            self.assertAlmostEqual(preds[0], 40.0, places=5)
            self.assertAlmostEqual(preds[1], 43.333333, places=5)
        finally:
            os.unlink(model_path)


if __name__ == "__main__":
    unittest.main()
