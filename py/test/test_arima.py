#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the ARIMA model.
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
from models.arima import (
    _format_predictions,
    _load_state,
    predict,
    train_and_evaluate,
)


class TestFormatPredictions(unittest.TestCase):
    """Test the _format_predictions helper function."""

    def test_format_predictions(self):
        """Test predictions are formatted correctly with confidence intervals."""
        preds = [10.0, 20.0, 30.0]
        conf_intervals = [[8.0, 12.0], [18.0, 22.0], [28.0, 32.0]]

        result = _format_predictions(preds, conf_intervals)

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0]["prediction"], 10.0, places=5)
        # std_dev ≈ (12 - 8) / 3.92 ≈ 1.02
        self.assertAlmostEqual(result[0]["std_dev"], (12.0 - 8.0) / 3.92, places=5)

    def test_empty_predictions(self):
        """Test formatting empty predictions list."""
        result = _format_predictions([], [])
        self.assertEqual(result, [])


class TestTrainAndEvaluate(unittest.TestCase):
    """Test the train_and_evaluate function."""

    def test_basic_train_evaluate(self):
        """Test basic training and evaluation with auto search."""
        # Create a simple linear trend for testing
        np.random.seed(42)
        y = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0])
        timestamps = [f"M{i}" for i in range(10)]
        train_range = range(0, 7)
        eval_range = range(7, 10)
        d = 1  # First difference for linear trend

        metrics, model, order, search_summary, y_true, y_pred, conf_int = train_and_evaluate(
            y, timestamps, train_range, eval_range, d, max_p=3, max_q=3
        )

        # Check basic structure
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("aic", metrics)
        self.assertIn("bic", metrics)
        self.assertIn("converged", metrics)

        # Check order
        self.assertIn("p", order)
        self.assertIn("d", order)
        self.assertIn("q", order)
        self.assertEqual(order["d"], d)

        # Check search summary
        self.assertIn("aic_optimized_model", search_summary)
        self.assertIn("bic_optimized_model", search_summary)
        self.assertIn("selected_model", search_summary)

        # Check predictions
        self.assertEqual(len(y_pred), len(eval_range))
        self.assertEqual(len(conf_int), len(eval_range))

    def test_aic_bic_comparison(self):
        """Test that AIC and BIC models are both considered."""
        np.random.seed(123)
        # Create data with slight noise
        y = np.array([1.0, 2.1, 3.0, 3.9, 5.1, 6.0, 7.1, 8.0, 9.1, 10.0])
        timestamps = [f"T{i}" for i in range(10)]

        metrics, model, order, search_summary, _, _, _ = train_and_evaluate(
            y, timestamps, range(0, 7), range(7, 10), d=0, max_p=2, max_q=2
        )

        # Both models should be searched
        self.assertIsNotNone(search_summary["aic_optimized_model"])
        self.assertIsNotNone(search_summary["bic_optimized_model"])

        # One should be selected
        self.assertIn(search_summary["selected_model"], ["aic_optimized", "bic_optimized"])

        # Selected model should have valid AIC and BIC
        self.assertIsInstance(metrics["aic"], float)
        self.assertIsInstance(metrics["bic"], float)

    def test_invalid_d(self):
        """Test that invalid d raises error."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        timestamps = ["M1", "M2", "M3", "M4", "M5"]

        with self.assertRaises(ValueError) as ctx:
            train_and_evaluate(y, timestamps, range(0, 3), range(3, 5), d=-1)
        self.assertIn("d must be >= 0", str(ctx.exception))

    def test_empty_train_range(self):
        """Test that empty training range raises error."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        timestamps = ["M1", "M2", "M3", "M4", "M5"]

        with self.assertRaises(ValueError) as ctx:
            train_and_evaluate(y, timestamps, range(0, 0), range(1, 3), d=0)
        self.assertIn("Training range is empty", str(ctx.exception))

    def test_empty_eval_range(self):
        """Test that empty evaluation range raises error."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        timestamps = ["M1", "M2", "M3", "M4", "M5"]

        with self.assertRaises(ValueError) as ctx:
            train_and_evaluate(y, timestamps, range(0, 3), range(4, 4), d=0)
        self.assertIn("Evaluation range is empty", str(ctx.exception))


class TestPredictFunction(unittest.TestCase):
    """Test the predict function with model loading."""

    def test_predict_with_trained_model(self):
        """Test prediction from a trained model."""
        # Train a simple model
        np.random.seed(42)
        y = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        timestamps = [f"M{i}" for i in range(7)]

        metrics, model, order, _, _, _, _ = train_and_evaluate(
            y, timestamps, range(0, 5), range(5, 7), d=1, max_p=2, max_q=2
        )

        # Save model
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            from serialization import dump_model
            state = {
                "model_type": "arima",
                "model": model,
                "order": order,
                "d": 1,
            }
            dump_model(state, f.name)
            model_path = f.name

        try:
            # Predict
            preds, conf_int = predict(model_path, steps=2)

            self.assertEqual(len(preds), 2)
            self.assertEqual(len(conf_int), 2)
            # Predictions should be reasonable
            for pred in preds:
                self.assertIsInstance(pred, float)
            # Confidence intervals should have [lower, upper]
            for ci in conf_int:
                self.assertEqual(len(ci), 2)
                self.assertLess(ci[0], ci[1])  # lower < upper
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
            self.assertIn("Invalid ARIMA model", str(ctx.exception))
        finally:
            os.unlink(model_path)

    def test_predict_with_invalid_steps(self):
        """Test that invalid steps raises error."""
        # Create a dummy valid model
        np.random.seed(42)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        timestamps = [f"M{i}" for i in range(5)]

        metrics, model, order, _, _, _, _ = train_and_evaluate(
            y, timestamps, range(0, 3), range(3, 5), d=0, max_p=1, max_q=1
        )

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            from serialization import dump_model
            state = {"model_type": "arima", "model": model, "order": order, "d": 0}
            dump_model(state, f.name)
            model_path = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                predict(model_path, steps=0)
            self.assertIn("steps must be > 0", str(ctx.exception))
        finally:
            os.unlink(model_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for the ARIMA model."""

    def test_full_workflow(self):
        """Test complete train, save, load, predict workflow."""
        np.random.seed(42)
        # Create trending data
        t = np.arange(20)
        y = 10 + 2 * t + np.random.normal(0, 0.5, 20)
        timestamps = [f"2024-{i:02d}" for i in range(1, 21)]

        # Train
        metrics, model, order, search_summary, y_true, y_pred, conf_int = train_and_evaluate(
            y, timestamps, range(0, 15), range(15, 20), d=1, max_p=3, max_q=3
        )

        # Verify training results
        self.assertGreater(metrics["evaluated_points"], 0)
        self.assertIn("rmse", metrics)
        self.assertIn("aic", metrics)
        self.assertTrue(metrics.get("converged", False))

        # Verify AIC/BIC comparison happened
        self.assertIn("aic_optimized_model", search_summary)
        self.assertIn("bic_optimized_model", search_summary)

        # Save and load
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            from serialization import dump_model
            state = {"model_type": "arima", "model": model, "order": order, "d": 1}
            dump_model(state, f.name)
            model_path = f.name

        try:
            # Predict future
            future_preds, future_conf = predict(model_path, steps=3)

            self.assertEqual(len(future_preds), 3)
            self.assertEqual(len(future_conf), 3)
            # Predictions should continue the trend
            for pred in future_preds:
                self.assertGreater(pred, 0)
        finally:
            os.unlink(model_path)


if __name__ == "__main__":
    unittest.main()
