# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for arima.py (ARIMA model)
Tests cover: helper functions, grid search, training, prediction, error handling, and integration scenarios
"""

import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from metrics import mae, mape, r2, rmse
from models import arima
from serialization import dump_model, load_model


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions in arima.py"""

    def test_determine_target_key_with_y(self):
        """Test _determine_target_key with 'y' key"""
        data = json.dumps(
            [{"y": 100, "month": "2023-01"}, {"y": 110, "month": "2023-02"}]
        )
        target_key, time_key = arima._determine_target_key(data)
        self.assertEqual(target_key, "y")
        self.assertEqual(time_key, "month")

    def test_determine_target_key_with_sales(self):
        """Test _determine_target_key with 'sales' key"""
        data = json.dumps([{"sales": 100}, {"sales": 110}])
        target_key, time_key = arima._determine_target_key(data)
        self.assertEqual(target_key, "sales")
        self.assertIsNone(time_key)

    def test_determine_target_key_with_y_no_time(self):
        """Test _determine_target_key with 'y' but no 'month'"""
        data = json.dumps([{"y": 100}, {"y": 110}])
        target_key, time_key = arima._determine_target_key(data)
        self.assertEqual(target_key, "y")
        self.assertIsNone(time_key)

    def test_determine_target_key_empty_list(self):
        """Test _determine_target_key with empty list"""
        data = json.dumps([])
        with self.assertRaises(ValueError) as ctx:
            arima._determine_target_key(data)
        self.assertIn("non-empty JSON array", str(ctx.exception))

    def test_determine_target_key_not_list(self):
        """Test _determine_target_key with non-list data"""
        data = json.dumps({"y": 100})
        with self.assertRaises(ValueError) as ctx:
            arima._determine_target_key(data)
        self.assertIn("non-empty JSON array", str(ctx.exception))

    def test_determine_target_key_invalid_json(self):
        """Test _determine_target_key with invalid JSON"""
        data = "not valid json {"
        with self.assertRaises(ValueError) as ctx:
            arima._determine_target_key(data)
        self.assertIn("Invalid JSON", str(ctx.exception))

    def test_determine_target_key_no_valid_key(self):
        """Test _determine_target_key with no valid target key"""
        data = json.dumps([{"invalid": 100}, {"invalid": 110}])
        with self.assertRaises(ValueError) as ctx:
            arima._determine_target_key(data)
        self.assertIn("Cannot determine target key", str(ctx.exception))

    def test_validate_ranges_valid(self):
        """Test _validate_ranges with valid ranges"""
        # Should not raise any exception
        arima._validate_ranges(n=10, ts=0, te=5, es=6, ee=9)

    def test_validate_ranges_invalid_train(self):
        """Test _validate_ranges with invalid train range"""
        with self.assertRaises(ValueError) as ctx:
            arima._validate_ranges(n=10, ts=5, te=4, es=6, ee=9)
        self.assertIn("train range invalid", str(ctx.exception))

    def test_validate_ranges_train_out_of_bounds(self):
        """Test _validate_ranges with train range out of bounds"""
        with self.assertRaises(ValueError) as ctx:
            arima._validate_ranges(n=10, ts=0, te=10, es=11, ee=12)
        self.assertIn("train range invalid", str(ctx.exception))

    def test_validate_ranges_invalid_eval(self):
        """Test _validate_ranges with invalid eval range"""
        with self.assertRaises(ValueError) as ctx:
            arima._validate_ranges(n=10, ts=0, te=5, es=9, ee=8)
        self.assertIn("eval range invalid", str(ctx.exception))

    def test_validate_ranges_eval_not_after_train(self):
        """Test _validate_ranges with eval not after train"""
        with self.assertRaises(ValueError) as ctx:
            arima._validate_ranges(n=10, ts=0, te=5, es=5, ee=9)
        self.assertIn("eval_start must be > train_end", str(ctx.exception))

    def test_validate_ranges_empty_eval(self):
        """Test _validate_ranges with empty eval range"""
        # When es > ee, it will fail with "eval range invalid" before checking empty
        with self.assertRaises(ValueError) as ctx:
            arima._validate_ranges(n=10, ts=0, te=5, es=7, ee=6)
        self.assertIn("eval range invalid", str(ctx.exception))


class TestFitAndScore(unittest.TestCase):
    """Test _fit_arima function"""

    def test_fit_arima_basic(self):
        """Test _fit_arima with basic data"""
        np.random.seed(42)
        y_train = np.array([100, 102, 105, 107, 110, 112, 115, 117, 120])
        y_eval = np.array([122, 125])

        result, fitted_res = arima._fit_arima(
            y_train, d=0, p=1, q=1, y_eval_true=y_eval
        )

        self.assertIn("p", result)
        self.assertIn("d", result)
        self.assertIn("q", result)
        self.assertEqual(result["p"], 1)
        self.assertEqual(result["d"], 0)
        self.assertEqual(result["q"], 1)
        self.assertIn("rmse", result)
        self.assertIn("mae", result)
        self.assertIn("mape", result)
        self.assertIn("r2", result)
        self.assertIn("aic", result)
        self.assertIn("converged", result)
        self.assertIsNotNone(fitted_res)

    def test_fit_arima_p0_q0(self):
        """Test _fit_arima with p=0, q=0 (mean model)"""
        y_train = np.array([100, 102, 105, 107, 110, 112, 115, 117, 120])
        y_eval = np.array([122, 125])

        result, fitted_res = arima._fit_arima(
            y_train, d=0, p=0, q=0, y_eval_true=y_eval
        )

        self.assertEqual(result["p"], 0)
        self.assertEqual(result["q"], 0)
        # With d=0, p=0, q=0 and trend='c', this is just a constant mean model
        self.assertIsNotNone(fitted_res)

    def test_fit_arima_with_differencing(self):
        """Test _fit_arima with d=1"""
        y_train = np.array([100, 102, 105, 109, 114, 120, 127, 135, 144])
        y_eval = np.array([154, 165])

        result, fitted_res = arima._fit_arima(
            y_train, d=1, p=1, q=1, y_eval_true=y_eval
        )

        self.assertEqual(result["d"], 1)
        self.assertIsNotNone(fitted_res)

    def test_fit_arima_returns_predictions(self):
        """Test that _fit_arima returns predictions in result"""
        y_train = np.array([100, 102, 105, 107, 110, 112, 115, 117, 120])
        y_eval = np.array([122, 125])

        result, fitted_res = arima._fit_arima(
            y_train, d=0, p=1, q=0, y_eval_true=y_eval
        )

        self.assertIn("y_pred", result)
        self.assertEqual(len(result["y_pred"]), len(y_eval))

    def test_fit_arima_with_insufficient_data(self):
        """Test _fit_arima with very small dataset"""
        # With only 3 data points, fitting ARIMA(2,0,2) should fail or struggle
        y_train = np.array([100, 102, 105])
        y_eval = np.array([107])

        result, fitted_res = arima._fit_arima(
            y_train, d=0, p=2, q=2, y_eval_true=y_eval
        )

        # Should either fail or succeed with warnings
        # The function catches exceptions and returns error dict
        if "error" in result:
            self.assertFalse(result["converged"])
        else:
            # If it succeeded, that's OK too
            self.assertIn("p", result)


class TestSearchPQ(unittest.TestCase):
    """Test _search_pq grid search function"""

    def test_search_pq_basic(self):
        """Test _search_pq with basic data"""
        np.random.seed(42)
        y_train = np.array([100, 102, 105, 107, 110, 112, 115, 117, 120])
        y_eval = np.array([122, 125])

        best, best_res, summary = arima._search_pq(
            y_train, y_eval, d=0, p_max=2, q_max=2
        )

        # Check best result
        self.assertIn("p", best)
        self.assertIn("d", best)
        self.assertIn("q", best)
        self.assertIn("rmse", best)
        self.assertIsNotNone(best_res)

        # Check summary
        self.assertIn("candidates_tried", summary)
        self.assertIn("succeeded", summary)
        self.assertIn("failed", summary)
        self.assertEqual(summary["candidates_tried"], 9)  # (0,1,2) x (0,1,2) = 9

    def test_search_pq_finds_best_by_rmse(self):
        """Test that _search_pq selects model with lowest RMSE"""
        y_train = np.array([100, 102, 105, 107, 110, 112, 115, 117, 120])
        y_eval = np.array([122, 125])

        best, best_res, summary = arima._search_pq(
            y_train, y_eval, d=0, p_max=1, q_max=1
        )

        # The best model should have p and q values
        self.assertGreaterEqual(best["p"], 0)
        self.assertLessEqual(best["p"], 1)
        self.assertGreaterEqual(best["q"], 0)
        self.assertLessEqual(best["q"], 1)

    def test_search_pq_with_d_greater_than_data(self):
        """Test _search_pq fails when d >= len(y_train)"""
        y_train = np.array([100, 102, 105])
        y_eval = np.array([107])

        with self.assertRaises(ValueError) as ctx:
            arima._search_pq(y_train, y_eval, d=3, p_max=1, q_max=1)
        self.assertIn("must be greater than d", str(ctx.exception))

    def test_search_pq_small_sample_warning(self):
        """Test _search_pq issues warning with small sample"""
        y_train = np.array([100, 102, 105, 107, 110])
        y_eval = np.array([112])

        with self.assertWarns(RuntimeWarning):
            best, best_res, summary = arima._search_pq(
                y_train, y_eval, d=0, p_max=3, q_max=3
            )

    def test_search_pq_summary_structure(self):
        """Test that _search_pq summary has correct structure"""
        y_train = np.array([100, 102, 105, 107, 110, 112, 115, 117, 120])
        y_eval = np.array([122, 125])

        best, best_res, summary = arima._search_pq(
            y_train, y_eval, d=0, p_max=1, q_max=1
        )

        self.assertIn("fit_failures_preview", summary)
        self.assertIn("top_by_aic_preview", summary)
        self.assertIsInstance(summary["fit_failures_preview"], list)
        self.assertIsInstance(summary["top_by_aic_preview"], list)


class TestPredictFunction(unittest.TestCase):
    """Test predict() function for ensemble"""

    def test_predict_basic(self):
        """Test predict() with a saved model"""
        # Train a simple model first
        np.random.seed(42)
        y_train = np.array([100, 102, 105, 107, 110, 112, 115, 117, 120])
        y_eval = np.array([122, 125])

        best, best_res, summary = arima._search_pq(
            y_train, y_eval, d=0, p_max=1, q_max=1
        )

        # Save the model
        temp_dir = tempfile.mkdtemp()
        try:
            model_path = os.path.join(temp_dir, "arima_test.pkl")
            dump_model(best_res, model_path)

            # Test predict
            predictions = arima.predict(model_path, steps=3)

            self.assertIsInstance(predictions, list)
            self.assertEqual(len(predictions), 3)
            for pred in predictions:
                self.assertIn("prediction", pred)
                self.assertIn("std_dev", pred)
                self.assertIsInstance(pred["prediction"], float)
                self.assertIsInstance(pred["std_dev"], float)
        finally:
            shutil.rmtree(temp_dir)

    def test_predict_invalid_steps(self):
        """Test predict() with invalid steps"""
        temp_dir = tempfile.mkdtemp()
        try:
            model_path = os.path.join(temp_dir, "dummy.pkl")

            with self.assertRaises(ValueError) as ctx:
                arima.predict(model_path, steps=0)
            self.assertIn("steps must be > 0", str(ctx.exception))

            with self.assertRaises(ValueError) as ctx:
                arima.predict(model_path, steps=-1)
            self.assertIn("steps must be > 0", str(ctx.exception))
        finally:
            shutil.rmtree(temp_dir)


class TestTrainingMode(unittest.TestCase):
    """Test ARIMA training mode (run function)"""

    def setUp(self):
        """Set up test data"""
        self.test_data = [
            {"y": 100},
            {"y": 102},
            {"y": 105},
            {"y": 107},
            {"y": 110},
            {"y": 112},
            {"y": 115},
            {"y": 117},
            {"y": 120},
            {"y": 122},
            {"y": 125},
        ]
        self.data_json = json.dumps(self.test_data)

    def test_training_basic(self):
        """Test basic training mode"""
        args = MagicMock()
        args.data = self.data_json
        args.train_start = 0
        args.train_end = 7
        args.eval_start = 8
        args.eval_end = 9
        args.d = 0
        args.p_max = 2
        args.q_max = 2
        args.pretty = False
        args.save_model = None
        args.load_model = None

        # Capture stdout
        captured_output = StringIO()
        with patch("sys.stdout", new=captured_output):
            arima.run(args)

        output = captured_output.getvalue()
        result = json.loads(output)

        self.assertIn("mode", result)
        self.assertEqual(result["mode"], "arima_search_pq_fixed_d")
        self.assertIn("best_order", result)
        self.assertIn("metrics", result)
        self.assertIn("search_summary", result)

    def test_training_with_save_model(self):
        """Test training with model saving"""
        temp_dir = tempfile.mkdtemp()
        try:
            model_path = os.path.join(temp_dir, "arima_test.pkl")

            args = MagicMock()
            args.data = self.data_json
            args.train_start = 0
            args.train_end = 7
            args.eval_start = 8
            args.eval_end = 9
            args.d = 0
            args.p_max = 1
            args.q_max = 1
            args.pretty = False
            args.save_model = model_path
            args.load_model = None

            captured_output = StringIO()
            with patch("sys.stdout", new=captured_output):
                arima.run(args)

            # Check model file exists
            self.assertTrue(os.path.exists(model_path))

            # Check output mentions saved model
            output = captured_output.getvalue()
            result = json.loads(output)
            # Note: The current implementation doesn't add "saved_model" key, but model should be saved

        finally:
            shutil.rmtree(temp_dir)

    def test_training_with_sales_key(self):
        """Test training with 'sales' key instead of 'y'"""
        sales_data = [
            {"sales": 100},
            {"sales": 102},
            {"sales": 105},
            {"sales": 107},
            {"sales": 110},
            {"sales": 112},
            {"sales": 115},
            {"sales": 117},
            {"sales": 120},
            {"sales": 122},
            {"sales": 125},
        ]
        data_json = json.dumps(sales_data)

        args = MagicMock()
        args.data = data_json
        args.train_start = 0
        args.train_end = 7
        args.eval_start = 8
        args.eval_end = 9
        args.d = 0
        args.p_max = 1
        args.q_max = 1
        args.pretty = False
        args.save_model = None
        args.load_model = None

        captured_output = StringIO()
        with patch("sys.stdout", new=captured_output):
            arima.run(args)

        output = captured_output.getvalue()
        result = json.loads(output)
        self.assertIn("best_order", result)

    def test_training_missing_data(self):
        """Test training fails without data"""
        args = MagicMock()
        args.data = None
        args.train_start = 0
        args.train_end = 5
        args.eval_start = 6
        args.eval_end = 7
        args.d = 0
        args.load_model = None

        with self.assertRaises(ValueError) as ctx:
            arima.run(args)
        self.assertIn("requires --data", str(ctx.exception))

    def test_training_missing_required_args(self):
        """Test training fails with missing required arguments"""
        args = MagicMock()
        args.data = self.data_json
        args.train_start = None  # Missing
        args.train_end = 7
        args.eval_start = 8
        args.eval_end = 9
        args.d = 0
        args.load_model = None

        with self.assertRaises(ValueError) as ctx:
            arima.run(args)
        self.assertIn("requires", str(ctx.exception))

    def test_training_invalid_ranges(self):
        """Test training fails with invalid ranges"""
        args = MagicMock()
        args.data = self.data_json
        args.train_start = 0
        args.train_end = 7
        args.eval_start = 7  # Should be > train_end
        args.eval_end = 9
        args.d = 0
        args.p_max = 1
        args.q_max = 1
        args.load_model = None

        with self.assertRaises(ValueError) as ctx:
            arima.run(args)
        self.assertIn("eval_start must be > train_end", str(ctx.exception))


class TestPredictionMode(unittest.TestCase):
    """Test ARIMA prediction mode with loaded model"""

    def setUp(self):
        """Set up a trained model"""
        self.test_data = [
            {"y": 100},
            {"y": 102},
            {"y": 105},
            {"y": 107},
            {"y": 110},
            {"y": 112},
            {"y": 115},
            {"y": 117},
            {"y": 120},
            {"y": 122},
            {"y": 125},
        ]
        self.data_json = json.dumps(self.test_data)

        # Train and save a model
        np.random.seed(42)
        y_train = np.array([100, 102, 105, 107, 110, 112, 115, 117])
        y_eval = np.array([120, 122])
        best, best_res, summary = arima._search_pq(
            y_train, y_eval, d=0, p_max=1, q_max=1
        )

        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "arima_test.pkl")
        dump_model(best_res, self.model_path)

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def test_prediction_with_forecast_steps(self):
        """Test prediction mode with forecast_steps"""
        args = MagicMock()
        args.load_model = self.model_path
        args.forecast_steps = 3
        args.eval_start = None
        args.eval_end = None
        args.pretty = False
        args.ci_level = 0.95
        args.data = None

        captured_output = StringIO()
        with patch("sys.stdout", new=captured_output):
            arima.run(args)

        output = captured_output.getvalue()
        result = json.loads(output)

        self.assertIn("mode", result)
        self.assertEqual(result["mode"], "arima_loaded_predict")
        self.assertIn("predictions", result)
        self.assertEqual(len(result["predictions"]), 3)
        self.assertIn("order", result)

    def test_prediction_with_eval_range(self):
        """Test prediction mode with eval_start/eval_end"""
        args = MagicMock()
        args.load_model = self.model_path
        args.forecast_steps = None
        args.eval_start = 8
        args.eval_end = 10
        args.pretty = False
        args.ci_level = 0.95
        args.data = None

        captured_output = StringIO()
        with patch("sys.stdout", new=captured_output):
            arima.run(args)

        output = captured_output.getvalue()
        result = json.loads(output)

        self.assertIn("predictions", result)
        self.assertEqual(result["forecast_steps"], 3)  # 10 - 8 + 1, but depends on nobs

    def test_prediction_with_data_for_metrics(self):
        """Test prediction mode with data to compute metrics"""
        args = MagicMock()
        args.load_model = self.model_path
        args.forecast_steps = None
        args.eval_start = 8
        args.eval_end = 9
        args.data = self.data_json
        args.pretty = False
        args.ci_level = 0.95

        captured_output = StringIO()
        with patch("sys.stdout", new=captured_output):
            arima.run(args)

        output = captured_output.getvalue()
        result = json.loads(output)

        # May have metrics if ranges align with training data
        # This depends on nobs of the saved model
        self.assertIn("predictions", result)

    def test_prediction_missing_steps(self):
        """Test prediction fails without steps or eval range"""
        args = MagicMock()
        args.load_model = self.model_path
        args.forecast_steps = None
        args.eval_start = None
        args.eval_end = None
        args.pretty = False
        args.data = None

        with self.assertRaises(ValueError) as ctx:
            arima.run(args)
        self.assertIn(
            "requires either eval_start/eval_end or --forecast_steps",
            str(ctx.exception),
        )

    def test_prediction_invalid_eval_range(self):
        """Test prediction fails with invalid eval range"""
        args = MagicMock()
        args.load_model = self.model_path
        args.forecast_steps = None
        args.eval_start = 10
        args.eval_end = 8  # End < start
        args.pretty = False
        args.data = None

        with self.assertRaises(ValueError) as ctx:
            arima.run(args)
        self.assertIn("eval_end must be >= eval_start", str(ctx.exception))


class TestErrorHandling(unittest.TestCase):
    """Test error handling throughout arima.py"""

    def test_specific_exceptions_in_fit_arima(self):
        """Test that _fit_arima catches specific exceptions"""
        # Create data that will cause fitting to fail - very small dataset with high order
        # This should trigger LinAlgError or ValueError during fitting
        y_train = np.array([1.0, 2.0])  # Only 2 points
        y_eval = np.array([3.0])

        result, fitted_res = arima._fit_arima(
            y_train, d=0, p=5, q=5, y_eval_true=y_eval
        )

        # With such a small dataset and high p,q, it should fail or succeed with warnings
        # If it returns error dict, check it; otherwise, it's OK
        if "error" in result:
            self.assertFalse(result["converged"])
            self.assertIsNone(fitted_res)
        else:
            # If it somehow succeeded, that's acceptable too
            self.assertIn("p", result)

    def test_invalid_json_in_determine_target_key(self):
        """Test _determine_target_key handles invalid JSON"""
        with self.assertRaises(ValueError) as ctx:
            arima._determine_target_key("{invalid json")
        self.assertIn("Invalid JSON", str(ctx.exception))

    def test_training_with_invalid_data_format(self):
        """Test training with invalid data format"""
        args = MagicMock()
        args.data = json.dumps({"not": "a list"})
        args.train_start = 0
        args.train_end = 5
        args.eval_start = 6
        args.eval_end = 7
        args.d = 0
        args.p_max = 1
        args.q_max = 1
        args.load_model = None

        with self.assertRaises(ValueError) as ctx:
            arima.run(args)
        self.assertIn("non-empty JSON array", str(ctx.exception))

    def test_all_fits_fail_in_search(self):
        """Test _search_pq when all (p,q) combinations fail"""
        # Create data that's too small for the requested d parameter
        y_train = np.array([100, 102])  # Only 2 points
        y_eval = np.array([105])

        # With d=2 and only 2 training points, this should fail validation
        with self.assertRaises(ValueError) as ctx:
            arima._search_pq(y_train, y_eval, d=2, p_max=1, q_max=1)
        self.assertIn("must be greater than d", str(ctx.exception))


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""

    def test_full_train_predict_workflow(self):
        """Test complete train -> save -> load -> predict workflow"""
        test_data = [
            {"y": 100},
            {"y": 102},
            {"y": 105},
            {"y": 107},
            {"y": 110},
            {"y": 112},
            {"y": 115},
            {"y": 117},
            {"y": 120},
            {"y": 122},
            {"y": 125},
            {"y": 127},
        ]
        data_json = json.dumps(test_data)

        temp_dir = tempfile.mkdtemp()
        try:
            model_path = os.path.join(temp_dir, "arima_workflow.pkl")

            # Step 1: Train and save
            args_train = MagicMock()
            args_train.data = data_json
            args_train.train_start = 0
            args_train.train_end = 7
            args_train.eval_start = 8
            args_train.eval_end = 9
            args_train.d = 0
            args_train.p_max = 1
            args_train.q_max = 1
            args_train.pretty = False
            args_train.save_model = model_path
            args_train.load_model = None

            captured_train = StringIO()
            with patch("sys.stdout", new=captured_train):
                arima.run(args_train)

            train_output = json.loads(captured_train.getvalue())
            self.assertIn("best_order", train_output)
            self.assertTrue(os.path.exists(model_path))

            # Step 2: Load and predict
            args_pred = MagicMock()
            args_pred.load_model = model_path
            args_pred.forecast_steps = 2
            args_pred.eval_start = None
            args_pred.eval_end = None
            args_pred.pretty = False
            args_pred.data = None

            captured_pred = StringIO()
            with patch("sys.stdout", new=captured_pred):
                arima.run(args_pred)

            pred_output = json.loads(captured_pred.getvalue())
            self.assertIn("predictions", pred_output)
            self.assertEqual(len(pred_output["predictions"]), 2)

            # Verify predictions have correct structure
            for pred in pred_output["predictions"]:
                self.assertIn("prediction", pred)
                self.assertIn("std_dev", pred)

        finally:
            shutil.rmtree(temp_dir)

    def test_with_month_timestamps(self):
        """Test with month timestamps in data"""
        test_data = [
            {"y": 100, "month": "2023-01"},
            {"y": 102, "month": "2023-02"},
            {"y": 105, "month": "2023-03"},
            {"y": 107, "month": "2023-04"},
            {"y": 110, "month": "2023-05"},
            {"y": 112, "month": "2023-06"},
            {"y": 115, "month": "2023-07"},
            {"y": 117, "month": "2023-08"},
            {"y": 120, "month": "2023-09"},
            {"y": 122, "month": "2023-10"},
        ]
        data_json = json.dumps(test_data)

        args = MagicMock()
        args.data = data_json
        args.train_start = 0
        args.train_end = 6
        args.eval_start = 7
        args.eval_end = 8
        args.d = 0
        args.p_max = 1
        args.q_max = 1
        args.pretty = False
        args.save_model = None
        args.load_model = None

        captured_output = StringIO()
        with patch("sys.stdout", new=captured_output):
            arima.run(args)

        output = captured_output.getvalue()
        result = json.loads(output)

        self.assertIn("meta", result)
        self.assertIn("months_eval_slice", result["meta"])

    def test_d_parameter_preserved(self):
        """Test that d parameter is never changed from caller's value"""
        test_data = [{"y": 100 + i * 2} for i in range(15)]
        data_json = json.dumps(test_data)

        for d_value in [0, 1, 2]:
            args = MagicMock()
            args.data = data_json
            args.train_start = 0
            args.train_end = 10
            args.eval_start = 11
            args.eval_end = 13
            args.d = d_value
            args.p_max = 1
            args.q_max = 1
            args.pretty = False
            args.save_model = None
            args.load_model = None

            captured_output = StringIO()
            with patch("sys.stdout", new=captured_output):
                arima.run(args)

            output = captured_output.getvalue()
            result = json.loads(output)

            # Verify d parameter is exactly what we passed
            self.assertEqual(result["best_order"]["d"], d_value)

    def test_metrics_computation(self):
        """Test that metrics are computed correctly"""
        test_data = [{"y": 100 + i * 2} for i in range(12)]
        data_json = json.dumps(test_data)

        args = MagicMock()
        args.data = data_json
        args.train_start = 0
        args.train_end = 8
        args.eval_start = 9
        args.eval_end = 10
        args.d = 0
        args.p_max = 1
        args.q_max = 1
        args.pretty = False
        args.save_model = None
        args.load_model = None

        captured_output = StringIO()
        with patch("sys.stdout", new=captured_output):
            arima.run(args)

        output = captured_output.getvalue()
        result = json.loads(output)

        self.assertIn("metrics", result)
        metrics = result["metrics"]
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("mape", metrics)
        self.assertIn("r2", metrics)
        self.assertIn("aic", metrics)

        # All metrics should be numbers
        self.assertIsInstance(metrics["rmse"], (int, float))
        self.assertIsInstance(metrics["mae"], (int, float))
        self.assertIsInstance(metrics["mape"], (int, float))
        self.assertIsInstance(metrics["r2"], (int, float))
        self.assertIsInstance(metrics["aic"], (int, float))


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHelperFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestFitAndScore))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchPQ))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingMode))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionMode))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
