# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for ma.py (Moving Average model)
Tests cover: metrics, core functions, edge cases, error handling, and integration scenarios
"""
import unittest
import json
import sys
import os
import tempfile
import shutil
import math
from unittest.mock import MagicMock, patch
from io import StringIO

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models import ma
from serialization import dump_model, load_model
from metrics import rmse, mae, mape, r2


class TestMetrics(unittest.TestCase):
    """Test metric calculation functions"""

    def test_rmse_normal(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.1, 2.9, 4.2, 4.8]
        result = rmse(y_true, y_pred)
        expected = math.sqrt(((0.1**2) + (0.1**2) + (0.1**2) + (0.2**2) + (0.2**2)) / 5)
        self.assertAlmostEqual(result, expected, places=5)

    def test_rmse_empty(self):
        result = rmse([], [])
        self.assertTrue(math.isnan(result))

    def test_rmse_perfect_match(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        result = rmse(y_true, y_pred)
        self.assertEqual(result, 0.0)

    def test_mae_normal(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.1, 2.9, 4.2, 4.8]
        result = mae(y_true, y_pred)
        expected = (0.1 + 0.1 + 0.1 + 0.2 + 0.2) / 5
        self.assertAlmostEqual(result, expected, places=5)

    def test_mae_empty(self):
        result = mae([], [])
        self.assertTrue(math.isnan(result))

    def test_mape_normal(self):
        y_true = [100.0, 200.0, 300.0]
        y_pred = [110.0, 190.0, 330.0]
        result = mape(y_true, y_pred)
        # (10/100 + 10/200 + 30/300) / 3 * 100 = (0.1 + 0.05 + 0.1) / 3 * 100
        expected = 100.0 * (0.1 + 0.05 + 0.1) / 3
        self.assertAlmostEqual(result, expected, places=5)

    def test_mape_with_zero_true_value(self):
        y_true = [0.0, 100.0]
        y_pred = [10.0, 110.0]
        result = mape(y_true, y_pred)
        # Should use eps for zero values
        self.assertFalse(math.isnan(result))
        self.assertTrue(result > 0)

    def test_mape_empty(self):
        result = mape([], [])
        self.assertTrue(math.isnan(result))

    def test_r2_normal(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.1, 2.9, 4.2, 4.8]
        result = r2(y_true, y_pred)
        # Calculate manually
        mean_y = sum(y_true) / len(y_true)  # 3.0
        ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
        ss_tot = sum((t - mean_y) ** 2 for t in y_true)
        expected = 1.0 - ss_res / ss_tot
        self.assertAlmostEqual(result, expected, places=5)

    def test_r2_perfect_match(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        result = r2(y_true, y_pred)
        self.assertEqual(result, 1.0)

    def test_r2_constant_true_values(self):
        y_true = [5.0, 5.0, 5.0]
        y_pred = [4.0, 5.0, 6.0]
        result = r2(y_true, y_pred)
        # ss_tot = 0, should return 0
        self.assertEqual(result, 0.0)

    def test_r2_empty(self):
        result = r2([], [])
        self.assertTrue(math.isnan(result))


class TestMovingAverageForecast(unittest.TestCase):
    """Test the core moving_average_forecast function"""

    def test_normal_forecast(self):
        series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        eval_range = range(3, 6)  # Predict for indices 3, 4, 5
        window = 3
        idxs, preds = ma.moving_average_forecast(series, eval_range, window, min_index=0)

        self.assertEqual(idxs, [3, 4, 5])
        # t=3: mean([1,2,3]) = 2.0
        # t=4: mean([2,3,4]) = 3.0
        # t=5: mean([3,4,5]) = 4.0
        self.assertEqual(preds, [2.0, 3.0, 4.0])

    def test_window_too_large_skips_early_points(self):
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        eval_range = range(1, 5)  # Try to predict 1, 2, 3, 4
        window = 3
        idxs, preds = ma.moving_average_forecast(series, eval_range, window, min_index=0)

        # t=1: need [0:1] but window=3, need 3 points, skip
        # t=2: need [0:2] but window=3, need 3 points, skip
        # t=3: mean([0,1,2]) = (1+2+3)/3 = 2.0
        # t=4: mean([1,2,3]) = (2+3+4)/3 = 3.0
        self.assertEqual(idxs, [3, 4])
        self.assertEqual(preds, [2.0, 3.0])

    def test_min_index_constraint(self):
        series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        eval_range = range(3, 6)
        window = 2
        min_index = 2  # Cannot use data before index 2
        idxs, preds = ma.moving_average_forecast(series, eval_range, window, min_index)

        # t=3: need [1:3], but min_index=2, so skip (1 < 2)
        # t=4: need [2:4] = [3.0, 4.0], mean = 3.5
        # t=5: need [3:5] = [4.0, 5.0], mean = 4.5
        self.assertEqual(idxs, [4, 5])
        self.assertEqual(preds, [3.5, 4.5])

    def test_empty_eval_range(self):
        series = [1.0, 2.0, 3.0]
        eval_range = range(5, 5)  # Empty range
        window = 2
        idxs, preds = ma.moving_average_forecast(series, eval_range, window, min_index=0)

        self.assertEqual(idxs, [])
        self.assertEqual(preds, [])

    def test_all_points_skipped(self):
        series = [1.0, 2.0, 3.0]
        eval_range = range(0, 3)
        window = 5  # Too large, will skip all points
        idxs, preds = ma.moving_average_forecast(series, eval_range, window, min_index=0)

        self.assertEqual(idxs, [])
        self.assertEqual(preds, [])

    def test_window_zero_raises_error(self):
        series = [1.0, 2.0, 3.0]
        eval_range = range(1, 3)
        with self.assertRaises(ValueError) as ctx:
            ma.moving_average_forecast(series, eval_range, window=0, min_index=0)
        self.assertIn("[VALIDATION]", str(ctx.exception))
        self.assertIn("greater than 0", str(ctx.exception))

    def test_negative_window_raises_error(self):
        series = [1.0, 2.0, 3.0]
        eval_range = range(1, 3)
        with self.assertRaises(ValueError) as ctx:
            ma.moving_average_forecast(series, eval_range, window=-1, min_index=0)
        self.assertIn("[VALIDATION]", str(ctx.exception))


class TestForecastFuture(unittest.TestCase):
    """Test the _forecast_future helper function"""

    def test_forecast_without_future_values(self):
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        window = 3
        steps = 3
        preds = ma._forecast_future(history, window, steps)

        # Step 1: mean([3,4,5]) = 4.0, append 4.0
        # Step 2: mean([4,5,4]) = 4.333..., append 4.333...
        # Step 3: mean([5,4,4.333]) = 4.444..., append 4.444...
        self.assertEqual(len(preds), 3)
        self.assertAlmostEqual(preds[0], 4.0, places=5)

    def test_forecast_with_future_values(self):
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        window = 3
        steps = 3
        future_values = [6.0, 7.0, 8.0]
        preds = ma._forecast_future(history, window, steps, future_values)

        # Step 1: mean([3,4,5]) = 4.0, use actual 6.0
        # Step 2: mean([4,5,6]) = 5.0, use actual 7.0
        # Step 3: mean([5,6,7]) = 6.0, use actual 8.0
        self.assertEqual(len(preds), 3)
        self.assertEqual(preds, [4.0, 5.0, 6.0])

    def test_forecast_partial_future_values(self):
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        window = 3
        steps = 4
        future_values = [6.0, 7.0]  # Only 2 values for 4 steps
        preds = ma._forecast_future(history, window, steps, future_values)

        # Steps 1-2 use actual values, steps 3-4 use predictions
        self.assertEqual(len(preds), 4)
        self.assertEqual(preds[0], 4.0)  # mean([3,4,5])
        self.assertEqual(preds[1], 5.0)  # mean([4,5,6])

    def test_forecast_with_window_zero(self):
        history = [1.0, 2.0, 3.0]
        window = 0
        steps = 2
        preds = ma._forecast_future(history, window, steps)

        # With window=0, uses entire buffer
        self.assertEqual(len(preds), 2)

    def test_steps_zero_raises_error(self):
        history = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError) as ctx:
            ma._forecast_future(history, window=2, steps=0)
        self.assertIn("[VALIDATION]", str(ctx.exception))
        self.assertIn("steps must be > 0", str(ctx.exception))

    def test_negative_steps_raises_error(self):
        history = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError) as ctx:
            ma._forecast_future(history, window=2, steps=-1)
        self.assertIn("[VALIDATION]", str(ctx.exception))

    def test_empty_history_raises_error(self):
        history = []
        with self.assertRaises(ValueError) as ctx:
            ma._forecast_future(history, window=2, steps=3)
        self.assertIn("[DATA]", str(ctx.exception))
        self.assertIn("empty", str(ctx.exception).lower())


class TestLoadState(unittest.TestCase):
    """Test model state loading"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_valid_state(self):
        model_path = os.path.join(self.temp_dir, "model.pkl")
        state = {
            "model_type": "ma",
            "window": 3,
            "history": [1.0, 2.0, 3.0],
            "train_end_index": 10,
            "std_dev_residuals": 0.5
        }
        dump_model(state, model_path)

        loaded = ma._load_state(model_path)
        self.assertEqual(loaded["model_type"], "ma")
        self.assertEqual(loaded["window"], 3)

    def test_load_invalid_model_type(self):
        model_path = os.path.join(self.temp_dir, "model.pkl")
        state = {
            "model_type": "arima",  # Wrong type
            "window": 3
        }
        dump_model(state, model_path)

        with self.assertRaises(ValueError) as ctx:
            ma._load_state(model_path)
        self.assertIn("[VALIDATION]", str(ctx.exception))
        self.assertIn("Invalid MA model", str(ctx.exception))

    def test_load_not_dict(self):
        model_path = os.path.join(self.temp_dir, "model.pkl")
        dump_model("not a dict", model_path)

        with self.assertRaises(ValueError) as ctx:
            ma._load_state(model_path)
        self.assertIn("[VALIDATION]", str(ctx.exception))


class TestPredictFunction(unittest.TestCase):
    """Test the standalone predict() function"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "model.pkl")

        # Create a valid model state
        self.state = {
            "model_type": "ma",
            "window": 3,
            "history": [1.0, 2.0, 3.0, 4.0, 5.0],
            "train_end_index": 4,
            "std_dev_residuals": 0.5,
            "value_key": "y"
        }
        dump_model(self.state, self.model_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_predict_without_data(self):
        preds, std_dev = ma.predict(self.model_path, steps=3)

        self.assertEqual(len(preds), 3)
        self.assertEqual(std_dev, 0.5)
        # First prediction should be mean of last 3 values: [3,4,5] = 4.0
        self.assertAlmostEqual(preds[0], 4.0, places=5)

    def test_predict_with_future_data(self):
        future_data = json.dumps([
            {"y": 1.0}, {"y": 2.0}, {"y": 3.0}, {"y": 4.0}, {"y": 5.0},
            {"y": 6.0}, {"y": 7.0}, {"y": 8.0}  # Future values after train_end_index=4
        ])
        preds, std_dev = ma.predict(self.model_path, steps=3, data=future_data)

        self.assertEqual(len(preds), 3)
        # Should use actual future values [6.0, 7.0, 8.0]
        # Step 1: mean([3,4,5]) = 4.0, use actual 6.0
        # Step 2: mean([4,5,6]) = 5.0, use actual 7.0
        # Step 3: mean([5,6,7]) = 6.0, use actual 8.0
        self.assertEqual(preds, [4.0, 5.0, 6.0])

    def test_predict_with_sales_field(self):
        # Test backward compatibility with "sales" field
        future_data = json.dumps([
            {"sales": 1.0}, {"sales": 2.0}, {"sales": 3.0}, {"sales": 4.0}, {"sales": 5.0},
            {"sales": 6.0}, {"sales": 7.0}
        ])
        preds, std_dev = ma.predict(self.model_path, steps=2, data=future_data)

        self.assertEqual(len(preds), 2)

    def test_predict_invalid_window(self):
        bad_state = self.state.copy()
        bad_state["window"] = 0
        bad_model_path = os.path.join(self.temp_dir, "bad_model.pkl")
        dump_model(bad_state, bad_model_path)

        with self.assertRaises(ValueError) as ctx:
            ma.predict(bad_model_path, steps=3)
        self.assertIn("[VALIDATION]", str(ctx.exception))
        self.assertIn("window must be >= 1", str(ctx.exception))

    def test_predict_empty_history(self):
        bad_state = self.state.copy()
        bad_state["history"] = []
        bad_model_path = os.path.join(self.temp_dir, "bad_model.pkl")
        dump_model(bad_state, bad_model_path)

        with self.assertRaises(ValueError) as ctx:
            ma.predict(bad_model_path, steps=3)
        self.assertIn("[DATA]", str(ctx.exception))
        self.assertIn("empty", str(ctx.exception).lower())

    def test_predict_history_shorter_than_window(self):
        bad_state = self.state.copy()
        bad_state["window"] = 10
        bad_state["history"] = [1.0, 2.0]  # Only 2 values but window=10
        bad_model_path = os.path.join(self.temp_dir, "bad_model.pkl")
        dump_model(bad_state, bad_model_path)

        with self.assertRaises(ValueError) as ctx:
            ma.predict(bad_model_path, steps=3)
        self.assertIn("[VALIDATION]", str(ctx.exception))
        self.assertIn("shorter than window", str(ctx.exception))


class TestTrainAndEvaluate(unittest.TestCase):
    """Test the train_and_evaluate function"""

    def test_normal_training(self):
        sales = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        months = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05",
                  "2020-06", "2020-07", "2020-08", "2020-09", "2020-10"]
        train_range = range(0, 7)  # 0-6
        eval_range = range(7, 10)  # 7-9
        window = 3

        metrics, std_dev, y_true, y_pred = ma.train_and_evaluate(
            sales, months, train_range, eval_range, window
        )

        self.assertEqual(metrics["window"], 3)
        self.assertEqual(metrics["evaluated_points"], 3)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("mape", metrics)
        self.assertIn("r2", metrics)
        self.assertEqual(len(y_true), 3)
        self.assertEqual(len(y_pred), 3)
        self.assertIsInstance(std_dev, float)

    def test_no_valid_evaluation_points(self):
        sales = [1.0, 2.0, 3.0, 4.0, 5.0]
        months = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05"]
        train_range = range(0, 3)
        eval_range = range(1, 3)  # Overlaps with train but window too large
        window = 5  # Too large

        with self.assertRaises(ValueError) as ctx:
            ma.train_and_evaluate(sales, months, train_range, eval_range, window)
        self.assertIn("[VALIDATION]", str(ctx.exception))
        self.assertIn("No evaluation points", str(ctx.exception))


class TestRunFunction(unittest.TestCase):
    """Integration tests for the main run() function"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "model.pkl")

        # Sample data
        self.sample_data = json.dumps([
            {"y": float(i), "month": f"2020-{i:02d}"} for i in range(1, 101)
        ])

        self.sample_data_sales = json.dumps([
            {"sales": float(i), "month": f"2020-{i:02d}"} for i in range(1, 101)
        ])

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _capture_run_output(self, args):
        """Helper to capture JSON output from run()"""
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            ma.run(args)
            output = captured.getvalue()
            return json.loads(output)
        finally:
            sys.stdout = old_stdout

    def test_training_mode_basic(self):
        args = MagicMock()
        args.data = self.sample_data
        args.train_start = 0
        args.train_end = 79
        args.eval_start = 80
        args.eval_end = 99
        args.window = 12
        args.save_model = self.model_path
        args.load_model = None
        args.pretty = False

        output = self._capture_run_output(args)

        self.assertEqual(output["mode"], "fixed_window_only")
        self.assertIn("metrics", output)
        self.assertIn("train_range", output)
        self.assertIn("evaluate_range", output)
        self.assertIn("eval_y_true", output)
        self.assertIn("eval_predictions", output)
        self.assertTrue(os.path.exists(self.model_path))

    def test_training_mode_with_sales_field(self):
        args = MagicMock()
        args.data = self.sample_data_sales
        args.train_start = 0
        args.train_end = 79
        args.eval_start = 80
        args.eval_end = 99
        args.window = 12
        args.save_model = None
        args.load_model = None
        args.pretty = False

        output = self._capture_run_output(args)

        self.assertEqual(output["mode"], "fixed_window_only")
        self.assertIn("metrics", output)

    def test_prediction_mode_with_forecast_steps(self):
        # First train a model
        train_args = MagicMock()
        train_args.data = self.sample_data
        train_args.train_start = 0
        train_args.train_end = 79
        train_args.eval_start = 80
        train_args.eval_end = 89
        train_args.window = 12
        train_args.save_model = self.model_path
        train_args.load_model = None
        train_args.pretty = False
        self._capture_run_output(train_args)

        # Now predict
        pred_args = MagicMock()
        pred_args.load_model = self.model_path
        pred_args.forecast_steps = 10
        pred_args.eval_start = None
        pred_args.eval_end = None
        pred_args.data = None
        pred_args.pretty = False

        output = self._capture_run_output(pred_args)

        self.assertEqual(output["mode"], "ma_loaded_predict")
        self.assertIn("predictions", output)
        self.assertEqual(len(output["predictions"]), 10)
        self.assertEqual(output["forecast_steps"], 10)
        # Check prediction format
        for pred in output["predictions"]:
            self.assertIn("prediction", pred)
            self.assertIn("std_dev", pred)

    def test_prediction_mode_with_eval_range(self):
        # First train
        train_args = MagicMock()
        train_args.data = self.sample_data
        train_args.train_start = 0
        train_args.train_end = 79
        train_args.eval_start = 80
        train_args.eval_end = 89
        train_args.window = 12
        train_args.save_model = self.model_path
        train_args.load_model = None
        train_args.pretty = False
        self._capture_run_output(train_args)

        # Predict with eval_start/eval_end
        pred_args = MagicMock()
        pred_args.load_model = self.model_path
        pred_args.forecast_steps = None
        pred_args.eval_start = 80
        pred_args.eval_end = 89
        pred_args.data = self.sample_data
        pred_args.pretty = False

        output = self._capture_run_output(pred_args)

        self.assertEqual(output["mode"], "ma_loaded_predict")
        self.assertIn("predictions", output)
        self.assertEqual(len(output["predictions"]), 10)
        # Should have metrics since we provided data
        self.assertIn("metrics", output)

    def test_validation_missing_data(self):
        args = MagicMock()
        args.data = None
        args.load_model = None
        args.pretty = False

        with self.assertRaises(ValueError) as ctx:
            self._capture_run_output(args)
        self.assertIn("[VALIDATION]", str(ctx.exception))
        self.assertIn("--data is required", str(ctx.exception))

    def test_validation_invalid_data_format(self):
        args = MagicMock()
        args.data = "not json"
        args.load_model = None
        args.pretty = False

        with self.assertRaises(ValueError) as ctx:
            self._capture_run_output(args)
        self.assertIn("[DATA]", str(ctx.exception))

    def test_validation_missing_target_key(self):
        args = MagicMock()
        args.data = json.dumps([{"value": 1.0}, {"value": 2.0}])
        args.load_model = None
        args.pretty = False

        with self.assertRaises(ValueError) as ctx:
            self._capture_run_output(args)
        self.assertIn("[DATA]", str(ctx.exception))
        self.assertIn("target key", str(ctx.exception).lower())

    def test_validation_invalid_train_range(self):
        args = MagicMock()
        args.data = self.sample_data
        args.train_start = 80
        args.train_end = 10  # end < start
        args.eval_start = 85
        args.eval_end = 89
        args.window = 5
        args.load_model = None
        args.pretty = False

        with self.assertRaises(ValueError) as ctx:
            self._capture_run_output(args)
        self.assertIn("[VALIDATION]", str(ctx.exception))

    def test_validation_invalid_eval_range(self):
        args = MagicMock()
        args.data = self.sample_data
        args.train_start = 0
        args.train_end = 79
        args.eval_start = 90
        args.eval_end = 80  # end < start
        args.window = 5
        args.load_model = None
        args.pretty = False

        with self.assertRaises(ValueError) as ctx:
            self._capture_run_output(args)
        self.assertIn("[VALIDATION]", str(ctx.exception))

    def test_validation_missing_window(self):
        args = MagicMock()
        args.data = self.sample_data
        args.train_start = 0
        args.train_end = 79
        args.eval_start = 80
        args.eval_end = 89
        args.window = None
        args.load_model = None
        args.pretty = False

        with self.assertRaises(ValueError) as ctx:
            self._capture_run_output(args)
        self.assertIn("[VALIDATION]", str(ctx.exception))
        self.assertIn("--window", str(ctx.exception))

    def test_validation_window_too_small(self):
        args = MagicMock()
        args.data = self.sample_data
        args.train_start = 0
        args.train_end = 79
        args.eval_start = 80
        args.eval_end = 89
        args.window = 0
        args.load_model = None
        args.pretty = False

        with self.assertRaises(ValueError) as ctx:
            self._capture_run_output(args)
        self.assertIn("[VALIDATION]", str(ctx.exception))
        self.assertIn("--window must be >= 1", str(ctx.exception))

    def test_validation_window_exceeds_training_length(self):
        args = MagicMock()
        args.data = self.sample_data
        args.train_start = 70
        args.train_end = 79  # Only 10 points
        args.eval_start = 80
        args.eval_end = 89
        args.window = 20  # Larger than training span
        args.load_model = None
        args.pretty = False

        with self.assertRaises(ValueError) as ctx:
            self._capture_run_output(args)
        self.assertIn("[VALIDATION]", str(ctx.exception))
        self.assertIn("cannot exceed training span", str(ctx.exception))

    def test_validation_load_mode_missing_steps(self):
        # First train
        train_args = MagicMock()
        train_args.data = self.sample_data
        train_args.train_start = 0
        train_args.train_end = 79
        train_args.eval_start = 80
        train_args.eval_end = 89
        train_args.window = 12
        train_args.save_model = self.model_path
        train_args.load_model = None
        train_args.pretty = False
        self._capture_run_output(train_args)

        # Try to predict without steps
        pred_args = MagicMock()
        pred_args.load_model = self.model_path
        pred_args.forecast_steps = None
        pred_args.eval_start = None
        pred_args.eval_end = None
        pred_args.pretty = False

        with self.assertRaises(ValueError) as ctx:
            self._capture_run_output(pred_args)
        self.assertIn("[VALIDATION]", str(ctx.exception))
        self.assertIn("eval_start/eval_end or --forecast_steps", str(ctx.exception))

    def test_pretty_output(self):
        args = MagicMock()
        args.data = self.sample_data
        args.train_start = 0
        args.train_end = 79
        args.eval_start = 80
        args.eval_end = 89
        args.window = 12
        args.save_model = None
        args.load_model = None
        args.pretty = True

        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            ma.run(args)
            output_str = captured.getvalue()
        finally:
            sys.stdout = old_stdout

        # Pretty output should have indentation
        self.assertIn("\n  ", output_str)  # Indented JSON
        # Should still be valid JSON
        json.loads(output_str)


class TestEdgeCasesAndBoundaries(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_single_training_point(self):
        """Window=1 with single training point"""
        args = MagicMock()
        args.data = json.dumps([{"y": float(i)} for i in range(1, 11)])
        args.train_start = 0
        args.train_end = 0  # Single point
        args.eval_start = 1
        args.eval_end = 5
        args.window = 1
        args.load_model = None
        args.save_model = None
        args.pretty = False

        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            ma.run(args)
            output = json.loads(captured.getvalue())
        finally:
            sys.stdout = old_stdout

        # Should work, each prediction uses the previous value
        self.assertIn("metrics", output)

    def test_large_window_small_data(self):
        """Window approaching data size"""
        args = MagicMock()
        args.data = json.dumps([{"y": float(i)} for i in range(1, 21)])
        args.train_start = 0
        args.train_end = 14
        args.eval_start = 15
        args.eval_end = 19
        args.window = 15  # Exactly the training length
        args.load_model = None
        args.save_model = None
        args.pretty = False

        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            ma.run(args)
            output = json.loads(captured.getvalue())
        finally:
            sys.stdout = old_stdout

        self.assertIn("metrics", output)

    def test_data_with_negative_values(self):
        """MA should work with negative values"""
        data = json.dumps([{"y": float(i - 10)} for i in range(1, 21)])
        args = MagicMock()
        args.data = data
        args.train_start = 0
        args.train_end = 9
        args.eval_start = 10
        args.eval_end = 14
        args.window = 3
        args.load_model = None
        args.save_model = None
        args.pretty = False

        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            ma.run(args)
            output = json.loads(captured.getvalue())
        finally:
            sys.stdout = old_stdout

        self.assertIn("metrics", output)
        # Predictions should also be negative
        self.assertTrue(any(p < 0 for p in output["eval_predictions"]))

    def test_data_with_very_large_values(self):
        """MA should handle large values"""
        data = json.dumps([{"y": float(i * 1e6)} for i in range(1, 21)])
        args = MagicMock()
        args.data = data
        args.train_start = 0
        args.train_end = 9
        args.eval_start = 10
        args.eval_end = 14
        args.window = 3
        args.load_model = None
        args.save_model = None
        args.pretty = False

        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            ma.run(args)
            output = json.loads(captured.getvalue())
        finally:
            sys.stdout = old_stdout

        self.assertIn("metrics", output)
        self.assertTrue(all(p > 1e6 for p in output["eval_predictions"]))


class TestPickleCompatibility(unittest.TestCase):
    """Test model serialization and deserialization"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_model(self):
        """Test complete save/load cycle"""
        model_path = os.path.join(self.temp_dir, "test_model.pkl")

        state = {
            "model_type": "ma",
            "window": 5,
            "history": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "train_end_index": 50,
            "std_dev_residuals": 1.23,
            "value_key": "y"
        }

        dump_model(state, model_path)
        loaded = load_model(model_path)

        self.assertEqual(loaded["window"], 5)
        self.assertEqual(loaded["history"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(loaded["train_end_index"], 50)
        self.assertAlmostEqual(loaded["std_dev_residuals"], 1.23)


if __name__ == '__main__':
    # Run with verbosity
    unittest.main(verbosity=2)
