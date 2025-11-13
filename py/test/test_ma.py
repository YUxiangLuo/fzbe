# py/test/test_ma.py
import unittest
import json
import sys
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO

# Add src to path to be able to import ma from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models import ma

class TestMaMetrics(unittest.TestCase):
    def test_rmse(self):
        self.assertAlmostEqual(ma.rmse([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertAlmostEqual(ma.rmse([1, 2, 3], [2, 3, 4]), 1.0)
        self.assertTrue(np.isnan(ma.rmse([], [])))

    def test_mae(self):
        self.assertAlmostEqual(ma.mae([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertAlmostEqual(ma.mae([1, 2, 3], [2, 3, 4]), 1.0)
        self.assertTrue(np.isnan(ma.mae([], [])))

    def test_mape(self):
        self.assertAlmostEqual(ma.mape([100, 200], [110, 220]), 10.0)
        self.assertAlmostEqual(ma.mape([100], [90]), 10.0)
        # Test with zero in true values
        self.assertAlmostEqual(ma.mape([0, 100], [10, 110]), (10/1e-12 * 100 + 10)/2)

    def test_r2_score(self):
        self.assertAlmostEqual(ma.r2_score([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(ma.r2_score([1, 2, 3], [1.1, 2.2, 2.9]), 0.97)


class TestMaCoreLogic(unittest.TestCase):
    def setUp(self):
        self.series = [10.0, 12.0, 15.0, 11.0, 14.0, 16.0, 18.0, 20.0]
        self.months = [f"2022-{i+1:02d}" for i in range(len(self.series))]
        self.data_sales = [{"month": m, "sales": s} for m, s in zip(self.months, self.series)]
        self.data_y = [{"ts": m, "y": s} for m, s in zip(self.months, self.series)]

    def test_extract_series(self):
        months, sales, value_key = ma._extract_series(self.data_sales)
        self.assertEqual(sales, self.series)
        self.assertEqual(months, self.months)
        self.assertEqual(value_key, "sales")

        months_y, sales_y, value_key_y = ma._extract_series(self.data_y)
        self.assertEqual(sales_y, self.series)
        self.assertEqual(value_key_y, "y")

        with self.assertRaisesRegex(ValueError, r"\[DATA\] .* must not be empty"):
            ma._extract_series([])
        with self.assertRaisesRegex(ValueError, r"\[DATA\] .* must contain 'sales' or 'y'"):
            ma._extract_series([{"a": 1}])

    def test_moving_average_forecast(self):
        # Test normal forecast
        idxs, preds = ma.moving_average_forecast(self.series, range(3, 6), window=3)
        self.assertEqual(idxs, [3, 4, 5])
        self.assertAlmostEqual(preds[0], (10 + 12 + 15) / 3) # pred for index 3
        self.assertAlmostEqual(preds[1], (12 + 15 + 11) / 3) # pred for index 4
        self.assertAlmostEqual(preds[2], (15 + 11 + 14) / 3) # pred for index 5

        # Test with min_index skipping points
        idxs_skip, _ = ma.moving_average_forecast(self.series, range(0, 4), window=3, min_index=0)
        self.assertEqual(idxs_skip, [3]) # Should skip 0, 1, 2

        # Test validation
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] .* must be greater than 0"):
            ma.moving_average_forecast(self.series, range(3, 6), window=0)

    def test_forecast_future(self):
        history = [10, 12, 14]
        # Test without future values
        preds = ma._forecast_future(history, window=2, steps=3)
        self.assertEqual(len(preds), 3)
        self.assertAlmostEqual(preds[0], 13.0) # (12+14)/2
        self.assertAlmostEqual(preds[1], (14 + 13.0) / 2) # 13.5
        self.assertAlmostEqual(preds[2], (13.0 + 13.5) / 2) # 13.25

        # Test with future values
        future_actuals = [15, 18]
        preds_w_actuals = ma._forecast_future(history, window=2, steps=3, future_values=future_actuals)
        self.assertEqual(len(preds_w_actuals), 3)
        self.assertAlmostEqual(preds_w_actuals[0], 13.0) # (12+14)/2
        self.assertAlmostEqual(preds_w_actuals[1], (14 + 15) / 2) # 14.5, uses first actual
        self.assertAlmostEqual(preds_w_actuals[2], (15 + 18) / 2) # 16.5, uses second actual

        # Test validation
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] steps must be > 0 for forecasting\."):
            ma._forecast_future(history, window=2, steps=0)
        with self.assertRaisesRegex(ValueError, r"\[DATA\] History is empty"):
            ma._forecast_future([], window=2, steps=2)


class TestMaTrainAndPredict(unittest.TestCase):
    def setUp(self):
        self.series = [10, 12, 15, 11, 14, 16, 18, 20, 22, 25]
        self.months = [f"2022-{i+1:02d}" for i in range(len(self.series))]
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_train_and_evaluate(self):
        metrics, std_dev, y_true, y_pred = ma.train_and_evaluate(
            self.series, self.months, train_range=range(0, 7), eval_range=range(7, 10), window=3
        )
        self.assertEqual(metrics["evaluated_points"], 3)
        self.assertEqual(y_true, [20, 22, 25])
        self.assertAlmostEqual(y_pred[0], (14 + 16 + 18) / 3) # pred for index 7
        self.assertTrue(std_dev > 0)

        # Test validation for no evaluation points
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] No evaluation points"):
            ma.train_and_evaluate(self.series, self.months, range(0, 3), range(1, 3), window=3)

    def test_predict(self):
        model_path = os.path.join(self.temp_dir.name, "test_ma.pkl")
        state = {
            "model_type": "ma",
            "window": 3,
            "history": [16, 18, 20],
            "train_end_index": 7,
            "std_dev_residuals": 1.5
        }
        ma.dump_joblib(state, model_path)

        preds, std_dev = ma.predict(model_path, steps=2)
        self.assertEqual(len(preds), 2)
        self.assertAlmostEqual(std_dev, 1.5)
        self.assertAlmostEqual(preds[0], (16 + 18 + 20) / 3) # 18.0

        # Test loading invalid model
        ma.dump_joblib({"model_type": "wrong"}, model_path)
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] Invalid MA model file"):
            ma.predict(model_path, steps=2)


class TestMaRunFunction(unittest.TestCase):
    def setUp(self):
        self.series = [10, 12, 15, 11, 14, 16, 18, 20, 22, 25]
        self.data_str = json.dumps([{"month": f"2022-{i+1:02d}", "sales": s} for i, s in enumerate(self.series)])
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_run_training_success(self):
        args = MagicMock()
        args.method = "ma"
        args.data = self.data_str
        args.train_start = 0
        args.train_end = 6
        args.eval_start = 7
        args.eval_end = 9
        args.window = 3
        args.save_model = os.path.join(self.temp_dir.name, "model.pkl")
        args.load_model = None
        args.pretty = True

        with patch('sys.stdout', new=StringIO()) as fake_out:
            ma.run(args)
            output = json.loads(fake_out.getvalue())

        self.assertEqual(output["mode"], "fixed_window_only")
        self.assertEqual(output["metrics"]["window"], 3)
        self.assertEqual(output["metrics"]["evaluated_points"], 3)
        self.assertIn("saved_model", output)
        self.assertTrue(os.path.exists(output["saved_model"]))

    def test_run_training_validation_error(self):
        args = MagicMock()
        args.data = self.data_str
        args.train_start = 0
        args.train_end = 6
        args.eval_start = 7
        args.eval_end = 9
        args.window = 10 # Window > train length
        args.load_model = None

        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] --window .* cannot exceed training span length"):
            ma.run(args)

    def test_run_loading_predict_success(self):
        # First, save a model
        model_path = os.path.join(self.temp_dir.name, "model.pkl")
        state = {"model_type": "ma", "window": 3, "history": [18, 20, 22], "train_end_index": 8, "std_dev_residuals": 1.0}
        ma.dump_joblib(state, model_path)

        args = MagicMock()
        args.load_model = model_path
        args.forecast_steps = 4
        # Mock the absence of optional args
        args.data = None
        args.eval_start = None
        args.eval_end = None
        args.pretty = False

        with patch('sys.stdout', new=StringIO()) as fake_out:
            ma.run(args)
            output = json.loads(fake_out.getvalue())

        self.assertEqual(output["mode"], "ma_loaded_predict")
        self.assertEqual(output["forecast_steps"], 4)
        self.assertEqual(len(output["predictions"]), 4)
        self.assertAlmostEqual(output["predictions"][0]["prediction"], (18+20+22)/3)
        self.assertAlmostEqual(output["predictions"][0]["std_dev"], 1.0)
        self.assertNotIn("metrics", output)

    def test_run_loading_predict_and_eval_with_metrics_error(self):
        # Test the case where the new metrics_error field appears
        model_path = os.path.join(self.temp_dir.name, "model.pkl")
        state = {"model_type": "ma", "window": 3, "history": [18, 20, 22], "train_end_index": 8, "std_dev_residuals": 1.0}
        ma.dump_joblib(state, model_path)

        args = MagicMock()
        args.load_model = model_path
        args.data = self.data_str
        args.eval_start = 9 # Set eval_start to trigger metrics calculation
        args.eval_end = 9   # Set eval_end to trigger metrics calculation (y_true length = 1)
        args.forecast_steps = 2 # Request 2 forecast steps (preds length = 2)
        args.pretty = False

        # To trigger metrics_error, we need to provide data for evaluation
        # but ensure its length doesn't match forecast_steps.
        # We will use eval_start=9, eval_end=9 in the data_str to get y_true length 1.
        # Since forecast_steps is 2, len(y_true) != len(preds) will be true.

        with patch('sys.stdout', new=StringIO()) as fake_out:
            ma.run(args)
            output = json.loads(fake_out.getvalue())
        
        self.assertEqual(output["mode"], "ma_loaded_predict")
        self.assertIn("metrics_error", output)
        self.assertIn("Length of true values and predictions do not match", output["metrics_error"])


if __name__ == '__main__':
    unittest.main()
