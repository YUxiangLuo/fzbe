# py/test/test_es.py
import unittest
import json
import sys
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO

# Add src to path to be able to import es from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models import es

class TestEsMetrics(unittest.TestCase):
    def test_rmse(self):
        self.assertAlmostEqual(es.rmse([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertAlmostEqual(es.rmse([1, 2, 3], [2, 3, 4]), 1.0)
        self.assertTrue(np.isnan(es.rmse([], [])))

    def test_mae(self):
        self.assertAlmostEqual(es.mae([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertAlmostEqual(es.mae([1, 2, 3], [2, 3, 4]), 1.0)
        self.assertTrue(np.isnan(es.mae([], [])))

    def test_mape(self):
        self.assertAlmostEqual(es.mape([100, 200], [110, 220]), 10.0)
        self.assertAlmostEqual(es.mape([100], [90]), 10.0)
        # Test with zero in true values
        self.assertAlmostEqual(es.mape([0, 100], [10, 110]), (10/1e-12 * 100 + 10)/2)

    def test_r2_score(self):
        self.assertAlmostEqual(es.r2_score([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(es.r2_score([1, 2, 3], [1.1, 2.2, 2.9]), 0.97)


class TestEsCoreLogic(unittest.TestCase):
    def setUp(self):
        self.series = [10.0, 12.0, 15.0, 11.0, 14.0, 16.0, 18.0, 20.0]
        self.months = [f"2022-{i+1:02d}" for i in range(len(self.series))]
        self.data_sales = [{"month": m, "sales": s} for m, s in zip(self.months, self.series)]
        self.data_y = [{"ts": m, "y": s} for m, s in zip(self.months, self.series)]

    def test_extract_series(self):
        months, sales, value_key = es._extract_series(self.data_sales)
        self.assertEqual(sales, self.series)
        self.assertEqual(months, self.months)
        self.assertEqual(value_key, "sales")

        months_y, sales_y, value_key_y = es._extract_series(self.data_y)
        self.assertEqual(sales_y, self.series)
        self.assertEqual(value_key_y, "y")

        with self.assertRaisesRegex(ValueError, r"\[DATA\] `--data` must not be empty"):
            es._extract_series([])
        with self.assertRaisesRegex(ValueError, r"\[DATA\] Data items must contain 'sales' or 'y'"):
            es._extract_series([{"a": 1}])

    def test_load_state(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        model_path = os.path.join(temp_dir.name, "test_es.pkl")
        
        # Test valid state
        valid_state = {"model_type": "es", "alpha": 0.5, "level": 10.0, "train_end_index": 5, "std_dev_residuals": 1.0}
        es.dump_joblib(valid_state, model_path)
        loaded_state = es._load_state(model_path)
        self.assertEqual(loaded_state, valid_state)

        # Test invalid model_type
        invalid_state = {"model_type": "ma", "alpha": 0.5, "level": 10.0, "train_end_index": 5, "std_dev_residuals": 1.0}
        es.dump_joblib(invalid_state, model_path)
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] Invalid ES model file"):
            es._load_state(model_path)


class TestEsTrainAndPredict(unittest.TestCase):
    def setUp(self):
        self.series = [10, 12, 15, 11, 14, 16, 18, 20, 22, 25]
        self.months = [f"2022-{i+1:02d}" for i in range(len(self.series))]
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_train_and_evaluate(self):
        metrics, std_dev, final_level, y_true, y_pred = es.train_and_evaluate(
            self.series, self.months, train_range=range(0, 7), eval_range=range(7, 10), alpha=0.5
        )
        self.assertEqual(metrics["rmse"], es.rmse(y_true, y_pred))
        self.assertEqual(len(y_true), 3)
        self.assertAlmostEqual(y_true[0], 20.0)
        self.assertTrue(std_dev > 0)
        self.assertTrue(final_level > 0)

        # Test validation for no evaluation points
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] No evaluation points"):
            es.train_and_evaluate(self.series, self.months, train_range=range(0, 1), eval_range=range(0, 1), alpha=0.5) # eval_start <= train_start

    def test_predict(self):
        model_path = os.path.join(self.temp_dir.name, "test_es.pkl")
        state = {
            "model_type": "es",
            "alpha": 0.5,
            "level": 18.0, # Level at train_end
            "train_end_index": 7,
            "std_dev_residuals": 1.5
        }
        es.dump_joblib(state, model_path)

        # Test with future actuals
        # The series is [10, 12, 15, 11, 14, 16, 18, 20, 22, 25]
        # train_end_index is 7, so sales[7] is 20.
        # We need future values starting from index 8.
        future_data_str = json.dumps([{"month": "2022-09", "sales": 22}, {"month": "2022-10", "sales": 25}, {"month": "2022-11", "sales": 28}])
        raw_preds, _ = es.predict(model_path, steps=3, data=future_data_str)
        preds_w_actuals = raw_preds
        self.assertEqual(len(preds_w_actuals), 3)
        self.assertAlmostEqual(preds_w_actuals[0], 18.0) # First pred is still last level
        self.assertAlmostEqual(preds_w_actuals[1], 20.0) # Second prediction uses the updated level
        self.assertAlmostEqual(preds_w_actuals[2], 22.5) # This should be the level after consuming 25

        # Test validation
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] steps must be > 0 for forecasting"):
            es.predict(model_path, steps=0)


class TestEsRunFunction(unittest.TestCase):
    def setUp(self):
        self.series = [10, 12, 15, 11, 14, 16, 18, 20, 22, 25]
        self.data_str = json.dumps([{"month": f"2022-{i+1:02d}", "sales": s} for i, s in enumerate(self.series)])
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_run_training_success(self):
        args = MagicMock()
        args.method = "es"
        args.data = self.data_str
        args.train_start = 0
        args.train_end = 6
        args.eval_start = 7
        args.eval_end = 9
        args.alpha = 0.5
        args.save_model = os.path.join(self.temp_dir.name, "model.pkl")
        args.load_model = None
        args.pretty = True

        with patch('sys.stdout', new=StringIO()) as fake_out:
            es.run(args)
            output = json.loads(fake_out.getvalue())

        self.assertEqual(output["mode"], "ses_fixed_alpha")
        self.assertEqual(output["alpha"], 0.5)
        self.assertEqual(output["metrics"]["rmse"], es.rmse([20, 22, 25], output["eval_predictions"]))
        self.assertIn("saved_model", output)
        self.assertTrue(os.path.exists(output["saved_model"]))

    def test_run_training_validation_error(self):
        args = MagicMock()
        args.data = self.data_str
        args.train_start = 0
        args.train_end = 6
        args.eval_start = 0 # eval_start < train_start + 1
        args.eval_end = 2
        args.alpha = 0.5
        args.load_model = None

        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] No evaluation points"):
            es.run(args)

    def test_run_loading_predict_success(self):
        # First, save a model
        model_path = os.path.join(self.temp_dir.name, "model.pkl")
        state = {"model_type": "es", "alpha": 0.5, "level": 18.0, "train_end_index": 7, "std_dev_residuals": 1.0}
        es.dump_joblib(state, model_path)

        args = MagicMock()
        args.load_model = model_path
        args.forecast_steps = 4
        args.data = None
        args.eval_start = None
        args.eval_end = None
        args.pretty = False

        with patch('sys.stdout', new=StringIO()) as fake_out:
            es.run(args)
            output = json.loads(fake_out.getvalue())

        self.assertEqual(output["mode"], "es_loaded_predict")
        self.assertEqual(output["forecast_steps"], 4)
        self.assertEqual(len(output["predictions"]), 4)
        self.assertAlmostEqual(output["predictions"][0]["prediction"], 18.0)
        self.assertAlmostEqual(output["predictions"][0]["std_dev"], 1.0)
        self.assertNotIn("metrics", output)

    def test_run_loading_predict_and_eval_with_metrics_error(self):
        # Test the case where the new metrics_error field appears
        model_path = os.path.join(self.temp_dir.name, "model.pkl")
        state = {"model_type": "es", "alpha": 0.5, "level": 18.0, "train_end_index": 7, "std_dev_residuals": 1.0}
        es.dump_joblib(state, model_path)

        args = MagicMock()
        args.load_model = model_path
        args.data = self.data_str
        args.eval_start = 9 # y_true length = 1
        args.eval_end = 9
        args.forecast_steps = 2 # preds length = 2
        args.pretty = False

        with patch('sys.stdout', new=StringIO()) as fake_out:
            es.run(args)
            output = json.loads(fake_out.getvalue())
        
        self.assertEqual(output["mode"], "es_loaded_predict")
        self.assertIn("metrics_error", output)
        self.assertIn("Length of true values and predictions do not match", output["metrics_error"])


if __name__ == '__main__':
    unittest.main()
