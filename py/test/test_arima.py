# py/test/test_arima.py
import unittest
import json
import sys
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Add src to path to be able to import arima from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models import arima
from metrics import rmse, mae, mape, r2

class TestArimaUtilities(unittest.TestCase):
    def setUp(self):
        self.valid_data_str = json.dumps([{"month": "2022-01", "sales": 10}, {"month": "2022-02", "sales": 12}])
        self.valid_data_no_month_str = json.dumps([{"sales": 10}, {"sales": 12}])
        self.series = np.array([10.0, 12.0, 15.0, 11.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0], dtype=np.float64)

    def test_parse_series(self):
        y, months = arima._parse_series(self.valid_data_str)
        np.testing.assert_array_almost_equal(y, np.array([10.0, 12.0]))
        self.assertEqual(months, ["2022-01", "2022-02"])

        y_no_month, months_no_month = arima._parse_series(self.valid_data_no_month_str)
        np.testing.assert_array_almost_equal(y_no_month, np.array([10.0, 12.0]))
        self.assertIsNone(months_no_month)

        with self.assertRaisesRegex(ValueError, r"\[DATA\] --data must be a JSON array of objects: Expecting value: line 1 column 1 \(char 0\)"):
            arima._parse_series("invalid json")
        with self.assertRaisesRegex(ValueError, r"^\[DATA\] --data must be a JSON array$"):
            arima._parse_series("{\"sales\": 10}")
        with self.assertRaisesRegex(ValueError, r"^\[DATA\] item\[0\] must be an object$"):
            arima._parse_series("[10]")
        with self.assertRaisesRegex(ValueError, r"^\[DATA\] item\[0\] missing 'sales'$"):
            arima._parse_series("[{\"month\": \"2022-01\"}]")
        with self.assertRaisesRegex(ValueError, r"^\[DATA\] item\[0\]\.sales must be a number$"):
            arima._parse_series("[{\"sales\": \"abc\"}]")
        with self.assertRaisesRegex(ValueError, r"^\[DATA\] item\[0\]\.sales must be finite$"):
            arima._parse_series(json.dumps([{"sales": float("inf") }]))
        with self.assertRaisesRegex(ValueError, r"^\[DATA\] no data points in --data$"):
            arima._parse_series("[]")
        with self.assertRaisesRegex(ValueError, r"^\[DATA\] length of 'months' must match 'sales'$"):
            arima._parse_series(json.dumps([{"month": "2022-01", "sales": 10}, {"sales": 12}]))

    def test_validate_ranges(self):
        n = len(self.series)
        arima._validate_ranges(n, 0, 7, 8, 9) # Valid

        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] train range invalid: ts=-1, te=7, n=10"):
            arima._validate_ranges(n, -1, 7, 8, 9)
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] eval range invalid: es=8, ee=10, n=10"):
            arima._validate_ranges(n, 0, 7, 8, n)
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] eval_start must be > train_end for ARIMA to ensure genuine out-of-sample evaluation."):
            arima._validate_ranges(n, 0, 7, 7, 9)
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] eval range invalid: es=8, ee=7, n=10"):
            arima._validate_ranges(n, 0, 7, 8, 7)


class TestArimaCoreLogic(unittest.TestCase):
    def setUp(self):
        self.y_train = np.array([10, 12, 15, 11, 14, 16, 18, 20], dtype=np.float64)
        self.y_eval_true = np.array([22, 25], dtype=np.float64)
        self.d = 0 # Assuming stationary data for simplicity in these tests

    @patch('models.arima.SARIMAX')
    def test_fit_and_score_success(self, MockSARIMAX):
        mock_results_instance = MagicMock()
        mock_results_instance.aic = 10.0
        mock_results_instance.mle_retvals = {"converged": True}
        
        mock_forecast_instance = MagicMock()
        mock_forecast_instance.predicted_mean = np.array([21.0, 24.0])
        mock_forecast_instance.conf_int.return_value = np.array([[20, 22], [23, 25]])
        mock_forecast_instance.se_mean = np.array([0.5, 0.6])
        mock_results_instance.get_forecast.return_value = mock_forecast_instance

        MockSARIMAX.return_value.fit.return_value = mock_results_instance

        result, res_obj = arima._fit_and_score(self.y_train, self.y_eval_true, self.d, 1, 1)

        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["aic"], 10.0)
        self.assertAlmostEqual(result["rmse"], rmse(self.y_eval_true.tolist(), [21.0, 24.0]))
        np.testing.assert_array_almost_equal(result["y_pred"], np.array([21.0, 24.0]))
        self.assertIsNotNone(res_obj)

    @patch('models.arima.SARIMAX')
    def test_fit_and_score_failure(self, MockSARIMAX):
        MockSARIMAX.return_value.fit.side_effect = Exception("Fit failed")
        # Ensure get_forecast is also mocked to avoid errors if it's called before the exception is caught
        MockSARIMAX.return_value.fit.return_value.get_forecast.return_value.predicted_mean = np.array([0.0, 0.0])

        result, res_obj = arima._fit_and_score(self.y_train, self.y_eval_true, self.d, 1, 1)

        self.assertIn("error", result)
        self.assertFalse(result["converged"])
        self.assertIsNone(res_obj)

    @patch('models.arima.SARIMAX')
    def test_search_pq_success(self, MockSARIMAX):
        # Mock _fit_and_score to return successful results for some (p,q)
        def mock_fit_and_score(y_train, y_eval_true, d, p, q):
            if (p, q) == (1, 0): # Best model
                return {"p": 1, "d": d, "q": 0, "rmse": 1.0, "aic": 5.0, "converged": True, "y_pred": np.array([21.0, 24.0]), "conf_int": np.array([[20, 22], [23, 25]])}, MagicMock()
            elif (p, q) == (0, 1):
                return {"p": 0, "d": d, "q": 1, "rmse": 1.5, "aic": 6.0, "converged": True, "y_pred": np.array([21.0, 24.0]), "conf_int": np.array([[20, 22], [23, 25]])}, MagicMock()
            else:
                return {"p": p, "d": d, "q": q, "error": "Fit failed", "converged": False}, None

        arima._fit_and_score = MagicMock(side_effect=mock_fit_and_score)

        best_item, best_res, search_summary = arima._search_pq(self.y_train, self.y_eval_true, self.d, p_max=1, q_max=1)

        self.assertEqual(best_item["p"], 1)
        self.assertEqual(best_item["q"], 0)
        self.assertEqual(search_summary["succeeded"], 2)
        self.assertEqual(search_summary["failed"], 2) # (0,0) and (1,1) failed

    @patch('models.arima.SARIMAX')
    def test_search_pq_all_failures(self, MockSARIMAX):
        # Mock _fit_and_score to always fail
        arima._fit_and_score = MagicMock(return_value=(
            {"p": 0, "d": self.d, "q": 0, "error": "Fit failed", "converged": False}, None
        ))

        with self.assertRaisesRegex(ValueError, r"^\[VALIDATION\] All \(p,q\) fits failed for d="):
            arima._search_pq(self.y_train, self.y_eval_true, self.d, p_max=1, q_max=1)

    def test_search_pq_train_data_length_validation(self):
        with self.assertRaisesRegex(ValueError, r"^\[VALIDATION\] Training data length .* must be greater than d"):
            arima._search_pq(np.array([10]), self.y_eval_true, d=1, p_max=1, q_max=1)


class TestArimaPredict(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.model_path = os.path.join(self.temp_dir.name, "test_arima_model.pkl")

        # Create a dummy SARIMAXResultsWrapper object for mocking
        self.mock_res = MagicMock()
        self.mock_res.model.order = (1, 0, 1)
        self.mock_res.nobs = 10 # Example training data length
        self.mock_res.get_forecast.side_effect = lambda steps: MagicMock(
            predicted_mean=np.arange(21.0, 21.0 + steps * 3, 3.0)[:steps],
            conf_int=MagicMock(return_value=np.array([[20, 22]] * steps)),
            se_mean=np.array([0.5] * steps),
        )
        self.mock_res.get_prediction.side_effect = lambda start, end: MagicMock(
            predicted_mean=np.arange(19.0, 19.0 + (end - start + 1) * 2, 2.0)[:(end - start + 1)],
            conf_int=MagicMock(return_value=np.array([[18, 20]] * (end - start + 1))),
            se_mean=np.array([0.4] * (end - start + 1)),
        )

    @patch('models.arima.load_joblib')
    def test_predict_function(self, mock_load_joblib):
        mock_load_joblib.return_value = self.mock_res
        preds = arima.predict(self.model_path, steps=2)
        self.assertEqual(len(preds), 2)
        self.assertAlmostEqual(preds[0]["prediction"], 21.0)
        self.assertAlmostEqual(preds[0]["std_dev"], 0.5)

        with self.assertRaisesRegex(ValueError, r"^\[VALIDATION\] steps must be > 0"):
            arima.predict(self.model_path, steps=0)

    @patch('models.arima.load_joblib')
    def test_handle_prediction_oos_forecast_steps(self, mock_load_joblib):
        mock_load_joblib.return_value = self.mock_res
        args = MagicMock()
        args.load_model = self.model_path
        args.forecast_steps = 3
        args.eval_start = None
        args.eval_end = None
        args.ci_level = 0.95
        args.data = None
        args.pretty = False

        with patch('sys.stdout', new=StringIO()) as fake_out:
            arima._handle_prediction(args, pretty=False)
            output = json.loads(fake_out.getvalue())

        self.assertEqual(output["mode"], "arima_loaded_predict")
        self.assertEqual(output["forecast_steps"], 3)
        self.assertEqual(len(output["predictions"]), 3)
        self.assertAlmostEqual(output["predictions"][0]["prediction"], 21.0)
        self.assertAlmostEqual(output["predictions"][0]["std_dev"], 0.5)
        self.assertIn("conf_int", output)
        self.assertEqual(output["order"], {"p": 1, "d": 0, "q": 1})

    @patch('models.arima.load_joblib')
    def test_handle_prediction_in_sample(self, mock_load_joblib):
        mock_load_joblib.return_value = self.mock_res
        args = MagicMock()
        args.load_model = self.model_path
        args.eval_start = 8
        args.eval_end = 9
        args.forecast_steps = None
        args.ci_level = 0.95
        args.data = None
        args.pretty = False

        with patch('sys.stdout', new=StringIO()) as fake_out:
            arima._handle_prediction(args, pretty=False)
            output = json.loads(fake_out.getvalue())

        self.assertEqual(output["mode"], "arima_loaded_predict")
        self.assertEqual(output["forecast_steps"], 2)
        self.assertEqual(len(output["predictions"]), 2)
        self.assertAlmostEqual(output["predictions"][0]["prediction"], 19.0)
        self.assertAlmostEqual(output["predictions"][0]["std_dev"], 0.4)
        self.assertEqual(output["meta"]["mode"], "in_sample")

    @patch('models.arima.load_joblib')
    def test_handle_prediction_mixed_sample(self, mock_load_joblib):
        mock_load_joblib.return_value = self.mock_res
        args = MagicMock()
        args.load_model = self.model_path
        args.eval_start = 9 # In-sample (index 9)
        args.eval_end = 10 # Out-of-sample (index 10, nobs=10)
        args.forecast_steps = None
        args.ci_level = 0.95
        args.data = None
        args.pretty = False

        with patch('sys.stdout', new=StringIO()) as fake_out:
            arima._handle_prediction(args, pretty=False)
            output = json.loads(fake_out.getvalue())

        self.assertEqual(output["mode"], "arima_loaded_predict")
        self.assertEqual(output["forecast_steps"], 2)
        self.assertEqual(len(output["predictions"]), 2)
        # First prediction is in-sample (index 9)
        self.assertAlmostEqual(output["predictions"][0]["prediction"], 19.0) # Assuming get_prediction for index 9
        # Second prediction is OOS (index 10)
        self.assertAlmostEqual(output["predictions"][1]["prediction"], 21.0) # Assuming get_forecast for 1 step
        self.assertEqual(output["meta"]["mode"], "mixed")

    @patch('models.arima.load_joblib')
    def test_handle_prediction_metrics_error(self, mock_load_joblib):
        mock_load_joblib.return_value = self.mock_res
        args = MagicMock()
        args.load_model = self.model_path
        args.eval_start = 8
        args.eval_end = 9
        args.forecast_steps = None
        args.ci_level = 0.95
        args.data = json.dumps([{"sales": 10}]) # Data with mismatching length
        args.pretty = False

        with patch('sys.stdout', new=StringIO()) as fake_out:
            arima._handle_prediction(args, pretty=False)
            output = json.loads(fake_out.getvalue())
        
        self.assertEqual(output["mode"], "arima_loaded_predict")
        self.assertIn("metrics_error", output)
        self.assertIn("[VALIDATION] Series length mismatch for metrics", output["metrics_error"])


class TestArimaRunFunction(unittest.TestCase):
    def setUp(self):
        self.series = [10, 12, 15, 11, 14, 16, 18, 20, 22, 25]
        self.data_str = json.dumps([{"month": f"2022-{i+1:02d}", "sales": s} for i, s in enumerate(self.series)])
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    @patch('models.arima._search_pq')
    @patch('models.arima.dump_joblib')
    def test_run_training_success(self, mock_dump_joblib, mock_search_pq):
        mock_search_pq.return_value = (
            {"p": 1, "d": 0, "q": 1, "rmse": 1.0, "mae": 0.8, "mape": 5.0, "r2": 0.9, "aic": 5.0, "converged": True, "y_pred": np.array([21.0, 24.0]), "conf_int": np.array([[20, 22], [23, 25]])}, 
            MagicMock(), 
            {"succeeded": 1, "failed": 0}
        )

        args = MagicMock()
        args.method = "arima"
        args.data = self.data_str
        args.train_start = 0
        args.train_end = 7
        args.eval_start = 8
        args.eval_end = 9
        args.d = 0
        args.p_max = 1
        args.q_max = 1
        args.save_model = os.path.join(self.temp_dir.name, "model.pkl")
        args.load_model = None
        args.pretty = True

        with patch('sys.stdout', new=StringIO()) as fake_out:
            arima.run(args)
            output = json.loads(fake_out.getvalue())

        mock_dump_joblib.assert_called_once()
    def test_run_training_validation_error(self):
        args = MagicMock()
        args.data = None # Missing data
        args.load_model = None

        with self.assertRaisesRegex(ValueError, r"^\[VALIDATION\] ARIMA training requires --data"):
            arima.run(args)

    @patch('models.arima._handle_prediction')
    def test_run_loading_predict_success(self, mock_handle_prediction):
        args = MagicMock()
        args.load_model = "dummy_path.pkl"
        args.pretty = False

        arima.run(args)
        mock_handle_prediction.assert_called_once_with(args, pretty=False)


if __name__ == '__main__':
    unittest.main()
