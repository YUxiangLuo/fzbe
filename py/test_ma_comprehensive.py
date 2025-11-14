# Comprehensive unit tests for ma.py (optimized)
import json
import os
import sys
import tempfile
import time
import unittest

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import numpy as np
from models.ma import (
    MODEL_TYPE,
    _add_metrics_to_output,
    _forecast_future,
    _load_state,
    _validate_training_args,
    moving_average_forecast,
    predict,
    run,
    train_and_evaluate,
)


class TestMetrics(unittest.TestCase):
    """Test metrics functions."""

    def test_rmse_normal(self):
        from metrics import rmse

        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1]
        self.assertAlmostEqual(rmse(y_true, y_pred), 0.1, places=5)


class TestMovingAverageForecast(unittest.TestCase):
    """Test moving average forecast function with vectorization."""

    def test_normal_forecast(self):
        series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        eval_range = range(5, 10)
        idxs, preds = moving_average_forecast(series, eval_range, 3)
        self.assertEqual(idxs, [5, 6, 7, 8, 9])
        expected_preds = [4.0, 5.0, 6.0, 7.0, 8.0]
        self.assertEqual(preds, expected_preds)

    def test_window_zero_raises_error(self):
        series = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            moving_average_forecast(series, range(1, 3), 0)


class TestForecastFuture(unittest.TestCase):
    """Test multi-step forecast function."""

    def test_forecast_without_future_values(self):
        history = [1, 2, 3, 4, 5]
        preds = _forecast_future(history, 3, 2)
        expected = [
            (3 + 4 + 5) / 3,
            (4 + 5 + 4) / 3,
        ]  # Correct logic: update buffer with prediction
        self.assertEqual(preds, expected)

    def test_steps_zero_raises_error(self):
        with self.assertRaises(ValueError):
            _forecast_future([1, 2, 3], 2, 0)


class TestLoadState(unittest.TestCase):
    """Test loading model state."""

    def setUp(self):
        from serialization import dump_model

        self.valid_state = {
            "model_type": MODEL_TYPE,
            "window": 3,
            "train_end_index": 9,
            "history": [8, 9, 10],
            "value_key": "sales",
            "std_dev_residuals": 0.5,
        }
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        dump_model(self.valid_state, self.temp_file.name)

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_load_valid_state(self):
        state = _load_state(self.temp_file.name)
        self.assertEqual(state, self.valid_state)


class TestPredictFunction(unittest.TestCase):
    """Test predict function."""

    def setUp(self):
        from serialization import dump_model

        self.state = {
            "model_type": MODEL_TYPE,
            "window": 3,
            "train_end_index": 9,
            "history": [8, 9, 10],
            "value_key": "sales",
            "std_dev_residuals": 0.5,
        }
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        dump_model(self.state, self.temp_file.name)

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_predict_without_data(self):
        preds, std_dev = predict(self.temp_file.name, 2)
        expected_preds = [(8 + 9 + 10) / 3, (9 + 10 + (8 + 9 + 10) / 3) / 3]
        self.assertEqual(preds, expected_preds)
        self.assertEqual(std_dev, 0.5)


class TestTrainAndEvaluate(unittest.TestCase):
    """Test train_and_evaluate function."""

    def test_normal_training(self):
        sales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        months = [str(i) for i in range(10)]
        train_range = range(0, 5)
        eval_range = range(5, 10)
        metrics, std_dev, y_true, y_pred = train_and_evaluate(
            sales, months, train_range, eval_range, 3
        )
        self.assertIn("rmse", metrics)
        self.assertEqual(len(y_true), len(y_pred))
        self.assertGreaterEqual(std_dev, 0)


class TestRunFunction(unittest.TestCase):
    """Test run function."""

    def test_training_mode_basic(self):
        data = json.dumps([{"sales": i, "month": f"2023-{i:02d}"} for i in range(10)])
        args = type(
            "Args",
            (),
            {
                "data": data,
                "train_start": 0,
                "train_end": 5,
                "eval_start": 6,
                "eval_end": 9,
                "window": 3,
                "save_model": None,
                "pretty": False,
            },
        )()
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            run(args)
        out = f.getvalue()
        self.assertTrue(out.strip())  # Check if output is not empty


class TestPerformance(unittest.TestCase):
    """Test performance of optimized functions."""

    def test_moving_average_forecast_performance(self):
        series = np.random.randn(10000)
        eval_range = range(1000, 2000)
        start_time = time.time()
        idxs, preds = moving_average_forecast(series, eval_range, 10)
        end_time = time.time()
        self.assertLess(end_time - start_time, 1.0)
        self.assertEqual(len(preds), 1000)


class TestPickleCompatibility(unittest.TestCase):
    """Test save and load compatibility."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_save_and_load_model(self):
        from serialization import dump_model

        state = {
            "model_type": MODEL_TYPE,
            "window": 3,
            "train_end_index": 9,
            "history": [7, 8, 9],
            "value_key": "sales",
            "std_dev_residuals": 0.2,
        }
        dump_model(state, self.temp_file.name)
        loaded_state = _load_state(self.temp_file.name)
        self.assertEqual(state, loaded_state)


if __name__ == "__main__":
    unittest.main()
