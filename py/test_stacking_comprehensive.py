# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for stacking.py (Stacking Ensemble model)
Tests cover: helper functions, training, prediction, error handling, and integration scenarios
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.stacking import (
    MODEL_TYPE,
    _train_base_models,
    _train_meta_model,
    predict,
    run,
    train_and_evaluate,
)


class TestTrainBaseModels(unittest.TestCase):
    """Test _train_base_models function."""

    def test_train_ma_model(self):
        """Test training MA base model."""
        sales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        months = [str(i) for i in range(10)]
        train_range = range(0, 5)
        eval_range = range(5, 10)
        base_models = ["ma"]
        base_params = {"ma": {"window": 3}}

        with patch(
            "models.stacking.ma_train",
            return_value=({"test": "metrics"}, 0.5, [6, 7, 8, 9, 10], [5, 6, 7, 8, 9]),
        ) as mock_ma:
            predictions = _train_base_models(
                sales, months, train_range, eval_range, base_models, base_params
            )
            self.assertIn("ma", predictions)
            self.assertEqual(len(predictions["ma"]), 5)  # eval_range length
            mock_ma.assert_called_once()

    def test_train_multiple_models(self):
        """Test training multiple base models."""
        sales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        months = [str(i) for i in range(10)]
        train_range = range(0, 5)
        eval_range = range(5, 10)
        base_models = ["ma", "es"]
        base_params = {"ma": {"window": 3}, "es": {"alpha": 0.5}}

        with (
            patch(
                "models.stacking.ma_train",
                return_value=({"test": "metrics"}, 0.5, [6, 7, 8, 9, 10], [4, 5, 6, 7]),
            ) as mock_ma,
            patch(
                "models.stacking.es_train",
                return_value=(
                    {"test": "metrics"},
                    0.5,
                    8.0,
                    [6, 7, 8, 9, 10],
                    [5.5, 6.5, 7.5, 8.5, 9.5],
                ),
            ) as mock_es,
        ):
            predictions = _train_base_models(
                sales, months, train_range, eval_range, base_models, base_params
            )
            self.assertIn("ma", predictions)
            self.assertIn("es", predictions)
            mock_ma.assert_called_once()
            mock_es.assert_called_once()

    def test_unsupported_model(self):
        """Test unsupported base model raises error."""
        sales = [1, 2, 3, 4, 5]
        months = ["1", "2", "3", "4", "5"]
        train_range = range(0, 3)
        eval_range = range(3, 5)
        base_models = ["invalid"]
        base_params = {}

        with self.assertRaises(ValueError) as cm:
            _train_base_models(
                sales, months, train_range, eval_range, base_models, base_params
            )
        self.assertIn("Unsupported base model", str(cm.exception))


class TestTrainMetaModel(unittest.TestCase):
    """Test _train_meta_model function."""

    def test_average_meta(self):
        """Test average meta-model."""
        base_predictions = {"ma": [1, 2, 3], "es": [4, 5, 6]}
        y_true = [2.5, 3.5, 4.5]
        meta_data = _train_meta_model(base_predictions, y_true, "average")
        self.assertEqual(meta_data["type"], "average")
        self.assertEqual(meta_data["weights"], [0.5, 0.5])

    def test_linear_meta(self):
        """Test linear meta-model."""
        base_predictions = {"ma": [1, 2, 3], "es": [4, 5, 6]}
        y_true = [2.5, 3.5, 4.5]
        meta_data = _train_meta_model(base_predictions, y_true, "linear")
        self.assertEqual(meta_data["type"], "linear")
        self.assertIn("coef", meta_data)
        self.assertIn("intercept", meta_data)

    def test_unsupported_meta_model(self):
        """Test unsupported meta model raises error."""
        base_predictions = {"ma": [1, 2, 3]}
        y_true = [1, 2, 3]
        with self.assertRaises(ValueError) as cm:
            _train_meta_model(base_predictions, y_true, "invalid")
        self.assertIn("Unsupported meta model", str(cm.exception))


class TestTrainAndEvaluate(unittest.TestCase):
    """Test train_and_evaluate function."""

    def test_normal_stacking(self):
        """Test normal stacking training."""
        sales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        months = [str(i) for i in range(10)]
        train_range = range(0, 5)
        eval_range = range(5, 10)
        base_models = ["ma", "es"]
        base_params = {"ma": {"window": 3}, "es": {"alpha": 0.5}}

        with (
            patch(
                "models.stacking._train_base_models",
                return_value={
                    "ma": [4, 5, 6, 7, 8, 9],
                    "es": [4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
                },
            ) as mock_base,
            patch(
                "models.stacking._train_meta_model",
                return_value={"type": "average", "weights": [0.5, 0.5]},
            ) as mock_meta,
        ):
            metrics, predictions = train_and_evaluate(
                sales,
                months,
                train_range,
                eval_range,
                base_models,
                "average",
                base_params,
            )
            self.assertIn("rmse", metrics)
            self.assertEqual(len(predictions), 5)  # eval length
            mock_base.assert_called_once()
            mock_meta.assert_called_once()

    def test_empty_base_models(self):
        """Test empty base models raises error."""
        sales = [1, 2, 3]
        months = ["1", "2", "3"]
        with self.assertRaises(ValueError) as cm:
            train_and_evaluate(sales, months, range(0, 2), range(2, 3), [], "average")
        self.assertIn("At least one base model", str(cm.exception))

    def test_single_base_model(self):
        """Test single base model raises error."""
        sales = [1, 2, 3]
        months = ["1", "2", "3"]
        with self.assertRaises(ValueError) as cm:
            train_and_evaluate(
                sales, months, range(0, 2), range(2, 3), ["ma"], "average"
            )
        self.assertIn("requires at least 2 base models", str(cm.exception))


class TestPredictFunction(unittest.TestCase):
    """Test predict function."""

    def setUp(self):
        from serialization import dump_model

        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        state = {
            "model_type": MODEL_TYPE,
            "base_models": ["ma"],
            "meta_model": "average",
            "meta_data": {"type": "average", "weights": [1.0]},
        }
        dump_model(state, self.temp_file.name)

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_predict_basic(self):
        """Test basic predict."""
        preds = predict(self.temp_file.name, 3)
        self.assertEqual(len(preds), 3)
        self.assertEqual(preds[0]["prediction"], 0.0)  # Placeholder

    def test_predict_invalid_steps(self):
        """Test invalid steps raises error."""
        with self.assertRaises(ValueError) as cm:
            predict(self.temp_file.name, 0)
        self.assertIn("steps must be > 0", str(cm.exception))


class TestRunFunction(unittest.TestCase):
    """Test run function."""

    def setUp(self):
        self.temp_save_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
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
        """Test basic training mode."""
        data = json.dumps([{"y": i + 1, "month": f"{i + 1:02d}"} for i in range(20)])
        args = type(
            "Args",
            (),
            {
                "data": data,
                "train_start": 0,
                "train_end": 14,
                "eval_start": 15,
                "eval_end": 19,
                "base_models": "ma,es",
                "meta_model": "average",
                "base_params": '{"ma": {"window": 3}, "es": {"alpha": 0.5}}',
                "target_key": "y",
                "save_model": None,
                "pretty": False,
            },
        )()

        with patch(
            "models.stacking.train_and_evaluate",
            return_value=({"rmse": 1.0}, [10, 11, 12, 13]),
        ) as mock_train:
            out = self._capture_run_output(args)
            self.assertEqual(out["mode"], "stacking_trained")
            self.assertIn("metrics", out)
            mock_train.assert_called_once()

    def test_prediction_mode_basic(self):
        """Test basic prediction mode."""
        args = type(
            "Args",
            (),
            {
                "load_model": "/tmp/fake.pkl",  # Use placeholder
                "forecast_steps": 5,
                "pretty": False,
            },
        )()

        with patch(
            "models.stacking.predict",
            return_value=[{"prediction": 1.0} for _ in range(5)],
        ) as mock_predict:
            out = self._capture_run_output(args)
            self.assertEqual(out["mode"], "stacking_loaded_predict")
            self.assertEqual(len(out["predictions"]), 5)
            mock_predict.assert_called_once()


class TestErrorHandling(unittest.TestCase):
    """Test error handling in stacking."""

    def test_training_missing_data(self):
        """Test training without data raises error."""
        args = type(
            "Args",
            (),
            {
                "data": None,
                "train_start": 0,
                "eval_start": 5,
            },
        )()
        with self.assertRaises(ValueError) as cm:
            run(args)
        self.assertIn("--data required", str(cm.exception))

    def test_invalid_base_params_json(self):
        """Test invalid base_params JSON raises error."""
        data = json.dumps([{"y": 1}, {"y": 2}])
        args = type(
            "Args",
            (),
            {
                "data": data,
                "train_start": 0,
                "train_end": 0,
                "eval_start": 1,
                "eval_end": 1,
                "base_models": "ma,es",
                "base_params": "invalid json",
            },
        )()
        with self.assertRaises(ValueError):
            run(args)


class TestIntegration(unittest.TestCase):
    """Test integration scenarios."""

    def test_full_workflow_ma_es(self):
        """Test full train with MA and ES."""
        sales = list(range(1, 31))  # 30 data points
        months = [f"2023-{i:02d}" for i in range(1, 31)]
        train_range = range(0, 20)
        eval_range = range(20, 30)

        with (
            patch(
                "models.stacking._train_base_models",
                return_value={
                    "ma": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                    "es": [20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5],
                },
            ) as mock_base,
            patch(
                "models.stacking._train_meta_model",
                return_value={"type": "average", "weights": [0.5, 0.5]},
            ) as mock_meta,
        ):
            metrics, predictions = train_and_evaluate(
                sales,
                months,
                train_range,
                eval_range,
                ["ma", "es"],
                "average",
                {"ma": {"window": 3}, "es": {"alpha": 0.5}},
            )
            self.assertIn("rmse", metrics)
            self.assertEqual(len(predictions), 10)  # eval length


if __name__ == "__main__":
    unittest.main()
