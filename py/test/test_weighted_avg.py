#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the Weighted Average ensemble model.
"""
import json
import os
import sys
import tempfile
import unittest
import warnings

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Suppress TensorFlow warnings for LSTM
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
from data_utils import ParsedData, load_data_from_json
from models.weighted_avg import (
    _validate_training_args,
    predict,
    train_and_evaluate,
)
from serialization import dump_model


class TestValidateTrainingArgs(unittest.TestCase):
    """Test the _validate_training_args function."""

    def _create_parsed_data(self, n_obs=50):
        """Helper to create a simple ParsedData object."""
        data = [{"y": float(i)} for i in range(n_obs)]
        return load_data_from_json(json.dumps(data))

    def test_valid_args(self):
        """Test validation passes for valid arguments."""
        parsed_data = self._create_parsed_data()
        models_to_use = ["ma", "es"]
        model_params = {
            "ma": {"window": 3},
            "es": {"alpha": 0.5}
        }
        # Should not raise
        _validate_training_args(parsed_data, 0, 30, 31, 40, models_to_use, model_params)

    def test_empty_models_list(self):
        """Test that empty models_to_use list raises error."""
        parsed_data = self._create_parsed_data()
        with self.assertRaisesRegex(ValueError, "models_to_use cannot be empty"):
            _validate_training_args(parsed_data, 0, 30, 31, 40, [], {})

    def test_unknown_model(self):
        """Test that an unknown model name raises error."""
        parsed_data = self._create_parsed_data()
        models_to_use = ["ma", "unknown"]
        model_params = {"ma": {"window": 3}, "unknown": {}}
        with self.assertRaisesRegex(ValueError, "Unknown model: unknown"):
            _validate_training_args(parsed_data, 0, 30, 31, 40, models_to_use, model_params)

    def test_invalid_ranges(self):
        """Test that invalid train/eval ranges raise errors."""
        parsed_data = self._create_parsed_data()
        models_to_use = ["ma", "es"]
        model_params = {"ma": {"window": 3}, "es": {"alpha": 0.5}}
        
        with self.assertRaisesRegex(ValueError, "Invalid train range"):
            _validate_training_args(parsed_data, 0, 50, 40, 45, models_to_use, model_params)
        
        with self.assertRaisesRegex(ValueError, "Invalid eval range"):
            _validate_training_args(parsed_data, 0, 30, 35, 55, models_to_use, model_params)
            
        with self.assertRaisesRegex(ValueError, "eval_start must be > train_end"):
            _validate_training_args(parsed_data, 0, 30, 30, 40, models_to_use, model_params)

    def test_missing_model_params(self):
        """Test that missing parameters for a model raises error."""
        parsed_data = self._create_parsed_data()
        models_to_use = ["ma", "es"]
        model_params = {"ma": {"window": 3}} # Missing params for 'es'
        with self.assertRaisesRegex(ValueError, "Missing parameters for model: es"):
            _validate_training_args(parsed_data, 0, 30, 31, 40, models_to_use, model_params)

    def test_invalid_arima_params(self):
        """Test invalid parameters for ARIMA."""
        parsed_data = self._create_parsed_data()
        models_to_use = ["arima", "ma"]
        model_params = {"arima": {"d": -1}, "ma": {"window": 3}}
        with self.assertRaisesRegex(ValueError, "ARIMA d must be >= 0"):
            _validate_training_args(parsed_data, 0, 30, 31, 40, models_to_use, model_params)

    def test_invalid_lstm_params(self):
        """Test training range smaller than LSTM look_back."""
        parsed_data = self._create_parsed_data()
        models_to_use = ["lstm", "ma"]
        model_params = {"lstm": {"look_back": 15, "epochs": 5}, "ma": {"window": 3}}
        with self.assertRaisesRegex(ValueError, "Training range must be larger than LSTM look_back"):
            _validate_training_args(parsed_data, 0, 10, 11, 20, models_to_use, model_params)


class TestTrainAndEvaluate(unittest.TestCase):
    """Test the train_and_evaluate function."""

    def _create_parsed_data(self, n_obs=50):
        """Helper to create a simple ParsedData object."""
        np.random.seed(42)
        data = [{"y": float(10 + i * 2 + np.random.randn())} for i in range(n_obs)]
        return load_data_from_json(json.dumps(data))

    def test_basic_workflow_ma_es(self):
        """Test basic training with MA and ES models."""
        parsed_data = self._create_parsed_data()
        models_to_use = ["ma", "es"]
        model_params = {
            "ma": {"window": 5},
            "es": {"alpha": 0.6}
        }

        metrics, base_states, weights, y_true, y_pred, std_dev = train_and_evaluate(
            parsed_data, range(0, 30), range(30, 40), models_to_use, model_params
        )

        # Check metrics
        self.assertIn("rmse", metrics)
        self.assertGreater(metrics["rmse"], 0)

        # Check weights
        self.assertIn("ma", weights)
        self.assertIn("es", weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        
        # Check predictions
        self.assertEqual(len(y_true), 10)
        self.assertEqual(len(y_pred), 10)
        
        # Check std_dev
        self.assertIsInstance(std_dev, float)
        self.assertGreater(std_dev, 0)

        # Check base model states
        self.assertIn("window", base_states["ma"])
        self.assertIn("alpha", base_states["es"])

    def test_workflow_with_arima(self):
        """Test training including the ARIMA model."""
        parsed_data = self._create_parsed_data()
        models_to_use = ["ma", "arima"]
        model_params = {
            "ma": {"window": 3},
            "arima": {"d": 1, "max_p": 2, "max_q": 2}
        }

        metrics, base_states, weights, y_true, y_pred, std_dev = train_and_evaluate(
            parsed_data, range(0, 30), range(30, 40), models_to_use, model_params
        )

        self.assertIn("rmse", metrics)
        self.assertIn("ma", weights)
        self.assertIn("arima", weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertIn("order", base_states["arima"])


class TestPredictAndFullWorkflow(unittest.TestCase):
    """Test the predict function and the full train-save-load-predict workflow."""

    def _create_parsed_data(self, n_obs=50):
        """Helper to create a simple ParsedData object."""
        np.random.seed(123)
        data = [{"y": float(50 + i * 1.5 + np.random.randn() * 2)} for i in range(n_obs)]
        return load_data_from_json(json.dumps(data))

    def test_full_workflow(self):
        """Test the complete train, save, load, and predict cycle."""
        parsed_data = self._create_parsed_data()
        models_to_use = ["ma", "es", "arima"]
        model_params = {
            "ma": {"window": 4},
            "es": {"alpha": 0.4},
            "arima": {"d": 1, "max_p": 2, "max_q": 2}
        }

        # Train the model
        metrics, base_states, weights, _, _, std_dev = train_and_evaluate(
            parsed_data, range(0, 35), range(35, 45), models_to_use, model_params
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "weighted_avg_test.pkl")

            # Manually handle LSTM model saving for the test state
            for name, state in base_states.items():
                if name == "lstm" and "keras_model" in state:
                    keras_model = state.pop("keras_model")
                    keras_path = os.path.splitext(model_path)[0] + "_lstm_keras_model.h5"
                    keras_model.save(keras_path)
                    state["keras_model_path"] = keras_path

            # Save the state
            state_to_save = {
                "model_type": "weighted_avg",
                "models": models_to_use,
                "base_model_states": base_states,
                "weights": weights,
                "std_dev_residuals": std_dev,
            }
            dump_model(state_to_save, model_path)

            # Check that the model file was created
            self.assertTrue(os.path.exists(model_path))

            # Predict using the saved model
            predictions, loaded_std_dev = predict(model_path, steps=5)

            # Verify predictions
            self.assertEqual(len(predictions), 5)
            for p in predictions:
                self.assertIsInstance(p, float)

            # Verify std_dev persistence
            self.assertAlmostEqual(std_dev, loaded_std_dev, places=5)

    def test_predict_invalid_steps(self):
        """Test that predict raises error for steps <= 0."""
        # We need a dummy model file to pass to predict
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
            model_path = tmp.name
            state = {
                "model_type": "weighted_avg", "models": [], "base_model_states": {},
                "weights": {}, "std_dev_residuals": 0.0
            }
            dump_model(state, model_path)
        
        try:
            with self.assertRaisesRegex(ValueError, "steps must be > 0"):
                predict(model_path, steps=0)
        finally:
            os.unlink(model_path)


if __name__ == "__main__":
    unittest.main()
