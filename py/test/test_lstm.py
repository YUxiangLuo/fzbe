#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the LSTM model.
"""
import json
import os
import sys
import tempfile
import unittest
import warnings

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from data_utils import ParsedData, load_data_from_json
from models.lstm import (
    _create_dataset,
    _format_predictions,
    _validate_training_args,
    train_and_evaluate,
)


class TestFormatPredictions(unittest.TestCase):
    """Test the _format_predictions helper function."""

    def test_format_predictions(self):
        """Test predictions are formatted correctly."""
        preds = [10.5, 20.3, 30.1]
        std_dev = 2.5

        result = _format_predictions(preds, std_dev)

        expected = [
            {"prediction": 10.5, "std_dev": 2.5},
            {"prediction": 20.3, "std_dev": 2.5},
            {"prediction": 30.1, "std_dev": 2.5},
        ]
        self.assertEqual(result, expected)

    def test_empty_predictions(self):
        """Test formatting empty predictions list."""
        result = _format_predictions([], 0.0)
        self.assertEqual(result, [])


class TestCreateDataset(unittest.TestCase):
    """Test the _create_dataset function."""

    def test_basic_dataset_creation(self):
        """Test creating LSTM input/output pairs."""
        # Create a simple 2D dataset (3 timesteps, 2 features)
        dataset = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
        ])
        look_back = 2

        X, y = _create_dataset(dataset, look_back)

        # Should have 3 samples (5 - 2 = 3)
        self.assertEqual(X.shape, (3, 2, 2))  # (samples, look_back, features)
        self.assertEqual(y.shape, (3,))

        # First sample: uses timesteps 0,1 to predict timestep 2
        np.testing.assert_array_equal(X[0], [[1.0, 2.0], [2.0, 3.0]])
        self.assertEqual(y[0], 3.0)  # Target is first column of timestep 2

        # Second sample: uses timesteps 1,2 to predict timestep 3
        np.testing.assert_array_equal(X[1], [[2.0, 3.0], [3.0, 4.0]])
        self.assertEqual(y[1], 4.0)

    def test_single_feature(self):
        """Test dataset creation with single feature."""
        dataset = np.array([[1.0], [2.0], [3.0], [4.0]])
        look_back = 2

        X, y = _create_dataset(dataset, look_back)

        self.assertEqual(X.shape, (2, 2, 1))
        self.assertEqual(y.shape, (2,))
        self.assertEqual(y[0], 3.0)
        self.assertEqual(y[1], 4.0)


class TestValidateTrainingArgs(unittest.TestCase):
    """Test the _validate_training_args function."""

    def _create_parsed_data(self, n_obs=20):
        """Helper to create a simple ParsedData object."""
        df = pd.DataFrame({
            'y': np.arange(n_obs, dtype=float),
            'month': [f'M{i}' for i in range(n_obs)]
        })
        return ParsedData(df=df, target_key='y', time_key='month')

    def test_valid_args(self):
        """Test validation passes for valid arguments."""
        parsed_data = self._create_parsed_data(n_obs=20)
        # Should not raise
        _validate_training_args(parsed_data, ts=0, te=10, es=11, ee=19, look_back=3)

    def test_invalid_train_range(self):
        """Test invalid training range raises error."""
        parsed_data = self._create_parsed_data(n_obs=10)
        with self.assertRaises(ValueError) as ctx:
            _validate_training_args(parsed_data, ts=5, te=3, es=6, ee=9, look_back=2)
        self.assertIn("Invalid train range", str(ctx.exception))

    def test_invalid_eval_range(self):
        """Test invalid evaluation range raises error."""
        parsed_data = self._create_parsed_data(n_obs=10)
        with self.assertRaises(ValueError) as ctx:
            _validate_training_args(parsed_data, ts=0, te=5, es=8, ee=7, look_back=2)
        self.assertIn("Invalid eval range", str(ctx.exception))

    def test_eval_before_train(self):
        """Test evaluation starting before/at training end raises error."""
        parsed_data = self._create_parsed_data(n_obs=10)
        with self.assertRaises(ValueError) as ctx:
            _validate_training_args(parsed_data, ts=0, te=5, es=4, ee=9, look_back=2)
        self.assertIn("eval_start must be > train_end", str(ctx.exception))

    def test_invalid_look_back(self):
        """Test invalid look_back raises error."""
        parsed_data = self._create_parsed_data(n_obs=10)
        with self.assertRaises(ValueError) as ctx:
            _validate_training_args(parsed_data, ts=0, te=5, es=6, ee=9, look_back=0)
        self.assertIn("look_back must be >= 1", str(ctx.exception))

    def test_train_range_too_small(self):
        """Test training range smaller than look_back raises error."""
        parsed_data = self._create_parsed_data(n_obs=20)
        with self.assertRaises(ValueError) as ctx:
            _validate_training_args(parsed_data, ts=0, te=2, es=3, ee=5, look_back=5)
        self.assertIn("Training range must be larger than look_back", str(ctx.exception))


class TestTrainAndEvaluate(unittest.TestCase):
    """Test the train_and_evaluate function."""

    def test_basic_train_single_feature(self):
        """Test basic training with single feature (target only)."""
        # Create simple linear data
        n = 30
        data = [{"y": float(i * 2 + 10)} for i in range(n)]
        data_json = json.dumps(data)

        parsed_data = load_data_from_json(data_json)
        feature_types = {"y": "numeric"}

        # Train with minimal epochs for speed
        metrics, model, preprocessor, y_true, y_pred, std_dev, history_df = train_and_evaluate(
            parsed_data, feature_types,
            range(0, 20), range(20, 30),
            look_back=3, epochs=5  # Few epochs for fast test
        )

        # Check basic structure
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)

        # Check predictions
        self.assertEqual(len(y_pred), 10)
        self.assertEqual(len(y_true), 10)
        self.assertIsInstance(std_dev, float)

        # Model should be a Keras model
        self.assertIsNotNone(model)

    def test_multi_feature_train(self):
        """Test training with multiple numeric features."""
        # Create data with multiple features
        n = 30
        data = []
        for i in range(n):
            data.append({
                "sales": float(100 + i * 5),
                "marketing": float(10 + i * 0.5),
                "price": float(50 - i * 0.1)
            })
        data_json = json.dumps(data)

        parsed_data = load_data_from_json(data_json, target_key="sales")
        feature_types = {
            "sales": "numeric",
            "marketing": "numeric",
            "price": "numeric"
        }

        metrics, model, preprocessor, y_true, y_pred, std_dev, history_df = train_and_evaluate(
            parsed_data, feature_types,
            range(0, 20), range(20, 30),
            look_back=3, epochs=5
        )

        # Check that it ran successfully
        self.assertEqual(len(y_pred), 10)
        self.assertIn("rmse", metrics)

    def test_empty_eval_range(self):
        """Test handling of empty evaluation range."""
        n = 20
        data = [{"y": float(i)} for i in range(n)]
        data_json = json.dumps(data)

        parsed_data = load_data_from_json(data_json)
        feature_types = {"y": "numeric"}

        metrics, model, preprocessor, y_true, y_pred, std_dev, history_df = train_and_evaluate(
            parsed_data, feature_types,
            range(0, 15), range(16, 16),  # Empty eval range
            look_back=3, epochs=5
        )

        # Should return empty results
        self.assertEqual(metrics, {})
        self.assertEqual(y_true, [])
        self.assertEqual(y_pred, [])
        self.assertEqual(std_dev, 0.0)


class TestPredictFunction(unittest.TestCase):
    """Test the predict function with model loading."""

    def test_predict_single_feature(self):
        """Test prediction with a trained single-feature model."""
        # Create and train a simple model
        n = 40
        np.random.seed(42)
        data = [{"y": float(10 + i * 0.5)} for i in range(n)]
        data_json = json.dumps(data)

        parsed_data = load_data_from_json(data_json)
        feature_types = {"y": "numeric"}

        # Train
        metrics, model, preprocessor, y_true, y_pred, std_dev, history_df = train_and_evaluate(
            parsed_data, feature_types,
            range(0, 30), range(30, 40),
            look_back=5, epochs=10
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "lstm_test.pkl")
            keras_model_path = os.path.join(tmpdir, "lstm_test_keras_model.h5")

            # Save model
            model.save(keras_model_path)
            from serialization import dump_model
            state = {
                "model_type": "lstm",
                "look_back": 5,
                "epochs": 10,
                "train_end_index": 29,
                "preprocessor": preprocessor,
                "history_df": history_df,
                "target_key": "y",
                "feature_keys": [],
                "keras_model_path": keras_model_path,
                "std_dev_residuals": std_dev,
            }
            dump_model(state, model_path)

            # Test predict function
            from models.lstm import predict
            predictions, pred_std_dev = predict(model_path, steps=5)

            # Verify predictions
            self.assertEqual(len(predictions), 5)
            self.assertIsInstance(pred_std_dev, float)
            # All predictions should be floats
            for pred in predictions:
                self.assertIsInstance(pred, float)
                # Predictions should be in a reasonable range
                self.assertGreater(pred, 0)

    def test_predict_multi_feature(self):
        """Test prediction with a multi-feature model."""
        # Create multi-feature data
        n = 40
        np.random.seed(123)
        data = []
        for i in range(n):
            data.append({
                "sales": float(100 + i * 5),
                "marketing": float(10 + i * 0.5),
                "price": float(50 - i * 0.1)
            })
        data_json = json.dumps(data)

        parsed_data = load_data_from_json(data_json, target_key="sales")
        feature_types = {
            "sales": "numeric",
            "marketing": "numeric",
            "price": "numeric"
        }

        # Train
        metrics, model, preprocessor, y_true, y_pred, std_dev, history_df = train_and_evaluate(
            parsed_data, feature_types,
            range(0, 30), range(30, 40),
            look_back=5, epochs=10
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "lstm_multi_test.pkl")
            keras_model_path = os.path.join(tmpdir, "lstm_multi_test_keras_model.h5")

            # Save model
            model.save(keras_model_path)
            from serialization import dump_model
            state = {
                "model_type": "lstm",
                "look_back": 5,
                "epochs": 10,
                "train_end_index": 29,
                "preprocessor": preprocessor,
                "history_df": history_df,
                "target_key": "sales",
                "feature_keys": ["marketing", "price"],
                "keras_model_path": keras_model_path,
                "std_dev_residuals": std_dev,
            }
            dump_model(state, model_path)

            # Test predict function
            from models.lstm import predict
            predictions, pred_std_dev = predict(model_path, steps=3)

            # Verify predictions
            self.assertEqual(len(predictions), 3)
            self.assertIsInstance(pred_std_dev, float)
            for pred in predictions:
                self.assertIsInstance(pred, float)

    def test_predict_invalid_steps(self):
        """Test that invalid steps raises error."""
        # Create a minimal valid model
        n = 20
        data = [{"y": float(i)} for i in range(n)]
        data_json = json.dumps(data)

        parsed_data = load_data_from_json(data_json)
        feature_types = {"y": "numeric"}

        metrics, model, preprocessor, y_true, y_pred, std_dev, history_df = train_and_evaluate(
            parsed_data, feature_types,
            range(0, 15), range(15, 20),
            look_back=3, epochs=5
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "lstm_test.pkl")
            keras_model_path = os.path.join(tmpdir, "lstm_test_keras_model.h5")

            model.save(keras_model_path)
            from serialization import dump_model
            state = {
                "model_type": "lstm",
                "look_back": 3,
                "epochs": 5,
                "train_end_index": 14,
                "preprocessor": preprocessor,
                "history_df": history_df,
                "target_key": "y",
                "feature_keys": [],
                "keras_model_path": keras_model_path,
                "std_dev_residuals": std_dev,
            }
            dump_model(state, model_path)

            # Test with invalid steps
            from models.lstm import predict
            with self.assertRaises(ValueError) as ctx:
                predict(model_path, steps=0)
            self.assertIn("steps must be > 0", str(ctx.exception))


class TestIntegration(unittest.TestCase):
    """Integration tests for the LSTM model."""

    def test_full_workflow_single_feature(self):
        """Test complete train, save, load, predict workflow with single feature."""
        # Create simple test data
        n = 40
        np.random.seed(42)
        data = [{"y": float(10 + i * 0.5 + np.random.randn() * 0.1)} for i in range(n)]
        data_json = json.dumps(data)

        parsed_data = load_data_from_json(data_json)
        feature_types = {"y": "numeric"}

        # Train
        metrics, model, preprocessor, y_true, y_pred, std_dev, history_df = train_and_evaluate(
            parsed_data, feature_types,
            range(0, 30), range(30, 40),
            look_back=5, epochs=10
        )

        # Verify training results
        self.assertGreater(len(y_pred), 0)
        self.assertIn("rmse", metrics)

        # Save and load workflow
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "lstm_test.pkl")
            keras_model_path = os.path.join(tmpdir, "lstm_test_keras_model.h5")

            # Save
            model.save(keras_model_path)
            from serialization import dump_model
            state = {
                "model_type": "lstm",
                "look_back": 5,
                "epochs": 10,
                "train_end_index": 29,
                "preprocessor": preprocessor,
                "history_df": history_df,
                "target_key": "y",
                "feature_keys": [],
                "keras_model_path": keras_model_path,
                "std_dev_residuals": std_dev,
            }
            dump_model(state, model_path)

            # Verify files exist
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(keras_model_path))

            # Load and predict
            from models.lstm import predict
            future_preds, future_std = predict(model_path, steps=5)

            # Verify predictions
            self.assertEqual(len(future_preds), 5)
            self.assertIsInstance(future_std, float)
            # Predictions should continue the trend
            for pred in future_preds:
                self.assertGreater(pred, 0)


if __name__ == "__main__":
    unittest.main()
