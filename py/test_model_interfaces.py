#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit test suite for verifying the consistency of model interfaces.
Ensures that all models adhere to a common API contract for `train_and_evaluate` and `predict`.
"""
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Import the necessary functions from each model
from data_utils import load_data_from_json
from models.arima import train_and_evaluate as arima_train, predict as arima_predict
from models.es import train_and_evaluate as es_train, predict as es_predict
from models.ma import train_and_evaluate as ma_train, predict as ma_predict
from models.lstm import train_and_evaluate as lstm_train, predict as lstm_predict

def create_test_data(n=60, categorical=False):
    """Helper to create test data."""
    data = []
    for i in range(n):
        record = {
            "sales": 150 + i * 3 + 8 * np.sin(i / 5) + np.random.randn(),
            "feature1": 50 + i * 0.5 + 3 * np.cos(i / 3) + np.random.randn()
        }
        if categorical:
            record["day_of_week"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i % 7]
        data.append(record)
    return data

class TestModelInterfaces(unittest.TestCase):

    def setUp(self):
        self.n_total = 60
        self.train_range = range(0, 45)
        self.eval_range = range(45, 60)
        self.forecast_steps = 10
        
        self.data = create_test_data(self.n_total, categorical=True)
        self.data_json = json.dumps(self.data)
        self.parsed_data = load_data_from_json(self.data_json)
        
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_arima_interface(self):
        """Verify ARIMA's train_and_evaluate and predict interfaces."""
        metrics, model, y_true, y_pred, std_devs = arima_train(
            self.parsed_data.y, self.parsed_data.timestamps, self.train_range, self.eval_range, order=(1, 1, 1)
        )
        
        # Test train_and_evaluate return signature
        self.assertIsInstance(y_pred, list)
        self.assertEqual(len(y_pred), len(self.eval_range))
        self.assertIsNotNone(model)

        # Test state consistency and predict return signature
        state = {"model": model}
        preds, _ = arima_predict(model_path=None, steps=self.forecast_steps, state=state)
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), self.forecast_steps)

    def test_es_interface(self):
        """Verify ES's train_and_evaluate and predict interfaces."""
        metrics, std_dev, level, y_true, y_pred = es_train(
            self.parsed_data.y, self.parsed_data.timestamps, self.train_range, self.eval_range, alpha=0.5
        )

        # Test train_and_evaluate return signature
        self.assertIsInstance(y_pred, list)
        self.assertEqual(len(y_pred), len(self.eval_range))
        self.assertIsNotNone(level)

        # Test state consistency and predict return signature
        state = {"alpha": 0.5, "level": level, "std_dev_residuals": std_dev}
        preds, _ = es_predict(model_path=None, steps=self.forecast_steps, state=state)
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), self.forecast_steps)

    def test_ma_interface(self):
        """Verify MA's train_and_evaluate and predict interfaces."""
        metrics, std_dev, y_true, y_pred = ma_train(
            self.parsed_data.y, self.parsed_data.timestamps, self.train_range, self.eval_range, window=5
        )

        # Test train_and_evaluate return signature
        self.assertIsInstance(y_pred, np.ndarray) # MA returns numpy array
        self.assertEqual(len(y_pred), len(self.eval_range))

        # Test state consistency and predict return signature
        history = self.parsed_data.y[self.train_range.stop - 5 : self.train_range.stop].tolist()
        state = {"window": 5, "history": history, "std_dev_residuals": std_dev, "train_end_index": self.train_range.stop -1}
        preds, _ = ma_predict(model_path=None, steps=self.forecast_steps, state=state)
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), self.forecast_steps)

    def test_lstm_interface(self):
        """Verify LSTM's train_and_evaluate and predict interfaces."""
        # Suppress TensorFlow logging for cleaner test output
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        feature_types = {
            "sales": "numeric",
            "feature1": "numeric",
            "day_of_week": "categorical"
        }
        
        metrics, model, preprocessor, y_true, y_pred, std_dev, history_df = lstm_train(
            self.parsed_data, feature_types, self.train_range, self.eval_range, look_back=5, epochs=1
        )

        # Test train_and_evaluate return signature
        self.assertIsInstance(y_pred, list)
        self.assertEqual(len(y_pred), len(self.eval_range))
        self.assertIsNotNone(model)
        self.assertIsNotNone(preprocessor)

        # Test state consistency and predict return signature
        # We need to save/load the keras model as it's not directly passable
        keras_model_path = os.path.join(self.tmpdir.name, "lstm_keras_model.h5")
        model.save(keras_model_path)

        state = {
            "preprocessor": preprocessor,
            "look_back": 5,
            "history_df": history_df,
            "keras_model_path": keras_model_path,
            "std_dev_residuals": std_dev
        }
        preds, _ = lstm_predict(model_path=None, steps=self.forecast_steps, state=state)
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), self.forecast_steps)

if __name__ == '__main__':
    unittest.main(verbosity=2)
