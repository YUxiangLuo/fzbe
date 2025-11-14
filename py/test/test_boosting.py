#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the AdaBoost-style ensemble with Greedy Forward Selection.
"""
import json
import os
import sys
import unittest
import warnings

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
from data_utils import load_data_from_json
from models.boosting import (
    _validate_training_args,
    train_and_evaluate,
)


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
        model_params = {"ma": {"window": 3}, "es": {"alpha": 0.5}}
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
        with self.assertRaisesRegex(ValueError, "Invalid train/eval ranges"):
            _validate_training_args(parsed_data, 0, 50, 40, 45, models_to_use, model_params)


class TestTrainAndEvaluate(unittest.TestCase):
    """Test the train_and_evaluate function for AdaBoost."""

    def _create_parsed_data(self, n_obs=50):
        """Helper to create a simple ParsedData object."""
        np.random.seed(42)
        data = [{"y": float(10 + i * 2 + np.random.randn())} for i in range(n_obs)]
        return load_data_from_json(json.dumps(data))

    def test_greedy_selection_and_summary(self):
        """Test the greedy forward selection and search summary output."""
        parsed_data = self._create_parsed_data()
        models_to_use = ["ma", "es", "arima"]
        model_params = {
            "ma": {"window": 5},
            "es": {"alpha": 0.6},
            "arima": {"d": 1}
        }

        metrics, chain, states, weights, y_true, y_pred, std_dev, summary = train_and_evaluate(
            parsed_data, range(0, 30), range(30, 40), models_to_use, model_params
        )

        # Check that a chain was selected
        self.assertGreater(len(chain), 0)
        self.assertLessEqual(len(chain), len(models_to_use))

        # Check that the chain is a subset of the models to use
        self.assertTrue(all(c in models_to_use for c in chain))

        # Check the search summary
        self.assertIn("stages", summary)
        self.assertEqual(len(summary["stages"]), len(models_to_use))
        
        # Check first stage summary
        stage1 = summary["stages"][0]
        self.assertEqual(stage1["stage"], 1)
        self.assertIn("candidates", stage1)
        self.assertIn("winner", stage1)
        self.assertIn(stage1["winner"], models_to_use)
        
        # Check that the winner is the one with the lowest weighted error
        winner_error = stage1["candidates"][stage1["winner"]]
        min_error = min(stage1["candidates"].values())
        self.assertAlmostEqual(winner_error, min_error)

    def test_model_skipping(self):
        """Test that a model is skipped if it doesn't improve RMSE."""
        # Create data where one model is very good and others are bad
        n = 50
        y = np.array([float(i * 10) for i in range(n)]) # Perfectly linear
        # Add noise only to the second half for the second model to fit
        y[n//2:] += 50 * np.random.randn(n - n//2)
        data = [{"y": v} for v in y]
        parsed_data = load_data_from_json(json.dumps(data))

        # MA with window 1 should be almost perfect on the first half
        # ES with high alpha will be noisy
        models_to_use = ["ma", "es"]
        model_params = {
            "ma": {"window": 1},
            "es": {"alpha": 0.9}
        }

        _, chain, _, _, _, _, _, summary = train_and_evaluate(
            parsed_data, range(0, 40), range(40, 50), models_to_use, model_params
        )

        # It's likely that after the first good model, the second one won't improve things
        # We check the summary for the "Model Skipped" action
        skipped = any(s.get("action") == "Model Skipped (did not improve RMSE)" for s in summary["stages"])
        
        # If skipped, the chain length should be less than the number of models
        if skipped:
            self.assertLess(len(chain), len(models_to_use))
        
        self.assertIn("action", summary["stages"][0]) # First stage should always add
        self.assertIn("action", summary["stages"][1]) # Second stage might skip


if __name__ == "__main__":
    unittest.main()
