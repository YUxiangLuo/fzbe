# py/test/test_lstm.py
import unittest
import json
import sys
import os
import tempfile
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from io import StringIO

# Add src to path to be able to import lstm from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models import lstm

class TestLstmUtils(unittest.TestCase):
    def test_parse_feature_types(self):
        self.assertEqual(lstm._parse_feature_types(None), {})
        self.assertEqual(lstm._parse_feature_types('{"a": "numeric", "b": "categorical"}'), {"a": "numeric", "b": "categorical"})
        with self.assertRaisesRegex(ValueError, r"\[DATA\] --feature_types must be a JSON object"):
            lstm._parse_feature_types("invalid json")
        with self.assertRaisesRegex(ValueError, r"\[DATA\] --feature_types must be a JSON object"):
            lstm._parse_feature_types("[1, 2]")

    def test_split_features(self):
        records = [{"x": {"a": 1, "b": 2}}, {"x": {"a": 3, "b": 4}}]
        self.assertEqual(lstm._split_features(records), ["a", "b"])
        with self.assertRaisesRegex(ValueError, r"\[DATA\] each record must contain 'x' dict"):
            lstm._split_features([{"y": 1}])
        with self.assertRaisesRegex(ValueError, r"\[DATA\] all records must share the same x keys and order"):
            lstm._split_features([{"x": {"a": 1}}, {"x": {"b": 2}}])

    def test_fit_numeric_scaler(self):
        values = np.array([1.0, 2.0, 3.0])
        _, tf, inv = lstm._fit_numeric_scaler(values, "zscore")
        np.testing.assert_almost_equal(tf(values), [-1.22474487, 0.0, 1.22474487])
        np.testing.assert_almost_equal(inv(tf(values)), values)
        _, tf, inv = lstm._fit_numeric_scaler(values, "minmax")
        np.testing.assert_almost_equal(tf(values), [0.0, 0.5, 1.0])
        np.testing.assert_almost_equal(inv(tf(values)), values)
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\] --norm must be one of none\|zscore\|minmax"):
            lstm._fit_numeric_scaler(values, "invalid")

class TestLstmCore(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"y": 10, "x": {"num": 1, "cat": "A"}},
            {"y": 12, "x": {"num": 2, "cat": "B"}},
            {"y": 15, "x": {"num": 3, "cat": "A"}},
            {"y": 11, "x": {"num": 4, "cat": "C"}},
            {"y": 14, "x": {"num": 5, "cat": "B"}},
            {"y": 16, "x": {"num": 6, "cat": "A"}},
        ]
        self.feature_types = {"num": "numeric", "cat": "categorical"}

    def test_prepare_arrays(self):
        prep = lstm._prepare_arrays(self.records, self.feature_types, "zscore", slice(0, 4), 2)
        self.assertEqual(prep["X_all"].shape[1], 1 + 4) # 1 numeric + 4 one-hot (A, B, C, UNK)
        X_seq, y_seq = prep["make_sequences"](0, 5)
        self.assertEqual(X_seq.shape, (4, 2, 5))
        self.assertEqual(y_seq.shape, (4,))

    def test_reconstruct_window_from_data(self):
        meta = {
            "lookback": 2,
            "input_dim": 5,
            "numeric_keys": ["num"],
            "categorical_keys": ["cat"],
            "num_params": [("z", 2.5, 1.118033988749895)],
            "cat_maps": {"cat": {"A": 1, "B": 2, "C": 3}},
        }
        window = lstm._reconstruct_window_from_data(self.records[-2:], meta)
        self.assertEqual(window.shape, (2, 5))

class TestLstmRun(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.records = [
            {"y": 10 + i, "x": {"num": i, "cat": "A" if i % 2 == 0 else "B"}}
            for i in range(20)
        ]
        self.data_str = json.dumps(self.records)
        self.feature_types_str = json.dumps({"num": "numeric", "cat": "categorical"})

    def test_run_training(self):
        args = MagicMock()
        args.data = self.data_str
        args.feature_types = self.feature_types_str
        args.train_start = 0
        args.train_end = 14
        args.eval_start = 15
        args.eval_end = 19
        args.lookback = 2
        args.epochs = 1
        args.save_model = os.path.join(self.temp_dir.name, "model.pt")
        args.load_model = None
        args.pretty = False
        args.norm = "zscore"

        with patch('sys.stdout', new=StringIO()) as fake_out:
            lstm.run(args)
            output = json.loads(fake_out.getvalue())

        self.assertEqual(output["mode"], "lstm_train_eval")
        self.assertIn("metrics", output)
        self.assertTrue(os.path.exists(output["saved_model"]))

    def test_run_prediction(self):
        # First, train and save a model
        train_args = MagicMock()
        train_args.data = self.data_str
        train_args.feature_types = self.feature_types_str
        train_args.train_start = 0
        train_args.train_end = 14
        train_args.eval_start = 15
        train_args.eval_end = 19
        train_args.lookback = 2
        train_args.epochs = 1
        model_path = os.path.join(self.temp_dir.name, "model.pt")
        train_args.save_model = model_path
        train_args.load_model = None
        train_args.pretty = False
        train_args.norm = "zscore"
        with patch('sys.stdout', new=StringIO()):
            lstm.run(train_args)

        # Now, load and predict
        predict_args = MagicMock()
        predict_args.load_model = model_path
        predict_args.forecast_steps = 5
        predict_args.data = None # Test with embedded window
        predict_args.pretty = False
        predict_args.norm = "zscore"

        with patch('sys.stdout', new=StringIO()) as fake_out:
            lstm.run(predict_args)
            output = json.loads(fake_out.getvalue())

        self.assertEqual(output["mode"], "lstm_loaded_predict")
        self.assertEqual(len(output["predictions"]), 5)

if __name__ == '__main__':
    unittest.main()
