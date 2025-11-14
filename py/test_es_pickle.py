# py/test_es_pickle.py
import unittest
import json
import sys
import os
import tempfile
import shutil
from unittest.mock import MagicMock
from io import StringIO

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models import es

class TestEsPickle(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_save_path = os.path.join(self.temp_dir, "es_model.pkl")
        self.data_path = "dummy_stacking_data.json"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_train_and_predict(self):
        # --- Test Training ---
        train_args = MagicMock()
        with open(self.data_path, 'r') as f:
            train_args.data = f.read()
        train_args.save_model = self.model_save_path
        train_args.train_start = 0
        train_args.train_end = 79
        train_args.eval_start = 80
        train_args.eval_end = 98
        train_args.trend = "add"
        train_args.seasonal = "add"
        train_args.seasonal_periods = 12
        train_args.norm = "zscore"
        train_args.seed = 42
        train_args.pretty = False
        train_args.load_model = None

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        es.run(train_args)
        sys.stdout = old_stdout
        
        train_output = json.loads(captured_output.getvalue())
        
        self.assertIn("metrics", train_output)
        self.assertTrue(os.path.exists(self.model_save_path))

        # --- Test Prediction ---
        predict_args = MagicMock()
        predict_args.load_model = self.model_save_path
        predict_args.forecast_steps = 10
        predict_args.pretty = False
        predict_args.eval_start = None
        predict_args.eval_end = None
        with open(self.data_path, 'r') as f:
            predict_args.data = f.read()

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        es.run(predict_args)
        sys.stdout = old_stdout
        
        predict_output = json.loads(captured_output.getvalue())
        
        self.assertIn("predictions", predict_output)
        self.assertEqual(len(predict_output["predictions"]), 10)

if __name__ == '__main__':
    unittest.main()
