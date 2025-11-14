# py/test_arima_pickle.py
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

from models import arima

class TestArimaPickle(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_save_path = os.path.join(self.temp_dir, "arima_model.pkl")
        self.data_path = "dummy_stacking_data.json" # Use the existing dummy data

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
        train_args.d = 1
        train_args.p_max = 3
        train_args.q_max = 3
        train_args.norm = "zscore"
        train_args.seed = 42
        train_args.pretty = False
        train_args.load_model = None # Ensure we are in training mode

        # Capture stdout to check the output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        arima.run(train_args)
        
        sys.stdout = old_stdout # Restore stdout
        
        train_output = json.loads(captured_output.getvalue())
        
        self.assertIn("metrics", train_output)
        self.assertIn("rmse", train_output["metrics"])
        self.assertTrue(os.path.exists(self.model_save_path))

        # --- Test Prediction ---
        predict_args = MagicMock()
        predict_args.load_model = self.model_save_path
        predict_args.forecast_steps = 10
        predict_args.pretty = False
        predict_args.eval_start = None
        predict_args.eval_end = None
        # Prediction doesn't need data, but let's provide it to be safe
        with open(self.data_path, 'r') as f:
            predict_args.data = f.read()

        sys.stdout = captured_output = StringIO()
        
        arima.run(predict_args)
        
        sys.stdout = old_stdout
        
        predict_output = json.loads(captured_output.getvalue())
        
        self.assertIn("predictions", predict_output)
        self.assertEqual(len(predict_output["predictions"]), 10)
        self.assertIn("prediction", predict_output["predictions"][0])

if __name__ == '__main__':
    unittest.main()