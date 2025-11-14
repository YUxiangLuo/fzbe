# py/test/test_stacking.py
import unittest
import json
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO

# Add src to path to be able to import stacking from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models import stacking

class Teststacking(unittest.TestCase):
    def setUp(self):
        self.records = [{"sales": 10 + i} for i in range(20)]
        self.data_str = json.dumps(self.records)

    @patch('models.stacking._run_base_model')
    def test_run_stacking(self, mock_run_base_model):
        # Mock the base model runner to return predictable results
        def mock_runner(model_name, data_str, train_indices, test_indices, model_args):
            if model_name == "ma":
                return np.full(len(test_indices), 1.0)
            elif model_name == "es":
                return np.full(len(test_indices), 2.0)
            return np.full(len(test_indices), 0.0)
        
        mock_run_base_model.side_effect = mock_runner

        args = MagicMock()
        args.data = self.data_str
        args.base_models = ["ma", "es"]
        args.eval_start = 15
        args.eval_end = 19
        args.meta_l2 = 0.1
        args.pretty = False

        with patch('sys.stdout', new=StringIO()) as fake_out:
            stacking.run(args)
            output = json.loads(fake_out.getvalue())

        self.assertEqual(output["mode"], "stacking_v2")
        self.assertIn("metrics", output)
        self.assertEqual(len(output["eval_predictions"]), 5)

if __name__ == '__main__':
    unittest.main()
