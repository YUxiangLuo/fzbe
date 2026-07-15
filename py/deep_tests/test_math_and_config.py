from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from py.deep_tests.support.harness import DATASETS, assert_prediction_contract, prediction_values, run_command
from py.deep_tests.support.oracles import metrics, nonnegative_line_search, recursive_ma, shrunk_inverse_mse_weights
from py.deep_tests.support.profiles import profile
from py.src.dynamic_config import derive_arima_search_limit, derive_boosting_max_rounds, derive_lstm_config, plan_internal_holdout
from py.src.metrics import mae, mape, r2, rmse
from py.src.models import boosting, lstm, ma, stacking, weighted_avg


class MathematicalGoldenTests(unittest.TestCase):
    @profile("fast")
    def test_recursive_ma_exact_and_not_teacher_forced(self):
        expected = [70 / 3, 250 / 9, 820 / 27]
        np.testing.assert_allclose(recursive_ma([10, 20, 40], 3, 3), expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(ma._forecast_future([10, 20, 40], 3, 3), expected, rtol=0, atol=1e-12)
        self.assertNotEqual(30.0, expected[1])

    @profile("fast")
    def test_metrics_match_independent_formula_and_boundaries(self):
        actual, predicted = [0, 2, 4, 6], [1, 2, 5, 4]
        expected = metrics(actual, predicted)
        for name, function in (("rmse", rmse), ("mae", mae), ("mape", mape), ("r2", r2)):
            self.assertAlmostEqual(expected[name], function(actual, predicted), places=12)
        self.assertEqual(1.0, r2([3, 3], [3, 3]))
        self.assertEqual(0.0, r2([3, 3], [2, 2]))
        self.assertEqual(0.0, mape([0, 0], [10, 20]))

    @profile("fast")
    def test_weighted_stacking_and_boosting_goldens(self):
        actual = np.asarray([10, 20, 30, 40, 50], dtype=float)
        columns = {"ma": [9, 19, 29, 39, 49], "es": [8, 21, 32, 39, 52]}
        expected = shrunk_inverse_mse_weights(actual, columns)
        observed, details = weighted_avg._compute_inverse_residual_mse_weights(actual, columns)
        for name in expected:
            self.assertAlmostEqual(expected[name], observed[name], places=12)
        self.assertAlmostEqual(1.0, sum(observed.values()), places=12)
        self.assertEqual(set(columns), set(details["residual_mse"]))

        x = np.asarray([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
        y = x @ np.asarray([0.25, 0.75])
        meta = stacking._fit_meta_model(x, y, ["ma", "es"])
        np.testing.assert_allclose(meta["coefficients"], [0.25, 0.75], atol=1e-10)
        np.testing.assert_allclose(stacking._predict_with_meta_model(meta, x), y, atol=1e-10)

        for residual, prediction in (([2, 4], [1, 2]), ([-2, -4], [1, 2]), ([1, 2], [0, 0])):
            self.assertAlmostEqual(nonnegative_line_search(residual, prediction), boosting._nonnegative_line_search(residual, prediction), places=12)
        np.testing.assert_equal(boosting._apply_stage_output_semantics(0, [-2, 3]), [0, 3])
        np.testing.assert_equal(boosting._apply_stage_output_semantics(1, [-2, 3]), [-2, 3])

    @profile("fast")
    def test_lstm_direct_windows_do_not_include_future_inputs(self):
        dataset = np.arange(24, dtype=float).reshape(12, 2)
        x, y = lstm._create_direct_dataset(dataset, look_back=3, forecast_horizon=2)
        self.assertEqual((8, 3, 2), x.shape)
        self.assertEqual((8, 2), y.shape)
        for index in range(len(x)):
            np.testing.assert_equal(x[index], dataset[index:index + 3])
            np.testing.assert_equal(y[index], dataset[index + 3:index + 5, 0])


class DynamicConfigurationTests(unittest.TestCase):
    @profile("fast")
    def test_arima_search_boundaries(self):
        with self.assertRaisesRegex(ValueError, "\\[VALIDATION\\]"):
            derive_arima_search_limit(6, 1)
        cases = [(6, 0, 1), (13, 1, 1), (14, 1, 2), (21, 1, 2), (22, 1, 3), (31, 1, 5), (37, 1, 5)]
        for train, d, expected in cases:
            with self.subTest(train=train, d=d):
                config = derive_arima_search_limit(train, d)
                self.assertEqual(expected, config["max_p"])
                self.assertEqual(expected, config["max_q"])

    @profile("fast")
    def test_lstm_schedule_budget_and_boundaries(self):
        for bad in ((5, 1, 1), (10, 0, 1), (10, 1, 0)):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "\\[VALIDATION\\]"):
                derive_lstm_config(*bad)
        for n in (10, 16, 24, 40, 64):
            config = derive_lstm_config(n, 3, 1)
            self.assertEqual(n - config["look_back"] - 3 + 1, config["windows"])
            self.assertLessEqual(config["parameter_count"], config["parameter_budget"])
            self.assertIn(config["units"], (2, 4, 8, 16))
            self.assertIn(config["batch_size"], (1, 2, 4, 8, 16))
            self.assertEqual(config, derive_lstm_config(n, 3, 1))

    @profile("fast")
    def test_holdout_and_boost_round_boundaries(self):
        feasible = plan_internal_holdout("weighted", 20, 2, lambda prefix, validation: True)
        self.assertEqual(4, feasible["validation_size"])
        self.assertEqual(4, plan_internal_holdout("stacking", 20, 2, lambda p, v: True)["validation_size"])
        with self.assertRaisesRegex(ValueError, "No chronological"):
            plan_internal_holdout("boosting", 20, 2, lambda p, v: False)
        self.assertEqual([1, 3, 4], [derive_boosting_max_rounds(2, value) for value in (2, 4, 10)])


if __name__ == "__main__":
    unittest.main()
