from __future__ import annotations

import math
import os
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from py.deep_tests.support.harness import DATASETS, assert_prediction_contract, prediction_values, run_command
from py.deep_tests.support.oracles import metrics, nonnegative_line_search, recursive_ma, shrunk_inverse_mse_weights
from py.deep_tests.support.profiles import profile
from py.src.dynamic_config import derive_arima_search_limit, derive_boosting_max_rounds, derive_lstm_config, plan_internal_holdout
from py.src.data_utils import ParsedData
from py.src.ensemble_utils import (
    controlled_ensemble_map,
    plan_ensemble_internal_holdout,
    rolling_origin_ranges,
)
from py.src.metrics import mae, mape, r2, rmse
from py.src.model_info import attach_method_metadata, ensure_state_metadata
from py.src.models import arima, boosting, lstm, ma, stacking, weighted_avg


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
    def test_checkpointed_boosting_folds_match_the_pooled_candidate_core(self):
        fold_states = [
            {
                "fold_index": 0,
                "prefix_range": range(0, 3),
                "validation_range": range(3, 5),
                "residual_val": np.asarray([4.0, 2.0]),
            },
            {
                "fold_index": 1,
                "prefix_range": range(0, 5),
                "validation_range": range(5, 7),
                "residual_val": np.asarray([3.0, 1.0]),
            },
        ]
        evaluated = {
            0: {
                "fold_predictions": np.asarray([2.0, 1.0]),
                "fold_fitted_train_predictions": np.asarray([0.0, 1.0, 1.5]),
            },
            1: {
                "fold_predictions": np.asarray([1.0, 2.0]),
                "fold_fitted_train_predictions": np.asarray([0.0, 1.0, 1.5, 2.0, 2.5]),
            },
        }
        checkpointed = boosting.finalize_rolling_origin_candidate(
            fold_states,
            "ma",
            [evaluated[0], evaluated[1]],
        )
        expected_gamma = nonnegative_line_search(
            [4.0, 2.0, 3.0, 1.0],
            [2.0, 1.0, 1.0, 2.0],
        )
        self.assertAlmostEqual(expected_gamma, checkpointed["coefficient"], places=12)

        def fake_fold(_parsed, fold, _model, _params, _round):
            return evaluated[fold["fold_index"]]

        with patch.object(
            boosting,
            "evaluate_rolling_origin_fold_candidate",
            side_effect=fake_fold,
        ):
            monolithic = boosting.evaluate_rolling_origin_candidate(
                object(), fold_states, "ma", {}, 0
            )
        self.assertAlmostEqual(
            checkpointed["coefficient"], monolithic["coefficient"], places=12
        )
        self.assertAlmostEqual(checkpointed["rmse"], monolithic["rmse"], places=12)
        for left, right in zip(
            checkpointed["next_residuals"], monolithic["next_residuals"]
        ):
            np.testing.assert_allclose(left, right, rtol=0, atol=0)

    @profile("fast")
    def test_lstm_direct_windows_do_not_include_future_inputs(self):
        dataset = np.arange(24, dtype=float).reshape(12, 2)
        x, y = lstm._create_direct_dataset(dataset, look_back=3, forecast_horizon=2)
        self.assertEqual((8, 3, 2), x.shape)
        self.assertEqual((8, 2), y.shape)
        for index in range(len(x)):
            np.testing.assert_equal(x[index], dataset[index:index + 3])
            np.testing.assert_equal(y[index], dataset[index + 3:index + 5, 0])

        metadata = lstm.attach_lstm_metadata({})
        self.assertEqual("historical_only", metadata["feature_timing"])
        self.assertIs(False, metadata["known_future_features"])
        with self.assertRaisesRegex(ValueError, "Only direct historical-window"):
            lstm.attach_lstm_metadata({}, lstm.LEGACY_FORECAST_STRATEGY)
        with self.assertRaisesRegex(ValueError, "Obsolete autoregressive"):
            lstm._forecast_from_state(
                {"forecast_strategy": lstm.LEGACY_FORECAST_STRATEGY},
                1,
            )

    @profile("fast")
    def test_arima_deployment_intervals_do_not_claim_eval_bias_calibration(self):
        point = arima._format_predictions([10.0], [[8.0, 12.0]])[0]
        self.assertEqual("arima_forecast_distribution", point["calibration_source"])
        self.assertIsNone(point["calibration_mean_error"])
        self.assertIsNone(point["calibration_count"])

    @profile("fast")
    def test_artifact_runtime_manifest_is_required_exactly(self):
        state = attach_method_metadata({}, "test", "test_strategy", ["test note"])
        self.assertIn("python", state["runtime_compatibility"])
        self.assertIn("packages", state["runtime_compatibility"])
        self.assertEqual(
            state,
            ensure_state_metadata(state, "test", "test_strategy", ["test note"]),
        )

        incompatible = deepcopy(state)
        incompatible["runtime_compatibility"]["python"] = "0.0"
        with self.assertRaisesRegex(ValueError, "runtime dependencies"):
            ensure_state_metadata(
                incompatible,
                "test",
                "test_strategy",
                ["test note"],
            )

        wrong_strategy = deepcopy(state)
        wrong_strategy["forecast_strategy"] = "different_strategy"
        with self.assertRaisesRegex(ValueError, "method metadata"):
            ensure_state_metadata(
                wrong_strategy,
                "test",
                "test_strategy",
                ["test note"],
            )


class DynamicConfigurationTests(unittest.TestCase):
    @profile("fast")
    def test_controlled_ensemble_parallelism_is_bounded_and_ordered(self):
        rendezvous = threading.Barrier(2)

        def task(value: int):
            def run() -> int:
                rendezvous.wait(timeout=10)
                return value
            return run

        with patch.dict(os.environ, {"FZ_ENSEMBLE_MAX_WORKERS": "8"}):
            self.assertEqual(
                [0, 1],
                controlled_ensemble_map(
                    [task(0), task(1)],
                    contains_lstm=True,
                ),
            )
        for invalid in ("0", "not-an-integer"):
            with self.subTest(invalid=invalid), patch.dict(
                os.environ,
                {"FZ_ENSEMBLE_MAX_WORKERS": invalid},
            ), self.assertRaisesRegex(ValueError, "positive integer"):
                controlled_ensemble_map([lambda: 1])

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

    @profile("fast")
    def test_ensemble_holdout_expands_into_nonoverlapping_rolling_origins(self):
        parsed = ParsedData(
            df=pd.DataFrame({"time": range(36), "y": range(36)}),
            target_key="y",
            time_key="time",
            feature_keys=[],
            time_frequency="numeric_step:1",
        )
        params = {"ma": {"window": 3}, "es": {"alpha": 0.2}}
        plan = plan_ensemble_internal_holdout(
            parsed,
            range(36),
            ["ma", "es"],
            params,
            "weighted",
        )
        self.assertEqual("expanding_window_rolling_origin", plan["scheme"])
        self.assertEqual(3, plan["fold_count"])
        self.assertEqual(24, plan["calibration_size"])
        self.assertIs(False, plan["single_fold_fallback"])
        ranges = rolling_origin_ranges(range(36), plan)
        self.assertEqual(
            [(0, 12, 12, 20), (0, 20, 20, 28), (0, 28, 28, 36)],
            [
                (prefix.start, prefix.stop, validation.start, validation.stop)
                for prefix, validation in ranges
            ],
        )
        validation_points = [point for _prefix, validation in ranges for point in validation]
        self.assertEqual(len(validation_points), len(set(validation_points)))

        small = ParsedData(
            df=pd.DataFrame({"time": range(8), "y": range(8)}),
            target_key="y",
            time_key="time",
            feature_keys=[],
            time_frequency="numeric_step:1",
        )
        fallback = plan_ensemble_internal_holdout(
            small,
            range(8),
            ["ma", "es"],
            params,
            "weighted",
        )
        self.assertEqual(1, fallback["fold_count"])
        self.assertIs(True, fallback["single_fold_fallback"])


if __name__ == "__main__":
    unittest.main()
