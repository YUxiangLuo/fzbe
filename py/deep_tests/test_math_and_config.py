from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from py.deep_tests.support.oracles import metrics, nonnegative_line_search, recursive_ma, shrunk_inverse_mse_weights
from py.deep_tests.support.profiles import profile
from py.src.baselines import (
    baseline_fitted_values,
    baseline_forecast,
    fit_baseline_state,
    forecast_from_baseline_state,
)
from py.src.dynamic_config import derive_arima_search_limit, derive_boosting_max_rounds, derive_lstm_config, plan_internal_holdout
from py.src.ensemble_utils import plan_expanding_window_oof_folds
from py.src.metrics import mae, mape, r2, rmse
from py.src.model_info import MODEL_ALGORITHM_REVISION
from py.src.models import boosting, es, lstm, ma, stacking, weighted_avg
import production_prepare_model


class MathematicalGoldenTests(unittest.TestCase):
    @profile("fast")
    def test_es_alpha_comparison_is_diagnostic_and_chronological(self):
        comparison = es.build_alpha_comparison(
            np.linspace(10.0, 30.0, 16),
            0.35,
        )
        self.assertEqual("training_tail_chronological_holdout", comparison["scope"])
        self.assertEqual(3, len(comparison["candidates"]))
        self.assertEqual(1, sum(row["selected"] for row in comparison["candidates"]))
        self.assertEqual(0.35, comparison["selected_alpha"])
        self.assertEqual(
            min(comparison["candidates"], key=lambda row: row["rmse"])["alpha"],
            comparison["best_alpha_by_rmse"],
        )
        self.assertEqual(
            "diagnostic_only_user_choice_preserved",
            comparison["selection_policy"],
        )
        for selected_alpha in (0.200000001, 0.500000001, 0.799999999):
            with self.subTest(selected_alpha=selected_alpha):
                near_reference = es.build_alpha_comparison(
                    np.linspace(10.0, 30.0, 16),
                    selected_alpha,
                )
                selected_rows = [
                    row
                    for row in near_reference["candidates"]
                    if row["selected"]
                ]
                self.assertEqual(3, len(near_reference["candidates"]))
                self.assertEqual(1, len(selected_rows))
                self.assertEqual(selected_alpha, selected_rows[0]["alpha"])

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
        self.assertIsNone(mape([0, 0], [10, 20]))

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

        collinear_x = np.column_stack([
            np.linspace(10.0, 20.0, 12),
            np.linspace(10.0, 20.0, 12) + 1e-9,
        ])
        collinear_y = np.linspace(10.5, 20.5, 12)
        stable = stacking._fit_meta_model(
            collinear_x,
            collinear_y,
            ["left", "right"],
        )
        perturbed = stacking._fit_meta_model(
            collinear_x,
            collinear_y + np.linspace(-1e-8, 1e-8, 12),
            ["left", "right"],
        )
        self.assertTrue(stable["regularized"])
        self.assertEqual(
            "ridge_regularized_nonnegative_least_squares",
            stable["strategy"],
        )
        self.assertTrue(np.isfinite(stable["coefficients"]).all())
        self.assertLess(
            np.linalg.norm(
                np.asarray(stable["coefficients"])
                - np.asarray(perturbed["coefficients"])
            ),
            1e-3,
        )

        for residual, prediction in (([2, 4], [1, 2]), ([-2, -4], [1, 2]), ([1, 2], [0, 0])):
            self.assertAlmostEqual(nonnegative_line_search(residual, prediction), boosting._nonnegative_line_search(residual, prediction), places=12)
        np.testing.assert_equal(boosting._apply_stage_output_semantics(0, [-2, 3]), [0, 3])
        np.testing.assert_equal(boosting._apply_stage_output_semantics(1, [-2, 3]), [-2, 3])
        np.testing.assert_equal(
            boosting._apply_stage_output_semantics(
                0,
                [-2, 3],
                residual_target=True,
            ),
            [-2, 3],
        )

    @profile("fast")
    def test_causal_baselines_and_empty_boosting_chain(self):
        history = np.asarray([10, 20, 30, 40], dtype=float)
        np.testing.assert_equal(
            baseline_fitted_values(history, "naive"),
            [10, 10, 20, 30],
        )
        np.testing.assert_equal(
            baseline_forecast(history, 3, "seasonal_naive", 2),
            [30, 40, 30],
        )
        calibration_residuals = [1.0, -1.0, 2.0, -2.0]
        predictions, profile, summary = boosting.predict(
            "",
            3,
            {
                "model_chain": [],
                "stage_coefficients": [],
                "base_model_states": {},
                "baseline": {
                    "kind": "naive",
                    "last_value": 40.0,
                },
                "uncertainty": {
                    "calibration_residuals": calibration_residuals,
                    "calibration_source": "post_selection_calibration",
                    "reference_series": history.tolist(),
                },
            },
        )
        self.assertEqual([40.0, 40.0, 40.0], predictions)
        self.assertEqual("post_selection_calibration", summary["calibration_source"])
        self.assertEqual("empirical", summary["source"])
        self.assertFalse(summary["fallback"])
        self.assertEqual([4, 4, 4], [
            horizon["calibration_count"]
            for horizon in summary["horizons"]
        ])
        np.testing.assert_allclose(
            profile,
            [np.std(calibration_residuals, ddof=1)] * 3,
        )

    @profile("fast")
    def test_saved_baseline_state_rejects_boolean_numeric_fields_and_bad_calendar_metadata(self):
        with self.assertRaisesRegex(ValueError, "invalid training_mean"):
            forecast_from_baseline_state({"kind": "mean", "training_mean": True}, 2)
        with self.assertRaisesRegex(ValueError, "period is invalid"):
            forecast_from_baseline_state({
                "kind": "seasonal_naive",
                "seasonal_period": True,
                "seasonal_tail": [1.0, 2.0],
            }, 2)
        with self.assertRaisesRegex(ValueError, "tail is invalid"):
            forecast_from_baseline_state({
                "kind": "seasonal_naive",
                "seasonal_period": 2,
                "seasonal_tail": [1.0, float("nan")],
            }, 2)

        quarterly_state = fit_baseline_state(
            [10.0, 20.0, 30.0, 40.0, 11.0, 21.0, 31.0, 41.0],
            "seasonal_naive",
            4,
            [
                "2024-01", "2024-04", "2024-07", "2024-10",
                "2025-01", "2025-04", "2025-07", "2025-10",
            ],
        )
        np.testing.assert_equal(
            forecast_from_baseline_state(quarterly_state, 4),
            [11.0, 21.0, 31.0, 41.0],
        )
        with self.assertRaisesRegex(ValueError, "calendar metadata is invalid"):
            forecast_from_baseline_state({
                **quarterly_state,
                "seasonal_phase_remainder": 2,
            }, 2)

    @profile("fast")
    def test_production_source_validation_uses_every_model_specific_loader(self):
        obsolete_revision = MODEL_ALGORITHM_REVISION - 1
        source_states = {
            "ma": {
                "model_type": "ma",
                "window": 2,
            },
            "es": {
                "model_type": "es",
            },
            "arima": {
                "model_type": "arima",
            },
            "lstm": {
                "model_type": "lstm",
                "forecast_strategy": lstm.DIRECT_FORECAST_STRATEGY,
            },
            "weighted_avg": {
                "model_type": "weighted_avg",
                "models": ["ma", "es"],
            },
            "stacking": {
                "model_type": "stacking",
                "models": ["ma", "es"],
            },
            "boosting": {
                "model_type": "boosting",
                "boosting_variant": "residual",
            },
        }
        self.assertEqual(
            set(production_prepare_model.ALL_MODELS),
            set(source_states),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model_type, source_state in source_states.items():
                with self.subTest(model_type=model_type):
                    artifact = root / f"obsolete-{model_type}.pkl"
                    source_state["algorithm_revision"] = obsolete_revision
                    with artifact.open("wb") as handle:
                        pickle.dump(source_state, handle)
                    with self.assertRaisesRegex(
                        ValueError,
                        "obsolete algorithm revision",
                    ):
                        production_prepare_model._load_validated_source_state(
                            model_type,
                            str(artifact),
                        )

    @profile("fast")
    def test_baseline_only_boosting_production_can_finalize_without_refit_work(self):
        records = [
            {
                "月份": f"2024-{index + 1:02d}",
                "销售数量": float(10 + index),
            }
            for index in range(8)
        ]
        source_state = {
            "model_type": "boosting",
            "boosting_variant": "residual",
            "model_chain": [],
            "stage_coefficients": [],
            "base_model_states": {},
            "baseline": {
                "kind": "naive",
                "last_value": 17.0,
            },
            "uncertainty": {
                "calibration_residuals": [1.0, -1.0, 2.0, -2.0],
                "calibration_source": "post_selection_calibration",
                "reference_series": [10.0, 11.0, 12.0, 13.0],
            },
        }
        boosting.attach_method_metadata(
            source_state,
            boosting.METHOD_NAME,
            boosting.FORECAST_STRATEGY,
            boosting.IMPLEMENTATION_NOTES,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_file = root / "data.json"
            source_model = root / "source.pkl"
            production_model = root / "production.pkl"
            data_file.write_text(
                json.dumps(records, ensure_ascii=False),
                encoding="utf-8",
            )
            with source_model.open("wb") as handle:
                pickle.dump(source_state, handle)

            args = SimpleNamespace(
                model_type="boosting",
                data_file=str(data_file),
                source_model=str(source_model),
                save_model=str(production_model),
                train_start=0,
                train_end=7,
                forecast_steps=3,
            )
            checkpoint = {}
            initialized = production_prepare_model._initialize(args, checkpoint)
            self.assertEqual([], initialized["work_items"])
            self.assertTrue(initialized["direct_finalize"])

            finalized = production_prepare_model._finalize(checkpoint)
            self.assertEqual(str(production_model), finalized["saved_model"])
            self.assertEqual(3, finalized["prepared_points"])
            with production_model.open("rb") as handle:
                production_state = pickle.load(handle)
            self.assertEqual([], production_state["model_chain"])
            self.assertEqual({}, production_state["base_model_states"])
            self.assertEqual(
                "post_selection_calibration",
                production_state["uncertainty"]["calibration_source"],
            )
            self.assertEqual(
                [4, 4, 4],
                [
                    point["calibration_count"]
                    for point in production_state["prepared_predictions"]
                ],
            )

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

        folds = plan_expanding_window_oof_folds(range(0, 12), range(12, 16))
        self.assertEqual(3, len(folds))
        self.assertEqual(12, folds[0]["train_end_exclusive"])
        self.assertEqual(12, folds[0]["holdout_start"])
        self.assertEqual(16, folds[-1]["holdout_end_exclusive"])


if __name__ == "__main__":
    unittest.main()
