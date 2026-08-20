from __future__ import annotations

import itertools
import json
import math
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from py.deep_tests.support.harness import (
    DATASETS,
    PY_ROOT,
    IsolatedWorkspace,
    assert_prediction_contract,
    run_command,
    run_loaded,
)
from py.deep_tests.support.oracles import shrunk_inverse_mse_weights
from py.deep_tests.support.profiles import profile
from py.src.data_utils import ParsedData
from py.src.ensemble_utils import (
    plan_ensemble_selection_calibration_split,
    plan_expanding_window_oof_folds,
)
from py.src.models import boosting, stacking, weighted_avg
from py.src.serialization import load_model


def _prediction_payload(predictions: list[float], std_dev: float = 1.0):
    return {
        "predictions": [
            {"prediction": value, "std_dev": std_dev} for value in predictions
        ]
    }


class WeightedAverageDegenerateCaseTests(unittest.TestCase):
    @profile("fast")
    def test_zero_error_identical_columns_and_scale_equivalence_stay_finite(self):
        zero_targets = np.zeros(6, dtype=float)
        equal_columns = {
            "ma": np.zeros(6, dtype=float),
            "es": np.zeros(6, dtype=float),
            "arima": np.zeros(6, dtype=float),
        }
        weights, details = weighted_avg._compute_inverse_residual_mse_weights(
            zero_targets,
            equal_columns,
        )
        self.assertEqual(set(equal_columns), set(weights))
        np.testing.assert_allclose(list(weights.values()), [1 / 3] * 3, atol=1e-15)
        self.assertAlmostEqual(1.0, sum(weights.values()), places=15)
        self.assertEqual(0.0, details["epsilon"])
        self.assertEqual(
            "equal_raw_weight_among_exact_zero_members",
            details["zero_error_policy"],
        )
        self.assertEqual(set(equal_columns), set(details["exact_zero_members"]))
        self.assertTrue(
            np.isfinite(
                list(details["raw_weights"].values())
                + list(details["final_weights"].values())
            ).all()
        )

        targets = np.asarray([2.0, 4.0, 6.0, 8.0, 10.0])
        columns = {
            "perfect": targets.copy(),
            "biased": targets + 3.0,
            "noisy": np.asarray([8.0, 1.0, 9.0, 2.0, 12.0]),
        }
        base_weights, base_details = weighted_avg._compute_inverse_residual_mse_weights(
            targets,
            columns,
        )
        for multiplier in (1e-15, 1.0, 1e15):
            scaled_weights, scaled_details = weighted_avg._compute_inverse_residual_mse_weights(
                targets * multiplier,
                {name: values * multiplier for name, values in columns.items()},
            )
            for name in columns:
                self.assertAlmostEqual(base_weights[name], scaled_weights[name], places=13)
                self.assertAlmostEqual(
                    base_details["residual_mse_normalized"][name],
                    scaled_details["residual_mse_normalized"][name],
                    places=13,
                )
            self.assertAlmostEqual(
                base_details["normalization_scale"] * multiplier,
                scaled_details["normalization_scale"],
                places=13,
            )
        self.assertEqual(1.0, base_details["raw_weights"]["perfect"])
        self.assertEqual(0.0, base_details["raw_weights"]["biased"])
        self.assertEqual(0.0, base_details["raw_weights"]["noisy"])
        self.assertGreater(base_weights["perfect"], base_weights["biased"])
        self.assertLess(base_weights["perfect"], 1.0)
        self.assertGreater(min(base_weights.values()), 0.0)

        no_exact_columns = {
            "near": targets + np.asarray([0.25, -0.5, 0.75, -1.0, 1.25]),
            "biased": targets + 3.0,
            "noisy": np.asarray([8.0, 1.0, 9.0, 2.0, 12.0]),
        }
        reference_weights, reference_details = (
            weighted_avg._compute_inverse_residual_mse_weights(
                targets,
                no_exact_columns,
            )
        )
        for multiplier in (1e-15, 1.0, 1e15):
            observed, diagnostics = weighted_avg._compute_inverse_residual_mse_weights(
                targets * multiplier,
                {
                    name: values * multiplier
                    for name, values in no_exact_columns.items()
                },
            )
            np.testing.assert_allclose(
                list(observed.values()),
                list(reference_weights.values()),
                rtol=2e-14,
                atol=2e-14,
            )
            np.testing.assert_allclose(
                list(diagnostics["residual_mse_normalized"].values()),
                list(reference_details["residual_mse_normalized"].values()),
                rtol=2e-14,
                atol=2e-14,
            )
            self.assertEqual(
                "no_epsilon_normalized_inverse_ratio",
                diagnostics["epsilon_policy"],
            )
            self.assertTrue(np.isfinite(list(observed.values())).all())

        tiny_targets = np.asarray([1.0, 0.0])
        tiny_columns = {
            "near": np.asarray([1.0, 1e-200]),
            "far": np.asarray([1.0, 2e-200]),
        }
        tiny_weights, tiny_details = weighted_avg._compute_inverse_residual_mse_weights(
            tiny_targets,
            tiny_columns,
        )
        oracle_weights = shrunk_inverse_mse_weights(tiny_targets, tiny_columns)
        self.assertEqual([], tiny_details["exact_zero_members"])
        self.assertAlmostEqual(0.8, tiny_details["raw_weights"]["near"], places=14)
        self.assertAlmostEqual(0.2, tiny_details["raw_weights"]["far"], places=14)
        self.assertAlmostEqual(0.62, tiny_weights["near"], places=14)
        self.assertAlmostEqual(0.38, tiny_weights["far"], places=14)
        np.testing.assert_allclose(
            list(tiny_weights.values()),
            list(oracle_weights.values()),
            rtol=1e-14,
            atol=1e-14,
        )
        self.assertEqual({"near": None, "far": None}, tiny_details["residual_mse_normalized"])
        self.assertGreater(tiny_details["residual_rms_normalized"]["far"], 0.0)

        mixed_targets = np.asarray([1e308, 0.0])
        mixed_columns = {
            "near": np.asarray([1e308, 1e-200]),
            "far": np.asarray([1e308, 2e-200]),
        }
        mixed_weights, mixed_details = weighted_avg._compute_inverse_residual_mse_weights(
            mixed_targets,
            mixed_columns,
        )
        mixed_oracle = shrunk_inverse_mse_weights(mixed_targets, mixed_columns)
        self.assertEqual([], mixed_details["exact_zero_members"])
        self.assertAlmostEqual(0.8, mixed_details["raw_weights"]["near"], places=14)
        self.assertAlmostEqual(0.2, mixed_details["raw_weights"]["far"], places=14)
        self.assertAlmostEqual(0.62, mixed_weights["near"], places=14)
        self.assertAlmostEqual(0.38, mixed_weights["far"], places=14)
        np.testing.assert_allclose(
            list(mixed_weights.values()),
            list(mixed_oracle.values()),
            rtol=1e-14,
            atol=1e-14,
        )
        self.assertEqual({"near": None, "far": None}, mixed_details["residual_mse"])
        self.assertEqual(
            {"near": None, "far": None},
            mixed_details["residual_mse_normalized"],
        )
        json.dumps(mixed_details, allow_nan=False)

        extreme_targets = np.asarray([1e308, -1e308])
        extreme_columns = {
            "half": np.asarray([5e307, -5e307]),
            "zero": np.asarray([0.0, 0.0]),
            "opposite": np.asarray([-1e308, 1e308]),
        }
        extreme_weights, extreme_details = (
            weighted_avg._compute_inverse_residual_mse_weights(
                extreme_targets,
                extreme_columns,
            )
        )
        self.assertTrue(np.isfinite(list(extreme_weights.values())).all())
        self.assertAlmostEqual(1.0, sum(extreme_weights.values()), places=15)
        self.assertTrue(np.isfinite(
            list(extreme_details["residual_mse_normalized"].values())
        ).all())
        self.assertIn(None, extreme_details["residual_mse"].values())

        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\].*at least one"):
            weighted_avg._compute_inverse_residual_mse_weights(
                np.asarray([1.0, 2.0]),
                {},
            )
        with self.assertRaisesRegex(ValueError, "match the target length"):
            weighted_avg._compute_inverse_residual_mse_weights(
                np.asarray([1.0, 2.0]),
                {"ma": [1.0]},
            )
        for invalid_targets, invalid_predictions in (
            ([1.0, float("nan")], {"ma": [1.0, 2.0]}),
            ([1.0, 2.0], {"ma": [1.0, float("inf")]}),
        ):
            with self.assertRaisesRegex(ValueError, "finite"):
                weighted_avg._compute_inverse_residual_mse_weights(
                    np.asarray(invalid_targets),
                    invalid_predictions,
                )


class StackingDegenerateCaseTests(unittest.TestCase):
    @profile("fast")
    def test_zero_collinear_constrained_and_three_member_meta_models(self):
        zero_matrix = np.zeros((6, 2), dtype=float)
        zero_meta = stacking._fit_meta_model(
            zero_matrix,
            np.zeros(6, dtype=float),
            ["ma", "es"],
        )
        self.assertTrue(zero_meta["regularized"])
        self.assertIsNone(zero_meta["condition_number"])
        np.testing.assert_equal(zero_meta["coefficients"], [0.0, 0.0])
        np.testing.assert_equal(
            stacking._predict_with_meta_model(zero_meta, zero_matrix),
            np.zeros(6),
        )

        matrix = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        )
        expected_coefficients = np.asarray([0.2, 0.3, 0.5])
        target = matrix @ expected_coefficients
        meta = stacking._fit_meta_model(
            matrix,
            target,
            ["ma", "es", "arima"],
        )
        self.assertFalse(meta["regularized"])
        np.testing.assert_allclose(
            meta["coefficients"],
            expected_coefficients,
            rtol=0,
            atol=1e-12,
        )
        self.assertTrue(all(value >= 0 for value in meta["coefficients"]))

        negative_target_meta = stacking._fit_meta_model(
            matrix,
            -np.ones(len(matrix)),
            ["ma", "es", "arima"],
        )
        np.testing.assert_allclose(
            negative_target_meta["coefficients"],
            np.zeros(3),
            atol=1e-15,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"\[VALIDATION\].*model_count \+ 2",
        ):
            stacking._fit_meta_model(
                np.ones((3, 2)),
                np.ones(3),
                ["ma", "es"],
            )
        for corrupt in (
            {"kind": "wrong", "coefficients": [1.0, 1.0]},
            {"kind": "nnls", "coefficients": [1.0]},
            {"kind": "nnls", "coefficients": [1.0, math.nan]},
        ):
            with self.subTest(corrupt=corrupt):
                with self.assertRaisesRegex(ValueError, r"\[VALIDATION\]"):
                    stacking._predict_with_meta_model(
                        corrupt,
                        np.ones((4, 2)),
                    )


class BoostingStageAndResidualTests(unittest.TestCase):
    @profile("fast")
    def test_warmup_padding_signed_values_and_repeated_stage_artifacts(self):
        target = np.asarray([10.0, 20.0, 40.0, 80.0, 160.0])
        fitted = boosting._compute_ma_in_sample_fitted(target, 2)
        np.testing.assert_equal(fitted, [0.0, 0.0, 15.0, 30.0, 60.0])

        np.testing.assert_equal(
            boosting._normalize_fitted_values([math.nan, 2.0], 4),
            [0.0, 0.0, 0.0, 2.0],
        )
        np.testing.assert_equal(
            boosting._normalize_fitted_values([1.0, 2.0, 3.0], 2),
            [2.0, 3.0],
        )
        with self.assertRaisesRegex(ValueError, r"\[DATA\].*infinite"):
            boosting._normalize_fitted_values([1.0, math.inf], 2)

        state = {
            "model_chain": ["ma", "es", "ma"],
            "stage_coefficients": [0.5, 0.25, 0.125],
            "base_model_states": {
                "ma": {"models": [{"round": 1}, {"round": 2}]},
                "es": {"models": [{"round": 1}]},
            },
        }
        stages = list(boosting._iter_stage_models(state))
        self.assertEqual(
            [
                (0, "ma", 0.5, {"round": 1}),
                (1, "es", 0.25, {"round": 1}),
                (2, "ma", 0.125, {"round": 2}),
            ],
            stages,
        )

        for residual, candidate, expected in (
            ([-2.0, -4.0], [-1.0, -2.0], 2.0),
            ([-2.0, -4.0], [1.0, 2.0], 0.0),
            ([1.0, 2.0], [math.nan, 1.0], 0.0),
            ([1.0, 2.0], [math.inf, 1.0], 0.0),
        ):
            with self.subTest(residual=residual, candidate=candidate):
                self.assertEqual(
                    expected,
                    boosting._nonnegative_line_search(residual, candidate),
                )
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\].*different lengths"):
            boosting._nonnegative_line_search([1.0], [1.0, 2.0])

    @profile("fast")
    def test_corrupt_repeated_stage_state_is_rejected_before_prediction(self):
        states = (
            {
                "model_chain": ["ma"],
                "stage_coefficients": [],
                "base_model_states": {"ma": {"models": [{}]}},
            },
            {
                "model_chain": ["ma"],
                "stage_coefficients": [math.nan],
                "base_model_states": {"ma": {"models": [{}]}},
            },
            {
                "model_chain": ["ma", "ma"],
                "stage_coefficients": [1.0, 1.0],
                "base_model_states": {"ma": {"models": [{}]}},
            },
        )
        for state in states:
            with self.subTest(state=state):
                with self.assertRaisesRegex(ValueError, r"\[VALIDATION\]"):
                    list(boosting._iter_stage_models(state))


class EnsembleChronologyAndReplayTests(unittest.TestCase):
    @profile("fast")
    def test_oof_folds_cover_nonzero_selection_exactly_without_overlap(self):
        for selection_size, max_folds in ((2, 2), (5, 3), (11, 5)):
            with self.subTest(
                selection_size=selection_size,
                max_folds=max_folds,
            ):
                prefix = range(7, 24)
                selection = range(24, 24 + selection_size)
                folds = plan_expanding_window_oof_folds(
                    prefix,
                    selection,
                    max_folds=max_folds,
                )
                covered: list[int] = []
                prior_stop = selection.start
                for fold_index, fold in enumerate(folds, start=1):
                    self.assertEqual(fold_index, fold["fold"])
                    self.assertEqual(prefix.start, fold["train_start"])
                    self.assertEqual(fold["holdout_start"], fold["train_end_exclusive"])
                    self.assertEqual(prior_stop, fold["holdout_start"])
                    holdout = list(
                        range(
                            fold["holdout_start"],
                            fold["holdout_end_exclusive"],
                        )
                    )
                    covered.extend(holdout)
                    prior_stop = fold["holdout_end_exclusive"]
                self.assertEqual(list(selection), covered)

        invalid_ranges = (
            (range(0, 0), range(0, 2), 2),
            (range(0, 5), range(6, 8), 2),
            (range(0, 5), range(5, 6), 3),
            (range(0, 5), range(5, 8), 1),
        )
        for prefix, selection, max_folds in invalid_ranges:
            with self.subTest(
                prefix=prefix,
                selection=selection,
                max_folds=max_folds,
            ):
                with self.assertRaisesRegex(ValueError, r"\[VALIDATION\]"):
                    plan_expanding_window_oof_folds(
                        prefix,
                        selection,
                        max_folds=max_folds,
                    )

    @profile("fast")
    def test_all_three_split_planners_partition_nonzero_training_range(self):
        parsed = ParsedData(
            pd.DataFrame({"y": [80.0 + 0.5 * index for index in range(72)]}),
            target_key="y",
        )
        train_range = range(7, 61)
        models = ["ma", "es", "arima"]
        params = {
            "ma": {"window": 4},
            "es": {"alpha": 0.35},
            "arima": {"d": 1},
        }
        for purpose in ("weighted", "stacking", "boosting"):
            with self.subTest(purpose=purpose):
                plan = plan_ensemble_selection_calibration_split(
                    parsed,
                    train_range,
                    models,
                    params,
                    purpose,
                )
                self.assertEqual(len(train_range), plan["train_size"])
                self.assertEqual(
                    len(train_range),
                    plan["prefix_size"]
                    + plan["selection_size"]
                    + plan["calibration_size"],
                )
                self.assertEqual(
                    plan["prefix_size"],
                    plan["selection_start_offset"],
                )
                self.assertEqual(
                    plan["prefix_size"] + plan["selection_size"],
                    plan["calibration_start_offset"],
                )
                self.assertGreaterEqual(plan["selection_size"], 2)
                if purpose == "stacking":
                    prefix = range(
                        train_range.start,
                        train_range.start + plan["prefix_size"],
                    )
                    selection = range(
                        prefix.stop,
                        prefix.stop + plan["selection_size"],
                    )
                    folds = plan_expanding_window_oof_folds(prefix, selection)
                    covered = [
                        index
                        for fold in folds
                        for index in range(
                            fold["holdout_start"],
                            fold["holdout_end_exclusive"],
                        )
                    ]
                    self.assertEqual(list(selection), covered)

    @profile("fast")
    def test_stacking_planner_uses_two_folds_when_one_row_lstm_folds_are_infeasible(self):
        feature_names = ["x1", "x2", "x3", "x4"]
        parsed = ParsedData(
            pd.DataFrame(
                {
                    "y": [100.0 + index for index in range(20)],
                    **{
                        name: [float(index + offset) for index in range(20)]
                        for offset, name in enumerate(feature_names, start=1)
                    },
                }
            ),
            target_key="y",
            feature_keys=feature_names,
        )
        params = {
            "ma": {"window": 3},
            "lstm": {
                "feature_types": {
                    "y": "numeric",
                    **{name: "numeric" for name in feature_names},
                },
                "normalization": "minmax",
                "lstm_target_feature": "y",
                "lstm_features": feature_names,
            },
        }

        plan = plan_ensemble_selection_calibration_split(
            parsed,
            range(0, 13),
            ["ma", "lstm"],
            params,
            "stacking",
        )

        self.assertEqual(2, plan["oof_fold_count"])
        self.assertEqual([2, 2], [
            fold["holdout_end_exclusive"] - fold["holdout_start"]
            for fold in plan["oof_folds"]
        ])
        self.assertEqual(
            list(range(9, 13)),
            [
                index
                for fold in plan["oof_folds"]
                for index in range(
                    fold["holdout_start"],
                    fold["holdout_end_exclusive"],
                )
            ],
        )

    @profile("fast")
    def test_all_three_prediction_paths_handle_four_members_and_signed_boosting_stages(
        self,
    ):
        member_predictions = {
            "ma": [-2.0, 2.0, 4.0],
            "es": [10.0, 10.0, 10.0],
            "arima": [1.0, 3.0, 5.0],
            "lstm": [4.0, 6.0, 8.0],
        }

        def fake_member_predict(**kwargs):
            model_file = str(kwargs["model_file"])
            values = member_predictions[model_file]
            return _prediction_payload(values, std_dev=0.5)

        models = ["ma", "es", "arima", "lstm"]
        base_states = {model_name: {"model_file": model_name} for model_name in models}
        uncertainty = {
            "calibration_residuals": [1.0, -1.0, 2.0, -2.0],
            "calibration_source": "post_selection_calibration",
            "reference_series": [10.0, 11.0, 12.0, 13.0],
        }

        weighted_state = {
            "models": models,
            "weights": {"ma": 0.1, "es": 0.2, "arima": 0.3, "lstm": 0.4},
            "base_model_states": base_states,
            "uncertainty": uncertainty,
        }
        with patch.object(
            weighted_avg,
            "run_base_model_predict_subprocess",
            side_effect=fake_member_predict,
        ):
            predictions, profile_values, summary = weighted_avg.predict(
                "",
                3,
                weighted_state,
            )
        expected_matrix = np.column_stack(
            [np.maximum(member_predictions[name], 0.0) for name in models]
        )
        expected_weighted = expected_matrix @ np.asarray([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(predictions, expected_weighted, atol=1e-12)
        self.assertEqual(3, len(profile_values))
        self.assertEqual("post_selection_calibration", summary["calibration_source"])

        coefficients = [0.2, 0.3, 0.1, 0.4]
        stacking_state = {
            "models": models,
            "meta_model": {
                "kind": "nnls",
                "model_names": models,
                "coefficients": coefficients,
            },
            "base_model_states": base_states,
            "uncertainty": uncertainty,
        }
        with patch.object(
            stacking,
            "run_base_model_predict_subprocess",
            side_effect=fake_member_predict,
        ):
            predictions, profile_values, summary = stacking.predict(
                "",
                3,
                stacking_state,
            )
        np.testing.assert_allclose(
            predictions,
            expected_matrix @ np.asarray(coefficients),
            atol=1e-12,
        )
        self.assertEqual(3, len(profile_values))
        self.assertEqual("post_selection_calibration", summary["calibration_source"])

        stage_predictions = {
            "ma-stage-1": [-10.0, 1.0, 2.0],
            "es-stage-1": [4.0, -4.0, 0.0],
            "ma-stage-2": [1.0, 1.0, -10.0],
        }

        def fake_stage_predict(**kwargs):
            return _prediction_payload(
                stage_predictions[str(kwargs["model_file"])],
                std_dev=0.25,
            )

        boosting_state = {
            "model_chain": ["ma", "es", "ma"],
            "stage_coefficients": [1.0, 0.5, 2.0],
            "base_model_states": {
                "ma": {
                    "models": [
                        {"model_file": "ma-stage-1"},
                        {"model_file": "ma-stage-2"},
                    ]
                },
                "es": {"models": [{"model_file": "es-stage-1"}]},
            },
            "baseline": {"kind": "naive", "last_value": 5.0},
            "uncertainty": uncertainty,
        }
        with patch.object(
            boosting,
            "run_base_model_predict_subprocess",
            side_effect=fake_stage_predict,
        ):
            predictions, profile_values, summary = boosting.predict(
                "",
                3,
                boosting_state,
            )
        np.testing.assert_allclose(predictions, [0.0, 6.0, 0.0], atol=1e-12)
        self.assertEqual(3, len(profile_values))
        self.assertEqual("post_selection_calibration", summary["calibration_source"])
        self.assertTrue(np.isfinite(profile_values).all())

    @profile("nightly")
    def test_each_real_ensemble_trains_three_classical_members_from_nonzero_origin(
        self,
    ):
        source = DATASETS / "seeded_arx_96.json"
        with IsolatedWorkspace() as temp:
            for ensemble_name in ("weighted_avg", "stacking", "boosting"):
                with self.subTest(ensemble=ensemble_name):
                    artifact = temp / f"{ensemble_name}-three-classical.pkl"
                    trained = run_command(
                        [
                            sys.executable,
                            PY_ROOT / "run_model.py",
                            ensemble_name,
                            "--data-file",
                            source,
                            "--train-start",
                            "8",
                            "--train-end",
                            "63",
                            "--eval-start",
                            "68",
                            "--eval-end",
                            "73",
                            "--models",
                            "ma,es,arima",
                            "--window",
                            "4",
                            "--alpha",
                            "0.35",
                            "--d",
                            "0",
                            "--save-model",
                            artifact,
                        ],
                        timeout=1800,
                    )
                    assert_prediction_contract(trained.payload, 6)
                    state = load_model(str(artifact))
                    self.assertEqual(ensemble_name, state["model_type"])
                    self.assertTrue(state["split_plan"]["calibration_available"])
                    self.assertGreaterEqual(
                        state["split_plan"]["selection_size"],
                        2,
                    )
                    if ensemble_name == "weighted_avg":
                        self.assertEqual(
                            {"ma", "es", "arima"},
                            set(state["weights"]),
                        )
                        self.assertAlmostEqual(
                            1.0,
                            sum(state["weights"].values()),
                            places=10,
                        )
                    elif ensemble_name == "stacking":
                        self.assertEqual(
                            ["ma", "es", "arima"],
                            state["meta_model"]["model_names"],
                        )
                        self.assertEqual(
                            3,
                            len(state["meta_model"]["coefficients"]),
                        )
                        self.assertTrue(
                            np.isfinite(state["meta_model"]["coefficients"]).all()
                        )
                    else:
                        self.assertEqual(
                            len(state["model_chain"]),
                            len(state["stage_coefficients"]),
                        )
                        self.assertTrue(np.isfinite(state["stage_coefficients"]).all())

                    loaded = run_loaded(ensemble_name, artifact, steps=6)
                    assert_prediction_contract(loaded.payload, 6)


    # The point check above pins one 54-point range with three members. These
    # invariants are the ones a boundary bug would break, so they are swept
    # across every training size and member subset the planner accepts: a split
    # that overlaps, leaves a gap, or reports offsets that disagree with its own
    # segment sizes would silently train on data it also calibrates against.
    @profile("fast")
    def test_split_planner_invariants_hold_across_every_accepted_configuration(self):
        frame = pd.DataFrame({
            "month": [f"{2020 + index // 12}-{(index % 12) + 1:02d}" for index in range(160)],
            "y": [80.0 + 0.5 * index for index in range(160)],
        })
        parsed = ParsedData(frame, target_key="y", time_key="month")
        params = {
            "ma": {"window": 3},
            "es": {"alpha": 0.3},
            "arima": {"d": 1},
            "lstm": {
                "feature_types": {"y": "numeric"},
                "lstm_target_feature": "y",
                "lstm_features": "[]",
                "normalization": "minmax",
            },
        }
        members = ("ma", "es", "arima", "lstm")

        planned = 0
        for train_size in range(6, 121):
            train_range = range(0, train_size)
            for size in range(1, len(members) + 1):
                for models in itertools.combinations(members, size):
                    for purpose in ("weighted", "boosting", "stacking"):
                        try:
                            plan = plan_ensemble_selection_calibration_split(
                                parsed, train_range, list(models), params, purpose,
                            )
                        except ValueError:
                            continue
                        planned += 1
                        prefix = int(plan["prefix_size"])
                        selection = int(plan["selection_size"])
                        calibration = int(plan["calibration_size"])
                        label = f"n={train_size} {'+'.join(models)} {purpose}"

                        with self.subTest(label):
                            # Exhaustive and gap-free over the training range.
                            self.assertEqual(train_size, prefix + selection + calibration)
                            self.assertGreaterEqual(prefix, 2)
                            self.assertGreaterEqual(selection, 2)
                            # Calibration is either absent or large enough to mean something.
                            if calibration != 0:
                                self.assertGreaterEqual(calibration, 2)
                            # Offsets must agree with the segment sizes they describe.
                            self.assertEqual(prefix, int(plan["selection_start_offset"]))
                            self.assertEqual(
                                prefix + selection if calibration else train_size,
                                int(plan["calibration_start_offset"]),
                            )
                            self.assertEqual(calibration > 0, bool(plan["calibration_available"]))

                            if purpose == "stacking":
                                cursor = prefix
                                for fold in plan["oof_folds"]:
                                    # Expanding window: train always from the prefix
                                    # start, and never past its own holdout.
                                    self.assertEqual(0, int(fold["train_start"]))
                                    self.assertEqual(cursor, int(fold["holdout_start"]))
                                    self.assertEqual(
                                        int(fold["holdout_start"]),
                                        int(fold["train_end_exclusive"]),
                                    )
                                    self.assertLess(
                                        int(fold["holdout_start"]),
                                        int(fold["holdout_end_exclusive"]),
                                    )
                                    cursor = int(fold["holdout_end_exclusive"])
                                # The folds must tile the selection set exactly.
                                self.assertEqual(prefix + selection, cursor)

        self.assertGreater(planned, 3_000, "sweep did not exercise a meaningful range")


if __name__ == "__main__":
    unittest.main()
