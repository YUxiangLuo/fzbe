from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import guided_ensemble_model
import production_prepare_model
from baselines import fit_baseline_state
from data_utils import ParsedData, load_data_from_json
from py.deep_tests.support.profiles import profile
from py.src.models import boosting
from py.src.uncertainty import ensemble_growth_profile


CALIBRATION_RESIDUALS = np.asarray([-2.0, -1.0, 1.0, 2.0])


def _parsed_data():
    values = [10.0, 10.0, 10.0, 11.0, 11.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    records = [
        {
            "月份": f"2025-{index + 1:02d}",
            "销售数量": value,
        }
        for index, value in enumerate(values)
    ]
    return load_data_from_json(json.dumps(records, ensure_ascii=False))


def _expected_profile(stage_coefficient: float) -> np.ndarray:
    calibration_std = float(np.std(CALIBRATION_RESIDUALS, ddof=1))
    growth = (1.0 + stage_coefficient * 100.0) / (
        1.0 + stage_coefficient
    )
    return np.asarray([calibration_std, calibration_std * growth])


def _calendar_parsed_data(months: list[str]):
    records = [
        {
            "年份": int(month.split("-")[0]),
            "月份": int(month.split("-")[1]),
            "销售数量": float(int(month.split("-")[1])),
        }
        for month in months
    ]
    frame = pd.DataFrame(records)
    frame["timestamp"] = months
    # These tests exercise Boosting's internal calendar alignment directly.
    # Public JSON ingestion rejects sparse monthly input by contract.
    return records, ParsedData(
        df=frame,
        target_key="销售数量",
        time_key="timestamp",
        feature_keys=["年份", "月份"],
    )


class BoostingShrinkageMetadataTests(unittest.TestCase):
    @profile("fast")
    def test_policy_is_explicit_and_legacy_artifacts_are_backfilled(self):
        expected = {"enabled": False, "factor": 1.0}
        first = boosting.get_post_line_search_shrinkage_metadata()
        second = boosting.get_post_line_search_shrinkage_metadata({})

        self.assertEqual(expected, first)
        self.assertEqual(expected, second)
        self.assertIsNot(first, second)

        legacy_state = {
            "model_type": "boosting",
            "boosting_variant": "residual",
            "model_chain": [],
            "stage_coefficients": [],
            "base_model_states": {},
            "baseline": {"kind": "mean", "training_mean": 10.0},
        }
        boosting.attach_method_metadata(
            legacy_state,
            boosting.METHOD_NAME,
            boosting.FORECAST_STRATEGY,
            boosting.IMPLEMENTATION_NOTES,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "legacy.pkl"
            with artifact_path.open("wb") as handle:
                pickle.dump(legacy_state, handle)
            loaded = boosting._load_state(str(artifact_path))

        self.assertEqual(expected, loaded["post_line_search_shrinkage"])

    @profile("fast")
    def test_policy_rejects_ambiguous_or_incompatible_artifact_values(self):
        invalid_values = [
            {"enabled": True, "factor": 1.0},
            {"enabled": False, "factor": 0.1},
            {"enabled": False, "factor": True},
            {"enabled": False, "factor": 1.0, "learning_rate": 1.0},
            None,
            "disabled",
        ]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "invalid post-line-search shrinkage metadata",
            ):
                boosting.get_post_line_search_shrinkage_metadata(
                    {"post_line_search_shrinkage": value}
                )


class CalendarAlignedSeasonalBaselineTests(unittest.TestCase):
    @profile("fast")
    def test_timestamp_outside_selection_does_not_change_baseline_winner(self):
        months = [f"2024-{month:02d}" for month in range(1, 13)] + [
            "2025-02",
            "2025-03",
        ]
        _, parsed_data = _calendar_parsed_data(months)
        parsed_data.df.loc[13, parsed_data.time_key] = "not-a-date"

        state, _fitted, predictions, diagnostics = (
            boosting.choose_stage_zero_baseline(
                parsed_data,
                range(0, 12),
                range(12, 13),
            )
        )

        self.assertEqual("seasonal_naive", state["kind"])
        np.testing.assert_allclose(predictions, [2.0])
        self.assertEqual(0.0, diagnostics["candidates"]["seasonal_naive"]["rmse"])

    @profile("fast")
    def test_invalid_timestamp_inside_selection_rejects_seasonal_candidate(self):
        months = [f"2024-{month:02d}" for month in range(1, 13)] + [
            "2025-02",
            "2025-03",
        ]
        _, parsed_data = _calendar_parsed_data(months)
        parsed_data.df.loc[12, parsed_data.time_key] = "not-a-date"

        state, _fitted, _predictions, diagnostics = (
            boosting.choose_stage_zero_baseline(
                parsed_data,
                range(0, 12),
                range(12, 13),
            )
        )

        self.assertNotEqual("seasonal_naive", state["kind"])
        self.assertNotIn("seasonal_naive", diagnostics["candidates"])

    @profile("fast")
    def test_missing_month_between_train_and_selection_keeps_seasonal_phase(self):
        months = [f"2024-{month:02d}" for month in range(1, 13)] + [
            "2025-02"
        ]
        _, parsed_data = _calendar_parsed_data(months)

        state, _fitted, predictions, diagnostics = (
            boosting.choose_stage_zero_baseline(
                parsed_data,
                range(0, 12),
                range(12, 13),
            )
        )

        self.assertEqual("seasonal_naive", state["kind"])
        self.assertEqual(12, state["seasonal_period"])
        np.testing.assert_allclose(predictions, [2.0])
        self.assertEqual(0.0, diagnostics["candidates"]["seasonal_naive"]["rmse"])
        self.assertGreater(diagnostics["candidates"]["mean"]["rmse"], 0.0)

    @profile("fast")
    def test_calendar_forecast_aligns_each_sparse_evaluation_month(self):
        months = [f"2024-{month:02d}" for month in range(1, 13)] + [
            "2025-02",
            "2025-04",
            "2025-08",
        ]
        _, parsed_data = _calendar_parsed_data(months)

        predictions = boosting.baseline_forecast_for_range(
            parsed_data,
            range(0, 12),
            range(12, 15),
            "seasonal_naive",
            12,
        )

        np.testing.assert_allclose(predictions, [2.0, 4.0, 8.0])

    @profile("fast")
    def test_full_refit_with_missing_history_persists_next_calendar_cycle(self):
        months = [f"2024-{month:02d}" for month in range(1, 13)] + [
            "2025-02",
            "2025-03",
            "2025-04",
            "2025-05",
        ]
        _, parsed_data = _calendar_parsed_data(months)
        train_range = range(0, len(months))

        fitted = boosting.baseline_fitted_values_for_range(
            parsed_data,
            train_range,
            "seasonal_naive",
            12,
        )
        state = boosting.fit_baseline_state_for_range(
            parsed_data,
            train_range,
            "seasonal_naive",
            12,
        )

        # February 2025 is aligned with February 2024, not with the row that
        # happens to be twelve observations earlier (January 2024).
        self.assertEqual(2.0, fitted[12])
        self.assertEqual("calendar_month_slots_v1", state["seasonal_alignment"])
        # Training ends in May: consecutive production replay must start at
        # June and retain the most recent value for each calendar month.
        np.testing.assert_allclose(
            boosting.forecast_from_baseline_state(state, 12),
            [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )

    @profile("fast")
    def test_baseline_only_training_uses_calendar_eval_and_saved_replay(self):
        train_months = [f"2024-{month:02d}" for month in range(1, 13)] + [
            "2025-02",
            "2025-03",
            "2025-04",
            "2025-05",
        ]
        eval_months = ["2025-06", "2025-08"]
        _, parsed_data = _calendar_parsed_data(train_months + eval_months)
        split_plan = {
            "purpose": "boosting",
            "prefix_size": 12,
            "selection_size": 2,
            "validation_size": 2,
            "calibration_size": 2,
            "calibration_available": True,
        }

        with (
            patch.object(
                boosting,
                "plan_ensemble_selection_calibration_split",
                return_value=split_plan,
            ),
            patch.object(boosting, "_build_round_records", return_value=([], [])),
            patch.object(
                boosting,
                "_train_stage_model",
                return_value=("candidate.pkl", np.zeros(2), [1.0, 1.0]),
            ),
            patch.object(
                boosting,
                "_compute_in_sample_fitted_predictions",
                side_effect=lambda _name, _path, _data, train_range: np.zeros(
                    len(train_range)
                ),
            ),
            patch.object(boosting, "remove_member_model_artifact"),
        ):
            result = boosting.train_and_evaluate(
                parsed_data,
                range(0, len(train_months)),
                range(len(train_months), len(train_months) + len(eval_months)),
                ["ma"],
                {"ma": {"window": 2}},
                learning_rate=0.0,
                max_rounds=0,
            )

        self.assertEqual([], result[1])
        self.assertEqual(
            {"enabled": False, "factor": 1.0},
            result[7]["post_line_search_shrinkage"],
        )
        np.testing.assert_allclose(result[5], [6.0, 8.0])
        saved_baseline = result[8]["baseline"]
        np.testing.assert_allclose(
            boosting.forecast_from_baseline_state(saved_baseline, 3),
            [6.0, 7.0, 8.0],
        )

    @profile("fast")
    def test_production_initialize_replays_calendar_aligned_saved_baseline(self):
        months = [f"2024-{month:02d}" for month in range(1, 13)] + [
            f"2025-{month:02d}" for month in range(1, 6)
        ]
        records, parsed_data = _calendar_parsed_data(months)
        source_baseline = boosting.fit_baseline_state_for_range(
            parsed_data,
            range(0, len(months)),
            "seasonal_naive",
            12,
        )
        source_state = {
            "model_type": "boosting",
            "boosting_variant": "residual",
            "model_chain": [],
            "stage_coefficients": [],
            "base_model_states": {},
            "baseline": source_baseline,
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
            data_file.write_text(
                json.dumps(records, ensure_ascii=False),
                encoding="utf-8",
            )
            with source_model.open("wb") as handle:
                pickle.dump(source_state, handle)
            args = SimpleNamespace(
                data_file=str(data_file),
                source_model=str(source_model),
                model_type="boosting",
                save_model=str(root / "production.pkl"),
                train_start=0,
                train_end=len(months) - 1,
                forecast_steps=3,
            )
            state: dict = {}
            output = production_prepare_model._initialize(args, state)

        self.assertTrue(output["direct_finalize"])
        np.testing.assert_allclose(state["future_prediction"], [6.0, 7.0, 8.0])
        np.testing.assert_allclose(
            boosting.forecast_from_baseline_state(state["baseline"], 3),
            [6.0, 7.0, 8.0],
        )

    @profile("fast")
    def test_production_initialize_rejects_obsolete_baseline_only_container(self):
        records = [
            {
                "月份": f"2025-{index + 1:02d}",
                "销售数量": float(index + 1),
            }
            for index in range(6)
        ]
        obsolete_state = {
            "model_type": "boosting",
            "boosting_variant": "residual",
            "algorithm_revision": 3,
            "model_chain": [],
            "stage_coefficients": [],
            "base_model_states": {},
            "baseline": {
                "kind": "mean",
                "training_mean": 2.5,
            },
            "uncertainty": {},
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_file = root / "data.json"
            source_model = root / "obsolete.pkl"
            data_file.write_text(
                json.dumps(records, ensure_ascii=False),
                encoding="utf-8",
            )
            with source_model.open("wb") as handle:
                pickle.dump(obsolete_state, handle)
            args = SimpleNamespace(
                data_file=str(data_file),
                source_model=str(source_model),
                model_type="boosting",
                save_model=str(root / "production.pkl"),
                train_start=0,
                train_end=3,
                forecast_steps=2,
            )
            checkpoint: dict = {}
            with self.assertRaisesRegex(
                ValueError,
                "obsolete algorithm revision",
            ):
                production_prepare_model._initialize(args, checkpoint)

        self.assertEqual({}, checkpoint)

    @profile("fast")
    def test_legacy_seasonal_state_without_calendar_metadata_still_replays(self):
        legacy_state = {
            "kind": "seasonal_naive",
            "seasonal_period": 2,
            "seasonal_tail": [30.0, 40.0],
        }
        np.testing.assert_allclose(
            boosting.forecast_from_baseline_state(legacy_state, 5),
            [30.0, 40.0, 30.0, 40.0, 30.0],
        )


class BoostingBaselineGrowthTests(unittest.TestCase):
    @profile("fast")
    def test_guided_baseline_only_result_keeps_explicit_shrinkage_policy(self):
        parsed_data = _parsed_data()
        baseline = fit_baseline_state(
            np.asarray(parsed_data.y[range(0, 8)], dtype=float),
            "mean",
        )
        state = {
            "parsed_data": parsed_data,
            "train_range": {"start": 0, "stop": 8},
            "eval_range": {"start": 8, "stop": 10},
            "calibration_range": {"start": 8, "stop": 8},
            "models": ["ma"],
            "model_params": {"ma": {"window": 2}},
            "base_model_files": {},
            "model_chain": [],
            "stage_coefficients": [],
            "selection_coefficients": [],
            "baseline": baseline,
            "full_uncertainty_profiles": {},
            "uncertainty_coefficients": {},
            "final_eval_preds": boosting.forecast_from_baseline_state(baseline, 2),
            "eval_y_true": np.asarray(parsed_data.y[range(8, 10)]).tolist(),
            "search_summary": {},
            "split_plan": {
                "purpose": "boosting",
                "prefix_size": 6,
                "selection_size": 2,
                "calibration_size": 0,
            },
        }

        guided_ensemble_model._run_boosting("combine_evaluate", None, state)

        expected = {"enabled": False, "factor": 1.0}
        self.assertEqual([], state["final_result"]["model_chain"])
        self.assertEqual(
            expected,
            state["final_result"]["post_line_search_shrinkage"],
        )
        self.assertEqual(
            expected,
            state["final_result"]["search_summary"][
                "post_line_search_shrinkage"
            ],
        )

    @profile("fast")
    def test_corrupt_saved_baseline_fails_before_member_subprocess(self):
        invalid_baselines = [
            {"kind": "mean"},
            {"kind": "mean", "training_mean": True},
            {"kind": "mean", "training_mean": float("nan")},
            {"kind": "mean", "training_mean": float("inf")},
            {"kind": "mean", "training_mean": -float("inf")},
            {"kind": "mean", "training_mean": [1.0]},
            {"kind": "naive"},
            {"kind": "naive", "last_value": False},
            {"kind": "naive", "last_value": float("nan")},
            {"kind": "naive", "last_value": float("inf")},
            {"kind": "naive", "last_value": -float("inf")},
            {"kind": "naive", "last_value": [1.0]},
            {
                "kind": "seasonal_naive",
                "seasonal_period": True,
                "seasonal_tail": [1.0, 2.0],
            },
            {
                "kind": "seasonal_naive",
                "seasonal_period": 2.0,
                "seasonal_tail": [1.0, 2.0],
            },
            {
                "kind": "seasonal_naive",
                "seasonal_period": 1,
                "seasonal_tail": [1.0],
            },
            {
                "kind": "seasonal_naive",
                "seasonal_period": 2,
            },
            {
                "kind": "seasonal_naive",
                "seasonal_period": 2,
                "seasonal_tail": [1.0],
            },
            {
                "kind": "seasonal_naive",
                "seasonal_period": 2,
                "seasonal_tail": [[1.0, 2.0]],
            },
            {
                "kind": "seasonal_naive",
                "seasonal_period": 2,
                "seasonal_tail": [True, 2.0],
            },
            {
                "kind": "seasonal_naive",
                "seasonal_period": 2,
                "seasonal_tail": [float("nan"), 2.0],
            },
            {
                "kind": "seasonal_naive",
                "seasonal_period": 2,
                "seasonal_tail": [float("inf"), 2.0],
            },
        ]
        with patch.object(
            boosting,
            "run_base_model_predict_subprocess",
        ) as member_subprocess:
            for baseline in invalid_baselines:
                with self.subTest(baseline=baseline):
                    with self.assertRaisesRegex(ValueError, r"^\[VALIDATION\]"):
                        boosting.predict(
                            "",
                            2,
                            {
                                "model_chain": ["ma"],
                                "stage_coefficients": [1.0],
                                "base_model_states": {"ma": {"model_file": "member.pkl"}},
                                "baseline": baseline,
                            },
                        )
            member_subprocess.assert_not_called()

    @profile("fast")
    def test_legacy_zero_baseline_requires_a_nonempty_stage_chain(self):
        for state in (
            {
                "model_chain": [],
                "stage_coefficients": [],
                "baseline": {"kind": "zero"},
            },
            {
                "model_chain": [],
                "stage_coefficients": [],
            },
        ):
            with self.subTest(state=state):
                with self.assertRaisesRegex(
                    ValueError,
                    "require a non-empty stage chain",
                ):
                    boosting._get_baseline_state(state)

        self.assertEqual(
            {"kind": "zero", "label": "旧版零起点"},
            boosting._get_baseline_state({"model_chain": ["ma"]}),
        )
        self.assertEqual(
            {"kind": "zero"},
            boosting._get_baseline_state(
                {"model_chain": ["ma"], "baseline": {"kind": "zero"}}
            ),
        )

    @profile("fast")
    def test_explicit_baseline_growth_is_continuous_and_legacy_zero_is_compatible(self):
        explicit_profiles = {"stage_0_ma": [1.0, 100.0]}
        explicit_coefficients = {"stage_0_ma": 1e-9}
        boosting._add_baseline_uncertainty_member(
            explicit_profiles,
            explicit_coefficients,
            {"kind": "naive", "last_value": 100.0},
            2,
            has_member_stages=True,
        )
        self.assertEqual([1.0, 1.0], explicit_profiles["stage_0_baseline"])
        self.assertEqual(1.0, explicit_coefficients["stage_0_baseline"])

        near_empty_profile, _ = ensemble_growth_profile(
            explicit_profiles,
            explicit_coefficients,
            CALIBRATION_RESIDUALS,
            [100.0, 100.0],
            2,
            calibration_source="post_selection_calibration",
        )
        baseline_only_profile, _ = ensemble_growth_profile(
            {"stage_0_baseline": [1.0, 1.0]},
            {"stage_0_baseline": 1.0},
            CALIBRATION_RESIDUALS,
            [100.0, 100.0],
            2,
            calibration_source="post_selection_calibration",
        )
        np.testing.assert_allclose(
            near_empty_profile,
            baseline_only_profile,
            rtol=1e-6,
        )

        legacy_profiles = {"stage_0_ma": [1.0, 100.0]}
        legacy_coefficients = {"stage_0_ma": 0.01}
        boosting._add_baseline_uncertainty_member(
            legacy_profiles,
            legacy_coefficients,
            {"kind": "zero"},
            2,
            has_member_stages=True,
        )
        self.assertNotIn("stage_0_baseline", legacy_profiles)
        self.assertNotIn("stage_0_baseline", legacy_coefficients)

        boosting._add_baseline_uncertainty_member(
            legacy_profiles,
            legacy_coefficients,
            {"kind": "zero"},
            2,
            has_member_stages=False,
        )
        self.assertEqual([1.0, 1.0], legacy_profiles["stage_0_baseline"])

    @profile("fast")
    def test_loaded_nonempty_chain_includes_explicit_baseline_growth(self):
        state = {
            "model_type": "boosting",
            "boosting_variant": "residual",
            "model_chain": ["ma"],
            "stage_coefficients": [0.01],
            "base_model_states": {
                "ma": {
                    "models": [
                        {
                            "model_file": "unused-ma.pkl",
                            "coefficient": 0.01,
                        }
                    ]
                }
            },
            "baseline": {
                "kind": "naive",
                "last_value": 100.0,
            },
            "uncertainty": {
                "calibration_residuals": CALIBRATION_RESIDUALS.tolist(),
                "calibration_source": "post_selection_calibration",
                "reference_series": [100.0, 100.0],
            },
        }
        member_result = {
            "predictions": [
                {"prediction": 1.0, "std_dev": 1.0},
                {"prediction": 1.0, "std_dev": 100.0},
            ]
        }
        with patch.object(
            boosting,
            "run_base_model_predict_subprocess",
            return_value=member_result,
        ):
            predictions, uncertainty_profile, summary = boosting.predict(
                "",
                2,
                state,
            )

        np.testing.assert_allclose(predictions, [100.01, 100.01])
        np.testing.assert_allclose(
            uncertainty_profile,
            _expected_profile(0.01),
        )
        self.assertEqual(
            "post_selection_calibration",
            summary["calibration_source"],
        )

    @profile("fast")
    def test_regular_nonempty_chain_includes_explicit_baseline_growth(self):
        parsed_data = _parsed_data()
        split_plan = {
            "purpose": "boosting",
            "prefix_size": 3,
            "selection_size": 3,
            "calibration_size": 2,
        }
        selected_baseline = {
            "kind": "mean",
            "label": "mean",
            "seasonal_period": None,
        }
        baseline_selection_predictions = (
            np.asarray(parsed_data.y[range(3, 6)], dtype=float) - 1.0
        )
        stage_results = iter(
            (
                ("candidate-ma.pkl", np.ones(3), [1.0, 100.0, 100.0]),
                ("final-ma.pkl", np.ones(2), [1.0, 100.0]),
            )
        )
        training_events: list[str] = []

        def train_stage(*_args, **_kwargs):
            result = next(stage_results)
            training_events.append(result[0])
            return result

        calibration_predictions = (
            np.asarray(parsed_data.y[range(6, 8)], dtype=float)
            - CALIBRATION_RESIDUALS[:2]
        )

        def prepare_calibration(*_args, **_kwargs):
            training_events.append("calibration")
            return calibration_predictions, ["ma"], [0.01]

        with (
            patch.object(
                boosting,
                "plan_ensemble_selection_calibration_split",
                return_value=split_plan,
            ),
            patch.object(boosting, "derive_boosting_max_rounds", return_value=1),
            patch.object(
                boosting,
                "choose_stage_zero_baseline",
                return_value=(
                    selected_baseline,
                    np.asarray(parsed_data.y[range(0, 3)], dtype=float) - 1.0,
                    baseline_selection_predictions,
                    {"selected_rmse": 1.0},
                ),
            ),
            patch.object(boosting, "_build_round_records", return_value=([], [])),
            patch.object(
                boosting,
                "_train_stage_model",
                side_effect=train_stage,
            ),
            patch.object(
                boosting,
                "_compute_in_sample_fitted_predictions",
                side_effect=lambda _name, _path, _data, train_range: np.ones(
                    len(train_range)
                ),
            ),
            patch.object(
                boosting,
                "_nonnegative_line_search",
                side_effect=(1.0, 0.01),
            ),
            patch.object(boosting, "remove_member_model_artifact"),
            patch.object(
                boosting,
                "_refit_fixed_chain_for_holdout",
                side_effect=prepare_calibration,
            ),
        ):
            result = boosting.train_and_evaluate(
                parsed_data,
                range(0, 8),
                range(8, 10),
                ["ma"],
                {"ma": {"window": 2}},
                learning_rate=0.5,
                max_rounds=1,
            )

        self.assertEqual(["ma"], result[1])
        self.assertEqual(
            ["candidate-ma.pkl", "calibration", "final-ma.pkl"],
            training_events,
        )
        expected_residuals = np.asarray(parsed_data.y[range(6, 8)]) - (
            calibration_predictions
        )
        expected_std = float(np.std(expected_residuals, ddof=1))
        expected_growth = (1.0 + 0.01 * 100.0) / 1.01
        np.testing.assert_allclose(
            result[6],
            [expected_std, expected_std * expected_growth],
        )

    @profile("fast")
    def test_calibration_is_rejected_when_full_refit_changes_the_chain(self):
        residuals = np.asarray([1.0, -1.0], dtype=float)
        metadata = {
            "requested_chain": ["ma", "es"],
            "retained_chain": ["ma", "es"],
            "coefficients": [0.5, 0.25],
            "matched_selection_structure": True,
            "matched_final_structure": False,
            "prepared_before_full_refit": True,
        }

        finalized_residuals, source, finalized = (
            boosting._finalize_selected_chain_calibration(
                residuals,
                "post_selection_calibration",
                metadata,
                ["ma"],
            )
        )

        self.assertEqual([], finalized_residuals.tolist())
        self.assertEqual(
            "unavailable_after_full_refit_chain_change",
            source,
        )
        self.assertFalse(finalized["matched_final_structure"])

        unavailable_residuals, unavailable_source, _ = (
            boosting._finalize_selected_chain_calibration(
                residuals,
                "unavailable_missing_pre_refit_calibration",
                metadata,
                ["ma", "es"],
            )
        )
        self.assertEqual([], unavailable_residuals.tolist())
        self.assertEqual(
            "unavailable_missing_pre_refit_calibration",
            unavailable_source,
        )

    @profile("fast")
    def test_guided_nonempty_chain_includes_explicit_baseline_growth(self):
        parsed_data = _parsed_data()
        baseline = fit_baseline_state(
            np.asarray(parsed_data.y[range(0, 8)], dtype=float),
            "mean",
        )
        calibration_predictions = (
            np.asarray(parsed_data.y[range(6, 8)], dtype=float)
            - CALIBRATION_RESIDUALS[:2]
        )
        state = {
            "parsed_data": parsed_data,
            "train_range": {"start": 0, "stop": 8},
            "eval_range": {"start": 8, "stop": 10},
            "calibration_range": {"start": 6, "stop": 8},
            "models": ["ma"],
            "model_params": {"ma": {"window": 2}},
            "base_model_files": {},
            "model_chain": ["ma"],
            "selection_model_chain": ["ma"],
            "stage_coefficients": [0.01],
            "selection_coefficients": [0.01],
            "baseline": baseline,
            "full_uncertainty_profiles": {
                "stage_0_ma": [1.0, 100.0],
            },
            "uncertainty_coefficients": {"stage_0_ma": 0.01},
            "final_eval_preds": [14.0, 15.0],
            "eval_y_true": [14.0, 15.0],
            "search_summary": {},
            "split_plan": {
                "purpose": "boosting",
                "prefix_size": 3,
                "selection_size": 3,
                "calibration_size": 2,
            },
        }
        with patch.object(
            guided_ensemble_model.boosting,
            "_refit_fixed_chain_for_holdout",
            return_value=(calibration_predictions, ["ma"], [0.01]),
        ):
            guided_ensemble_model._run_boosting(
                "combine_evaluate",
                None,
                state,
            )

        expected_residuals = np.asarray(parsed_data.y[range(6, 8)]) - (
            calibration_predictions
        )
        expected_std = float(np.std(expected_residuals, ddof=1))
        expected_growth = (1.0 + 0.01 * 100.0) / 1.01
        np.testing.assert_allclose(
            state["uncertainty"]["profile"],
            [expected_std, expected_std * expected_growth],
        )
        self.assertEqual(
            {"enabled": False, "factor": 1.0},
            state["search_summary"]["post_line_search_shrinkage"],
        )
        self.assertEqual(
            {"enabled": False, "factor": 1.0},
            state["final_result"]["post_line_search_shrinkage"],
        )


class ProductionCalibrationInvalidationTests(unittest.TestCase):
    @profile("fast")
    def test_chain_change_persists_reason_and_adds_explicit_baseline_growth(self):
        parsed_data = _parsed_data()
        baseline = fit_baseline_state(
            np.asarray(parsed_data.y[range(0, 8)], dtype=float),
            "mean",
        )
        source_state = {
            "model_type": "boosting",
            "model_chain": ["ma", "es"],
            "split_plan": {},
            "uncertainty": {
                "calibration_residuals": CALIBRATION_RESIDUALS.tolist(),
                "calibration_source": "post_selection_calibration",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            save_model = Path(directory) / "production.pkl"
            state = {
                "model_type": "boosting",
                "forecast_steps": 2,
                "source_state": source_state,
                "parsed_data": parsed_data,
                "train_start": 0,
                "train_end": 7,
                "save_model": str(save_model),
                "future_prediction": np.asarray([14.0, 15.0]),
                "baseline": baseline,
                "refit_stages": [
                    {
                        "stage": 0,
                        "model": "ma",
                        "coefficient": 0.01,
                    }
                ],
                "member_profiles": {
                    "stage_0_ma": [1.0, 100.0],
                },
                "uncertainty_coefficients": {"stage_0_ma": 0.01},
                "member_states": {"ma": {"models": []}},
            }
            growth_impl = production_prepare_model.ensemble_growth_profile
            with patch.object(
                production_prepare_model,
                "ensemble_growth_profile",
                wraps=growth_impl,
            ) as growth_spy:
                production_prepare_model._finalize(state)

            member_profiles = growth_spy.call_args.args[0]
            coefficients = growth_spy.call_args.args[1]
            self.assertEqual(
                "training_history",
                growth_spy.call_args.kwargs["calibration_source"],
            )
            self.assertEqual(
                [1.0, 1.0],
                member_profiles["stage_0_baseline"],
            )
            self.assertEqual(1.0, coefficients["stage_0_baseline"])

            with save_model.open("rb") as handle:
                artifact = pickle.load(handle)

        uncertainty = artifact["uncertainty"]
        self.assertEqual(
            {"enabled": False, "factor": 1.0},
            artifact["post_line_search_shrinkage"],
        )
        self.assertEqual([], uncertainty["calibration_residuals"])
        self.assertEqual(
            "training_history",
            uncertainty["calibration_source"],
        )
        self.assertEqual(
            "unavailable_after_production_refit_chain_change",
            uncertainty["calibration_invalidation_reason"],
        )


if __name__ == "__main__":
    unittest.main()
