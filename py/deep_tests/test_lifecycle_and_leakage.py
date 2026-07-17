from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from py.deep_tests.support.harness import (
    DATASETS, MODEL_TYPES, IsolatedWorkspace, assert_prediction_contract, corrupt_copy,
    mutate_records, prediction_values, run_loaded, run_model,
)
from py.deep_tests.support.oracles import metrics
from py.deep_tests.support.profiles import profile
from py.src.serialization import load_model


class SevenModelLifecycleTests(unittest.TestCase):
    @profile("full")
    def test_all_seven_real_cli_train_save_load_and_corruption(self):
        with IsolatedWorkspace() as temp:
            source = DATASETS / "seasonal_pulse_60.json"
            for model in MODEL_TYPES:
                with self.subTest(model=model):
                    artifact = temp / f"{model}.pkl"
                    trained = run_model(model, source, artifact)
                    self.assertTrue(artifact.is_file())
                    assert_prediction_contract(trained.payload, 4)
                    self.assertEqual(4, trained.payload["metrics"].get("evaluated_points", len(trained.payload["eval_y_true"])))
                    expected_metrics = metrics(trained.payload["eval_y_true"], prediction_values(trained.payload))
                    for name, expected in expected_metrics.items():
                        self.assertTrue(math.isclose(expected, trained.payload["metrics"][name], rel_tol=1e-7, abs_tol=1e-7), (model, name))
                    state = load_model(str(artifact))
                    self.assertEqual(model, state["model_type"])
                    self.assertEqual(4, state["algorithm_revision"])
                    self.assertEqual(2, state["artifact_schema_revision"])
                    self.assertEqual("trusted_pickle_state", state["artifact_format"])
                    self.assertIn("packages", state["runtime_compatibility"])
                    if model == "lstm":
                        self.assertEqual("historical_only", state["feature_timing"])
                        self.assertIs(False, state["known_future_features"])
                    if model == "arima":
                        self.assertNotIn("calibration_mean_error", state)
                        self.assertNotIn("calibration_count", state)
                        self.assertTrue(all(
                            point["calibration_mean_error"] is None
                            and point["calibration_count"] is None
                            for point in trained.payload["predictions"]
                        ))
                    if model in {"weighted_avg", "stacking", "boosting"}:
                        split_plan = state["split_plan"]
                        self.assertGreaterEqual(split_plan["prefix_size"], 2)
                        self.assertGreaterEqual(split_plan["validation_size"], 2)
                        self.assertNotIn("scheme", split_plan)
                        self.assertNotIn("folds", split_plan)
                        expected_calibration_source = (
                            "stacking_meta_fit_holdout_reused"
                            if model == "stacking"
                            else (
                                "boosting_selection_holdout_reused"
                                if model == "boosting"
                                else "internal_validation"
                            )
                        )
                        self.assertEqual(
                            expected_calibration_source,
                            state["uncertainty"]["calibration_source"],
                        )
                        if model in {"stacking", "boosting"}:
                            self.assertIs(
                                False,
                                state["uncertainty"]["summary"]["coverage_guarantee"],
                            )
                            self.assertEqual(
                                1,
                                state["uncertainty"]["summary"]["calibration_origins"],
                            )
                            self.assertTrue(all(
                                point["coverage_guarantee"] is False
                                and point["upper_error_p99_kind"] == "uncalibrated_estimate"
                                and point["calibration_source"] == expected_calibration_source
                                and point["calibration_origins"] == 1
                                and point["interval_kind"]
                                == "censored_nonnegative_uncalibrated_empirical_residual_quantile"
                                and 0
                                <= point["interval_lower"]
                                <= point["prediction"]
                                <= point["interval_upper"]
                                for point in trained.payload["predictions"]
                            ))

                    first = run_loaded(model, artifact)
                    second = run_loaded(model, artifact)
                    assert_prediction_contract(first.payload, 4)
                    tolerance = 1e-5 if model == "lstm" else 1e-9
                    for left, right in zip(prediction_values(first.payload), prediction_values(second.payload)):
                        self.assertTrue(math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance))

                    corrupted = temp / f"{model}-corrupt.pkl"
                    corrupt_copy(artifact, corrupted)
                    failed = run_loaded(model, corrupted, expected_ok=False)
                    self.assertIn("[MODEL]", failed.stderr)

                    if model == "lstm":
                        sidecar = Path(state["keras_model_path"])
                        backup = temp / "lstm-sidecar-backup.h5"
                        shutil.copy2(sidecar, backup)
                        sidecar.unlink()
                        self.assertIn("[MODEL]", run_loaded(model, artifact, expected_ok=False).stderr)
                        shutil.copy2(backup, sidecar)
                    elif model in {"weighted_avg", "stacking"}:
                        member = Path(next(iter(state["base_model_states"].values()))["model_file"])
                        backup = temp / f"{model}-member-backup.pkl"
                        shutil.copy2(member, backup)
                        member.write_bytes(b"broken member")
                        self.assertNotEqual(0, run_loaded(model, artifact, expected_ok=False).returncode)
                        shutil.copy2(backup, member)
                    elif model == "boosting":
                        model_info = next(iter(state["base_model_states"].values()))
                        member = Path(model_info["models"][0]["model_file"])
                        backup = temp / "boosting-member-backup.pkl"
                        shutil.copy2(member, backup)
                        member.unlink()
                        self.assertNotEqual(0, run_loaded(model, artifact, expected_ok=False).returncode)
                        shutil.copy2(backup, member)

    @profile("full")
    def test_eval_and_gap_labels_cannot_change_predictions_for_all_seven(self):
        with IsolatedWorkspace() as temp:
            source = DATASETS / "gap_probe_40.json"
            eval_mutation = temp / "eval-mutated.json"
            gap_mutation = temp / "gap-mutated.json"
            mutate_records(source, eval_mutation, range(32, 36), target_delta=100000)
            mutate_records(source, gap_mutation, range(28, 32), target_delta=200000)
            for model in MODEL_TYPES:
                with self.subTest(model=model):
                    # This fixture's split matches common_args except train is 0..35 there;
                    # create model-specific argv through copies padded so the invariant region is post-train.
                    # common_args trains through 35, therefore use the seasonal fixture split for this table.
                    baseline_source = DATASETS / "seasonal_pulse_60.json"
                    eval2 = temp / f"{model}-eval.json"
                    gap2 = temp / f"{model}-gap.json"
                    mutate_records(baseline_source, eval2, range(40, 44), target_delta=100000)
                    mutate_records(baseline_source, gap2, range(36, 40), target_delta=200000)
                    baseline = run_model(model, baseline_source).payload
                    changed_eval = run_model(model, eval2).payload
                    changed_gap = run_model(model, gap2).payload
                    self.assertNotEqual(baseline["eval_y_true"], changed_eval["eval_y_true"])
                    if model == "lstm":
                        for payload in (baseline, changed_eval, changed_gap):
                            self.assertEqual("historical_only", payload["feature_timing"])
                            self.assertIs(False, payload["known_future_features"])
                    tolerance = 1e-5 if model == "lstm" else 1e-8 if model == "arima" else 1e-10
                    for candidate in (changed_eval, changed_gap):
                        for left, right in zip(prediction_values(baseline), prediction_values(candidate)):
                            self.assertTrue(math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance), (model, left, right))


class NightlyLstmEnsembleTests(unittest.TestCase):
    @profile("nightly")
    def test_real_lstm_member_lifecycle_for_each_ensemble(self):
        with IsolatedWorkspace() as temp:
            source = DATASETS / "mixed_features_48.json"
            # Real ma+lstm fan-out is intentionally nightly; every artifact is temporary.
            from py.deep_tests.support.harness import PY_ROOT, run_command
            import sys
            for model in ("weighted_avg", "stacking", "boosting"):
                artifact = temp / f"{model}-lstm.pkl"
                argv = [sys.executable, PY_ROOT / "run_model.py", model, "--data-file", source,
                        "--train-start", "0", "--train-end", "35", "--eval-start", "40", "--eval-end", "43",
                        "--models", "ma,lstm", "--window", "3", "--lstm-target-feature", "y",
                        "--lstm-features", '["price","visits","region"]',
                        "--feature-types", '{"y":"numeric","price":"numeric","visits":"numeric","region":"categorical"}',
                        "--normalization", "minmax", "--save-model", artifact]
                result = run_command(argv, timeout=1800)
                assert_prediction_contract(result.payload, 4)
                assert_prediction_contract(run_loaded(model, artifact).payload, 4)


if __name__ == "__main__":
    unittest.main()
