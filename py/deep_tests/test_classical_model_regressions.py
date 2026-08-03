from __future__ import annotations

import json
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import guided_model

from py.deep_tests.support.harness import (
    DATASETS,
    PY_ROOT,
    IsolatedWorkspace,
    assert_prediction_contract,
    prediction_values,
    run_command,
    run_loaded,
)
from py.deep_tests.support.oracles import recursive_ma, ses
from py.deep_tests.support.profiles import profile
from py.src.forecast_utils import SIGNED_PREDICTION_DOMAIN
from py.src.models import arima, es, ma
from py.src.serialization import load_model


class MovingAverageSituationMatrixTests(unittest.TestCase):
    @profile("fast")
    def test_recursive_forecast_matches_independent_oracle_across_windows_scales_and_horizons(
        self,
    ):
        cases = (
            ([0.0, 0.0, 0.0, 0.0], 2),
            ([10.0, -20.0, 40.0, -80.0], 3),
            ([1e12, 1e12 + 3.0, 1e12 - 6.0, 1e12 + 9.0], 4),
        )
        for history, window in cases:
            for steps in (1, 2, 8):
                with self.subTest(history=history, window=window, steps=steps):
                    expected = recursive_ma(history, window, steps)
                    observed = ma._forecast_future(history, window, steps)
                    np.testing.assert_allclose(
                        observed,
                        expected,
                        rtol=1e-15,
                        atol=max(1e-12, max(map(abs, history)) * 1e-15),
                    )
                    self.assertTrue(np.isfinite(observed).all())

        invalid_cases = (
            ([1.0, 2.0], 1, 1),
            ([1.0], 2, 1),
            ([1.0, 2.0], 2, 0),
        )
        for history, window, steps in invalid_cases:
            with self.subTest(invalid=(history, window, steps)):
                with self.assertRaisesRegex(ValueError, r"\[(?:VALIDATION|DATA)\]"):
                    ma._forecast_future(history, window, steps)

    @profile("fast")
    def test_nonzero_training_origin_and_gap_use_only_recursive_training_history(self):
        prefix = [9e8, -9e8, 7e8, -7e8]
        training = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
        gap = [8e7, -8e7]
        evaluation = [1.0, 2.0, 3.0]
        values = np.asarray(prefix + training + gap + evaluation, dtype=float)
        train_range = range(4, 10)
        eval_range = range(12, 15)
        expected_all = recursive_ma(
            training[-3:], 3, eval_range.stop - train_range.stop
        )
        expected = expected_all[
            eval_range.start - train_range.stop : eval_range.start
            - train_range.stop
            + len(eval_range)
        ]

        result = ma.train_and_evaluate(
            values,
            [str(index) for index in range(len(values))],
            train_range,
            eval_range,
            3,
            SIGNED_PREDICTION_DOMAIN,
        )
        np.testing.assert_allclose(result[3], expected, rtol=0, atol=1e-12)

        mutated = values.copy()
        mutated[:4] *= -123.0
        mutated[10:12] += 1e15
        changed = ma.train_and_evaluate(
            mutated,
            [str(index) for index in range(len(values))],
            train_range,
            eval_range,
            3,
            SIGNED_PREDICTION_DOMAIN,
        )
        np.testing.assert_allclose(changed[3], expected, rtol=0, atol=1e-12)


class ExponentialSmoothingSituationMatrixTests(unittest.TestCase):
    @profile("fast")
    def test_guided_es_labels_the_estimated_initial_level_explicitly(self):
        state = {
            "y": np.asarray([10.0, 12.0, 11.0, 14.0, 13.0, 15.0]),
            "alpha": 0.35,
            "train_range": {"start_index": 0, "end_index": 3},
            "evaluate_range": {"start_index": 4, "end_index": 5},
        }

        with patch.object(guided_model, "es", es):
            output = guided_model._run_es("fit_model", None, state)

        self.assertIn("initial_level", output)
        self.assertNotIn("level", output)
        self.assertTrue(math.isfinite(output["initial_level"]))

    @profile("fast")
    def test_alpha_extremes_constant_signed_and_large_scale_series_remain_finite(self):
        series_cases = (
            np.zeros(12, dtype=float),
            np.asarray([10.0, -5.0, 20.0, -10.0, 40.0, -20.0] * 2),
            np.asarray([1e12 + index * 100.0 for index in range(12)]),
        )
        for values in series_cases:
            for alpha in (1e-8, 0.35, 1.0):
                with self.subTest(scale=float(np.max(np.abs(values))), alpha=alpha):
                    model = es._fit_textbook_ses(values, alpha)
                    prediction = float(model.forecast(1)[0])
                    initial_level = float(model.params["initial_level"])
                    expected_level = ses(values, alpha, initial_level)
                    self.assertTrue(math.isfinite(prediction))
                    self.assertTrue(
                        math.isclose(
                            prediction,
                            expected_level,
                            rel_tol=1e-10,
                            abs_tol=max(1e-8, float(np.max(np.abs(values))) * 1e-10),
                        )
                    )
                    if alpha == 1.0:
                        self.assertAlmostEqual(float(values[-1]), prediction, places=8)

    @profile("fast")
    def test_nonzero_training_origin_and_gap_do_not_affect_flat_level_forecast(self):
        values = np.asarray(
            [1e9, -1e9, 5e8]
            + [15.0, 18.0, 22.0, 21.0, 25.0, 29.0, 32.0, 35.0]
            + [7e8, -7e8]
            + [40.0, 41.0, 42.0],
            dtype=float,
        )
        train_range = range(3, 11)
        eval_range = range(13, 16)
        observed = es.train_and_evaluate(
            values,
            [str(index) for index in range(len(values))],
            train_range,
            eval_range,
            0.2,
            SIGNED_PREDICTION_DOMAIN,
        )
        self.assertEqual(3, len(observed[3]))
        np.testing.assert_allclose(observed[3], [observed[3][0]] * 3, atol=1e-12)

        mutated = values.copy()
        mutated[:3] *= -77.0
        mutated[11:13] += 3e14
        changed = es.train_and_evaluate(
            mutated,
            [str(index) for index in range(len(values))],
            train_range,
            eval_range,
            0.2,
            SIGNED_PREDICTION_DOMAIN,
        )
        np.testing.assert_allclose(changed[3], observed[3], rtol=0, atol=1e-12)


class ClassicalValidationBoundaryTests(unittest.TestCase):
    @profile("fast")
    def test_ma_es_and_arima_reject_parameter_and_range_boundaries(self):
        values = np.arange(14, dtype=float)

        for window, message in (
            (1, "Window must be >= 2"),
            (13, "Training range must contain at least 13 points"),
        ):
            with self.subTest(model="ma", window=window):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"\[VALIDATION\].*{message}",
                ):
                    ma._validate_training_args(values, 0, 11, 12, 13, window)

        for alpha in (0.0, -0.1, 1.000001, math.nan, math.inf):
            with self.subTest(model="es", alpha=alpha):
                with self.assertRaisesRegex(ValueError, r"\[VALIDATION\]"):
                    es._validate_training_args(values, 0, 7, 9, 10, alpha)

        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\].*d must be >= 0"):
            arima.train_and_evaluate(
                values,
                [str(index) for index in range(len(values))],
                range(0, 8),
                range(9, 11),
                -1,
            )
        with self.assertRaisesRegex(
            ValueError, r"\[VALIDATION\].*at least 6 effective"
        ):
            arima.train_and_evaluate(
                values,
                [str(index) for index in range(len(values))],
                range(0, 8),
                range(9, 11),
                3,
            )
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\].*steps must be > 0"):
            arima.predict("", 0, state={})

    @profile("fast")
    def test_arima_search_falls_through_to_next_converged_aicc_candidate(self):
        first = SimpleNamespace(
            order=(2, 0, 0),
            seasonal_order=(0, 0, 0, 0),
            with_intercept=True,
            trend=None,
            arima_res_=SimpleNamespace(mle_retvals={"converged": False}),
        )
        second = SimpleNamespace(
            order=(1, 0, 0),
            seasonal_order=(0, 0, 0, 0),
            with_intercept=True,
            trend=None,
            arima_res_=SimpleNamespace(mle_retvals={"converged": True}),
        )
        with (
            patch.object(
                arima,
                "_run_auto_arima_search",
                return_value=(first, second),
            ),
            patch.object(
                arima,
                "_ensure_converged",
                side_effect=[
                    ValueError("[MODEL] first candidate did not converge"),
                    (second, False),
                ],
            ),
        ):
            selected, retried, summary = arima._select_converged_auto_arima(
                np.arange(20, dtype=float),
                d=0,
                max_p=2,
                max_q=2,
                context="test search",
            )

        self.assertIs(second, selected)
        self.assertTrue(retried)
        self.assertEqual(2, summary["candidate_count"])
        self.assertEqual(2, summary["selected_rank"])
        self.assertTrue(summary["fallback_used"])
        self.assertEqual(
            [
                {
                    "rank": 1,
                    "order": [2, 0, 0],
                    "with_intercept": True,
                    "reason": "did_not_converge_after_retry",
                }
            ],
            summary["skipped_candidates"],
        )


class ArimaSituationMatrixTests(unittest.TestCase):
    @profile("fast")
    def test_guided_arima_propagates_deterministic_boundary_to_summary_and_metrics(
        self,
    ):
        best_model = SimpleNamespace(
            order=(0, 1, 0),
            seasonal_order=(0, 0, 0, 0),
            with_intercept=True,
            trend=None,
            arima_res_=SimpleNamespace(mle_retvals={"converged": False}),
            aic=lambda: 1.0,
            aicc=lambda: 1.5,
            bic=lambda: 2.0,
        )
        state = {
            "y": np.arange(12, dtype=float),
            "timestamps": [f"t{index}" for index in range(12)],
            "d": 1,
            "max_p": 2,
            "max_q": 2,
            "train_range": {"start_index": 0, "end_index": 7},
            "evaluate_range": {"start_index": 8, "end_index": 9},
            "data_points": 12,
            "save_model": None,
            "best_model": best_model,
            "convergence_retried": True,
            "candidate_selection": {"selected_rank": 1},
            "diagnostics": {},
        }

        with patch.object(guided_model, "arima", arima):
            selection = guided_model._run_arima("select_model", None, state)
            self.assertTrue(selection["convergence_boundary_case"])
            self.assertTrue(state["search_summary"]["convergence_boundary_case"])

            # Simulate resuming a session checkpoint written before this field
            # was added; evaluate_metrics must reconstruct and persist it.
            state.pop("convergence_boundary_case")
            state["search_summary"].pop("convergence_boundary_case")
            state.update(
                {
                    "y_true": [8.0, 9.0],
                    "y_pred": [8.0, 9.0],
                    "confidence_intervals": [[8.0, 8.0], [9.0, 9.0]],
                    "converged": False,
                }
            )
            evaluation = guided_model._run_arima("evaluate_metrics", None, state)

            # The same reconstruction is required when an older checkpoint is
            # resumed directly at its final save stage.
            state["metrics"].pop("convergence_boundary_case")
            state["search_summary"].pop("convergence_boundary_case")
            saved = guided_model._run_arima("save_artifact", None, state)

        self.assertFalse(evaluation["metrics"]["converged"])
        self.assertTrue(evaluation["metrics"]["convergence_boundary_case"])
        self.assertTrue(state["search_summary"]["convergence_boundary_case"])
        self.assertTrue(
            saved["final_result"]["metrics"]["convergence_boundary_case"]
        )

    @profile("full")
    def test_d_zero_stochastic_and_d_one_deterministic_cases_have_valid_search_diagnostics(
        self,
    ):
        seeded_records = json.loads(
            (DATASETS / "seeded_arx_96.json").read_text(encoding="utf-8")
        )
        seeded_values = np.asarray(
            [record["y"] for record in seeded_records],
            dtype=float,
        )
        seeded_months = [record["month"] for record in seeded_records]
        trend_values = np.asarray([80.0 + 2.0 * index for index in range(52)])
        trend_months = [f"t{index}" for index in range(len(trend_values))]

        cases = (
            (
                "stochastic-level",
                seeded_values,
                seeded_months,
                range(8, 56),
                range(60, 64),
                0,
                False,
            ),
            (
                "deterministic-trend",
                trend_values,
                trend_months,
                range(5, 40),
                range(44, 48),
                1,
                True,
            ),
        )
        for (
            name,
            values,
            months,
            train_range,
            eval_range,
            differencing,
            deterministic,
        ) in cases:
            with self.subTest(name=name):
                (
                    metrics,
                    _model,
                    order,
                    search,
                    y_true,
                    predictions,
                    intervals,
                ) = arima.train_and_evaluate(
                    values,
                    months,
                    train_range,
                    eval_range,
                    differencing,
                    max_p=2,
                    max_q=2,
                )
                self.assertEqual(differencing, order["d"])
                self.assertEqual(order, search["selected_order"])
                self.assertEqual(
                    len(train_range) - differencing, search["effective_train_size"]
                )
                self.assertTrue(
                    metrics["converged"] or search["convergence_boundary_case"]
                )
                if deterministic:
                    self.assertLess(metrics["rmse"], 1e-3)
                else:
                    self.assertTrue(metrics["converged"])
                self.assertEqual(len(eval_range), len(y_true))
                self.assertEqual(len(eval_range), len(predictions))
                self.assertEqual((len(eval_range), 2), np.asarray(intervals).shape)
                self.assertTrue(np.isfinite(predictions).all())
                self.assertTrue(np.isfinite(intervals).all())
                self.assertTrue(
                    np.all(np.asarray(intervals)[:, 0] <= np.asarray(intervals)[:, 1])
                )
                for criterion in ("aic", "aicc", "bic"):
                    self.assertTrue(math.isfinite(float(search[criterion])))
                candidate_selection = search["candidate_selection"]
                self.assertGreaterEqual(candidate_selection["candidate_count"], 1)
                self.assertGreaterEqual(candidate_selection["selected_rank"], 1)
                self.assertLessEqual(
                    candidate_selection["selected_rank"],
                    candidate_selection["candidate_count"],
                )
                self.assertEqual(
                    candidate_selection["selected_rank"] > 1,
                    candidate_selection["fallback_used"],
                )
                diagnostics = search["diagnostics"]
                if diagnostics["ljung_box_pvalue"] is not None:
                    self.assertGreaterEqual(diagnostics["ljung_box_pvalue"], 0.0)
                    self.assertLessEqual(diagnostics["ljung_box_pvalue"], 1.0)
                self.assertEqual(len(eval_range), metrics["evaluated_points"])

    @profile("full")
    def test_all_classical_clis_cover_nonzero_origin_gap_long_replay_and_d_two(self):
        records = [
            {
                "month": f"t{index:02d}",
                "y": 20.0 + 0.5 * index + 0.25 * index * index,
            }
            for index in range(64)
        ]
        with IsolatedWorkspace() as temp:
            data_file = temp / "quadratic.json"
            data_file.write_text(
                json.dumps(records, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            model_args = {
                "ma": ["--window", "4"],
                "es": ["--alpha", "0.05"],
                "arima": ["--d", "2"],
            }
            for model_name, extra_args in model_args.items():
                with self.subTest(model=model_name):
                    artifact = temp / f"{model_name}.pkl"
                    argv = [
                        sys.executable,
                        PY_ROOT / "run_model.py",
                        model_name,
                        "--data-file",
                        data_file,
                        "--train-start",
                        "8",
                        "--train-end",
                        "43",
                        "--eval-start",
                        "48",
                        "--eval-end",
                        "52",
                        "--save-model",
                        artifact,
                        *extra_args,
                    ]
                    trained = run_command(argv, timeout=300)
                    assert_prediction_contract(trained.payload, 5)
                    self.assertEqual(
                        [48, 49, 50, 51, 52],
                        trained.payload["metrics"]["evaluate_indices"],
                    )

                    state = load_model(str(artifact))
                    self.assertEqual(model_name, state["model_type"])
                    if model_name == "ma":
                        all_expected = recursive_ma(
                            [record["y"] for record in records[40:44]],
                            4,
                            9,
                        )
                        np.testing.assert_allclose(
                            prediction_values(trained.payload),
                            all_expected[4:9],
                            rtol=0,
                            atol=1e-10,
                        )
                    elif model_name == "es":
                        predictions = prediction_values(trained.payload)
                        np.testing.assert_allclose(
                            predictions,
                            [predictions[0]] * len(predictions),
                            rtol=0,
                            atol=1e-10,
                        )
                    else:
                        self.assertEqual(2, state["order"]["d"])
                        self.assertLess(trained.payload["metrics"]["rmse"], 1e-2)

                    one_step = run_loaded(model_name, artifact, steps=1)
                    twelve_steps = run_loaded(model_name, artifact, steps=12)
                    assert_prediction_contract(one_step.payload, 1)
                    assert_prediction_contract(twelve_steps.payload, 12)
                    self.assertAlmostEqual(
                        prediction_values(one_step.payload)[0],
                        prediction_values(twelve_steps.payload)[0],
                        places=8,
                    )


if __name__ == "__main__":
    unittest.main()
