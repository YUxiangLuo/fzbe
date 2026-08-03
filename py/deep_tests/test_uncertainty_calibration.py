from __future__ import annotations

import json
import math
import unittest

from py.deep_tests.support.harness import DATASETS
from py.deep_tests.support.profiles import profile
from py.src import uncertainty as U


def _series(fixture: str) -> list[float]:
    return [float(row["y"]) for row in json.loads((DATASETS / fixture).read_text(encoding="utf-8"))]


class NormalHorizonConstantTests(unittest.TestCase):
    @profile("fast")
    def test_normal_offsets_follow_documented_z_scores(self):
        for scale in (0.0, 1.0, 2.5, 40.0):
            with self.subTest(scale=scale):
                horizon = U._normal_horizon(scale, source=U.UNCERTAINTY_MODEL, calibration_source="training")
                self.assertEqual(scale, horizon["std_dev"])
                self.assertAlmostEqual(-U.NORMAL_95 * scale, horizon["interval_lower_offset"], places=12)
                self.assertAlmostEqual(U.NORMAL_95 * scale, horizon["interval_upper_offset"], places=12)
                self.assertAlmostEqual(U.NORMAL_99 * scale, horizon["upper_error_p99"], places=12)
                self.assertEqual(U.COVERAGE_CLAIM_NONE, horizon["coverage_claim"])
        with self.assertRaisesRegex(ValueError, r"\[MODEL\]"):
            U._normal_horizon(-1.0, source=U.UNCERTAINTY_MODEL, calibration_source="training")


class FallbackProfileTests(unittest.TestCase):
    @profile("fast")
    def test_stable_trend_uses_step_rms_floor_and_sqrt_horizon_growth(self):
        # A pure linear trend has zero centered variance but a real step size; the
        # floor stops the buffer from collapsing to a misleading "zero risk".
        values, metadata = U.fallback_profile(_series("trend_48.json"), 4)
        self.assertEqual("first_difference_rms_floor", metadata["reason"])
        self.assertGreater(values[0], 0.0)
        for index, value in enumerate(values):
            self.assertAlmostEqual(values[0] * math.sqrt(index + 1), value, places=12)

    @profile("fast")
    def test_truly_constant_series_reports_zero_scale(self):
        values, metadata = U.fallback_profile(_series("level_36.json"), 3)
        self.assertEqual("first_difference_scale", metadata["reason"])
        self.assertEqual([0.0, 0.0, 0.0], values)

    @profile("fast")
    def test_insufficient_history_uses_floored_baseline(self):
        # A single observation leaves no residuals; the scale floors at 5% of |mean|.
        values, metadata = U.fallback_profile([250.0], 3)
        self.assertEqual("insufficient_residuals", metadata["reason"])
        self.assertAlmostEqual(12.5, values[0], places=12)
        for index, value in enumerate(values):
            self.assertAlmostEqual(values[0] * math.sqrt(index + 1), value, places=12)


class SesProfileTests(unittest.TestCase):
    @profile("fast")
    def test_growth_follows_ann_variance_law_without_residuals(self):
        scale, alpha, steps = 4.0, 0.6, 5
        values, metadata = U.ses_profile(scale, alpha, steps)
        self.assertEqual("ets_ann_fallback_normal", metadata["method"])
        for horizon in range(1, steps + 1):
            expected = scale * math.sqrt(1.0 + alpha * alpha * (horizon - 1))
            self.assertAlmostEqual(expected, values[horizon - 1], places=12)

    @profile("fast")
    def test_switches_to_empirical_quantiles_with_enough_residuals(self):
        values, metadata = U.ses_profile(3.0, 0.5, 3, calibration_residuals=[1, -1, 2, -2, 0.5, -0.5])
        self.assertEqual("ets_ann_empirical_residual_quantiles", metadata["method"])
        self.assertEqual(3, len(values))
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in values))

    @profile("fast")
    def test_negative_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[MODEL\]"):
            U.ses_profile(-1.0, 0.5, 3)


class FormatPredictionsTests(unittest.TestCase):
    @profile("fast")
    def test_intervals_center_on_the_point_and_match_normal_offsets(self):
        values, metadata = U.fallback_profile(_series("seasonal_pulse_60.json"), 3)
        points = U.format_predictions(
            [100.0, 100.0, 100.0], values, {"horizon_start": 1, "horizons": metadata["horizons"]}
        )
        self.assertEqual(3, len(points))
        for point in points:
            std = point["std_dev"]
            upper_half = point["interval_upper"] - point["prediction"]
            lower_half = point["prediction"] - point["interval_lower"]
            self.assertAlmostEqual(upper_half, lower_half, places=9)
            self.assertAlmostEqual(U.NORMAL_95 * std, upper_half, places=9)
            self.assertAlmostEqual(U.NORMAL_99 * std, point["upper_error_p99"], places=9)
            self.assertEqual(U.INTERVAL_LEVEL, point["interval_level"])
            self.assertEqual(U.COVERAGE_CLAIM_NONE, point["coverage_claim"])

    @profile("fast")
    def test_structural_guards_reject_bad_shapes(self):
        with self.assertRaisesRegex(ValueError, r"\[MODEL\]"):  # length mismatch
            U.format_predictions([1.0, 2.0], [1.0], {})
        with self.assertRaisesRegex(ValueError, r"\[MODEL\]"):  # horizon_start must be >= 1
            U.format_predictions([1.0], [1.0], {"horizon_start": 0})
        with self.assertRaisesRegex(ValueError, r"\[MODEL\]"):  # std_dev must be nonnegative
            U.format_predictions([1.0], [-1.0], {})


class ProfileForStepsTests(unittest.TestCase):
    @profile("fast")
    def test_valid_saved_horizons_are_preserved_then_extended_with_fallback(self):
        reference = _series("seasonal_pulse_60.json")
        saved_profile, saved_metadata = U.ses_profile(2.0, 0.4, 2, calibration_residuals=[1, -1, 2, -2, 0.3, -0.3])
        extended, metadata = U.profile_for_steps(saved_profile, 4, reference, saved_metadata)
        self.assertEqual(4, len(extended))
        for index in range(2):
            self.assertAlmostEqual(saved_profile[index], extended[index], places=12)
        self.assertEqual(4, len(metadata["horizons"]))

    @profile("fast")
    def test_invalid_saved_horizon_falls_back_immediately(self):
        reference = _series("seasonal_pulse_60.json")
        bad_metadata = {"method": "x", "calibration_source": "y", "horizons": [{"missing": "fields"}]}
        extended, _ = U.profile_for_steps([5.0], 3, reference, bad_metadata)
        self.assertEqual(3, len(extended))

    @profile("fast")
    def test_saved_empirical_horizon_normalizes_legacy_bias_provenance(self):
        horizon = U._empirical_horizon(
            [1.0, -1.0, 2.0, -2.0],
            calibration_source="post_selection_calibration",
        )
        horizon.pop("bias_diagnostic_source")

        normalized = U._valid_saved_horizon(horizon)

        self.assertIsNotNone(normalized)
        self.assertEqual(
            "post_selection_calibration",
            normalized["bias_diagnostic_source"],
        )
        self.assertEqual(U.COVERAGE_CLAIM_NONE, normalized["coverage_claim"])

    @profile("fast")
    def test_saved_horizon_rejects_claimed_coverage_or_blank_bias_provenance(self):
        horizon = U._empirical_horizon(
            [1.0, -1.0, 2.0, -2.0],
            calibration_source="post_selection_calibration",
        )
        self.assertIsNone(U._valid_saved_horizon({
            **horizon,
            "coverage_claim": "model_nominal",
        }))
        self.assertIsNone(U._valid_saved_horizon({
            **horizon,
            "bias_diagnostic_source": "",
        }))

    @profile("fast")
    def test_nonpositive_steps_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[MODEL\]"):
            U.profile_for_steps([1.0], 0, [1.0, 2.0, 3.0])


class MovingAverageProfileTests(unittest.TestCase):
    @profile("fast")
    def test_constant_series_has_zero_rolling_origin_spread(self):
        values, metadata = U.moving_average_profile(_series("level_36.json"), 3, 4)
        self.assertEqual("rolling_origin_horizon_residual_quantiles", metadata["method"])
        self.assertEqual(4, len(values))
        for value in values:
            self.assertLess(value, 1e-9)

    @profile("fast")
    def test_variable_series_has_positive_finite_spread(self):
        values, _ = U.moving_average_profile(_series("seasonal_pulse_60.json"), 3, 4)
        self.assertEqual(4, len(values))
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in values))
        self.assertTrue(any(value > 0 for value in values))


if __name__ == "__main__":
    unittest.main()
