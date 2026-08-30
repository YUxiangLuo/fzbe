from __future__ import annotations

import unittest

import numpy as np

from py.deep_tests.support.profiles import profile
from py.src.forecast_utils import (
    DEMAND_PREDICTION_DOMAIN,
    SIGNED_PREDICTION_DOMAIN,
    apply_prediction_domain,
    normalize_demand_forecast,
    resolve_prediction_domain,
)
from py.src.models import arima


class OutputDomainTests(unittest.TestCase):
    @profile("fast")
    def test_demand_clamps_negatives_while_signed_preserves_them(self):
        raw = [-5.0, -0.0, 0.0, 3.5, 12.0]
        np.testing.assert_array_equal(apply_prediction_domain(raw, DEMAND_PREDICTION_DOMAIN), [0, 0, 0, 3.5, 12])
        np.testing.assert_array_equal(apply_prediction_domain(raw, SIGNED_PREDICTION_DOMAIN), [-5, 0, 0, 3.5, 12])
        np.testing.assert_array_equal(normalize_demand_forecast(raw), [0, 0, 0, 3.5, 12])

    @profile("fast")
    def test_arima_clamp_widens_only_the_upper_bound_and_preserves_negative_tail(self):
        [point] = arima._format_predictions([-4.0], [[-7.0, -1.0]])
        self.assertEqual(0.0, point["prediction"])
        self.assertEqual(-7.0, point["interval_lower"])
        self.assertEqual(0.0, point["interval_upper"])
        self.assertLess(point["interval_lower"], 0.0)

    @profile("fast")
    def test_arima_signed_formatter_preserves_negative_points_and_intervals(self):
        points = arima._format_predictions(
            [-6.0, 4.0],
            [[-8.0, -4.0], [2.0, 6.0]],
            prediction_domain=SIGNED_PREDICTION_DOMAIN,
        )
        self.assertEqual([-6.0, 4.0], [point["prediction"] for point in points])
        self.assertEqual([-8.0, -4.0], [points[0]["interval_lower"], points[0]["interval_upper"]])

    @profile("fast")
    def test_unknown_domain_is_a_validation_error(self):
        for domain in ("weird", None, ""):
            with self.subTest(domain=domain), self.assertRaisesRegex(ValueError, r"\[VALIDATION\]"):
                apply_prediction_domain([1.0], domain)

    @profile("fast")
    def test_resolve_prefers_explicit_signed_request_over_stored_domain(self):
        self.assertEqual(DEMAND_PREDICTION_DOMAIN, resolve_prediction_domain(None))
        self.assertEqual(DEMAND_PREDICTION_DOMAIN, resolve_prediction_domain({}))
        self.assertEqual(SIGNED_PREDICTION_DOMAIN, resolve_prediction_domain({"prediction_domain": SIGNED_PREDICTION_DOMAIN}))
        # An explicit allow-negative request wins even when the state stored demand,
        # which is exactly how a Boosting residual learner reuses a demand base model.
        self.assertEqual(
            SIGNED_PREDICTION_DOMAIN,
            resolve_prediction_domain({"prediction_domain": DEMAND_PREDICTION_DOMAIN}, allow_negative_predictions=True),
        )
        with self.assertRaisesRegex(ValueError, r"\[VALIDATION\]"):
            resolve_prediction_domain({"prediction_domain": "weird"})

    @profile("fast")
    def test_boosting_stage_semantics_reduce_to_the_shared_domain_rule(self):
        # Stage 0 forecasts physical demand (clamped); later stages learn signed residuals.
        stage0 = apply_prediction_domain([-2.0, 3.0], resolve_prediction_domain(None))
        stage1 = apply_prediction_domain([-2.0, 3.0], resolve_prediction_domain(None, allow_negative_predictions=True))
        np.testing.assert_array_equal(stage0, [0.0, 3.0])
        np.testing.assert_array_equal(stage1, [-2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
