"""每个融合模型把成员预测组合成最终预测的那一步，与独立参考实现逐值比对。

已有的 golden 测试覆盖的是**组件**——MA 递推、误差指标、逆 MSE 权重、非负线搜索。
组合本身没有被独立复算过，而这正是本次改动最大的地方：Boosting 引入了第 0 阶段
基线并把后续阶段改成有符号残差，Stacking 换成折外预测矩阵上的 NNLS。

三者的截断语义**故意不同**，这是最容易在重构中被抹平的一点：

  加权 / Stacking   逐成员截断 → 组合 → 再截断
  Boosting          基线 + Σ γ×阶段（阶段是有符号残差，不截断）→ 只截断总和

成员预测里特意放了负值，用来把这个差异钉住。
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from py.deep_tests.support.oracles import (
    baseline_fitted,
    baseline_forecast,
    boosting_compose,
    stacking_blend,
    weighted_blend,
)
from py.deep_tests.support.profiles import profile
from py.src.baselines import (
    baseline_fitted_values as production_baseline_fitted,
    baseline_forecast as production_baseline_forecast,
)
from py.src.models import boosting, stacking, weighted_avg


def _member_result(predictions: list[float]) -> dict:
    return {
        "predictions": [
            {"prediction": value, "std_dev": 1.0} for value in predictions
        ]
    }


# One member deliberately forecasts below zero so the clamping semantics differ
# visibly between the ensembles.
MEMBER_OUTPUTS = {
    "ma": [120.0, 130.0, 140.0],
    "es": [-20.0, 110.0, 90.0],
}
MEMBER_ORDER = ["ma", "es"]

UNCERTAINTY = {
    "calibration_residuals": [1.0, -1.0, 2.0, -2.0],
    "calibration_source": "post_selection_calibration",
    "reference_series": [100.0, 105.0, 110.0, 115.0],
}


def _member_side_effect(**kwargs):
    return _member_result(MEMBER_OUTPUTS[kwargs["model_type"]])


class BaselineOracleTests(unittest.TestCase):
    @profile("fast")
    def test_stage_zero_baselines_match_an_independent_derivation(self):
        history = [100.0, 90.0, 130.0, 110.0, 95.0, 140.0, 105.0, 120.0]
        cases = (
            ("mean", None),
            ("naive", None),
            ("seasonal_naive", 4),
            ("seasonal_naive", 2),
        )
        for kind, period in cases:
            with self.subTest(kind=kind, period=period):
                np.testing.assert_allclose(
                    production_baseline_fitted(history, kind, period),
                    baseline_fitted(history, kind, period),
                    rtol=0,
                    atol=1e-12,
                )
                # Forecast beyond one season so the seasonal cycle has to wrap.
                np.testing.assert_allclose(
                    production_baseline_forecast(history, 9, kind, period),
                    baseline_forecast(history, 9, kind, period),
                    rtol=0,
                    atol=1e-12,
                )

    # The oracle above is a second implementation of the same algorithm, which
    # catches transcription and off-by-one errors but would not notice if both
    # copies drifted from what the baseline is supposed to *be*. These assert
    # the definitions directly.
    @profile("fast")
    def test_baselines_satisfy_their_definitions_not_just_a_second_implementation(self):
        history = [100.0, 90.0, 130.0, 110.0, 95.0, 140.0, 105.0, 120.0]
        size = len(history)

        mean = sum(history) / size
        fitted_mean = production_baseline_fitted(history, "mean")
        # Definition: the mean baseline predicts the training mean, always.
        for value in fitted_mean:
            self.assertAlmostEqual(mean, float(value), places=12)
        for value in production_baseline_forecast(history, 5, "mean"):
            self.assertAlmostEqual(mean, float(value), places=12)

        fitted_naive = production_baseline_fitted(history, "naive")
        # Definition: today's fitted value is yesterday's observation.
        for index in range(1, size):
            self.assertAlmostEqual(history[index - 1], float(fitted_naive[index]), places=12)
        # Definition: the forecast continues the last observation.
        for value in production_baseline_forecast(history, 5, "naive"):
            self.assertAlmostEqual(history[-1], float(value), places=12)

        for period in (2, 4):
            with self.subTest(period=period):
                fitted_seasonal = production_baseline_fitted(history, "seasonal_naive", period)
                # Definition: once a full season exists, repeat the same
                # position one season earlier.
                for index in range(period, size):
                    self.assertAlmostEqual(
                        history[index - period],
                        float(fitted_seasonal[index]),
                        places=12,
                    )
                # Definition: the forecast replays the final season, in order.
                forecast = production_baseline_forecast(history, 2 * period + 1, "seasonal_naive", period)
                final_season = history[-period:]
                for step, value in enumerate(forecast):
                    self.assertAlmostEqual(final_season[step % period], float(value), places=12)


class EnsembleCompositionOracleTests(unittest.TestCase):
    @profile("fast")
    def test_weighted_average_blend_matches_the_oracle_and_clamps_each_member(self):
        weights = {"ma": 0.4, "es": 0.6}
        state = {
            "models": MEMBER_ORDER,
            "weights": weights,
            "base_model_states": {
                name: {"model_file": f"{name}.pkl"} for name in MEMBER_ORDER
            },
            "uncertainty": UNCERTAINTY,
        }

        with patch.object(
            weighted_avg,
            "run_base_model_predict_subprocess",
            side_effect=_member_side_effect,
        ):
            predictions, _profile, _summary = weighted_avg.predict("", 3, state)

        expected = weighted_blend(MEMBER_OUTPUTS, weights, MEMBER_ORDER)
        np.testing.assert_allclose(predictions, expected, rtol=0, atol=1e-9)
        # ES forecast −20 must be clamped to 0 before weighting, so period 1 is
        # 0.4 × 120 rather than 0.4 × 120 + 0.6 × (−20).
        self.assertAlmostEqual(48.0, predictions[0], places=9)

    @profile("fast")
    def test_stacking_meta_blend_matches_the_oracle_and_clamps_each_member(self):
        coefficients = [0.5, 0.25]
        state = {
            "models": MEMBER_ORDER,
            "meta_model": {
                "kind": "nnls",
                "strategy": "nonnegative_least_squares",
                "model_names": MEMBER_ORDER,
                "coefficients": coefficients,
            },
            "base_model_states": {
                name: {"model_file": f"{name}.pkl"} for name in MEMBER_ORDER
            },
            "uncertainty": UNCERTAINTY,
        }

        with patch.object(
            stacking,
            "run_base_model_predict_subprocess",
            side_effect=_member_side_effect,
        ):
            predictions, _profile, _summary = stacking.predict("", 3, state)

        expected = stacking_blend(
            MEMBER_OUTPUTS,
            dict(zip(MEMBER_ORDER, coefficients)),
            MEMBER_ORDER,
        )
        np.testing.assert_allclose(predictions, expected, rtol=0, atol=1e-9)
        # No intercept, and the clamped ES member contributes nothing here.
        self.assertAlmostEqual(60.0, predictions[0], places=9)

    @profile("fast")
    def test_boosting_composition_keeps_stage_corrections_signed(self):
        history = [100.0, 90.0, 130.0, 110.0]
        stage_coefficients = [0.5, 0.25]
        state = {
            "model_type": "boosting",
            "boosting_variant": "residual",
            "baseline": {
                "kind": "mean",
                "label": "训练均值",
                "training_count": len(history),
                "training_mean": float(np.mean(history)),
                "last_value": history[-1],
                "seasonal_period": None,
                "seasonal_tail": [],
            },
            "model_chain": MEMBER_ORDER,
            "stage_coefficients": stage_coefficients,
            "base_model_states": {
                name: {"models": [{"model_file": f"{name}.pkl", "round": index}]}
                for index, name in enumerate(MEMBER_ORDER)
            },
            "uncertainty": UNCERTAINTY,
        }

        with patch.object(
            boosting,
            "run_base_model_predict_subprocess",
            side_effect=_member_side_effect,
        ):
            predictions, _profile, _summary = boosting.predict("", 3, state)

        expected = boosting_compose(
            baseline_forecast(history, 3, "mean"),
            [MEMBER_OUTPUTS[name] for name in MEMBER_ORDER],
            stage_coefficients,
        )
        np.testing.assert_allclose(predictions, expected, rtol=0, atol=1e-9)
        # The −20 correction must survive with its sign: 107.5 + 0.5×120 +
        # 0.25×(−20) = 162.5. Clamping the stage first would give 167.5.
        self.assertAlmostEqual(162.5, predictions[0], places=9)


if __name__ == "__main__":
    unittest.main()
