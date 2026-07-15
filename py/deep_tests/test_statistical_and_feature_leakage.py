from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

from py.deep_tests.support.harness import (
    DATASETS,
    PY_ROOT,
    IsolatedWorkspace,
    assert_prediction_contract,
    mutate_records,
    prediction_values,
    run_command,
    run_model,
)
from py.deep_tests.support.profiles import profile


def feature_model_args(model: str, source: Path) -> list[str | Path]:
    args: list[str | Path] = [
        sys.executable,
        PY_ROOT / "run_model.py",
        model,
        "--data-file", source,
        "--train-start", "0",
        "--train-end", "63",
        "--eval-start", "68",
        "--eval-end", "75",
    ]
    lstm_args: list[str] = [
        "--lstm-target-feature", "y",
        "--lstm-features", '["driver","promo","region"]',
        "--feature-types", '{"y":"numeric","driver":"numeric","promo":"numeric","region":"categorical"}',
        "--normalization", "minmax",
    ]
    if model == "lstm":
        return args + lstm_args
    return args + ["--models", "ma,lstm", "--window", "3"] + lstm_args


class StatisticalAndFeatureLeakageTests(unittest.TestCase):
    @profile("nightly")
    def test_seeded_arx_arima_retraining_is_stable(self):
        source = DATASETS / "seeded_arx_96.json"
        first = run_model("arima", source).payload
        second = run_model("arima", source).payload
        assert_prediction_contract(first, 4)
        assert_prediction_contract(second, 4)
        for left, right in zip(prediction_values(first), prediction_values(second)):
            self.assertTrue(math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-8), (left, right))

    @profile("nightly")
    def test_future_exogenous_poison_cannot_change_lstm_family_training(self):
        with IsolatedWorkspace() as temp:
            source = DATASETS / "seeded_arx_96.json"
            poisoned = temp / "future-feature-poison.json"
            mutate_records(source, poisoned, range(64, 76), features=True)
            for model in ("lstm", "weighted_avg", "stacking", "boosting"):
                with self.subTest(model=model):
                    baseline = run_command(feature_model_args(model, source), timeout=1800).payload
                    changed = run_command(feature_model_args(model, poisoned), timeout=1800).payload
                    assert_prediction_contract(baseline, 8)
                    assert_prediction_contract(changed, 8)
                    tolerance = 1e-5 if model == "lstm" else 1e-4
                    for left, right in zip(prediction_values(baseline), prediction_values(changed)):
                        self.assertTrue(math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance), (model, left, right))


if __name__ == "__main__":
    unittest.main()
