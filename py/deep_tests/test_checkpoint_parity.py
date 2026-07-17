from __future__ import annotations

import math
import unittest
from pathlib import Path

from py.deep_tests.support.checkpoints import run_guided, run_production
from py.deep_tests.support.harness import DATASETS, ENSEMBLES, MODEL_TYPES, IsolatedWorkspace, assert_prediction_contract, common_args, prediction_values, run_command, run_loaded, run_model
from py.deep_tests.support.profiles import profile
from py.src.serialization import load_model


def production_ensemble_oracle(model: str, artifact: Path) -> list[float]:
    """Recombine independently loaded production members without ensemble code."""
    state = load_model(str(artifact))
    steps = int(state["prepared_forecast_steps"])
    if model in {"weighted_avg", "stacking"}:
        columns = {
            name: prediction_values(run_loaded(name, Path(state["base_model_states"][name]["model_file"]), steps).payload)
            for name in state["models"]
        }
        if model == "weighted_avg":
            coefficients = state["weights"]
            intercept = 0.0
        else:
            meta = state["meta_model"]
            coefficients = dict(zip(meta["model_names"], meta["coefficients"]))
            intercept = float(meta.get("intercept", 0.0))
        return [
            max(0.0, intercept + math.fsum(float(coefficients[name]) * columns[name][index] for name in columns))
            for index in range(steps)
        ]

    counters: dict[str, int] = {}
    total = [0.0] * steps
    for stage_index, (name, coefficient) in enumerate(zip(state["model_chain"], state["stage_coefficients"])):
        member_index = counters.get(name, 0)
        counters[name] = member_index + 1
        member = state["base_model_states"][name]["models"][member_index]
        values = prediction_values(run_loaded(name, Path(member["model_file"]), steps, allow_negative=True).payload)
        if stage_index == 0:
            values = [max(0.0, value) for value in values]
        for index, value in enumerate(values):
            total[index] += float(coefficient) * value
    return [max(0.0, value) for value in total]


class CheckpointModeParityTests(unittest.TestCase):
    @profile("nightly")
    def test_all_seven_standalone_guided_and_production_contracts(self):
        source = DATASETS / "seasonal_pulse_60.json"
        with IsolatedWorkspace() as temp:
            for model in MODEL_TYPES:
                with self.subTest(model=model):
                    standalone_artifact = temp / f"standalone-{model}.pkl"
                    guided_artifact = temp / f"guided-{model}.pkl"
                    production_artifact = temp / f"production-{model}.pkl"
                    standalone = run_model(model, source, standalone_artifact).payload
                    guided_dir = temp / f"guided-checkpoints-{model}"; guided_dir.mkdir()
                    guided = run_guided(model, source, guided_artifact, guided_dir)
                    assert_prediction_contract(standalone, 4)
                    assert_prediction_contract(guided, 4)
                    tolerance = 1e-5 if model == "lstm" else 1e-7
                    for left, right in zip(prediction_values(standalone), prediction_values(guided)):
                        self.assertTrue(math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance), (model, left, right))

                    production_dir = temp / f"production-checkpoints-{model}"; production_dir.mkdir()
                    production = run_production(model, source, standalone_artifact, production_artifact, production_dir)
                    self.assertEqual(4, production["prepared_points"])
                    self.assertEqual(production_artifact.resolve(), Path(production["saved_model"]).resolve())
                    loaded = run_loaded(model, production_artifact)
                    assert_prediction_contract(loaded.payload, 4)

                    if model in ENSEMBLES:
                        # Production ensembles intentionally preserve their backtest
                        # weights/meta-model while refitting members on the longer range.
                        # Recombine those independently loaded members with a local oracle.
                        expected = production_ensemble_oracle(model, production_artifact)
                        standalone_state = load_model(str(standalone_artifact))
                        production_state = load_model(str(production_artifact))
                        if model == "boosting":
                            self.assertEqual(
                                standalone_state["stage_coefficients"],
                                production_state["stage_coefficients"],
                            )
                            self.assertEqual(
                                standalone_state["uncertainty"]["calibration_residuals"],
                                production_state["uncertainty"]["calibration_residuals"],
                            )
                            self.assertEqual(
                                "internal_validation",
                                production_state["uncertainty"]["calibration_source"],
                            )
                    else:
                        # Base-model production is a longer-range refit, so compare it
                        # with an independent monolithic fit over that same range.
                        independent_args = common_args(model, source)
                        independent_args[independent_args.index("--train-end") + 1] = "47"
                        independent_args[independent_args.index("--eval-start") + 1] = "48"
                        independent_args[independent_args.index("--eval-end") + 1] = "51"
                        independent = run_command(
                            independent_args,
                            timeout=600 if model == "lstm" else 300,
                        ).payload
                        assert_prediction_contract(independent, 4)
                        expected = prediction_values(independent)
                    # Every fresh LSTM fit resets TensorFlow's seed at the fit boundary,
                    # so loading the source sidecar cannot perturb production refits.
                    production_tolerance = tolerance
                    for left, right in zip(prediction_values(loaded.payload), expected):
                        self.assertTrue(math.isclose(left, right, rel_tol=production_tolerance, abs_tol=production_tolerance), (model, left, right))


if __name__ == "__main__":
    unittest.main()
