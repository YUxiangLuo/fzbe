from __future__ import annotations

import json
import math
import sys
import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd

from py.deep_tests.support.harness import (
    DATASETS,
    PY_ROOT,
    IsolatedWorkspace,
    assert_prediction_contract,
    prediction_values,
    run_command,
    run_loaded,
)
from py.deep_tests.support.profiles import profile
from py.src.data_utils import ParsedData, load_data_from_json
from py.src.dynamic_config import derive_lstm_config
from py.src.models import lstm
from py.src.serialization import load_model


@dataclass
class _FixedDirectModel:
    scaled_predictions: list[float]
    last_input_shape: tuple[int, ...] | None = None

    def predict(self, values, verbose=0):
        del verbose
        self.last_input_shape = tuple(np.asarray(values).shape)
        return np.asarray([self.scaled_predictions], dtype=float)


def _mixed_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(30):
        records.append(
            {
                "month": f"t{index:02d}",
                "y": 50.0 + 1.5 * index,
                "price": f"{1000 + 3 * index:,}",
                "region": "north" if index % 2 == 0 else "south",
                "signed,orders": float((index % 5) - 2),
            }
        )
    records[25]["region"] = "future-only-region"
    return records


class LstmFeatureAndPreprocessingTests(unittest.TestCase):
    @profile("fast")
    def test_feature_parser_preserves_json_names_with_commas_and_requires_explicit_empty(
        self,
    ):
        self.assertEqual(
            ["signed,orders", "region"],
            lstm._parse_feature_key_list('["signed,orders", "region"]'),
        )
        self.assertEqual(
            ["price", "region"],
            lstm._parse_feature_key_list("price, region"),
        )
        self.assertEqual([], lstm._parse_feature_key_list("[]"))
        for missing in (None, "", "   "):
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(
                    ValueError,
                    r"\[VALIDATION\].*explicit feature list",
                ):
                    lstm._parse_feature_key_list(missing)

    @profile("fast")
    def test_minmax_and_zscore_preserve_feature_order_and_ignore_unseen_category(self):
        records = _mixed_records()
        parsed = load_data_from_json(
            json.dumps(records, ensure_ascii=False),
            target_key="y",
            feature_keys=["signed,orders", "price", "region"],
        )
        feature_types = {
            "y": "numeric",
            "signed,orders": "numeric",
            "price": "numeric",
            "region": "categorical",
        }
        train_range = range(4, 24)

        for normalization in ("minmax", "zscore"):
            with self.subTest(normalization=normalization):
                preprocessor, transformed = lstm._build_preprocessor(
                    parsed,
                    feature_types.copy(),
                    train_range,
                    normalization,
                )
                self.assertTrue(np.isfinite(transformed).all())
                self.assertEqual(len(records), transformed.shape[0])
                self.assertEqual(
                    ["y", "signed,orders", "price"],
                    preprocessor.transformers_[0][2],
                )
                self.assertEqual(["region"], preprocessor.transformers_[1][2])
                future_row = preprocessor.transform(parsed.df.iloc[[25]])
                if hasattr(future_row, "toarray"):
                    future_row = future_row.toarray()
                self.assertEqual((1, transformed.shape[1]), future_row.shape)
                self.assertTrue(np.isfinite(future_row).all())

    @profile("fast")
    def test_direct_forecast_handles_both_normalizers_multiple_steps_and_horizon_boundary(
        self,
    ):
        records = _mixed_records()
        parsed = load_data_from_json(
            json.dumps(records, ensure_ascii=False),
            target_key="y",
            feature_keys=["signed,orders", "price", "region"],
        )
        feature_types = {
            "y": "numeric",
            "signed,orders": "numeric",
            "price": "numeric",
            "region": "categorical",
        }
        train_range = range(3, 24)
        desired_targets = np.asarray([-5.0, 75.0, 160.0], dtype=float)

        for normalization in ("minmax", "zscore"):
            with self.subTest(normalization=normalization):
                preprocessor, transformed = lstm._build_preprocessor(
                    parsed,
                    feature_types.copy(),
                    train_range,
                    normalization,
                )
                target_scaler = preprocessor.named_transformers_["numeric"].named_steps[
                    "scaler"
                ]
                scaler_inputs = pd.DataFrame(
                    np.zeros(
                        (len(desired_targets), target_scaler.n_features_in_),
                        dtype=float,
                    ),
                    columns=["y", "signed,orders", "price"],
                )
                scaler_inputs["y"] = desired_targets
                scaled_targets = target_scaler.transform(scaler_inputs)[:, 0]
                fake_model = _FixedDirectModel(scaled_targets.tolist())
                state = {
                    "keras_model": fake_model,
                    "preprocessor": preprocessor,
                    "look_back": 5,
                    "history_df": parsed.df.iloc[train_range],
                    "forecast_horizon": 3,
                    "uncertainty_profile": [0.5, 1.0, 1.5],
                    "uncertainty_summary": {
                        "source": "normal_approximation",
                        "fallback": True,
                    },
                    "reference_series": parsed.df.iloc[train_range]["y"].tolist(),
                }

                observed, uncertainty = lstm._forecast_direct_from_state(state, 3)
                np.testing.assert_allclose(
                    observed,
                    desired_targets,
                    rtol=0,
                    atol=1e-10,
                )
                self.assertEqual(3, len(uncertainty))
                self.assertTrue(np.isfinite(uncertainty).all())
                self.assertTrue(all(value >= 0 for value in uncertainty))
                self.assertEqual(
                    (1, 5, transformed.shape[1]),
                    fake_model.last_input_shape,
                )

                prefix, prefix_uncertainty = lstm._forecast_direct_from_state(
                    state,
                    1,
                )
                np.testing.assert_allclose(prefix, desired_targets[:1], atol=1e-10)
                self.assertEqual(uncertainty[:1], prefix_uncertainty)
                with self.assertRaisesRegex(
                    ValueError,
                    r"\[VALIDATION\].*exceeds the saved LSTM horizon",
                ):
                    lstm._forecast_direct_from_state(state, 4)

    @profile("fast")
    def test_numeric_validation_reports_target_and_feature_failures_before_keras(self):
        cases = (
            (
                "target-null",
                pd.DataFrame({"y": [1.0, None, 3.0], "x": [1.0, 2.0, 3.0]}),
                {"y": "numeric", "x": "numeric"},
                "y",
            ),
            (
                "feature-string",
                pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, "bad", 3.0]}),
                {"y": "numeric", "x": "numeric"},
                "x",
            ),
            (
                "feature-infinity",
                pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, math.inf, 3.0]}),
                {"y": "numeric", "x": "numeric"},
                "x",
            ),
        )
        for name, frame, feature_types, expected_column in cases:
            with self.subTest(name=name):
                parsed = ParsedData(
                    frame,
                    target_key="y",
                    feature_keys=["x"],
                )
                with self.assertRaisesRegex(
                    ValueError,
                    rf"\[DATA\].*{expected_column}",
                ):
                    lstm._validate_lstm_numeric_columns(parsed, feature_types)

        parsed = ParsedData(
            pd.DataFrame({"y": [1.0, 2.0], "region": ["a", "b"]}),
            target_key="y",
            feature_keys=["region"],
        )
        with self.assertRaisesRegex(
            ValueError,
            r"\[VALIDATION\].*target feature must be numeric",
        ):
            lstm._validate_lstm_feature_types(
                parsed,
                {"y": "categorical", "region": "categorical"},
            )
        with self.assertRaisesRegex(
            ValueError,
            r"\[VALIDATION\].*numeric.*categorical",
        ):
            lstm._validate_lstm_feature_types(
                parsed,
                {"y": "numeric", "region": "ordinal"},
            )


class LstmConfigurationBoundaryMatrixTests(unittest.TestCase):
    @profile("fast")
    def test_short_histories_large_horizons_and_feature_budgets_are_rejected(self):
        invalid_cases = (
            (6, 3, 1, "too short"),
            (12, 10, 1, "too short"),
            (12, 2, 100, "too complex"),
            (30, 0, 1, "at least 1"),
            (30, 2, 0, "at least the target"),
        )
        for train_size, horizon, features, message in invalid_cases:
            with self.subTest(
                train_size=train_size,
                horizon=horizon,
                features=features,
            ):
                with self.assertRaisesRegex(ValueError, rf"\[VALIDATION\].*{message}"):
                    derive_lstm_config(train_size, horizon, features)

        configs = [
            derive_lstm_config(train_size, horizon, feature_count)
            for train_size, horizon, feature_count in (
                (12, 1, 1),
                (24, 4, 3),
                (48, 8, 7),
                (96, 12, 12),
            )
        ]
        for config in configs:
            self.assertGreaterEqual(config["look_back"], 2)
            self.assertGreaterEqual(config["windows"], 4)
            self.assertLessEqual(
                config["parameter_count"],
                config["parameter_budget"],
            )


class LstmRealTrainingVariantTests(unittest.TestCase):
    @profile("full")
    def test_zscore_mixed_features_unseen_category_comma_name_and_saved_horizon(self):
        records = json.loads(
            (DATASETS / "mixed_features_48.json").read_text(encoding="utf-8")
        )
        for index, record in enumerate(records):
            record["signed,orders"] = float((index % 7) - 3)
            if index >= 36:
                record["region"] = "仅评估期出现"

        features = ["signed,orders", "price", "visits", "region"]
        feature_types = {
            "y": "numeric",
            "signed,orders": "numeric",
            "price": "numeric",
            "visits": "numeric",
            "region": "categorical",
        }
        with IsolatedWorkspace() as temp:
            data_file = temp / "lstm-zscore-mixed.json"
            artifact = temp / "lstm-zscore-mixed.pkl"
            data_file.write_text(
                json.dumps(records, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            trained = run_command(
                [
                    sys.executable,
                    PY_ROOT / "run_model.py",
                    "lstm",
                    "--data-file",
                    data_file,
                    "--train-start",
                    "4",
                    "--train-end",
                    "35",
                    "--eval-start",
                    "40",
                    "--eval-end",
                    "43",
                    "--lstm-target-feature",
                    "y",
                    "--lstm-features-json",
                    json.dumps(features, ensure_ascii=False),
                    "--feature-types",
                    json.dumps(feature_types, ensure_ascii=False),
                    "--normalization",
                    "zscore",
                    "--save-model",
                    artifact,
                ],
                timeout=900,
            )
            assert_prediction_contract(trained.payload, 4)
            self.assertEqual("zscore", trained.payload["normalization"])
            self.assertEqual(8, trained.payload["forecast_horizon"])
            self.assertEqual(features, trained.payload["feature_keys"])
            self.assertEqual(
                features, trained.payload["training_metadata"]["feature_keys"]
            )
            self.assertEqual(
                [40, 41, 42, 43], trained.payload["metrics"]["evaluate_indices"]
            )

            state = load_model(str(artifact))
            self.assertEqual("zscore", state["normalization"])
            self.assertEqual(features, state["feature_keys"])
            self.assertEqual(feature_types, state["feature_types"])
            self.assertEqual(8, state["forecast_horizon"])
            self.assertNotIn(
                "仅评估期出现",
                state["preprocessor"]
                .named_transformers_["categorical"]
                .named_steps["onehot"]
                .categories_[0]
                .tolist(),
            )

            one_step = run_loaded("lstm", artifact, steps=1)
            full_horizon = run_loaded("lstm", artifact, steps=8)
            assert_prediction_contract(one_step.payload, 1)
            assert_prediction_contract(full_horizon.payload, 8)
            self.assertAlmostEqual(
                prediction_values(one_step.payload)[0],
                prediction_values(full_horizon.payload)[0],
                places=5,
            )

            too_far = run_loaded(
                "lstm",
                artifact,
                steps=9,
                expected_ok=False,
            )
            self.assertIn("[VALIDATION]", too_far.stderr)
            self.assertIn("exceeds the saved LSTM horizon", too_far.stderr)


    # `_create_direct_dataset` is already pinned to put X strictly before Y for
    # one window size. What was never checked is the split that follows it:
    # `train_and_evaluate` fits on the first `selection_train_windows` and
    # validates on the last `validation_windows`, refitting the scaler on a
    # prefix. If those two window sets touched, early stopping would select an
    # epoch count using data the model had already fitted — the teaching copy
    # claims the network "只读取预测起点前的历史特征", so this is the invariant
    # behind that claim.
    @profile("fast")
    def test_standard_regime_never_fits_on_a_validation_target_window(self):
        checked = 0
        for train_size in range(8, 121):
            for horizon in (2, 3, 6, 12):
                for features in (1, 3, 8):
                    try:
                        config = derive_lstm_config(train_size, horizon, features)
                    except (TypeError, ValueError):
                        continue
                    if config.get("regime") != "standard":
                        continue

                    look_back = int(config["look_back"])
                    validation_windows = int(config["validation_windows"])
                    selection_windows = int(config["selection_train_windows"])
                    total_windows = train_size - look_back - horizon + 1
                    label = f"n={train_size} h={horizon} f={features}"

                    with self.subTest(label):
                        self.assertGreater(total_windows, 0)
                        self.assertGreater(selection_windows, 0)
                        self.assertGreater(validation_windows, 0)

                        validation_start = total_windows - validation_windows
                        self.assertGreaterEqual(validation_start, 0)

                        # Window i covers [i, i + look_back + horizon) on the raw
                        # timeline; its target starts at i + look_back.
                        last_fitted_index = (selection_windows - 1) + look_back + horizon
                        first_validation_target = validation_start + look_back
                        self.assertLessEqual(last_fitted_index, first_validation_target)

                        # The validation preprocessor is refit on this prefix; it
                        # must stop before the first validation target as well.
                        selection_fit_points = selection_windows + look_back + horizon - 1
                        self.assertLessEqual(selection_fit_points, first_validation_target)

                    checked += 1

        self.assertGreater(checked, 1_000, "sweep did not exercise a meaningful range")


if __name__ == "__main__":
    unittest.main()
