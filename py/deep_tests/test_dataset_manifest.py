from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from py.deep_tests.support.generate_datasets import DATASET_DIR, DEFAULT_SEED, check, generate_all
from py.deep_tests.support.profiles import profile
from py.src.data_utils import (
    ParsedData,
    _canonical_provenance_scalar,
    build_training_provenance,
    build_numeric_subprocess_records,
    load_data_from_json,
)
from py.src.subprocess_utils import (
    run_base_model_fit_subprocess,
    run_base_model_subprocess,
)


class DatasetManifestTests(unittest.TestCase):
    @profile("fast")
    def test_training_provenance_is_typed_range_bound_and_feature_scoped(self):
        records = [
            {
                "month": f"2024-{index + 1:02d}",
                "y": str(100 + index),
                "feature": -0.0 if index == 0 else index % 2,
                "ignored": f"metadata-{index}",
            }
            for index in range(8)
        ]
        parsed = load_data_from_json(json.dumps(records))
        train_range = range(0, 5)
        target_only = build_training_provenance(parsed, train_range, feature_keys=[])
        feature_bound = build_training_provenance(
            parsed,
            train_range,
            feature_keys=["feature"],
        )

        reordered = [
            {
                "ignored": row["ignored"],
                "feature": 0.0 if row["feature"] == 0 else row["feature"],
                "y": int(row["y"]),
                "month": row["month"],
            }
            for row in records
        ]
        reordered_parsed = load_data_from_json(json.dumps(reordered))
        self.assertEqual(
            target_only,
            build_training_provenance(reordered_parsed, train_range, feature_keys=[]),
        )
        self.assertEqual(
            feature_bound,
            build_training_provenance(
                reordered_parsed,
                train_range,
                feature_keys=["feature"],
            ),
        )

        future_changed = json.loads(json.dumps(records))
        future_changed[7]["y"] = "9999"
        future_changed[7]["feature"] = 9999.5
        future_parsed = load_data_from_json(json.dumps(future_changed))
        self.assertEqual(
            feature_bound,
            build_training_provenance(future_parsed, train_range, feature_keys=["feature"]),
        )

        prefix_changed = json.loads(json.dumps(records))
        prefix_changed[2]["feature"] = 99
        changed_parsed = load_data_from_json(json.dumps(prefix_changed))
        self.assertEqual(
            target_only,
            build_training_provenance(changed_parsed, train_range, feature_keys=[]),
        )
        self.assertNotEqual(
            feature_bound["sha256"],
            build_training_provenance(
                changed_parsed,
                train_range,
                feature_keys=["feature"],
            )["sha256"],
        )
        self.assertNotEqual(
            target_only["sha256"],
            build_training_provenance(parsed, range(1, 5), feature_keys=[])["sha256"],
        )

    @profile("fast")
    def test_training_provenance_preserves_roles_feature_order_and_exact_types(self):
        records = [
            {
                "month": f"2025-{index + 1:02d}",
                "y": 10 + index,
                "first": index,
                "second": index + 0.5,
            }
            for index in range(4)
        ]
        parsed = load_data_from_json(json.dumps(records))
        forward = build_training_provenance(
            parsed,
            range(0, 4),
            feature_keys=["first", "second"],
        )
        reverse = build_training_provenance(
            parsed,
            range(0, 4),
            feature_keys=["second", "first"],
        )
        self.assertEqual(["first", "second"], forward["feature_keys"])
        self.assertEqual(["y", "month", "first", "second"], forward["columns"])
        self.assertNotEqual(forward["sha256"], reverse["sha256"])

        timestamp_changed = json.loads(json.dumps(records))
        for index, row in enumerate(timestamp_changed):
            row["month"] = f"2026-{index + 1:02d}"
        shifted = load_data_from_json(json.dumps(timestamp_changed))
        self.assertNotEqual(
            build_training_provenance(parsed, range(0, 4), feature_keys=[])["sha256"],
            build_training_provenance(shifted, range(0, 4), feature_keys=[])["sha256"],
        )

        for invalid_features in (["first", "first"], ["y"], ["month"]):
            with self.subTest(invalid_features=invalid_features), self.assertRaisesRegex(
                ValueError,
                "duplicate|collide",
            ):
                build_training_provenance(
                    parsed,
                    range(0, 4),
                    feature_keys=invalid_features,
                )

        integer_future_records = [
            {"month": f"period-{index}", "y": 10 + index, "feature": index}
            for index in range(8)
        ]
        fractional_future_records = json.loads(json.dumps(integer_future_records))
        fractional_future_records[7]["y"] = 17.5
        fractional_future_records[7]["feature"] = 7.5
        self.assertEqual(
            build_training_provenance(
                load_data_from_json(json.dumps(integer_future_records)),
                range(0, 6),
                feature_keys=["feature"],
            ),
            build_training_provenance(
                load_data_from_json(json.dumps(fractional_future_records)),
                range(0, 6),
                feature_keys=["feature"],
            ),
        )

        numeric_int_records = [
            {"month": "period-a", "y": 1, "feature": 2},
            {"month": "period-b", "y": 3, "feature": 4},
        ]
        numeric_float_records = [
            {"month": "period-a", "y": 1.0, "feature": 2.0},
            {"month": "period-b", "y": 3.0, "feature": 4.0},
        ]
        self.assertEqual(
            build_training_provenance(
                load_data_from_json(json.dumps(numeric_int_records)),
                range(0, 2),
                feature_keys=["feature"],
                feature_types={"feature": "numeric"},
            ),
            build_training_provenance(
                load_data_from_json(json.dumps(numeric_float_records)),
                range(0, 2),
                feature_keys=["feature"],
                feature_types={"feature": "numeric"},
            ),
        )

        self.assertEqual(
            {"type": "integer", "value": "9007199254740993"},
            _canonical_provenance_scalar(9007199254740993),
        )
        self.assertNotEqual(
            _canonical_provenance_scalar(9007199254740992),
            _canonical_provenance_scalar(9007199254740993),
        )
        self.assertEqual(
            {"type": "float", "value": "0x0.0p+0"},
            _canonical_provenance_scalar(-0.0),
        )
        encoded_boundaries = [
            _canonical_provenance_scalar(value)
            for value in (True, None, "1", 1, 1.0)
        ]
        self.assertEqual(5, len({json.dumps(value, sort_keys=True) for value in encoded_boundaries}))
        for value in (float("nan"), float("inf"), np.float64("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "non-finite"):
                _canonical_provenance_scalar(value)

        large_integer_frame = pd.DataFrame({
            "y": [1, 2],
            "month": ["period-a", "period-b"],
            "exact": [9007199254740992, 9007199254740993],
        })
        large_integer_data = ParsedData(
            large_integer_frame,
            target_key="y",
            time_key="month",
            feature_keys=["exact"],
        )
        exact = build_training_provenance(
            large_integer_data,
            range(0, 2),
            feature_keys=["exact"],
            feature_types={"exact": "categorical"},
        )
        swapped = large_integer_frame.copy()
        swapped.loc[1, "exact"] = 9007199254740992
        self.assertNotEqual(
            exact["sha256"],
            build_training_provenance(
                ParsedData(swapped, "y", "month", ["exact"]),
                range(0, 2),
                feature_keys=["exact"],
                feature_types={"exact": "categorical"},
            )["sha256"],
        )

    @profile("fast")
    def test_member_ordinary_and_fit_only_artifacts_bind_the_timestamp_prefix(self):
        records = [
            {"month": f"2024-{index + 1:02d}", "y": 20 + index}
            for index in range(12)
        ]
        parsed = load_data_from_json(json.dumps(records))
        parent_provenance = build_training_provenance(
            parsed,
            range(0, 8),
            feature_keys=[],
        )
        member_records, member_features = build_numeric_subprocess_records(parsed)
        self.assertEqual([], member_features)
        self.assertEqual("2024-01", member_records[0]["month"])

        with tempfile.TemporaryDirectory(prefix="fz-provenance-member-") as temp:
            root = Path(temp)
            ordinary_path = root / "ordinary.pkl"
            fit_path = root / "fit.pkl"
            shifted_fit_path = root / "shifted-fit.pkl"
            run_base_model_subprocess(
                "ma",
                member_records,
                0,
                7,
                8,
                9,
                {"window": 2, "save_model": str(ordinary_path)},
            )
            run_base_model_fit_subprocess(
                "ma",
                member_records,
                0,
                7,
                str(ordinary_path),
                str(fit_path),
            )
            shifted_records = [dict(row) for row in member_records]
            for index, row in enumerate(shifted_records):
                row["month"] = f"2025-{index + 1:02d}"
            run_base_model_fit_subprocess(
                "ma",
                shifted_records,
                0,
                7,
                str(ordinary_path),
                str(shifted_fit_path),
            )

            with ordinary_path.open("rb") as handle:
                ordinary = pickle.load(handle)
            with fit_path.open("rb") as handle:
                fitted = pickle.load(handle)
            with shifted_fit_path.open("rb") as handle:
                shifted = pickle.load(handle)
            self.assertEqual("month", ordinary["training_provenance"]["time_key"])
            self.assertEqual(parent_provenance, ordinary["training_provenance"])
            self.assertEqual(
                ordinary["training_provenance"],
                fitted["training_provenance"],
            )
            self.assertNotEqual(
                fitted["training_provenance"]["sha256"],
                shifted["training_provenance"]["sha256"],
            )

    @profile("fast")
    def test_training_provenance_rejects_non_finite_values_in_any_role(self):
        for column in ("y", "month", "feature"):
            frame = pd.DataFrame({
                "y": pd.Series([1.0], dtype=object),
                "month": pd.Series(["period-a"], dtype=object),
                "feature": pd.Series([2.0], dtype=object),
            })
            frame.loc[0, column] = float("inf")
            parsed = ParsedData(frame, "y", "month", ["feature"])
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "non-finite"):
                build_training_provenance(
                    parsed,
                    range(0, 1),
                    feature_keys=["feature"],
                )

    @profile("fast")
    def test_committed_fixtures_reproduce_byte_for_byte(self):
        self.assertEqual([], check())

    @profile("fast")
    def test_manifest_hash_schema_formula_and_splits(self):
        manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(DEFAULT_SEED, manifest["seed"])
        for entry in manifest["fixtures"]:
            with self.subTest(entry["file"]):
                path = DATASET_DIR / entry["file"]
                records = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: self.fail(value))
                self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(entry["rows"], len(records))
                self.assertTrue(entry["formula"])
                self.assertEqual(entry["columns"], sorted({key for row in records for key in row}))
                splits = entry["splits"]
                if "eval" in splits:
                    self.assertLess(splits["train"][1], splits["eval"][0])
                    self.assertLess(splits["eval"][1], len(records))
                    if "gap" in splits:
                        self.assertEqual(splits["train"][1] + 1, splits["gap"][0])
                        self.assertEqual(splits["gap"][1] + 1, splits["eval"][0])

    @profile("fast")
    def test_all_fixture_numbers_are_finite_strict_json(self):
        def visit(value):
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value))
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        for path in DATASET_DIR.glob("*.json"):
            visit(json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: self.fail(value)))

    @profile("fast")
    def test_generator_is_hash_seed_independent(self):
        script = DATASET_DIR.parent / "support" / "generate_datasets.py"
        for seed in ("1", "98765"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            result = subprocess.run([sys.executable, script, "--check"], text=True, capture_output=True, env=env)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    @profile("fast")
    def test_seed_controls_only_the_declared_stochastic_fixture(self):
        first = generate_all(DEFAULT_SEED)
        repeated = generate_all(DEFAULT_SEED)
        changed = generate_all(DEFAULT_SEED + 1)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first["seeded_arx_96.json"], changed["seeded_arx_96.json"])
        for name in first.keys() - {"seeded_arx_96.json"}:
            self.assertEqual(first[name], changed[name], name)

        values = [float(row["y"]) for row in first["seeded_arx_96.json"]]
        mean = sum(values) / len(values)
        lag_covariance = sum((values[i] - mean) * (values[i - 1] - mean) for i in range(1, len(values)))
        self.assertGreater(lag_covariance, 0)


if __name__ == "__main__":
    unittest.main()
