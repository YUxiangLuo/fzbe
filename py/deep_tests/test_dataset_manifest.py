from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

from py.deep_tests.support.generate_datasets import DATASET_DIR, DEFAULT_SEED, canonical_json_bytes, check, generate_all
from py.deep_tests.support.profiles import profile


class DatasetManifestTests(unittest.TestCase):
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
                for child in value.values(): visit(child)
            elif isinstance(value, list):
                for child in value: visit(child)
        for path in DATASET_DIR.glob("*.json"):
            visit(json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: self.fail(value)))

    @profile("fast")
    def test_generator_is_hash_seed_independent(self):
        script = DATASET_DIR.parent / "support" / "generate_datasets.py"
        for seed in ("1", "98765"):
            env = os.environ.copy(); env["PYTHONHASHSEED"] = seed
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
