from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from py.deep_tests.support.harness import DATASETS, PY_ROOT, assert_prediction_contract, parse_strict_json, run_command
from py.deep_tests.support.profiles import profile
from py.src.data_utils import build_numeric_subprocess_records, build_subprocess_records, load_data_from_json


class DataContractTests(unittest.TestCase):
    def _run_ma_text(self, text: str, expected_ok: bool):
        with tempfile.TemporaryDirectory(prefix="fz-deep-contract-") as temp:
            path = Path(temp) / "input.json"
            path.write_text(text, encoding="utf-8")
            return run_command([sys.executable, PY_ROOT / "run_model.py", "ma", "--data-file", path,
                                "--train-start", "0", "--train-end", "5", "--eval-start", "6", "--eval-end", "7", "--window", "2"],
                               expected_ok=expected_ok)

    @profile("fast")
    def test_numeric_strings_coerce_and_categories_are_path_specific(self):
        records = [{"month": f"2024-{i + 1:02d}", "y": f"{1000 + i:,}", "visits": f"{2000 + 3*i:,}", "region": "北" if i % 2 else "南"} for i in range(8)]
        result = self._run_ma_text(json.dumps(records, ensure_ascii=False), True)
        assert_prediction_contract(result.payload, 2)
        parsed = load_data_from_json(json.dumps(records, ensure_ascii=False))
        numeric, _ = build_numeric_subprocess_records(parsed)
        rich, _ = build_subprocess_records(parsed)
        self.assertNotIn("region", numeric[0])
        self.assertEqual(records[0]["region"], rich[0]["region"])
        self.assertEqual(1000.0, parsed.y[0])

    @profile("fast")
    def test_bad_json_data_errors(self):
        cases = {
            "empty": "[]",
            "object": "{}",
            "missing": '[{"month":"2024-01","x":1}]',
            "non_numeric": '[{"y":"bad"},{"y":2}]',
            "nan": '[{"y":1},{"y":NaN}]',
            "infinity": '[{"y":1},{"y":Infinity}]',
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                result = self._run_ma_text(text, False)
                self.assertIn("[DATA]", result.stderr)
                self.assertEqual("", result.stdout.strip())

    @profile("fast")
    def test_strict_json_rejects_constants_without_rejecting_string_content(self):
        payload = parse_strict_json('{"implementation_notes":"NaN and Infinity are forbidden numeric constants"}')
        self.assertIn("Infinity", payload["implementation_notes"])
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant), self.assertRaises(ValueError):
                parse_strict_json(f'{{"value":{constant}}}')

    @profile("fast")
    def test_invalid_ranges_are_validation_errors(self):
        path = DATASETS / "level_36.json"
        cases = (("0", "20", "20", "23"), ("0", "40", "41", "42"), ("5", "2", "10", "12"))
        for ts, te, es, ee in cases:
            with self.subTest(case=(ts, te, es, ee)):
                result = run_command([sys.executable, PY_ROOT / "run_model.py", "ma", "--data-file", path,
                                      "--train-start", ts, "--train-end", te, "--eval-start", es, "--eval-end", ee, "--window", "3"], expected_ok=False)
                self.assertIn("[VALIDATION]", result.stderr)

    @profile("fast")
    def test_es_alpha_one_flat_cli_golden_and_metrics(self):
        path = DATASETS / "trend_48.json"
        result = run_command([sys.executable, PY_ROOT / "run_model.py", "es", "--data-file", path,
                              "--train-start", "0", "--train-end", "35", "--eval-start", "40", "--eval-end", "43", "--alpha", "1"])
        assert_prediction_contract(result.payload, 4)
        self.assertEqual([150.0] * 4, [point["prediction"] for point in result.payload["predictions"]])
        self.assertEqual([160.0, 162.0, 164.0, 166.0], result.payload["eval_y_true"])


if __name__ == "__main__":
    unittest.main()
