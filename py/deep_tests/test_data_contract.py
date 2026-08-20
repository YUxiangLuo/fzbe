from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from py.deep_tests.support.harness import DATASETS, PY_ROOT, assert_prediction_contract, parse_strict_json, run_command
from py.deep_tests.support.profiles import profile
from py.src.data_utils import (
    attach_training_provenance,
    build_numeric_subprocess_records,
    build_subprocess_records,
    build_training_provenance,
    infer_feature_types,
    load_data_from_json,
    validate_training_provenance,
)


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
        numeric, numeric_features = build_numeric_subprocess_records(parsed)
        rich, rich_features = build_subprocess_records(parsed)
        self.assertNotIn("region", numeric[0])
        self.assertEqual(records[0]["region"], rich[0]["region"])
        self.assertEqual(records[0]["month"], numeric[0]["month"])
        self.assertEqual(records[0]["month"], rich[0]["month"])
        self.assertNotIn("month", numeric_features)
        self.assertNotIn("month", rich_features)
        self.assertEqual("month", load_data_from_json(json.dumps(numeric)).time_key)
        self.assertEqual("month", load_data_from_json(json.dumps(rich)).time_key)
        self.assertEqual(1000.0, parsed.y[0])

    @profile("fast")
    def test_lstm_member_records_preserve_consumed_provenance_types(self):
        records = [
            {
                "month": f"2024-{index + 1:02d}",
                "y": 100 + index,
                "numeric": index,
                "category_code": 9007199254740992 + index,
            }
            for index in range(8)
        ]
        feature_types = {
            "numeric": "numeric",
            "category_code": "categorical",
        }
        parent = load_data_from_json(json.dumps(records))
        member_records, member_features = build_subprocess_records(
            parent,
            feature_keys=["numeric", "category_code"],
            feature_types=feature_types,
        )
        member = load_data_from_json(
            json.dumps(member_records),
            feature_keys=member_features,
        )
        self.assertEqual(records[0]["category_code"], member_records[0]["category_code"])
        parent_provenance = build_training_provenance(
            parent,
            range(0, 6),
            feature_keys=member_features,
            feature_types=feature_types,
        )
        member_provenance = build_training_provenance(
            member,
            range(0, 6),
            feature_keys=member_features,
            feature_types=feature_types,
        )
        self.assertEqual(parent_provenance, member_provenance)
        member_state: dict = {}
        attach_training_provenance(
            member_state,
            member,
            range(0, 6),
            feature_keys=member_features,
            feature_types=feature_types,
        )
        self.assertEqual(
            parent_provenance,
            validate_training_provenance(
                member_state,
                parent,
                range(0, 6),
                feature_keys=member_features,
                feature_types=feature_types,
            ),
        )

    @profile("fast")
    def test_monthly_calendar_cadence_rejects_gaps_and_accepts_year_rollover(self):
        rollover = [
            {"month": "2024-12", "y": 1},
            {"month": "2025-01", "y": 2},
        ]
        self.assertEqual(
            ["2024-12", "2025-01"],
            load_data_from_json(json.dumps(rollover)).timestamps,
        )

        invalid_pairs = (
            ("2024-01", "2024-03"),
            ("2024-02", "2024-02"),
            ("2024-03", "2024-02"),
        )
        for first, second in invalid_pairs:
            with self.subTest(first=first, second=second), self.assertRaisesRegex(
                ValueError,
                "Monthly calendar timestamps must be contiguous",
            ):
                load_data_from_json(json.dumps([
                    {"month": first, "y": 1},
                    {"month": second, "y": 2},
                ]))

    @profile("fast")
    def test_invalid_calendar_shaped_values_fail_without_blocking_generic_time_keys(self):
        invalid_calendar_cases = (
            [{"month": "2024-13", "y": 1}, {"month": "2024-14", "y": 2}],
            [{"month": "2024-12", "y": 1}, {"month": "2025-13", "y": 2}],
            [{"month": "2024-13", "y": 1}],
        )
        for records in invalid_calendar_cases:
            with self.subTest(records=records), self.assertRaisesRegex(
                ValueError,
                "contains invalid values",
            ):
                load_data_from_json(json.dumps(records))

        invalid_separate_parts = (
            [{"年份": 2024, "月份": 13, "销售数量": 1}],
            [{"年份": "bad", "月份": 1, "销售数量": 1}],
            [{"年份": 2024, "月份": 1.5, "销售数量": 1}],
        )
        for records in invalid_separate_parts:
            with self.subTest(records=records), self.assertRaisesRegex(
                ValueError,
                "Separate year/month columns contain invalid calendar values",
            ):
                load_data_from_json(json.dumps(records, ensure_ascii=False))

        generic = [
            {"month": "period-13", "y": 1},
            {"month": "period-14", "y": 2},
        ]
        self.assertEqual(
            ["period-13", "period-14"],
            load_data_from_json(json.dumps(generic)).timestamps,
        )

    @profile("fast")
    def test_feature_type_inference_uses_only_declared_training_rows(self):
        rows = []
        for index in range(30):
            rows.append({
                "month": f"{2024 + index // 12:04d}-{index % 12 + 1:02d}",
                "y": 100 + index,
                "mystery": index % 2 if index < 20 else index,
            })

        training_view = load_data_from_json(json.dumps(rows))
        full_view = load_data_from_json(json.dumps(rows))
        training_types = infer_feature_types(
            training_view,
            feature_keys=["mystery"],
            inference_range=range(0, 20),
        )
        full_types = infer_feature_types(full_view, feature_keys=["mystery"])

        self.assertEqual("categorical", training_types["mystery"])
        self.assertEqual("numeric", full_types["mystery"])

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
