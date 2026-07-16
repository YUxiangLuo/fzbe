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
    def test_time_axis_rejects_duplicates_reordering_and_missing_periods(self):
        invalid_months = {
            "duplicate": ["2024-01", "2024-02", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07"],
            "reordered": ["2024-01", "2024-03", "2024-02", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08"],
            "missing": ["2024-01", "2024-02", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09"],
        }
        for name, months in invalid_months.items():
            with self.subTest(name=name):
                records = [{"month": month, "y": 10 + index} for index, month in enumerate(months)]
                result = self._run_ma_text(json.dumps(records), False)
                self.assertIn("[DATA]", result.stderr)
                self.assertIn("Time column 'month'", result.stderr)

    @profile("fast")
    def test_time_axis_accepts_monthly_and_fixed_numeric_sequences(self):
        monthly = [{"month": f"2024-{index + 1:02d}", "y": index} for index in range(8)]
        parsed_monthly = load_data_from_json(json.dumps(monthly))
        self.assertEqual("monthly", parsed_monthly.time_frequency)

        numeric = [{"date": 100 + 5 * index, "y": index} for index in range(8)]
        parsed_numeric = load_data_from_json(json.dumps(numeric))
        self.assertEqual("numeric_step:5", parsed_numeric.time_frequency)

    @profile("fast")
    def test_composite_year_month_is_strict_and_consecutive(self):
        valid = [
            {"年份": 2024 + (index + 10) // 12, "月份": (index + 10) % 12 + 1, "销售数量": index}
            for index in range(8)
        ]
        parsed = load_data_from_json(json.dumps(valid, ensure_ascii=False))
        self.assertEqual("timestamp", parsed.time_key)
        self.assertEqual("monthly", parsed.time_frequency)

        invalid = list(valid)
        invalid[3] = {**invalid[3], "月份": 13}
        with self.assertRaisesRegex(ValueError, r"\[DATA\].*年份.*月份"):
            load_data_from_json(json.dumps(invalid, ensure_ascii=False))

        low_year = [{**row, "年份": 999} for row in valid]
        with self.assertRaisesRegex(ValueError, r"\[DATA\].*年份.*月份"):
            load_data_from_json(json.dumps(low_year, ensure_ascii=False))

        fractional_cases = {
            "year": [{**row, "年份": 2024.01} if index == 0 else row for index, row in enumerate(valid)],
            "month": [{**row, "月份": float(row["月份"]) + 0.00001} if index == 0 else row for index, row in enumerate(valid)],
        }
        for name, records in fractional_cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, r"\[DATA\].*年份.*月份"):
                load_data_from_json(json.dumps(records, ensure_ascii=False))

    @profile("fast")
    def test_combined_month_legacy_rows_take_precedence_over_redundant_year(self):
        records = [
            {
                "年份": str(2024 + (index + 10) // 12),
                "月份": f"{2024 + (index + 10) // 12}-{(index + 10) % 12 + 1:02d}",
                "销售数量": index,
            }
            for index in range(8)
        ]
        parsed = load_data_from_json(json.dumps(records, ensure_ascii=False))
        self.assertEqual("月份", parsed.time_key)
        self.assertEqual("monthly", parsed.time_frequency)
        self.assertEqual(records[0]["月份"], parsed.df.iloc[0]["月份"])

    @profile("fast")
    def test_combined_month_rejects_years_before_1000(self):
        for year in ("0000", "0999"):
            for time_key in ("month", "period"):
                with self.subTest(year=year, time_key=time_key):
                    records = [
                        {time_key: f"{year}-{month:02d}", "y": month}
                        for month in range(1, 3)
                    ]
                    kwargs = {} if time_key == "month" else {"time_key": time_key}
                    with self.assertRaisesRegex(ValueError, r"\[DATA\].*Time column"):
                        load_data_from_json(json.dumps(records), **kwargs)

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
