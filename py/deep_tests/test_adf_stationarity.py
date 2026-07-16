from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from py.deep_tests.support.harness import DATASETS, PY_ROOT
from py.deep_tests.support.profiles import profile
from py.src.adf.core import adf_fixed_d


def _series(fixture: str, key: str = "y") -> list[float]:
    records = json.loads((DATASETS / fixture).read_text(encoding="utf-8"))
    return [float(row[key]) for row in records]


def _run_adf_cli(argv: list[str]) -> subprocess.CompletedProcess[str]:
    # The backend invokes `python -m src.adf <input> --pretty` with cwd=py; mirror it.
    return subprocess.run(
        [sys.executable, "-m", "src.adf", *argv],
        cwd=PY_ROOT, text=True, capture_output=True,
    )


class AdfCoreContractTests(unittest.TestCase):
    @profile("fast")
    def test_diff_order_validation_rejects_bad_specs(self):
        trend = _series("trend_48.json")
        cases = {
            "empty": (),
            "duplicate": (0, 0),
            "negative": (-1,),
            "boolean": (True,),      # bool is an int subclass but never a valid order
            "fractional": (0.5,),
        }
        for name, orders in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                adf_fixed_d(trend, orders)

    @profile("fast")
    def test_short_and_nonfinite_series_are_rejected_before_adf(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            adf_fixed_d([1.0], (0, 2))
        with self.assertRaisesRegex(ValueError, "NaN or inf"):
            adf_fixed_d([1.0, float("inf"), 3.0], (0,))
        with self.assertRaises(ValueError):  # a 2-D input is not a 1-D series
            adf_fixed_d([[1.0, 2.0], [3.0, 4.0]], (0,))

    @profile("fast")
    def test_result_is_one_ordered_structured_entry_per_diff_order(self):
        orders = (2, 0, 1)  # entries keep request order rather than being sorted
        results = adf_fixed_d(_series("seeded_arx_96.json"), orders)
        self.assertEqual(list(orders), [entry["diff_order"] for entry in results])
        for entry in results:
            for key in ("diff_order", "statistic", "p_value", "used_lags", "n_obs", "stationary", "critical_values"):
                self.assertIn(key, entry)
            self.assertEqual({"1%", "5%", "10%"}, set(entry["critical_values"]))

    @profile("fast")
    def test_pure_trend_is_nonstationary_and_collapses_when_differenced(self):
        # y = 80 + 2t has a unit root at level and a degenerate (constant) first difference.
        results = {entry["diff_order"]: entry for entry in adf_fixed_d(_series("trend_48.json"), (0, 1, 2))}
        self.assertFalse(results[0]["stationary"])
        self.assertGreater(results[0]["p_value"], 0.05)
        for d in (1, 2):
            self.assertIsNone(results[d]["stationary"])
            self.assertTrue(results[d]["error"])

    @profile("fast")
    def test_constant_series_cannot_be_tested_at_any_order(self):
        for entry in adf_fixed_d(_series("level_36.json"), (0, 1, 2)):
            self.assertIsNone(entry["p_value"])
            self.assertIsNone(entry["stationary"])
            self.assertTrue(entry["error"])

    @profile("fast")
    def test_seeded_ar_process_is_flagged_stationary_at_level(self):
        # A committed AR(1) process with |phi|=0.65 (<1) is stationary; the p-value
        # margin is many orders of magnitude below alpha, so the verdict is stable.
        level = adf_fixed_d(_series("seeded_arx_96.json"), (0,))[0]
        self.assertTrue(level["stationary"])
        self.assertLess(level["p_value"], 0.05)
        self.assertLess(level["statistic"], level["critical_values"]["5%"])


class AdfCliCrossLanguageContractTests(unittest.TestCase):
    @profile("full")
    def test_cli_json_argument_satisfies_backend_parser_contract(self):
        # Field checks mirror fzbe/src/utils/adf.ts::parseAdfResultPayload for a
        # stationary series that yields three valid rows.
        records = json.loads((DATASETS / "seeded_arx_96.json").read_text(encoding="utf-8"))
        payload = json.dumps([{"month": row["month"], "sales": row["y"]} for row in records])
        process = _run_adf_cli([payload, "--pretty"])
        self.assertEqual(0, process.returncode, process.stderr)
        rows = json.loads(process.stdout)
        self.assertEqual(3, len(rows))
        seen: set[int] = set()
        for row in rows:
            self.assertIn(row["diff_order"], (0, 1, 2))
            self.assertNotIn(row["diff_order"], seen)
            seen.add(row["diff_order"])
            self.assertTrue(math.isfinite(row["statistic"]))
            self.assertTrue(0.0 <= row["p_value"] <= 1.0)
            self.assertIsInstance(row["used_lags"], int)
            self.assertGreaterEqual(row["used_lags"], 0)
            self.assertIsInstance(row["n_obs"], int)
            self.assertGreaterEqual(row["n_obs"], 1)
            self.assertIsInstance(row["stationary"], bool)
            for band in ("1%", "5%", "10%"):
                self.assertTrue(math.isfinite(row["critical_values"][band]))

    @profile("full")
    def test_cli_reads_file_input_with_custom_value_key(self):
        with tempfile.TemporaryDirectory(prefix="fz-deep-adf-") as temp:
            path = Path(temp) / "series.json"
            path.write_text(json.dumps([{"y": value} for value in _series("seeded_arx_96.json")]), encoding="utf-8")
            process = _run_adf_cli([str(path), "--value-key", "y", "--diff-orders", "0,1"])
            self.assertEqual(0, process.returncode, process.stderr)
            self.assertEqual([0, 1], [row["diff_order"] for row in json.loads(process.stdout)])

    @profile("full")
    def test_cli_reports_missing_file_without_crashing(self):
        process = _run_adf_cli([str(DATASETS / "does-not-exist.json")])
        self.assertEqual(1, process.returncode)
        self.assertIn("File not found", process.stderr)
        self.assertEqual("", process.stdout.strip())


if __name__ == "__main__":
    unittest.main()
