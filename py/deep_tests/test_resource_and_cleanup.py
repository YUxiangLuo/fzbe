from __future__ import annotations

import os
import resource
import unittest
from pathlib import Path

from py.deep_tests.support.harness import DATASETS, IsolatedWorkspace, assert_prediction_contract, repository_artifacts, run_model
from py.deep_tests.support.profiles import profile


@unittest.skipUnless(os.name == "posix" and Path("/proc/meminfo").exists(), "Linux resource contract")
class LinuxResourceTests(unittest.TestCase):
    @profile("resource")
    def test_representative_ensemble_peak_rss_disk_and_cleanup(self):
        before = repository_artifacts()
        with IsolatedWorkspace() as temp:
            result = run_model("weighted_avg", DATASETS / "seasonal_pulse_60.json", temp / "weighted.pkl", timeout=900)
            assert_prediction_contract(result.payload, 4)
            peak_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            available_kib = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")))
            # A broad safety ceiling catches runaway process trees without pretending to be a microbenchmark.
            limit_kib = min(8 * 1024 * 1024, int(available_kib * 0.70))
            self.assertLess(peak_kib, limit_kib, f"peak_rss_kib={peak_kib}, limit_kib={limit_kib}")
            disk_bytes = sum(path.stat().st_size for path in temp.rglob("*") if path.is_file())
            self.assertLess(disk_bytes, 512 * 1024 * 1024, f"temporary artifact bytes={disk_bytes}")
        self.assertEqual(before, repository_artifacts())

    @profile("resource")
    def test_temporary_workspace_removes_checkpoint_artifact_and_mutation_files(self):
        root: Path | None = None
        with IsolatedWorkspace() as temp:
            root = temp
            for name in ("candidate.pkl", "sidecar.h5", "receipt.json", "mutation.json"):
                (temp / name).write_bytes(b"temporary")
            self.assertTrue(all((temp / name).exists() for name in ("candidate.pkl", "sidecar.h5", "receipt.json", "mutation.json")))
        self.assertIsNotNone(root)
        self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
