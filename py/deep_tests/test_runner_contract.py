from __future__ import annotations

import unittest

from py.deep_tests.run import flatten, test_profile
from py.deep_tests.support.profiles import profile


class RunnerContractTests(unittest.TestCase):
    @profile("fast")
    def test_loader_failures_are_profile_agnostic(self):
        suite = unittest.defaultTestLoader.loadTestsFromName("fz_deep_definitely_missing.module")
        failed = list(flatten(suite))
        self.assertEqual(1, len(failed))
        self.assertEqual("loader-error", test_profile(failed[0]))
        for selected_profiles in ({"fast"}, {"resource"}, {"fast", "full", "nightly"}):
            self.assertTrue(test_profile(failed[0]) in selected_profiles or test_profile(failed[0]) == "loader-error")


if __name__ == "__main__":
    unittest.main()
