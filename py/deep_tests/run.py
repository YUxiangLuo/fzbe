#!/usr/bin/env python3
"""The only supported runner for fast/full/nightly/resource deep-test profiles."""
from __future__ import annotations

import argparse
import os
import sys
import time
import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
for path in (str(REPO_ROOT / "py"), str(REPO_ROOT / "py" / "src"), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

FIXED_ENV = {
    "PYTHONHASHSEED": "20250308", "TF_DETERMINISTIC_OPS": "1", "TF_CPP_MIN_LOG_LEVEL": "3",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
}


def flatten(suite: unittest.TestSuite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


def test_profile(test: unittest.TestCase) -> str:
    # Discovery/import failures must fail every profile, including resource-only
    # runs; treating unittest's synthetic _FailedTest as fast could hide them.
    if test.__class__.__module__ == "unittest.loader" and test.__class__.__name__ == "_FailedTest":
        return "loader-error"
    method = getattr(test, test._testMethodName)
    return getattr(method, "fz_deep_profile", getattr(test.__class__, "fz_deep_profile", "fast"))


def main() -> int:
    # Hash randomization is fixed only at interpreter startup, so re-exec before
    # discovery when the caller did not provide the suite's declared seed.
    if os.environ.get("PYTHONHASHSEED") != FIXED_ENV["PYTHONHASHSEED"]:
        env = os.environ.copy()
        env.update(FIXED_ENV)
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("fast", "full", "nightly", "resource"), default="fast")
    parser.add_argument("--list", action="store_true", help="list selected test ids without executing")
    args = parser.parse_args()
    os.environ.update(FIXED_ENV)
    discovered = unittest.defaultTestLoader.discover(str(PACKAGE_DIR), pattern="test_*.py", top_level_dir=str(REPO_ROOT))
    selected_profiles = {
        "fast": {"fast"},
        "full": {"fast", "full"},
        "nightly": {"fast", "full", "nightly"},
        "resource": {"resource"},
    }[args.profile]
    selected = [test for test in flatten(discovered) if test_profile(test) in selected_profiles or test_profile(test) == "loader-error"]
    if args.list:
        for test in selected:
            print(f"{test.id()} [{test_profile(test)}]")
        print(f"profile={args.profile} tests={len(selected)}")
        return 0
    print(f"FZ deep tests: profile={args.profile}, tests={len(selected)}, seed=20250308, threads=1", flush=True)
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2, failfast=False, buffer=False).run(unittest.TestSuite(selected))
    elapsed = time.monotonic() - started
    print(f"FZ deep tests complete: profile={args.profile}, tests={result.testsRun}, failures={len(result.failures)}, errors={len(result.errors)}, skipped={len(result.skipped)}, seconds={elapsed:.3f}", flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
