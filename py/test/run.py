#!/usr/bin/env python3
"""Run the backend Python workflow suites from one deterministic entry point."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parents[1]

FIXED_ENV = {
    "PYTHONHASHSEED": "20250308",
    "TF_DETERMINISTIC_OPS": "1",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
}


@dataclass(frozen=True)
class Workflow:
    script: str
    suite: str
    description: str
    args: tuple[str, ...] = ()


WORKFLOWS = (
    Workflow("metrics_smoke.py", "contract", "forecast metric formulas"),
    Workflow("data_contract_smoke.py", "contract", "finite CSV/JSON numeric contract"),
    Workflow("model_metadata_assertions.py", "contract", "model metadata and teaching copy"),
    Workflow("ma_csv_smoke.py", "smoke", "moving-average CSV lifecycle"),
    Workflow("es_csv_smoke.py", "smoke", "exponential-smoothing CSV lifecycle"),
    Workflow("arima_csv_smoke.py", "smoke", "ARIMA CSV lifecycle"),
    Workflow("lstm_csv_smoke.py", "smoke", "LSTM CSV lifecycle"),
    Workflow("weighted_avg_csv_smoke.py", "smoke", "weighted-average ensemble lifecycle"),
    Workflow("stacking_csv_smoke.py", "smoke", "stacking ensemble lifecycle"),
    Workflow("boosting_csv_smoke.py", "smoke", "boosting ensemble lifecycle"),
    Workflow("weighted_avg_lstm_csv_smoke.py", "smoke", "weighted-average with LSTM"),
    Workflow("stacking_lstm_csv_smoke.py", "smoke", "stacking with LSTM"),
    Workflow("boosting_lstm_csv_smoke.py", "smoke", "boosting with LSTM"),
    Workflow("ensemble_reuse_lstm_csv_smoke.py", "smoke", "saved LSTM ensemble reuse"),
    Workflow("production_preparation_smoke.py", "smoke", "resumable production preparation"),
    Workflow("guided_standalone_parity.py", "parity", "guided and standalone base-model parity"),
    Workflow("guided_ensemble_parity.py", "parity", "guided and regular ensemble parity"),
    Workflow("review_regressions.py", "regression", "model review regression reproductions"),
    Workflow("designed_model_diagnostics.py", "diagnostic", "designed model diagnostics"),
    Workflow("lstm_feature_diagnostics.py", "diagnostic", "LSTM feature diagnostics"),
    Workflow(
        "model_edge_cases.py",
        "stress",
        "focused model edge-case matrix",
        ("--profile", "focused"),
    ),
    Workflow("medium_model_stress.py", "stress", "medium-size model stress matrix"),
)

SUITES = tuple(dict.fromkeys(workflow.suite for workflow in WORKFLOWS))
PROFILES = {
    "quick": ("contract",),
    "ci": ("contract", "smoke", "parity", "regression"),
    "full": SUITES,
}


def registry_errors() -> list[str]:
    ignored = {"__init__.py", "run.py"}
    discovered = {
        script.name
        for script in TEST_DIR.glob("*.py")
        if script.name not in ignored
    }
    registered_names = [workflow.script for workflow in WORKFLOWS]
    registered = set(registered_names)
    errors = [
        *(f"unregistered workflow script: {name}" for name in sorted(discovered - registered)),
        *(f"registered workflow script is missing: {name}" for name in sorted(registered - discovered)),
        *(
            f"workflow script is registered {count} times: {name}"
            for name, count in sorted(Counter(registered_names).items())
            if count > 1
        ),
    ]
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="ci",
        help="named workflow profile (default: ci)",
    )
    selection.add_argument(
        "--suite",
        action="append",
        choices=SUITES,
        help="run one or more explicit suites; may be repeated",
    )
    parser.add_argument("--list", action="store_true", help="list selected scripts without executing")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failed script")
    return parser.parse_args()


def selected_workflows(args: argparse.Namespace) -> list[Workflow]:
    suites = tuple(args.suite) if args.suite else PROFILES[args.profile]
    return [workflow for workflow in WORKFLOWS if workflow.suite in suites]


def main() -> int:
    args = parse_args()
    errors = registry_errors()
    if errors:
        print("Python workflow registry is inconsistent:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 2

    selected = selected_workflows(args)
    label = ",".join(args.suite) if args.suite else args.profile

    if args.list:
        for workflow in selected:
            command = " ".join((workflow.script, *workflow.args))
            print(f"{workflow.suite:10} {command:52} {workflow.description}")
        print(f"selection={label} scripts={len(selected)}")
        return 0

    env = os.environ.copy()
    env.update(FIXED_ENV)
    failures: list[str] = []
    started = time.monotonic()
    print(
        f"FZ Python workflows: selection={label}, scripts={len(selected)}, "
        "seed=20250308, threads=1",
        flush=True,
    )

    for index, workflow in enumerate(selected, start=1):
        script = TEST_DIR / workflow.script
        if not script.is_file():
            failures.append(workflow.script)
            print(f"\n[{index}/{len(selected)}] MISSING {script}", file=sys.stderr, flush=True)
            if args.fail_fast:
                break
            continue

        print(
            f"\n[{index}/{len(selected)}] {workflow.suite}: "
            f"{workflow.script} — {workflow.description}",
            flush=True,
        )
        result = subprocess.run(
            [sys.executable, str(script), *workflow.args],
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            failures.append(workflow.script)
            print(
                f"FAILED: {workflow.script} (exit={result.returncode})",
                file=sys.stderr,
                flush=True,
            )
            if args.fail_fast:
                break

    elapsed = time.monotonic() - started
    print(
        f"\nFZ Python workflows complete: selection={label}, "
        f"scripts={len(selected)}, failures={len(failures)}, seconds={elapsed:.3f}",
        flush=True,
    )
    if failures:
        print("Failed scripts: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
