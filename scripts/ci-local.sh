#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bun="$project_root/scripts/run-pinned-bun.sh"
profile="${1:-full}"

export CI="${CI:-1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-20250308}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TF_DETERMINISTIC_OPS="${TF_DETERMINISTIC_OPS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"

run_python_setup() {
  "$project_root/py/install.sh" --test
}

run_dataset_check() {
  "$project_root/py/venv/bin/python" \
    "$project_root/py/deep_tests/support/generate_datasets.py" --check
}

run_full() {
  "$bun" install --frozen-lockfile
  "$bun" run test:check
  "$bun" run test:system:db-compat
  "$bun" run test:system:user-phone-canonical
  "$bun" run test:system:concurrency
  "$bun" audit

  run_python_setup
  "$bun" run python:lock:check
  "$bun" run lint:python
  "$bun" run test:deep:fast
  "$bun" run test:workflow
  run_dataset_check
  "$bun" run test:deep:full
}

cd "$project_root"
case "$profile" in
  full)
    run_full
    ;;
  mysql8)
    "$project_root/scripts/test-mysql8.sh"
    ;;
  nightly)
    run_python_setup
    run_dataset_check
    "$bun" run test:deep:nightly
    "$bun" run test:deep:resource
    ;;
  resource)
    run_python_setup
    "$bun" run test:deep:resource
    ;;
  *)
    echo "Usage: $0 [full|mysql8|nightly|resource]" >&2
    exit 2
    ;;
esac
