#!/usr/bin/env bash
# Smoke-style tests for the moving-average model with synthetic data.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python"
MAIN="$ROOT/src/main.py"

DATA="$("$PYTHON" - <<'PY'
import json
series = [103, 107, 110, 115, 119, 120, 124, 129, 133, 138, 140, 143]
data = [{"month": f"m{i}", "sales": float(v)} for i, v in enumerate(series)]
print(json.dumps(data))
PY
)"

run_pass() {
  local desc="$1"; shift
  echo ">>> $desc"
  "$PYTHON" "$MAIN" "$@" > /dev/null
  echo "    ok"
}

run_fail() {
  local desc="$1"; shift
  local expected_exit="${1:-2}"
  shift
  local stderr_file
  stderr_file="$(mktemp)"
  echo ">>> $desc (expect exit ${expected_exit})"
  set +e
  "$PYTHON" "$MAIN" "$@" > /dev/null 2> "$stderr_file"
  local status=$?
  set -e
  if [[ $status -ne $expected_exit ]]; then
    cat "$stderr_file"
    echo "    unexpected exit code: $status" >&2
    rm -f "$stderr_file"
    exit 1
  fi
  echo "    got expected exit code ${expected_exit}"
  cat "$stderr_file"
  rm -f "$stderr_file"
}

run_pass "Test A: baseline window=3 on train[0,7] eval[8,11]" \
  --method ma \
  --data "$DATA" \
  --train_start 0 --train_end 7 \
  --eval_start 8 --eval_end 11 \
  --window 3

run_pass "Test B: respect train_start when window=4" \
  --method ma \
  --data "$DATA" \
  --train_start 3 --train_end 10 \
  --eval_start 11 --eval_end 11 \
  --window 4

run_pass "Test C: skip early eval points inside training block" \
  --method ma \
  --data "$DATA" \
  --train_start 5 --train_end 10 \
  --eval_start 6 --eval_end 10 \
  --window 3

run_fail "Test D: insufficient history for eval window" 2 \
  --method ma \
  --data "$DATA" \
  --train_start 5 --train_end 7 \
  --eval_start 6 --eval_end 7 \
  --window 3

MODEL="$(mktemp "${TMPDIR:-/tmp}/ma_model_XXXXXX.pkl")"
trap 'rm -f "$MODEL"' EXIT

run_pass "Test E1: train and persist model artifact" \
  --method ma \
  --data "$DATA" \
  --train_start 4 --train_end 10 \
  --eval_start 11 --eval_end 11 \
  --window 3 \
  --save_model "$MODEL"

run_pass "Test E2: load persisted model for 2-step forecast" \
  --method ma \
  --load_model "$MODEL" \
  --forecast_steps 2

echo "✓ MA tests completed."
