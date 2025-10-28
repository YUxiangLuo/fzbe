#!/usr/bin/env bash
# Regression-style checks for ARIMA training & prediction.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python"
MAIN="$ROOT/src/main.py"

DATA="$("$PYTHON" - <<'PY'
import json
series = [103, 107, 110, 115, 119, 120, 124, 129, 133, 138, 140, 143, 147, 150]
print(json.dumps([{"month": f"m{i}", "sales": float(v)} for i, v in enumerate(series)]))
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
  local expect="${1:-2}"
  shift
  echo ">>> $desc (expect exit ${expect})"
  errfile="$(mktemp)"
  set +e
  "$PYTHON" "$MAIN" "$@" > /dev/null 2> "$errfile"
  status=$?
  set -e
  if [[ $status -ne $expect ]]; then
    cat "$errfile"
    rm -f "$errfile"
    echo "    unexpected exit code: $status" >&2
    exit 1
  fi
  echo "    got expected exit code ${expect}"
  cat "$errfile"
  rm -f "$errfile"
}

run_pass "ARIMA A: baseline train[0,9] eval[10,13]" \
  --method arima \
  --data "$DATA" \
  --train_start 0 --train_end 9 \
  --eval_start 10 --eval_end 13 \
  --d 1 \
  --p_max 2 --q_max 2

run_pass "ARIMA B: shorter eval horizon" \
  --method arima \
  --data "$DATA" \
  --train_start 2 --train_end 11 \
  --eval_start 12 --eval_end 13 \
  --d 1 \
  --p_max 2 --q_max 1

run_fail "ARIMA C: overlapping train/eval should error" 2 \
  --method arima \
  --data "$DATA" \
  --train_start 0 --train_end 7 \
  --eval_start 7 --eval_end 10 \
  --d 1

MODEL="$(mktemp "${TMPDIR:-/tmp}/arima_model_XXXXXX.pkl")"
trap 'rm -f "$MODEL"' EXIT

run_pass "ARIMA D1: train and save model" \
  --method arima \
  --data "$DATA" \
  --train_start 0 --train_end 9 \
  --eval_start 10 --eval_end 13 \
  --d 1 \
  --p_max 2 --q_max 2 \
  --save_model "$MODEL"

run_pass "ARIMA D2: load model with forecast_steps" \
  --method arima \
  --load_model "$MODEL" \
  --forecast_steps 3

run_pass "ARIMA E: load model with eval range & metrics" \
  --method arima \
  --load_model "$MODEL" \
  --data "$DATA" \
  --eval_start 10 --eval_end 13

echo "✓ ARIMA tests completed."
