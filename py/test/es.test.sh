#!/usr/bin/env bash
# Synthetic checks for the single-exponential-smoothing model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python"
MAIN="$ROOT/src/main.py"

DATA_BASE="$("$PYTHON" - <<'PY'
import json
series = [103, 107, 110, 115, 119, 120, 124, 129, 133, 138, 140, 143]
print(json.dumps([{"month": f"m{i}", "sales": float(v)} for i, v in enumerate(series)]))
PY
)"

DATA_EXTENDED="$("$PYTHON" - <<'PY'
import json
series = [103, 107, 110, 115, 119, 120, 124, 129, 133, 138, 140, 143, 147, 150]
print(json.dumps([{"month": f"m{i}", "sales": float(v)} for i, v in enumerate(series)]))
PY
)"

alpha=0.4

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
  tmp="$(mktemp)"
  set +e
  "$PYTHON" "$MAIN" "$@" > /dev/null 2> "$tmp"
  status=$?
  set -e
  if [[ $status -ne $expect ]]; then
    cat "$tmp"
    rm -f "$tmp"
    echo "    unexpected exit code: $status" >&2
    exit 1
  fi
  echo "    got expected exit code ${expect}"
  cat "$tmp"
  rm -f "$tmp"
}

run_pass "ES A: train[0,7] eval[8,11]" \
  --method es \
  --data "$DATA_BASE" \
  --train_start 0 --train_end 7 \
  --eval_start 8 --eval_end 11 \
  --alpha "$alpha"

run_pass "ES B: shifted training window" \
  --method es \
  --data "$DATA_BASE" \
  --train_start 3 --train_end 10 \
  --eval_start 11 --eval_end 11 \
  --alpha "$alpha"

run_pass "ES C: evaluation overlapping with training tail" \
  --method es \
  --data "$DATA_BASE" \
  --train_start 5 --train_end 10 \
  --eval_start 6 --eval_end 10 \
  --alpha "$alpha"

run_fail "ES D: insufficient history for evaluation" 2 \
  --method es \
  --data "$DATA_BASE" \
  --train_start 5 --train_end 7 \
  --eval_start 5 --eval_end 7 \
  --alpha "$alpha"

MODEL="$(mktemp "${TMPDIR:-/tmp}/es_model_XXXXXX.pkl")"
trap 'rm -f "$MODEL"' EXIT

run_pass "ES E1: train and save model" \
  --method es \
  --data "$DATA_BASE" \
  --train_start 4 --train_end 10 \
  --eval_start 11 --eval_end 11 \
  --alpha "$alpha" \
  --save_model "$MODEL"

run_pass "ES E2: load saved model for 2-step forecast" \
  --method es \
  --load_model "$MODEL" \
  --forecast_steps 2

run_pass "ES F: load with future data updates" \
  --method es \
  --load_model "$MODEL" \
  --forecast_steps 2 \
  --data "$DATA_EXTENDED"

echo "✓ ES tests completed."
