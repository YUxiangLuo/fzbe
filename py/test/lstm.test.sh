#!/usr/bin/env bash
# Targeted regression checks for the LSTM forecaster.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python"
MAIN="$ROOT/src/main.py"

# Synthetic dataset with a numeric and a categorical feature
D1="$("$PYTHON" - <<'PY'
import json
records=[]
for i in range(24):
    y = 200 + i * 3 + (5 if i % 4 == 0 else 0)
    records.append({
        "ts": f"t{i}",
        "y": float(y),
        "x": {
            "f1": float(i),
            "cat": "A" if i % 2 == 0 else "B"
        }
    })
print(json.dumps(records))
PY
)"

D2="$("$PYTHON" - <<'PY'
import json
records=[]
for i in range(30):
    y = 150 + i * 1.5 + (3 if i % 6 == 0 else 0)
    records.append({
        "ts": f"u{i}",
        "y": float(y),
        "x": {
            "f1": float(i),
            "cat": "X" if i % 3 == 0 else "Y"
        }
    })
print(json.dumps(records))
PY
)"

FEATURE_TYPES='{"f1":"numeric","cat":"categorical"}'

run_pass() {
  local desc="$1"; shift
  echo ">>> $desc"
  "$PYTHON" "$MAIN" "$@" > /dev/null
  echo "    ok"
}

run_fail() {
  local desc="$1"; shift
  local expected="${1:-2}"
  shift
  echo ">>> $desc (expect exit ${expected})"
  err="$(mktemp)"
  set +e
  "$PYTHON" "$MAIN" "$@" > /dev/null 2> "$err"
  status=$?
  set -e
  if [[ $status -ne $expected ]]; then
    cat "$err"
    rm -f "$err"
    echo "    unexpected exit code: $status" >&2
    exit 1
  fi
  echo "    got expected exit code ${expected}"
  cat "$err"
  rm -f "$err"
}

MODEL_Z="$(mktemp "${TMPDIR:-/tmp}/lstm_norm_zscore_XXXXXX.pt")"
MODEL_M="$(mktemp "${TMPDIR:-/tmp}/lstm_norm_minmax_XXXXXX.pt")"
MODEL_N="$(mktemp "${TMPDIR:-/tmp}/lstm_norm_none_XXXXXX.pt")"
trap 'rm -f "$MODEL_Z" "$MODEL_M" "$MODEL_N"' EXIT

# --- zscore normalization path ---
run_pass "LSTM zscore: train and save" \
  --method lstm \
  --data "$D1" \
  --train_start 0 --train_end 17 \
  --eval_start 18 --eval_end 21 \
  --norm zscore \
  --feature_types "$FEATURE_TYPES" \
  --save_model "$MODEL_Z"

run_pass "LSTM zscore: load with forecast_steps" \
  --method lstm \
  --load_model "$MODEL_Z" \
  --forecast_steps 3

run_pass "LSTM zscore: load with data window" \
  --method lstm \
  --load_model "$MODEL_Z" \
  --forecast_steps 2 \
  --data "$D1"

# --- minmax normalization path ---
run_pass "LSTM minmax: train and save" \
  --method lstm \
  --data "$D1" \
  --train_start 0 --train_end 17 \
  --eval_start 18 --eval_end 21 \
  --norm minmax \
  --feature_types "$FEATURE_TYPES" \
  --save_model "$MODEL_M"

run_pass "LSTM minmax: load with data window" \
  --method lstm \
  --load_model "$MODEL_M" \
  --forecast_steps 2 \
  --data "$D1"

# --- no normalization path ---
run_pass "LSTM none: train and save" \
  --method lstm \
  --data "$D2" \
  --train_start 0 --train_end 23 \
  --eval_start 24 --eval_end 27 \
  --norm none \
  --feature_types "$FEATURE_TYPES" \
  --save_model "$MODEL_N"

run_pass "LSTM none: load with data window" \
  --method lstm \
  --load_model "$MODEL_N" \
  --forecast_steps 2 \
  --data "$D2"

# --- validation failure ---
run_fail "LSTM: overlapping train/eval rejected" 2 \
  --method lstm \
  --data "$D1" \
  --train_start 0 --train_end 18 \
  --eval_start 18 --eval_end 21 \
  --norm zscore \
  --feature_types "$FEATURE_TYPES"

echo "✓ LSTM tests completed."
