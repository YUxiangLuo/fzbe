#!/usr/bin/env bash
# Regression checks for stacking ensemble.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python"
MAIN="$ROOT/src/main.py"

DATA="$("$PYTHON" - <<'PY'
import json
records = []
for i in range(20):
    sales = 120 + i * 2 + (3 if i % 4 == 0 else 0)
    records.append({
        "month": f"m{i}",
        "sales": float(sales),
        "ts": f"t{i}",
        "y": float(sales),
        "x": {"f1": float(i)}
    })
print(json.dumps(records))
PY
)"

MA_MODEL=$(mktemp "${TMPDIR:-/tmp}/stack_ma_XXXXXX.pkl")
ES_MODEL=$(mktemp "${TMPDIR:-/tmp}/stack_es_XXXXXX.pkl")
ARIMA_MODEL=$(mktemp "${TMPDIR:-/tmp}/stack_arima_XXXXXX.pkl")
STACK_MODEL=$(mktemp "${TMPDIR:-/tmp}/stack_meta_XXXXXX.pkl")
trap 'rm -f "$MA_MODEL" "$ES_MODEL" "$ARIMA_MODEL" "$STACK_MODEL"' EXIT

# Train base models used in stacking
"$PYTHON" "$MAIN" \
  --method ma \
  --data "$DATA" \
  --train_start 0 --train_end 11 \
  --eval_start 12 --eval_end 14 \
  --window 3 \
  --save_model "$MA_MODEL" > /dev/null

"$PYTHON" "$MAIN" \
  --method es \
  --data "$DATA" \
  --train_start 0 --train_end 11 \
  --eval_start 12 --eval_end 14 \
  --alpha 0.5 \
  --save_model "$ES_MODEL" > /dev/null

"$PYTHON" "$MAIN" \
  --method arima \
  --data "$DATA" \
  --train_start 0 --train_end 11 \
  --eval_start 12 --eval_end 14 \
  --d 1 \
  --save_model "$ARIMA_MODEL" > /dev/null

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

# Train stacking meta-model
run_pass "Stacking: train and save meta-model" \
  --method stacking \
  --data "$DATA" \
  --models "$MA_MODEL" "$ES_MODEL" "$ARIMA_MODEL" \
  --eval_start 12 --eval_end 14 \
  --save_ensemble "$STACK_MODEL"

# Load for forecasting
run_pass "Stacking: load ensemble forecast" \
  --method stacking \
  --load_ensemble "$STACK_MODEL" \
  --forecast_steps 3

# Load ensemble with eval data to compute metrics
run_pass "Stacking: load ensemble with eval metrics" \
  --method stacking \
  --load_ensemble "$STACK_MODEL" \
  --forecast_steps 3 \
  --data "$DATA" \
  --eval_start 15 --eval_end 17

# Validation failure: require at least 2 base models
run_fail "Stacking: reject single base model" 2 \
  --method stacking \
  --data "$DATA" \
  --models "$MA_MODEL" \
  --eval_start 12 --eval_end 14

echo "✓ Stacking tests completed."
