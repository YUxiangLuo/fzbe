#!/usr/bin/env bash
# Regression checks for boosting ensemble.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python"
MAIN="$ROOT/src/main.py"

DATA="$($PYTHON - <<'PY'
import json
records = []
for i in range(20):
    sales = 150 + i * 2 + (5 if i % 5 == 0 else 0)
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

MA_MODEL=$(mktemp "${TMPDIR:-/tmp}/boost_ma_XXXXXX.pkl")
ES_MODEL=$(mktemp "${TMPDIR:-/tmp}/boost_es_XXXXXX.pkl")
ARIMA_MODEL=$(mktemp "${TMPDIR:-/tmp}/boost_arima_XXXXXX.pkl")
BOOST_MODEL=$(mktemp "${TMPDIR:-/tmp}/boost_ensemble_XXXXXX.pkl")
trap 'rm -f "$MA_MODEL" "$ES_MODEL" "$ARIMA_MODEL" "$BOOST_MODEL"' EXIT

# Train base models used for residual stages
$PYTHON "$MAIN" --method ma --data "$DATA" \
  --train_start 0 --train_end 13 --eval_start 14 --eval_end 16 \
  --window 3 --save_model "$MA_MODEL" > /dev/null

$PYTHON "$MAIN" --method es --data "$DATA" \
  --train_start 0 --train_end 13 --eval_start 14 --eval_end 16 \
  --alpha 0.5 --save_model "$ES_MODEL" > /dev/null

$PYTHON "$MAIN" --method arima --data "$DATA" \
  --train_start 0 --train_end 13 --eval_start 14 --eval_end 16 \
  --d 1 --save_model "$ARIMA_MODEL" > /dev/null

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

# Train boosting ensemble with shrinkage
run_pass "Boosting: train with learning_rate" \
  --method boosting \
  --data "$DATA" \
  --models "$MA_MODEL" "$ES_MODEL" "$ARIMA_MODEL" \
  --train_start 0 --train_end 13 \
  --eval_start 14 --eval_end 16 \
  --window 3 --alpha 0.5 --d 1 \
  --learning_rate 0.3 \
  --save_ensemble "$BOOST_MODEL"

# Load ensemble to forecast
run_pass "Boosting: load and forecast" \
  --method boosting \
  --load_ensemble "$BOOST_MODEL" \
  --forecast_steps 3

# Load ensemble with eval metrics
run_pass "Boosting: load with eval metrics" \
  --method boosting \
  --load_ensemble "$BOOST_MODEL" \
  --forecast_steps 3 \
  --data "$DATA" \
  --eval_start 17 --eval_end 19

# Validation failure check
run_fail "Boosting: detect invalid ranges" 2 \
  --method boosting \
  --data "$DATA" \
  --models "$MA_MODEL" "$ES_MODEL" \
  --train_start 0 --train_end 15 \
  --eval_start 14 --eval_end 16 \
  --window 3 --alpha 0.5 --d 1

echo "✓ Boosting tests completed."
