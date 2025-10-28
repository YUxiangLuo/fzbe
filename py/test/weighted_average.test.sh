#!/usr/bin/env bash
# Regression checks for the weighted average ensemble.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python"
MAIN="$ROOT/src/main.py"

DATA="$("$PYTHON" - <<'PY'
import json
series = [100, 105, 111, 119, 126, 133, 139, 145, 150, 156, 161, 165]
print(json.dumps([{"month": f"m{i}", "sales": float(v)} for i, v in enumerate(series)]))
PY
)"

MA_MODEL=$(mktemp "${TMPDIR:-/tmp}/wa_ma_XXXXXX.pkl")
ES_MODEL=$(mktemp "${TMPDIR:-/tmp}/wa_es_XXXXXX.pkl")
trap 'rm -f "$MA_MODEL" "$ES_MODEL" "$WA_DEFAULT" "$WA_EQUAL" "$WA_INV" "$WA_GRID" "$WA_LR"' EXIT

# Train base models (MA and ES) to be combined by the ensemble.
run_base_models() {
  "$PYTHON" "$MAIN" \
    --method ma \
    --data "$DATA" \
    --train_start 0 --train_end 7 \
    --eval_start 8 --eval_end 10 \
    --window 3 \
    --save_model "$MA_MODEL" > /dev/null

  "$PYTHON" "$MAIN" \
    --method es \
    --data "$DATA" \
    --train_start 0 --train_end 7 \
    --eval_start 8 --eval_end 10 \
    --alpha 0.4 \
    --save_model "$ES_MODEL" > /dev/null
}

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

run_base_models

WA_DEFAULT=$(mktemp "${TMPDIR:-/tmp}/wa_default_XXXXXX.pkl")
WA_EQUAL=$(mktemp "${TMPDIR:-/tmp}/wa_equal_XXXXXX.pkl")
WA_INV=$(mktemp "${TMPDIR:-/tmp}/wa_inverse_XXXXXX.pkl")
WA_GRID=$(mktemp "${TMPDIR:-/tmp}/wa_grid_XXXXXX.pkl")
WA_LR=$(mktemp "${TMPDIR:-/tmp}/wa_lr_XXXXXX.pkl")

# Default (inverse_variance) strategy
run_pass "Weighted avg default (inverse_variance): train and save" \
  --method weighted_average \
  --data "$DATA" \
  --models "$MA_MODEL" "$ES_MODEL" \
  --eval_start 8 --eval_end 10 \
  --save_ensemble "$WA_DEFAULT"

run_pass "Weighted avg default (inverse_variance): load forecast_steps" \
  --method weighted_average \
  --load_ensemble "$WA_DEFAULT" \
  --forecast_steps 3

# Equal weights strategy
run_pass "Weighted avg equal: train and save" \
  --method weighted_average \
  --data "$DATA" \
  --models "$MA_MODEL" "$ES_MODEL" \
  --eval_start 8 --eval_end 10 \
  --weight_strategy equal \
  --save_ensemble "$WA_EQUAL"

run_pass "Weighted avg equal: load forecast_steps" \
  --method weighted_average \
  --load_ensemble "$WA_EQUAL" \
  --forecast_steps 3

# Inverse RMSE strategy
run_pass "Weighted avg inverse_rmse: train and save" \
  --method weighted_average \
  --data "$DATA" \
  --models "$MA_MODEL" "$ES_MODEL" \
  --eval_start 8 --eval_end 10 \
  --weight_strategy inverse_rmse \
  --save_ensemble "$WA_INV"

# Grid search strategy (two models)
run_pass "Weighted avg grid_search: train and save" \
  --method weighted_average \
  --data "$DATA" \
  --models "$MA_MODEL" "$ES_MODEL" \
  --eval_start 8 --eval_end 10 \
  --weight_strategy grid_search \
  --save_ensemble "$WA_GRID"

# Linear regression strategy
run_pass "Weighted avg linear_regression: train and save" \
  --method weighted_average \
  --data "$DATA" \
  --models "$MA_MODEL" "$ES_MODEL" \
  --eval_start 8 --eval_end 10 \
  --weight_strategy linear_regression \
  --save_ensemble "$WA_LR"

run_pass "Weighted avg linear_regression: load with data window" \
  --method weighted_average \
  --load_ensemble "$WA_LR" \
  --data "$DATA" \
  --eval_start 8 --eval_end 10

# Validation failure: not enough models
run_fail "Weighted avg: requires at least two models" 2 \
  --method weighted_average \
  --data "$DATA" \
  --models "$MA_MODEL" \
  --eval_start 8 --eval_end 10

echo "✓ Weighted average tests completed."
