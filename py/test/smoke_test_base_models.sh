#!/usr/bin/env bash
#
# Smoke Test for the four base models (ARIMA, MA, ES, LSTM).
#
# This test verifies that each base model can:
#   1. Train on a minimal dataset and save a model file without crashing.
#   2. Load the saved model and run a prediction without crashing.
#
# It does NOT check the correctness of the output, only that the core
# train/predict pipeline runs successfully.
#
set -euo pipefail

# --- Setup ---
# Navigate to the project root directory
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Create a temporary directory for test artifacts
TEMP_DIR=$(mktemp -d)

# Ensure the temporary directory is cleaned up on script exit
trap 'rm -rf "$TEMP_DIR"' EXIT

echo ">>> Starting smoke tests for base models..."
echo "    Temporary models will be stored in: $TEMP_DIR"
echo

# --- Test Data ---
# A minimal, self-contained dataset for testing
# Simple format for ARIMA, MA, ES
DATA_SIMPLE='[
    {"month": "2023-01", "sales": 100}, {"month": "2023-02", "sales": 110},
    {"month": "2023-03", "sales": 105}, {"month": "2023-04", "sales": 120},
    {"month": "2023-05", "sales": 115}, {"month": "2023-06", "sales": 130},
    {"month": "2023-07", "sales": 125}, {"month": "2023-08", "sales": 140},
    {"month": "2023-09", "sales": 135}, {"month": "2023-10", "sales": 150}
]'
# Full format for LSTM (with dummy features)
DATA_FULL='[
    {"ts": "2023-01", "y": 100, "x": {"f1": 0}}, {"ts": "2023-02", "y": 110, "x": {"f1": 0}},
    {"ts": "2023-03", "y": 105, "x": {"f1": 0}}, {"ts": "2023-04", "y": 120, "x": {"f1": 0}},
    {"ts": "2023-05", "y": 115, "x": {"f1": 0}}, {"ts": "2023-06", "y": 130, "x": {"f1": 0}},
    {"ts": "2023-07", "y": 125, "x": {"f1": 0}}, {"ts": "2023-08", "y": 140, "x": {"f1": 0}},
    {"ts": "2023-09", "y": 135, "x": {"f1": 0}}, {"ts": "2023-10", "y": 150, "x": {"f1": 0}}
]'
FEATURE_TYPES='{"f1": "numeric"}'

# Define a small train/eval split
TRAIN_START=0
TRAIN_END=7
EVAL_START=8
EVAL_END=9

# --- Test Loop ---
BASE_MODELS=(arima ma es lstm)

for method in "${BASE_MODELS[@]}"; do
    echo -n ">>> Testing [$(echo "$method" | tr '[:lower:]' '[:upper:]')]... "
    
    # --- Configure model-specific arguments ---
    model_path=""
    data_var=""
    # Use an array for arguments to handle spaces and quotes correctly
    extra_args_array=()
    
    case "$method" in
        arima)
            model_path="$TEMP_DIR/arima.pkl"
            data_var="$DATA_SIMPLE"
            extra_args_array=(--d 1)
            ;;
        ma)
            model_path="$TEMP_DIR/ma.pkl"
            data_var="$DATA_SIMPLE"
            extra_args_array=(--window 2)
            ;;
        es)
            model_path="$TEMP_DIR/es.pkl"
            data_var="$DATA_SIMPLE"
            extra_args_array=(--alpha 0.5)
            ;;
        lstm)
            model_path="$TEMP_DIR/lstm.pt"
            data_var="$DATA_FULL"
            # Note: The JSON string is a single element in the array
            extra_args_array=(--lookback 2 --epochs 5 --feature_types "$FEATURE_TYPES")
            ;;
    esac

    # --- 1. Train and Save Model ---
    # We redirect stdout to /dev/null because we only care about success (exit code 0)
    python src/main.py \
        --method "$method" \
        --data "$data_var" \
        --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
        --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
        --save_model "$model_path" \
        "${extra_args_array[@]}" > /dev/null

    # --- 2. Load and Predict ---
    python src/main.py \
        --method "$method" \
        --load_model "$model_path" \
        --forecast_steps 1 > /dev/null

    echo "[OK]"
done

echo
echo "✅ All smoke tests passed successfully!"
