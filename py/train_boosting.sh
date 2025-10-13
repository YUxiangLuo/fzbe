#!/usr/bin/env bash
# Train a boosting ensemble model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo ">>> Generating dataset..."
DATA_FULL="$(python -c 'from gen_dataset import generate_dataset; import json; print(json.dumps(generate_dataset(start_year=2021, n_years=3)))')"

TRAIN_START=0
TRAIN_END=29
EVAL_START=30
EVAL_END=35

# Base models to use in the boosting sequence.
# The paths are used as templates/configurations.
# These do NOT need to be pre-trained.
BASE_MODELS=(
  "trained_models/arima.pkl"
  "trained_models/ma.pkl"
  "trained_models/es.pkl"
  "trained_models/lstm.pt"
)

MODEL_SAVE_LOC="trained_models/boosting_config.pkl"

echo ">>> Training boosting ensemble..."
# Provide hyperparameters for all base models, as they will be re-trained on residuals.
python src/main.py \
  --method boosting \
  --data "$DATA_FULL" \
  --models "${BASE_MODELS[@]}" \
  --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
  --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
  --save_ensemble "$MODEL_SAVE_LOC" \
  --d 1 \
  --window 4 \
  --alpha 0.5 \
  --lookback 4 \
  --norm zscore \
  --pretty

echo ">>> Boosting ensemble training complete. Config saved to $MODEL_SAVE_LOC"
