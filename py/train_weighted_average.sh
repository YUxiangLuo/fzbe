#!/usr/bin/env bash
# Train a weighted average ensemble model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo ">>> Generating dataset..."
DATA_FULL="$(python -c 'from gen_dataset import generate_dataset; import json; print(json.dumps(generate_dataset(start_year=2021, n_years=3)))')"

TRAIN_START=0
TRAIN_END=29
EVAL_START=30
EVAL_END=35

# These models should be pre-trained before running this script
MODELS=(
  "trained_models/arima.pkl"
  "trained_models/ma.pkl"
  "trained_models/es.pkl"
  "trained_models/lstm.pt"
)

MODEL_SAVE_LOC="trained_models/weighted_average_config.pkl"

echo ">>> Training weighted average ensemble..."
python src/main.py \
  --method weighted_average \
  --data "$DATA_FULL" \
  --models "${MODELS[@]}" \
  --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
  --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
  --weight_strategy "inverse_rmse" \
  --save_ensemble "$MODEL_SAVE_LOC" \
  --pretty

echo ">>> Weighted average ensemble training complete. Config saved to $MODEL_SAVE_LOC"
