#!/usr/bin/env bash
# Train a stacking ensemble model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo ">>> Generating dataset..."
DATA_FULL="$(python -c 'from gen_dataset import generate_dataset; import json; print(json.dumps(generate_dataset(start_year=2021, n_years=3)))')"

# The evaluation set is used to train the meta-model
EVAL_START=30
EVAL_END=35

# These models should be pre-trained before running this script
MODELS=(
  "trained_models/arima.pkl"
  "trained_models/ma.pkl"
  "trained_models/es.pkl"
  "trained_models/lstm.pt"
)

MODEL_SAVE_LOC="trained_models/stacking_config.pkl"

echo ">>> Training stacking ensemble (meta-model)..."
python src/main.py \
  --method stacking \
  --data "$DATA_FULL" \
  --models "${MODELS[@]}" \
  --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
  --save_ensemble "$MODEL_SAVE_LOC" \
  --pretty

echo ">>> Stacking ensemble training complete. Config saved to $MODEL_SAVE_LOC"
