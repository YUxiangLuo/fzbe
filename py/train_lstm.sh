#!/usr/bin/env bash
# Usage: ./train_lstm.sh <DATA_FULL> <FEATURE_TYPES> <TRAIN_START> <TRAIN_END> <EVAL_START> <EVAL_END> <NORM> <MODEL_SAVE_LOC>

if [ "$#" -ne 8 ]; then
  echo "Error: Requires 8 arguments" >&2
  echo "Usage: $0 <DATA_FULL> <FEATURE_TYPES> <TRAIN_START> <TRAIN_END> <EVAL_START> <EVAL_END> <NORM> <MODEL_SAVE_LOC>" >&2
  exit 1
fi

DATA_FULL="$1"
FEATURE_TYPES="$2"
TRAIN_START="$3"
TRAIN_END="$4"
EVAL_START="$5"
EVAL_END="$6"
NORM="$7"
MODEL_SAVE_LOC="$8"

python src/main.py --method lstm \
  --data "$DATA_FULL" --feature_types "$FEATURE_TYPES" \
  --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
  --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
  --norm "$NORM" --save_model "$MODEL_SAVE_LOC" --pretty