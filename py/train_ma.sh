#!/usr/bin/env bash
# Usage: ./train_ma.sh <DATA_SIMPLE> <TRAIN_START> <TRAIN_END> <EVAL_START> <EVAL_END> <WINDOW> <MODEL_SAVE_LOC>

if [ "$#" -ne 7 ]; then
  echo "Error: Requires 7 arguments" >&2
  echo "Usage: $0 <DATA_SIMPLE> <TRAIN_START> <TRAIN_END> <EVAL_START> <EVAL_END> <WINDOW> <MODEL_SAVE_LOC>" >&2
  exit 1
fi

DATA_SIMPLE="$1"
TRAIN_START="$2"
TRAIN_END="$3"
EVAL_START="$4"
EVAL_END="$5"
WINDOW="$6"
MODEL_SAVE_LOC="$7"

python src/main.py --method ma \
  --data "$DATA_SIMPLE" \
  --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
  --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
  --window "$WINDOW" --save_model "$MODEL_SAVE_LOC" --pretty