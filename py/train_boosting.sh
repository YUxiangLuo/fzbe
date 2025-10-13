#!/usr/bin/env bash
# Usage: ./train_boosting.sh <DATA_FULL> <TRAIN_START> <TRAIN_END> <EVAL_START> <EVAL_END> <D> <WINDOW> <ALPHA> <LOOKBACK> <NORM> <FEATURE_TYPES> <MODEL_SAVE_LOC> <MODEL1> <MODEL2> [<MODEL3> ...]

if [ "$#" -lt 14 ]; then
  echo "Error: Requires at least 14 arguments" >&2
  echo "Usage: $0 <DATA_FULL> <TRAIN_START> <TRAIN_END> <EVAL_START> <EVAL_END> <D> <WINDOW> <ALPHA> <LOOKBACK> <NORM> <FEATURE_TYPES> <MODEL_SAVE_LOC> <MODEL1> <MODEL2> [...]" >&2
  exit 1
fi

DATA_FULL="$1"
TRAIN_START="$2"
TRAIN_END="$3"
EVAL_START="$4"
EVAL_END="$5"
D="$6"
WINDOW="$7"
ALPHA="$8"
LOOKBACK="$9"
NORM="${10}"
FEATURE_TYPES="${11}"
MODEL_SAVE_LOC="${12}"
shift 12
BASE_MODELS=("$@")

python src/main.py \
  --method boosting \
  --data "$DATA_FULL" \
  --models "${BASE_MODELS[@]}" \
  --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
  --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
  --save_ensemble "$MODEL_SAVE_LOC" \
  --d "$D" \
  --window "$WINDOW" \
  --alpha "$ALPHA" \
  --lookback "$LOOKBACK" \
  --norm "$NORM" \
  --feature_types "$FEATURE_TYPES" \
  --pretty
