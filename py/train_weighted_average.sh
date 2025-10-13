#!/usr/bin/env bash
# Usage: ./train_weighted_average.sh <DATA_FULL> <EVAL_START> <EVAL_END> <WEIGHT_STRATEGY> <MODEL_SAVE_LOC> <MODEL1> <MODEL2> [<MODEL3> ...]

if [ "$#" -lt 7 ]; then
  echo "Error: Requires at least 7 arguments" >&2
  echo "Usage: $0 <DATA_FULL> <EVAL_START> <EVAL_END> <WEIGHT_STRATEGY> <MODEL_SAVE_LOC> <MODEL1> <MODEL2> [<MODEL3> ...]" >&2
  exit 1
fi

DATA_FULL="$1"
EVAL_START="$2"
EVAL_END="$3"
WEIGHT_STRATEGY="$4"
MODEL_SAVE_LOC="$5"
shift 5
MODELS=("$@")

python src/main.py \
  --method weighted_average \
  --data "$DATA_FULL" \
  --models "${MODELS[@]}" \
  --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
  --weight_strategy "$WEIGHT_STRATEGY" \
  --save_ensemble "$MODEL_SAVE_LOC" \
  --pretty
