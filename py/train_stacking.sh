#!/usr/bin/env bash
# Usage: ./train_stacking.sh <DATA_FULL> <EVAL_START> <EVAL_END> <MODEL_SAVE_LOC> <MODEL1> <MODEL2> [<MODEL3> ...]

if [ "$#" -lt 6 ]; then
  echo "Error: Requires at least 6 arguments" >&2
  echo "Usage: $0 <DATA_FULL> <EVAL_START> <EVAL_END> <MODEL_SAVE_LOC> <MODEL1> <MODEL2> [<MODEL3> ...]" >&2
  exit 1
fi

DATA_FULL="$1"
EVAL_START="$2"
EVAL_END="$3"
MODEL_SAVE_LOC="$4"
shift 4
MODELS=("$@")

python src/main.py \
  --method stacking \
  --data "$DATA_FULL" \
  --models "${MODELS[@]}" \
  --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
  --save_ensemble "$MODEL_SAVE_LOC" \
  --pretty
