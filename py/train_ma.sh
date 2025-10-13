#!/usr/bin/env bash
DATA_SIMPLE="$(python - <<'PY'
from gen_dataset import generate_dataset, generate_simple_format
import json
data = generate_dataset(start_year=2021, n_years=3)
print(json.dumps(generate_simple_format(data)))
PY
)"

TRAIN_START=0
TRAIN_END=29
EVAL_START=30
EVAL_END=35
WINDOW=4
MODEL_SAVE_LOC=./trained_models/ma.pkl

python src/main.py --method ma \
  --data "$DATA_SIMPLE" \
  --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
  --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
  --window $WINDOW --save_model "$MODEL_SAVE_LOC" --pretty