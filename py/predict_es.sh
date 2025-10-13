#!/usr/bin/env bash
# Usage: ./predict_es.sh <MODEL_LOC> <FORECAST_STEPS>

if [ "$#" -ne 2 ]; then
  echo "Error: Requires 2 arguments" >&2
  echo "Usage: $0 <MODEL_LOC> <FORECAST_STEPS>" >&2
  exit 1
fi

MODEL_LOC="$1"
FORECAST_STEPS="$2"

python src/main.py --method "es" --load_model "$MODEL_LOC" --forecast_steps "$FORECAST_STEPS" --pretty