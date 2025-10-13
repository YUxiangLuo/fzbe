#!/usr/bin/env bash
MODEL_LOC="trained_models/weighted_average_config.pkl"
FORECAST_STEPS=6

python src/main.py \
  --method weighted_average \
  --load_ensemble "$MODEL_LOC" \
  --forecast_steps "$FORECAST_STEPS" \
  --pretty
