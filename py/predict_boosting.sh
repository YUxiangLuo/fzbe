MODEL_LOC="trained_models/boosting_config.pkl"
FORECAST_STEPS=6

python src/main.py \
  --method boosting \
  --load_ensemble "$MODEL_LOC" \
  --forecast_steps "$FORECAST_STEPS" \
  --pretty
