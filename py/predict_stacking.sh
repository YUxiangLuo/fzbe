MODEL_LOC="trained_models/stacking_config.pkl"
FORECAST_STEPS=6
python src/main.py \
  --method stacking \
  --load_ensemble "$MODEL_LOC" \
  --forecast_steps "$FORECAST_STEPS" \
  --pretty
