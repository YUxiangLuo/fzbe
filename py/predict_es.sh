MODEL_LOC=./trained_models/es.pkl
FORECAST_STEPS=3
python src/main.py --method "es" --load_model "$MODEL_LOC" --forecast_steps "$FORECAST_STEPS" --pretty