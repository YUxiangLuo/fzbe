MODEL_LOC=./trained_models/lstm.pt
FORECAST_STEPS=3
python src/main.py --method "lstm" --load_model "$MODEL_LOC" --forecast_steps "$FORECAST_STEPS" --pretty