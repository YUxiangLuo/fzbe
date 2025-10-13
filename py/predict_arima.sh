MODEL_LOC=./trained_models/arima.pkl
FORECAST_STEPS=3
python src/main.py --method "arima" --load_model "$MODEL_LOC" --forecast_steps "$FORECAST_STEPS" --pretty