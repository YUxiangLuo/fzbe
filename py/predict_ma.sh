MODEL_LOC=./trained_models/ma.pkl
FORECAST_STEPS=3
python src/main.py --method "ma" --load_model "$MODEL_LOC" --forecast_steps "$FORECAST_STEPS" --pretty