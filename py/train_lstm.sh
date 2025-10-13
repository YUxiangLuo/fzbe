DATA_FULL="$(python - <<'PY'
from gen_dataset import generate_dataset
import json
data = generate_dataset(start_year=2021, n_years=3)
print(json.dumps(data))
PY
)"
FEATURE_TYPES='{"price":"numeric","discount_rate":"numeric","promo":"binary","is_holiday":"binary","temperature":"numeric","competitor_price":"numeric"}'

TRAIN_START=0
TRAIN_END=29
EVAL_START=30
EVAL_END=35
NORM=zscore
MODEL_SAVE_LOC=./trained_models/lstm.pt


# LSTM
python src/main.py --method lstm \
  --data "$DATA_FULL" --feature_types "$FEATURE_TYPES" \
  --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
  --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
  --norm $NORM --save_model $MODEL_SAVE_LOC --pretty