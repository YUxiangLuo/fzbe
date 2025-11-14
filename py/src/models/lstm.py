# py/src/models/lstm.py
# -*- coding: utf-8 -*-
"""
LSTM (Long Short-Term Memory) model for time series forecasting.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Import shared utilities
from data_utils import ParsedData, load_data_from_json
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model

MODEL_TYPE = "lstm"

# Type alias for input flexibility
ArrayLike = Union[List[float], np.ndarray]


# ---------- utilities ----------
def _load_state(model_path: str) -> Dict[str, Any]:
    """
    Load and validate a saved LSTM model state.
    """
    from tensorflow.keras.models import load_model as keras_load_model
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid LSTM model file.")
    
    # Load the Keras model from the dedicated file
    keras_model_path = state.get("keras_model_path")
    if not keras_model_path or not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Keras model file not found at {keras_model_path}")
    state["keras_model"] = keras_load_model(keras_model_path)
    
    return state


def _create_dataset(
    dataset: np.ndarray, look_back: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input and output datasets for LSTM.
    The target variable is assumed to be the first column after transformation.
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i : (i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0]) # Target is the first column
    return np.array(dataX), np.array(dataY)


# ---------- predict ----------
def predict(
    model_path: str,
    steps: int,
    data: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], float]:
    """
    Forecast using a saved LSTM model.
    """
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")
    state = state or _load_state(model_path)
    
    # If state is passed directly, the keras model might not be loaded yet
    if "keras_model" not in state:
        from tensorflow.keras.models import load_model as keras_load_model
        keras_model_path = state.get("keras_model_path")
        if not keras_model_path or not os.path.exists(keras_model_path):
            raise FileNotFoundError(f"Keras model file not found at {keras_model_path}")
        state["keras_model"] = keras_load_model(keras_model_path)

    keras_model = state["keras_model"]
    preprocessor = state["preprocessor"]
    look_back = state["look_back"]
    history_df = state["history_df"]
    std_dev_residuals = state.get("std_dev_residuals", 0.0)
    
    # The preprocessor expects a DataFrame, so we work with it
    scaled_history = preprocessor.transform(history_df)
    n_features_out = scaled_history.shape[1]

    # Use the last part of the history as the initial input
    input_seq = scaled_history[-look_back:].reshape(1, look_back, n_features_out)

    # Get the scaler for the target variable to perform inverse transform later
    target_scaler = preprocessor.named_transformers_['numeric'].named_steps['scaler']
    target_column_indices = preprocessor.output_indices_['numeric']
    if isinstance(target_column_indices, slice):
        target_column_index = target_column_indices.start
    else:
        target_column_index = target_column_indices[0]


    predictions_scaled = []
    for _ in range(steps):
        pred_scaled = keras_model.predict(input_seq, verbose=0)
        predictions_scaled.append(pred_scaled[0, 0])
        
        # Create the next input step (as a scaled vector)
        next_input_step_scaled = np.zeros((1, 1, n_features_out))
        next_input_step_scaled[0, 0, target_column_index] = pred_scaled[0, 0]
        
        # For other features, assume they stay constant at the last known value
        last_known_scaled_features = input_seq[0, -1, :]
        for i in range(n_features_out):
            if i != target_column_index:
                next_input_step_scaled[0, 0, i] = last_known_scaled_features[i]
            
        # Update the input sequence for the next prediction
        input_seq = np.append(input_seq[:, 1:, :], next_input_step_scaled, axis=1)

    # Inverse transform the predictions
    # Create a dummy array with the same shape as the scaler expects
    n_numeric_features = target_scaler.n_features_in_
    
    dummy_array = np.zeros((len(predictions_scaled), n_numeric_features))
    dummy_array[:, 0] = predictions_scaled
    unscaled_predictions = target_scaler.inverse_transform(dummy_array)[:, 0]
    
    return unscaled_predictions.flatten().tolist(), std_dev_residuals


# ---------- train and evaluate ----------
def train_and_evaluate(
    parsed_data: ParsedData,
    feature_types: Dict[str, str],
    train_range: range,
    eval_range: range,
    look_back: int,
    epochs: int,
) -> Tuple[Dict[str, Any], Any, Any, List[float], List[float], float, pd.DataFrame]:
    """
    Trains and evaluates the LSTM model with typed features.
    """
    from tensorflow.keras.layers import LSTM as KerasLSTM
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    df = parsed_data.df
    target_key = parsed_data.target_key
    
    # Separate features by type
    numeric_features = [k for k, v in feature_types.items() if v == 'numeric' and k != target_key]
    categorical_features = [k for k, v in feature_types.items() if v == 'categorical']

    # Create the preprocessing pipelines for numeric and categorical features
    # The target variable is always treated as the first numeric feature
    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler(feature_range=(0, 1)))])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, [target_key] + numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Fit the preprocessor and transform the data
    scaled_series = preprocessor.fit_transform(df)
    if isinstance(scaled_series, csr_matrix): # OneHotEncoder can return sparse matrix
        scaled_series = scaled_series.toarray()

    n_features_out = scaled_series.shape[1]

    # Split into train and test sets
    train_data = scaled_series[train_range]
    
    # Create dataset for training
    trainX, trainY = _create_dataset(train_data, look_back)
    
    # Build and train the LSTM model
    model = Sequential()
    model.add(KerasLSTM(4, input_shape=(look_back, n_features_out)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=0)

    # Make predictions for evaluation
    if not list(eval_range):
        return {}, model, preprocessor, [], [], 0.0, pd.DataFrame()

    eval_input_start = eval_range.start - look_back
    if eval_input_start < 0:
        raise ValueError("Not enough data before evaluation range to form input.")
        
    eval_input_data = scaled_series[eval_input_start : eval_range.stop]
    evalX, _ = _create_dataset(eval_input_data, look_back)

    preds_scaled = model.predict(evalX, verbose=0)
    
    # Inverse transform predictions
    target_scaler = preprocessor.named_transformers_['numeric'].named_steps['scaler']
    dummy_preds = np.zeros((len(preds_scaled), len(numeric_features) + 1))
    dummy_preds[:, 0] = preds_scaled.flatten()
    y_pred = target_scaler.inverse_transform(dummy_preds)[:, 0]
    
    y_true = df[target_key].values[eval_range]

    residuals = y_true - y_pred
    std_dev_residuals = float(np.std(residuals))

    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
    
    history_df_for_saving = df.iloc[train_range.stop - look_back : train_range.stop]
    
    return metrics, model, preprocessor, y_true.tolist(), y_pred.tolist(), std_dev_residuals, history_df_for_saving


# ---------- public entry ----------
def run(args) -> None:
    """
    Main entry point for LSTM model training and prediction.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if load_path:
        state = _load_state(load_path)
        steps = int(getattr(args, "forecast_steps", 0))
        if steps <= 0:
            raise ValueError("[VALIDATION] --forecast_steps must be > 0.")

        preds, std_dev = predict(load_path, steps, state=state)
        predictions_with_std = [{"prediction": p, "std_dev": std_dev} for p in preds]

        out: Dict[str, Any] = {
            "mode": "lstm_loaded_predict",
            "look_back": state["look_back"],
            "train_end_index": state["train_end_index"],
            "target_key": state.get("target_key"),
            "feature_keys": state.get("feature_keys"),
            "forecast_steps": steps,
            "predictions": predictions_with_std,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # Training Mode
    if not hasattr(args, "data") or not args.data:
        raise ValueError("[VALIDATION] --data is required for training.")

    target_key = getattr(args, "lstm_target_feature", None)
    feature_keys_arg = getattr(args, "lstm_features", None)
    
    try:
        feature_types_str = getattr(args, "feature_types", "{}")
        feature_types = json.loads(feature_types_str)
    except json.JSONDecodeError:
        raise ValueError("[VALIDATION] --feature_types is not a valid JSON string.")

    try:
        parsed_data = load_data_from_json(
            args.data, target_key=target_key, feature_keys=feature_keys_arg
        )
        months = parsed_data.timestamps or [str(i) for i in range(parsed_data.n_obs)]
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"[DATA] Failed to parse --data: {e}")

    # If feature_types is empty, infer from dataframe dtypes
    if not feature_types:
        for col in parsed_data.df.columns:
            if col == parsed_data.target_key or col == parsed_data.time_key:
                continue
            if pd.api.types.is_numeric_dtype(parsed_data.df[col]):
                feature_types[col] = 'numeric'
            else:
                feature_types[col] = 'categorical'
    
    # Add target key to feature_types if not present
    if parsed_data.target_key not in feature_types:
        feature_types[parsed_data.target_key] = 'numeric'


    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)
    look_back = int(getattr(args, "look_back", getattr(args, "lookback", 10)))
    epochs = int(args.epochs)

    _validate_training_args(parsed_data, ts, te, es, ee, look_back)

    metrics, model, preprocessor, y_true, y_pred, std_dev_residuals, history_df = train_and_evaluate(
        parsed_data, feature_types, range(ts, te + 1), range(es, ee + 1), look_back, epochs
    )
    
    predictions_with_std = [{"prediction": p, "std_dev": std_dev_residuals} for p in y_pred]

    out: Dict[str, Any] = {
        "mode": "lstm_train_eval",
        "data_points": parsed_data.n_obs,
        "target_key": parsed_data.target_key,
        "feature_types": feature_types,
        "look_back": look_back,
        "epochs": epochs,
        "train_range": {"start_index": ts, "end_index": te},
        "evaluate_range": {"start_index": es, "end_index": ee},
        "metrics": metrics,
        "eval_y_true": y_true,
        "predictions": predictions_with_std,
    }

    if save_path:
        keras_model_path = os.path.splitext(save_path)[0] + "_keras_model.h5"
        model.save(keras_model_path)
        
        state = {
            "model_type": MODEL_TYPE,
            "look_back": look_back,
            "epochs": epochs,
            "train_end_index": te,
            "preprocessor": preprocessor,
            "history_df": history_df,
            "target_key": parsed_data.target_key,
            "feature_keys": parsed_data.feature_keys,
            "keras_model_path": keras_model_path,
            "std_dev_residuals": std_dev_residuals,
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))


def _validate_training_args(
    parsed_data: ParsedData, ts: int, te: int, es: int, ee: int, look_back: int
) -> None:
    n = parsed_data.n_obs
    if not (0 <= ts <= te < n):
        raise ValueError(f"Invalid train range: [{ts},{te}]")
    if not (es == te + 1 and ee == te):
        if not (0 <= es <= ee < n):
            raise ValueError(f"Invalid eval range: [{es},{ee}]")
    if es <= te:
        raise ValueError("eval_start must be > train_end.")
    if look_back < 1:
        raise ValueError("--look_back must be >= 1.")
    if te - ts + 1 <= look_back:
        raise ValueError("Training range must be larger than look_back.")