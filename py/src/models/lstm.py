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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Import shared utilities
from data_utils import ParsedData, load_data_from_json
from logger import log, get_log_contents
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model

MODEL_TYPE = "lstm"

# Type alias for input flexibility
ArrayLike = Union[List[float], np.ndarray]


# ---------- utilities ----------
def _format_predictions(preds: List[float], std_dev: float) -> List[Dict[str, float]]:
    """
    Format predictions with standard deviation for JSON output.
    """
    return [{"prediction": float(p), "std_dev": float(std_dev)} for p in preds]


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
    
    # Ensure normalization is loaded for predict function
    if "normalization" not in state:
        state["normalization"] = 'minmax' # Default to minmax if not found (for older models)

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


def _estimate_std_dev(residuals: ArrayLike, reference_series: ArrayLike) -> float:
    residuals_array = np.asarray(residuals, dtype=float)
    if residuals_array.size > 1:
        return float(np.std(residuals_array, ddof=1))

    reference_array = np.asarray(reference_series, dtype=float)
    if reference_array.size > 1:
        return float(np.std(reference_array, ddof=1))

    baseline = abs(float(np.mean(reference_array))) if reference_array.size else 0.0
    return float(max(baseline * 0.05, 1.0))


def _forecast_from_state(state: Dict[str, Any], steps: int) -> Tuple[List[float], float]:
    keras_model = state["keras_model"]
    preprocessor = state["preprocessor"]
    look_back = state["look_back"]
    history_df = state["history_df"]
    std_dev_residuals = float(state.get("std_dev_residuals", 0.0))
    normalization = state.get("normalization", 'minmax')

    log.info(f"Generating {steps}-step forecast with look_back={look_back}, normalization='{normalization}'")

    scaled_history = preprocessor.transform(history_df)
    if isinstance(scaled_history, csr_matrix):
        scaled_history = scaled_history.toarray()
    n_features_out = scaled_history.shape[1]
    log.debug(f"Preprocessed history: {len(history_df)} rows, {n_features_out} features after transformation")

    input_seq = scaled_history[-look_back:].reshape(1, look_back, n_features_out)

    target_scaler = preprocessor.named_transformers_['numeric'].named_steps['scaler']
    target_column_indices = preprocessor.output_indices_['numeric']
    if isinstance(target_column_indices, slice):
        target_column_index = target_column_indices.start
    else:
        target_column_index = target_column_indices[0]

    predictions_scaled = []
    for step_idx in range(steps):
        pred_scaled = keras_model.predict(input_seq, verbose=0)
        predictions_scaled.append(pred_scaled[0, 0])
        log.debug(f"  Step {step_idx+1}/{steps}: predicted scaled value = {pred_scaled[0, 0]:.4f}")

        next_input_step_scaled = np.zeros((1, 1, n_features_out))
        next_input_step_scaled[0, 0, target_column_index] = pred_scaled[0, 0]

        last_known_scaled_features = input_seq[0, -1, :]
        for i in range(n_features_out):
            if i != target_column_index:
                next_input_step_scaled[0, 0, i] = last_known_scaled_features[i]

        input_seq = np.append(input_seq[:, 1:, :], next_input_step_scaled, axis=1)

    n_numeric_features = target_scaler.n_features_in_
    dummy_array = np.zeros((len(predictions_scaled), n_numeric_features))
    dummy_array[:, 0] = predictions_scaled
    unscaled_predictions = target_scaler.inverse_transform(dummy_array)[:, 0]

    log.info(f"Generated {len(unscaled_predictions)} predictions after inverse transform")

    return unscaled_predictions.flatten().tolist(), std_dev_residuals


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
    log.debug(f"LSTM predict called with steps={steps}, model_path={model_path}")
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
        log.debug(f"Loaded Keras model from {keras_model_path}")

    return _forecast_from_state(state, steps)


def _infer_feature_types(parsed_data: ParsedData) -> Dict[str, str]:
    feature_types: Dict[str, str] = {}
    time_key = parsed_data.time_key or ''

    for col in parsed_data.df.columns:
        if col == parsed_data.target_key or col == time_key:
            continue
        if pd.api.types.is_numeric_dtype(parsed_data.df[col]):
            feature_types[col] = 'numeric'
        else:
            feature_types[col] = 'categorical'

    feature_types[parsed_data.target_key] = 'numeric'
    return feature_types


def _fit_model(
    parsed_data: ParsedData,
    feature_types: Dict[str, str],
    train_range: range,
    look_back: int,
    epochs: int,
    normalization: str,
):
    from tensorflow.keras.layers import LSTM as KerasLSTM
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import LearningRateScheduler

    df = parsed_data.df
    target_key = parsed_data.target_key
    numeric_features = [k for k, v in feature_types.items() if v == 'numeric' and k != target_key]
    categorical_features = [k for k, v in feature_types.items() if v == 'categorical']

    log.debug(f"Target: '{target_key}', Numeric features: {numeric_features}, Categorical features: {categorical_features}")

    if normalization == 'zscore':
        scaler = StandardScaler()
        log.info("Using StandardScaler (z-score normalization)")
    elif normalization == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
        log.info("Using MinMaxScaler (0-1 normalization)")
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    numeric_transformer = Pipeline(steps=[('scaler', scaler)])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, [target_key] + numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    log.info("Fitting preprocessor on TRAINING data only...")
    train_df = df.iloc[train_range]
    preprocessor.fit(train_df)

    scaled_series = preprocessor.transform(df)
    if isinstance(scaled_series, csr_matrix):
        scaled_series = scaled_series.toarray()

    n_features_out = scaled_series.shape[1]
    log.info(f"Preprocessed data shape: {scaled_series.shape} (features after encoding: {n_features_out})")

    train_data = scaled_series[train_range]
    log.info(f"Creating LSTM dataset with look_back={look_back}...")
    trainX, trainY = _create_dataset(train_data, look_back)
    log.info(f"Training dataset created: X shape={trainX.shape}, Y shape={trainY.shape}")

    log.info(f"Building LSTM model: 3 layers, 288 units each, input_shape=({look_back}, {n_features_out})")
    model = Sequential()
    model.add(KerasLSTM(288, input_shape=(look_back, n_features_out), return_sequences=True))
    model.add(KerasLSTM(288, return_sequences=True))
    model.add(KerasLSTM(288))
    model.add(Dense(1))

    initial_learning_rate = 0.01
    decay_rate = 0.99
    optimizer = Adam(learning_rate=initial_learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    def lr_schedule(epoch):
        return initial_learning_rate * (decay_rate ** epoch)

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)

    log.info(f"Training LSTM model for {epochs} epochs with batch_size=6...")
    log.info(f"Initial learning rate: {initial_learning_rate}, decay rate: {decay_rate}")
    model.fit(trainX, trainY, epochs=epochs, batch_size=6, verbose=0, callbacks=[lr_scheduler])
    log.info("Model training completed")

    history_df_for_saving = df.iloc[train_range.stop - look_back : train_range.stop].copy()
    log.debug(f"Saved history for future predictions: {len(history_df_for_saving)} rows")

    return model, preprocessor, history_df_for_saving


# ---------- train and evaluate ----------
def train_and_evaluate(
    parsed_data: ParsedData,
    feature_types: Dict[str, str],
    train_range: range,
    eval_range: range,
    look_back: int,
    epochs: int,
    normalization: str = 'minmax',
    timestamps: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Any, Any, List[float], List[float], float, pd.DataFrame]:
    """
    Trains and evaluates the LSTM model with typed features.
    """
    log.info(f"Starting LSTM train_and_evaluate: look_back={look_back}, epochs={epochs}, normalization='{normalization}', train_range={train_range}, eval_range={eval_range}")

    df = parsed_data.df
    target_key = parsed_data.target_key
    train_target_values = df[target_key].values[train_range]

    log.debug(f"Total series length: {len(df)}")
    log.debug(f"Training data range: [{train_range.start}, {train_range.stop-1}], count={train_range.stop - train_range.start}")
    log.debug(f"Evaluation data range: [{eval_range.start}, {eval_range.stop-1}], count={eval_range.stop - eval_range.start}")

    model, preprocessor, history_df_for_saving = _fit_model(
        parsed_data,
        feature_types,
        train_range,
        look_back,
        epochs,
        normalization,
    )

    # Make predictions for evaluation
    if not list(eval_range):
        log.warning("Empty evaluation range, returning empty metrics")
        return {}, model, preprocessor, [], [], _estimate_std_dev([], train_target_values), history_df_for_saving

    log.info(f"Generating predictions for evaluation range...")
    steps_to_eval_end = eval_range.stop - train_range.stop
    forecast_offset = eval_range.start - train_range.stop
    state_for_eval = {
        "keras_model": model,
        "preprocessor": preprocessor,
        "look_back": look_back,
        "history_df": history_df_for_saving,
        "normalization": normalization,
    }
    full_forecast, _ = _forecast_from_state(state_for_eval, steps_to_eval_end)
    y_pred = full_forecast[forecast_offset: forecast_offset + len(eval_range)]
    if len(y_pred) != len(eval_range):
        raise ValueError("[VALIDATION] 无法为完整评估区间生成 LSTM 预测。")

    y_true = df[target_key].values[eval_range]
    log.info(f"Generated {len(y_pred)} predictions after inverse transform")

    residuals = y_true - y_pred
    std_dev_residuals = _estimate_std_dev(residuals, train_target_values)
    log.info(f"Calculated residuals: mean={np.mean(residuals):.4f}, std_dev={std_dev_residuals:.4f}")

    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "look_back": look_back,
        "epochs": epochs,
        "evaluated_points": len(y_true),
        "evaluate_indices": list(eval_range),
        "evaluate_months": [timestamps[i] for i in eval_range] if timestamps else [],
    }

    log.info(f"Evaluation metrics computed: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}, R²={metrics['r2']:.4f}")

    return metrics, model, preprocessor, y_true.tolist(), list(y_pred), std_dev_residuals, history_df_for_saving


# ---------- public entry ----------
def run(args) -> None:
    """
    Main entry point for LSTM model training and prediction.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    log.info(f"LSTM run started with args: {vars(args)}")

    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if getattr(args, "fit_only", False):
        if not hasattr(args, "data") or not args.data:
            raise ValueError("[VALIDATION] --data is required for fit-only mode.")
        if not save_path:
            raise ValueError("[VALIDATION] --save-model is required for fit-only mode.")
        if not load_path:
            raise ValueError("[VALIDATION] --load-model is required for fit-only mode.")

        state = _load_state(load_path)
        parsed_data = load_data_from_json(
            args.data,
            target_key=state.get("target_key"),
            feature_keys=state.get("feature_keys"),
        )

        ts, te = int(args.train_start), int(args.train_end)
        look_back = int(state["look_back"])
        epochs = int(state.get("epochs", 20))
        normalization = state.get("normalization", 'minmax')
        _validate_fit_args(parsed_data, ts, te, look_back)

        feature_types = _infer_feature_types(parsed_data)
        model, preprocessor, history_df = _fit_model(
            parsed_data,
            feature_types,
            range(ts, te + 1),
            look_back,
            epochs,
            normalization,
        )

        keras_model_path = os.path.splitext(save_path)[0] + "_keras_model.h5"
        temp_keras_model_path = os.path.splitext(save_path)[0] + f"_keras_model.tmp-{os.getpid()}.h5"
        try:
            model.save(temp_keras_model_path)
            os.replace(temp_keras_model_path, keras_model_path)
        finally:
            if os.path.exists(temp_keras_model_path):
                try:
                    os.remove(temp_keras_model_path)
                except OSError:
                    pass

        std_dev_residuals = float(
            state.get(
                "std_dev_residuals",
                _estimate_std_dev([], parsed_data.df[parsed_data.target_key].values[ts : te + 1]),
            )
        )
        next_state = {
            "model_type": MODEL_TYPE,
            "look_back": look_back,
            "epochs": epochs,
            "normalization": normalization,
            "train_end_index": te,
            "preprocessor": preprocessor,
            "history_df": history_df,
            "target_key": parsed_data.target_key,
            "feature_keys": parsed_data.feature_keys,
            "keras_model_path": keras_model_path,
            "std_dev_residuals": std_dev_residuals,
        }
        dump_model(next_state, save_path)

        out: Dict[str, Any] = {
            "mode": "lstm_fit_only",
            "look_back": look_back,
            "epochs": epochs,
            "normalization": normalization,
            "train_end_index": te,
            "target_key": parsed_data.target_key,
            "feature_keys": parsed_data.feature_keys,
            "saved_model": save_path,
        }
        out["logs"] = get_log_contents()
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    if load_path:
        log.info(f"Entering prediction mode with model: {load_path}")
        state = _load_state(load_path)
        steps = int(getattr(args, "forecast_steps", 0))
        if steps <= 0:
            log.error("forecast_steps must be > 0 for prediction mode.")
            raise ValueError("[VALIDATION] --forecast_steps must be > 0.")

        log.info(f"Starting prediction with {steps} forecast steps")
        preds, std_dev = predict(load_path, steps, state=state)
        predictions_with_std = _format_predictions(preds, std_dev)

        out: Dict[str, Any] = {
            "mode": "lstm_loaded_predict",
            "look_back": state["look_back"],
            "train_end_index": state["train_end_index"],
            "target_key": state.get("target_key"),
            "feature_keys": state.get("feature_keys"),
            "forecast_steps": steps,
            "predictions": predictions_with_std,
        }
        log.info("Prediction finished successfully")

        # Add captured logs to the final output
        out["logs"] = get_log_contents()

        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # Training Mode
    log.info("Entering training mode.")
    if not hasattr(args, "data") or not args.data:
        log.error("--data is required for training.")
        raise ValueError("[VALIDATION] --data is required for training.")

    target_key = getattr(args, "lstm_target_feature", None)
    feature_keys_str = getattr(args, "lstm_features", None)
    feature_keys_list = [k.strip() for k in feature_keys_str.split(',')] if feature_keys_str else None
    log.debug(f"Received target_key: {target_key}, feature_keys_str: {feature_keys_str}, parsed_list: {feature_keys_list}")

    try:
        feature_types_str = getattr(args, "feature_types", None) or "{}"
        feature_types = json.loads(feature_types_str)
        log.debug(f"Received feature_types: {feature_types}")

        parsed_data = load_data_from_json(
            args.data, target_key=target_key, feature_keys=feature_keys_list
        )
        months = parsed_data.timestamps or [str(i) for i in range(parsed_data.n_obs)]

        # If feature_types is empty, infer from dataframe dtypes
        if not feature_types:
            log.info("No feature_types provided, inferring from DataFrame.")
            feature_types = _infer_feature_types(parsed_data)
            log.info(f"Inferred feature_types: {feature_types}")

        if parsed_data.target_key not in feature_types:
            feature_types[parsed_data.target_key] = 'numeric'
            log.debug(f"Ensured target_key '{parsed_data.target_key}' is in feature_types.")

        ts, te = int(args.train_start), int(args.train_end)
        es, ee = int(args.eval_start), int(args.eval_end)

        look_back_arg = getattr(args, "look_back", None)
        epochs_arg = getattr(args, "epochs", None)

        # Auto-select look_back and epochs based on training data size (for educational small datasets)
        train_size = te - ts + 1
        if look_back_arg is not None:
            look_back = int(look_back_arg)
        else:
            look_back = max(3, min(12, train_size // 4))
            log.info(f"Auto-selected look_back={look_back} based on train_size={train_size}")

        if epochs_arg is not None:
            epochs = int(epochs_arg)
        else:
            epochs = 20  # Default as per client requirements
            log.info(f"Using default epochs={epochs}")

        normalization_arg = getattr(args, "normalization", None)
        normalization = normalization_arg if normalization_arg is not None else 'minmax'
        log.debug(f"Training params: look_back={look_back}, epochs={epochs}, normalization='{normalization}'")

        _validate_training_args(parsed_data, ts, te, es, ee, look_back)

        log.info("Starting model training and evaluation...")
        metrics, model, preprocessor, y_true, y_pred, std_dev_residuals, history_df = train_and_evaluate(
            parsed_data, feature_types, range(ts, te + 1), range(es, ee + 1), look_back, epochs, normalization, months
        )
        log.info(f"Training finished. Metrics: {metrics}")

        predictions_with_std = _format_predictions(y_pred, std_dev_residuals)

        out: Dict[str, Any] = {
            "mode": "lstm_train_eval",
            "data_points": parsed_data.n_obs,
            "target_key": parsed_data.target_key,
            "feature_types": feature_types,
            "look_back": look_back,
            "epochs": epochs,
            "normalization": normalization,
            "train_range": {"start_index": ts, "end_index": te},
            "evaluate_range": {"start_index": es, "end_index": ee},
            "metrics": metrics,
            "eval_y_true": y_true,
            "predictions": predictions_with_std,
        }

        if save_path:
            log.info(f"Saving model to {save_path}")
            keras_model_path = os.path.splitext(save_path)[0] + "_keras_model.h5"
            temp_keras_model_path = os.path.splitext(save_path)[0] + f"_keras_model.tmp-{os.getpid()}.h5"
            log.debug(f"Saving Keras model to temporary path {temp_keras_model_path}")
            try:
                model.save(temp_keras_model_path)
                os.replace(temp_keras_model_path, keras_model_path)
            finally:
                if os.path.exists(temp_keras_model_path):
                    try:
                        os.remove(temp_keras_model_path)
                    except OSError:
                        pass

            state = {
                "model_type": MODEL_TYPE,
                "look_back": look_back,
                "epochs": epochs,
                "normalization": normalization,
                "train_end_index": te,
                "preprocessor": preprocessor,
                "history_df": history_df,
                "target_key": parsed_data.target_key,
                "feature_keys": parsed_data.feature_keys,
                "keras_model_path": keras_model_path,
                "std_dev_residuals": std_dev_residuals,
            }
            dump_model(state, save_path)
            log.info(f"Model saved successfully to {save_path} (Keras: {keras_model_path})")
            out["saved_model"] = save_path

        # Add captured logs to the final output
        out["logs"] = get_log_contents()

        log.info("Run finished successfully.")
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))

    except (ValueError, TypeError, json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        log.error(f"An error occurred during data processing or training: {e}", exc_info=True)
        raise ValueError(f"[DATA] Failed to parse or process --data: {e}")


def _validate_fit_args(
    parsed_data: ParsedData, ts: int, te: int, look_back: int
) -> None:
    n = parsed_data.n_obs

    if not (0 <= ts <= te < n):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], must be within [0, {n-1}]")
    if look_back < 1:
        raise ValueError("[VALIDATION] look_back must be >= 1.")
    if te - ts + 1 <= look_back:
        raise ValueError("[VALIDATION] Training range must be larger than look_back.")


def _validate_training_args(
    parsed_data: ParsedData, ts: int, te: int, es: int, ee: int, look_back: int
) -> None:
    """
    Validate training and evaluation parameters for LSTM.
    """
    log.debug(f"Validating training arguments: ts={ts}, te={te}, es={es}, ee={ee}, look_back={look_back}")
    n = parsed_data.n_obs
    log.debug(f"Total data points: {n}")

    if not (0 <= ts <= te < n):
        log.error(f"Invalid training range: [{ts}, {te}], data size: {n}")
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], must be within [0, {n-1}]")
    if not (0 <= es <= ee < n):
        log.error(f"Invalid evaluation range: [{es}, {ee}], data size: {n}")
        raise ValueError(f"[VALIDATION] Invalid eval range: [{es},{ee}], must be within [0, {n-1}]")
    if es <= te:
        log.error(f"Evaluation start {es} must be after training end {te}")
        raise ValueError(f"[VALIDATION] Evaluation start ({es}) must be after training end ({te}).")
    if look_back < 1:
        log.error(f"look_back {look_back} must be >= 1")
        raise ValueError("[VALIDATION] look_back must be >= 1.")
    if te - ts + 1 <= look_back:
        log.error(f"Training range size {te - ts + 1} must be larger than look_back {look_back}")
        raise ValueError("[VALIDATION] Training range must be larger than look_back.")

    # Additional helpful validations
    train_size = te - ts + 1
    eval_size = ee - es + 1
    log.info(f"Validation passed: train_size={train_size}, eval_size={eval_size}, look_back={look_back}")
