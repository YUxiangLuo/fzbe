# py/src/models/lstm.py
# -*- coding: utf-8 -*-
"""
LSTM (Long Short-Term Memory) model for time series forecasting.
"""
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Import shared utilities
from data_utils import ParsedData, infer_feature_types as infer_parsed_feature_types, load_data_from_json
from durable_io import fsync_file, fsync_parent_directory
from logger import log, get_log_contents
from metrics import mae, mape, r2, rmse
from model_defaults import resolve_lstm_look_back
from model_info import attach_method_metadata, ensure_state_metadata
from model_validation import validate_minimum_evaluation_points
from serialization import dump_model, load_model

MODEL_TYPE = "lstm"
DIRECT_FORECAST_STRATEGY = "direct_multi_horizon"
LEGACY_FORECAST_STRATEGY = "autoregressive"
DEFAULT_PRODUCTION_FORECAST_HORIZON = 6
METHOD_NAME = "LSTM 多变量销量预测"
DIRECT_IMPLEMENTATION_NOTES = [
    "教科书 LSTM 定义的是带遗忘门、输入门和输出门的循环神经网络单元。",
    "本系统采用单层 LSTM + Dropout + L2 正则 + Dense 输出层，隐藏单元数随样本量自适应，以在小样本上抑制过拟合。",
    "为了服务未来多期销量预测，默认直接多步输出完整预测 horizon，减少递归预测的误差累积。",
    "训练中的 EarlyStopping 监控训练损失而非验证损失（教学数据过小，不再切分验证集），其作用是训练损失进入平台期即停止，并非教科书意义的验证集早停。",
    "预测值附带的 std_dev 是至少两个评估残差计算出的样本标准差（ddof=1），对各预测期保持常数；ARIMA 的区间会随步长展宽。",
]
AUTOREGRESSIVE_IMPLEMENTATION_NOTES = [
    "教科书 LSTM 定义的是带遗忘门、输入门和输出门的循环神经网络单元。",
    "本系统采用单层 LSTM + Dropout + L2 正则 + Dense 输出层，隐藏单元数随样本量自适应，以在小样本上抑制过拟合。",
    "当训练数据不足以拟合直接多步输出时，系统回退为一步 LSTM 并递归预测未来多期。",
    "训练中的 EarlyStopping 监控训练损失而非验证损失（教学数据过小，不再切分验证集），其作用是训练损失进入平台期即停止，并非教科书意义的验证集早停。",
    "预测值附带的 std_dev 是至少两个评估残差计算出的样本标准差（ddof=1），对各预测期保持常数；ARIMA 的区间会随步长展宽。",
]

# Type alias for input flexibility
ArrayLike = Union[List[float], np.ndarray]


# ---------- shared LSTM v2 architecture (single source of truth) ----------
# Both the standalone/ensemble path (this module) and the guided-training path
# (guided_model.py) build the LSTM through these helpers, so the architecture and
# hyper-parameters can never diverge between the two entry points again.
LSTM_DROPOUT_RATE = 0.2
LSTM_L2_REG = 1e-4
LSTM_INITIAL_LEARNING_RATE = 0.001
LSTM_LR_DECAY_RATE = 0.98


def select_lstm_units(train_sample_count: int) -> Tuple[int, int]:
    """Pick (recurrent_units, dense_units) scaled to the training-sample count.

    Keep enough capacity for direct multi-horizon learning while bounding model
    size on larger teaching datasets.  The direct dataset contains fewer samples
    than the original series, so scaling from that count needs a wider factor
    than a one-step model would use.
    """
    recurrent_units = int(max(16, min(80, train_sample_count * 5)))
    dense_units = int(max(8, recurrent_units // 2))
    return recurrent_units, dense_units


def build_lstm_network(
    look_back: int,
    n_features_out: int,
    recurrent_units: int,
    dense_units: int,
    output_units: int,
    learning_rate: float = LSTM_INITIAL_LEARNING_RATE,
):
    """Build and compile the canonical v2 LSTM network.

    Single LSTM layer (L2 on input & recurrent weights) + Dropout + ReLU Dense +
    linear output. ``output_units`` is the forecast horizon for direct multi-step
    models and 1 for autoregressive models.
    """
    from tensorflow.keras.layers import LSTM as KerasLSTM
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    model = Sequential([
        Input(shape=(int(look_back), int(n_features_out))),
        KerasLSTM(
            int(recurrent_units),
            kernel_regularizer=l2(LSTM_L2_REG),
            recurrent_regularizer=l2(LSTM_L2_REG),
        ),
        Dropout(LSTM_DROPOUT_RATE),
        Dense(int(dense_units), activation="relu"),
        Dense(int(output_units)),
    ])
    optimizer = Adam(learning_rate=float(learning_rate), clipnorm=1.0)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    return model


# ---------- utilities ----------
def _metadata_for_strategy(strategy: str) -> Tuple[str, List[str]]:
    if strategy == DIRECT_FORECAST_STRATEGY:
        return DIRECT_FORECAST_STRATEGY, DIRECT_IMPLEMENTATION_NOTES
    return LEGACY_FORECAST_STRATEGY, AUTOREGRESSIVE_IMPLEMENTATION_NOTES


def _format_predictions(preds: List[float], std_dev: float) -> List[Dict[str, float]]:
    """
    Format predictions with standard deviation for JSON output.
    """
    return [{"prediction": float(p), "std_dev": float(std_dev)} for p in preds]


def _parse_feature_key_list(raw_value: Optional[str]) -> Optional[List[str]]:
    if raw_value is None or raw_value == "":
        return None

    try:
        parsed = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [item.strip() for item in str(raw_value).split(',') if item.strip()]


def _summarize_args_for_log(args) -> Dict[str, Any]:
    summarized = dict(vars(args))
    data = summarized.get("data")
    if isinstance(data, str):
        summarized["data"] = f"<json payload chars={len(data)}>"
    return summarized


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
    state["keras_model"] = keras_load_model(keras_model_path, compile=False)
    
    # Ensure normalization is loaded for predict function
    if "normalization" not in state:
        state["normalization"] = 'minmax' # Default to minmax if not found (for older models)
    if "forecast_strategy" not in state:
        state["forecast_strategy"] = (
            DIRECT_FORECAST_STRATEGY
            if int(state.get("forecast_horizon", 0) or 0) > 0
            else LEGACY_FORECAST_STRATEGY
        )
    forecast_strategy, notes = _metadata_for_strategy(str(state["forecast_strategy"]))
    ensure_state_metadata(state, METHOD_NAME, forecast_strategy, notes)

    return state


def _create_autoregressive_dataset(
    dataset: np.ndarray,
    look_back: int = 1,
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


def _create_dataset(
    dataset: np.ndarray,
    look_back: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible alias for legacy callers that expect one-step targets.
    """
    return _create_autoregressive_dataset(dataset, look_back)


def _create_direct_dataset(
    dataset: np.ndarray,
    look_back: int,
    forecast_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create multi-horizon training samples.

    X uses the past `look_back` rows of the full multivariate history.
    Y is the next `forecast_horizon` target values.
    """
    if forecast_horizon < 1:
        raise ValueError("[VALIDATION] forecast_horizon must be >= 1.")

    dataX, dataY = [], []
    max_start = len(dataset) - look_back - forecast_horizon + 1
    for start_idx in range(max_start):
        end_history = start_idx + look_back
        end_target = end_history + forecast_horizon
        dataX.append(dataset[start_idx:end_history, :])
        dataY.append(dataset[end_history:end_target, 0])
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


def _set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _resolve_production_forecast_horizon(
    state: Optional[Dict[str, Any]] = None,
    eval_forecast_horizon: int = 0,
) -> int:
    configured_horizon = DEFAULT_PRODUCTION_FORECAST_HORIZON
    if state is not None:
        configured_horizon = int(
            state.get(
                "production_forecast_horizon",
                state.get("forecast_horizon", DEFAULT_PRODUCTION_FORECAST_HORIZON),
            )
            or DEFAULT_PRODUCTION_FORECAST_HORIZON
        )
    return max(int(eval_forecast_horizon or 0), configured_horizon)


def _can_fit_direct_forecast(train_size: int, look_back: int, forecast_horizon: int) -> bool:
    return forecast_horizon > 0 and train_size >= look_back + forecast_horizon


def _build_preprocessor(
    parsed_data: ParsedData,
    feature_types: Dict[str, str],
    train_range: range,
    normalization: str,
) -> Tuple[Any, np.ndarray]:
    df = parsed_data.df
    target_key = parsed_data.target_key
    feature_order = [key for key in parsed_data.feature_keys if key in df.columns and key != target_key]
    numeric_features = [key for key in feature_order if feature_types.get(key) == "numeric"]
    categorical_features = [key for key in feature_order if feature_types.get(key) == "categorical"]

    log.debug(
        "Target: '%s', Numeric features: %s, Categorical features: %s",
        target_key,
        numeric_features,
        categorical_features,
    )

    if normalization == 'zscore':
        scaler = StandardScaler()
        log.info("Using StandardScaler (z-score normalization)")
    elif normalization == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
        log.info("Using MinMaxScaler (0-1 normalization)")
    else:
        raise ValueError(f"[VALIDATION] Unknown normalization method: {normalization}")

    numeric_transformer = Pipeline(steps=[('scaler', scaler)])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, [target_key] + numeric_features),
            ('categorical', categorical_transformer, categorical_features),
        ],
        remainder='drop',
    )

    log.info("Fitting preprocessor on TRAINING data only...")
    train_df = df.iloc[train_range]
    preprocessor.fit(train_df)

    scaled_series = preprocessor.transform(df)
    if isinstance(scaled_series, csr_matrix):
        scaled_series = scaled_series.toarray()

    log.info(
        "Preprocessed data shape: %s (features after encoding: %s)",
        scaled_series.shape,
        scaled_series.shape[1],
    )
    return preprocessor, np.asarray(scaled_series, dtype=float)


def _validate_lstm_feature_types(parsed_data: ParsedData, feature_types: Dict[str, str]) -> None:
    valid_types = {"numeric", "categorical"}
    invalid_items = [
        f"{feature_name}={feature_type!r}"
        for feature_name, feature_type in feature_types.items()
        if feature_type not in valid_types
    ]
    if invalid_items:
        raise ValueError(
            "[VALIDATION] LSTM feature_types values must be 'numeric' or 'categorical': "
            + ", ".join(invalid_items)
        )

    target_type = feature_types.get(parsed_data.target_key, "numeric")
    if target_type != "numeric":
        raise ValueError("[VALIDATION] LSTM target feature must be numeric.")
    feature_types[parsed_data.target_key] = "numeric"


def _validate_lstm_numeric_columns(parsed_data: ParsedData, feature_types: Dict[str, str]) -> None:
    """
    Validate and normalize numeric columns before they enter sklearn/Keras.

    MinMaxScaler and StandardScaler preserve NaN values, and Keras can then train
    a model that emits NaN predictions. Rejecting invalid numeric data here keeps
    CLI output valid JSON and gives the caller a data-quality error instead.
    """
    numeric_columns = [parsed_data.target_key]
    numeric_columns.extend(
        feature_name
        for feature_name in parsed_data.feature_keys
        if feature_name != parsed_data.target_key and feature_types.get(feature_name) == "numeric"
    )

    invalid_columns: List[str] = []
    for column_name in numeric_columns:
        if column_name not in parsed_data.df.columns:
            continue

        numeric_values = pd.to_numeric(parsed_data.df[column_name], errors="coerce")
        numeric_array = numeric_values.to_numpy(dtype=float)
        invalid_mask = numeric_values.isna().to_numpy() | ~np.isfinite(numeric_array)
        if invalid_mask.any():
            invalid_rows = np.flatnonzero(invalid_mask)[:5]
            row_preview = ",".join(str(int(row_index)) for row_index in invalid_rows)
            invalid_columns.append(f"{column_name}(rows={row_preview})")
            continue

        parsed_data.df[column_name] = numeric_values

    if invalid_columns:
        raise ValueError(
            "[DATA] LSTM numeric columns contain missing, non-numeric, or non-finite values: "
            + "; ".join(invalid_columns)
        )


def _inverse_transform_target_values(preprocessor: Any, scaled_values: np.ndarray) -> np.ndarray:
    target_scaler = preprocessor.named_transformers_['numeric'].named_steps['scaler']
    flat_values = np.asarray(scaled_values, dtype=float).reshape(-1)
    n_numeric_features = target_scaler.n_features_in_
    dummy_array = np.zeros((len(flat_values), n_numeric_features))
    dummy_array[:, 0] = flat_values
    return target_scaler.inverse_transform(dummy_array)[:, 0]


def _forecast_direct_from_state(state: Dict[str, Any], steps: int) -> Tuple[List[float], float]:
    keras_model = state["keras_model"]
    preprocessor = state["preprocessor"]
    look_back = state["look_back"]
    history_df = state["history_df"]
    std_dev_residuals = float(state.get("std_dev_residuals", 0.0))
    forecast_horizon = int(state.get("forecast_horizon", 0))

    if forecast_horizon < 1:
        raise ValueError("[VALIDATION] Saved LSTM model is missing forecast_horizon.")
    if steps > forecast_horizon:
        raise ValueError(
            f"[VALIDATION] Requested forecast_steps ({steps}) exceeds the saved LSTM horizon ({forecast_horizon})."
        )

    log.info(
        "Generating %s-step direct forecast with look_back=%s, saved_horizon=%s",
        steps,
        look_back,
        forecast_horizon,
    )

    scaled_history = preprocessor.transform(history_df)
    if isinstance(scaled_history, csr_matrix):
        scaled_history = scaled_history.toarray()
    n_features_out = scaled_history.shape[1]
    if len(history_df) < look_back:
        raise ValueError("[VALIDATION] Saved history is shorter than look_back.")

    log.debug(f"Preprocessed history: {len(history_df)} rows, {n_features_out} features after transformation")

    input_seq = scaled_history[-look_back:].reshape(1, look_back, n_features_out)
    predictions_scaled = keras_model.predict(input_seq, verbose=0)[0]
    unscaled_predictions = _inverse_transform_target_values(preprocessor, predictions_scaled)

    log.info(f"Generated {len(unscaled_predictions)} predictions after inverse transform")
    return unscaled_predictions[:steps].flatten().tolist(), std_dev_residuals


def _forecast_autoregressive_from_state(state: Dict[str, Any], steps: int) -> Tuple[List[float], float]:
    keras_model = state["keras_model"]
    preprocessor = state["preprocessor"]
    look_back = state["look_back"]
    history_df = state["history_df"]
    std_dev_residuals = float(state.get("std_dev_residuals", 0.0))
    normalization = state.get("normalization", 'minmax')

    log.info(f"Generating {steps}-step autoregressive forecast with look_back={look_back}, normalization='{normalization}'")

    scaled_history = preprocessor.transform(history_df)
    if isinstance(scaled_history, csr_matrix):
        scaled_history = scaled_history.toarray()
    n_features_out = scaled_history.shape[1]
    log.debug(f"Preprocessed history: {len(history_df)} rows, {n_features_out} features after transformation")

    input_seq = scaled_history[-look_back:].reshape(1, look_back, n_features_out)

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

    unscaled_predictions = _inverse_transform_target_values(preprocessor, np.asarray(predictions_scaled, dtype=float))

    log.info(f"Generated {len(unscaled_predictions)} predictions after inverse transform")

    return unscaled_predictions.flatten().tolist(), std_dev_residuals


def _forecast_from_state(state: Dict[str, Any], steps: int) -> Tuple[List[float], float]:
    strategy = state.get("forecast_strategy", LEGACY_FORECAST_STRATEGY)
    if strategy == DIRECT_FORECAST_STRATEGY:
        return _forecast_direct_from_state(state, steps)
    return _forecast_autoregressive_from_state(state, steps)


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
        state["keras_model"] = keras_load_model(keras_model_path, compile=False)
        log.debug(f"Loaded Keras model from {keras_model_path}")

    return _forecast_from_state(state, steps)


def _fit_direct_model(
    parsed_data: ParsedData,
    feature_types: Dict[str, str],
    train_range: range,
    look_back: int,
    epochs: int,
    normalization: str,
    forecast_horizon: int,
):
    from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

    df = parsed_data.df
    preprocessor, scaled_series = _build_preprocessor(parsed_data, feature_types, train_range, normalization)
    n_features_out = scaled_series.shape[1]

    train_data = scaled_series[train_range]
    log.info(
        "Creating direct LSTM dataset with look_back=%s, forecast_horizon=%s...",
        look_back,
        forecast_horizon,
    )
    trainX, trainY = _create_direct_dataset(train_data, look_back, forecast_horizon)
    if trainX.size == 0:
        raise ValueError(
            "[VALIDATION] Training range is too short for the requested LSTM look_back and forecast horizon."
        )
    log.info(f"Training dataset created: X shape={trainX.shape}, Y shape={trainY.shape}")

    recurrent_units, dense_units = select_lstm_units(len(trainX))
    log.info(
        "Building direct LSTM (v2 single-layer) model: units=%s, dense=%s, horizon=%s, input_shape=(%s, %s)",
        recurrent_units,
        dense_units,
        forecast_horizon,
        look_back,
        n_features_out,
    )
    model = build_lstm_network(look_back, n_features_out, recurrent_units, dense_units, forecast_horizon)

    initial_learning_rate = LSTM_INITIAL_LEARNING_RATE
    decay_rate = LSTM_LR_DECAY_RATE

    def lr_schedule(epoch):
        return initial_learning_rate * (decay_rate ** epoch)

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)
    early_stopping = EarlyStopping(
        monitor="loss",
        patience=max(2, min(6, epochs // 4 or 2)),
        restore_best_weights=True,
    )
    batch_size = max(1, min(8, len(trainX)))

    log.info(f"Training direct LSTM model for {epochs} epochs with batch_size={batch_size}...")
    log.info(f"Initial learning rate: {initial_learning_rate}, decay rate: {decay_rate}")
    model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[lr_scheduler, early_stopping],
        shuffle=False,
    )
    log.info("Model training completed")

    history_df_for_saving = df.iloc[train_range.stop - look_back : train_range.stop].copy()
    log.debug(f"Saved history for future predictions: {len(history_df_for_saving)} rows")

    return model, preprocessor, history_df_for_saving


def _fit_autoregressive_model(
    parsed_data: ParsedData,
    feature_types: Dict[str, str],
    train_range: range,
    look_back: int,
    epochs: int,
    normalization: str,
):
    from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

    df = parsed_data.df
    preprocessor, scaled_series = _build_preprocessor(parsed_data, feature_types, train_range, normalization)
    n_features_out = scaled_series.shape[1]

    train_data = scaled_series[train_range]
    log.info(f"Creating autoregressive LSTM dataset with look_back={look_back}...")
    trainX, trainY = _create_autoregressive_dataset(train_data, look_back)
    if trainX.size == 0:
        raise ValueError("[VALIDATION] Training range is too short for autoregressive LSTM.")
    log.info(f"Training dataset created: X shape={trainX.shape}, Y shape={trainY.shape}")

    recurrent_units, dense_units = select_lstm_units(len(trainX))
    log.info(
        "Building autoregressive LSTM (v2 single-layer) model: units=%s, dense=%s, input_shape=(%s, %s)",
        recurrent_units,
        dense_units,
        look_back,
        n_features_out,
    )
    model = build_lstm_network(look_back, n_features_out, recurrent_units, dense_units, 1)

    initial_learning_rate = LSTM_INITIAL_LEARNING_RATE
    decay_rate = LSTM_LR_DECAY_RATE

    def lr_schedule(epoch):
        return initial_learning_rate * (decay_rate ** epoch)

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)
    early_stopping = EarlyStopping(
        monitor="loss",
        patience=max(2, min(6, epochs // 4 or 2)),
        restore_best_weights=True,
    )
    batch_size = max(1, min(8, len(trainX)))

    log.info(f"Training autoregressive LSTM model for {epochs} epochs with batch_size={batch_size}...")
    model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[lr_scheduler, early_stopping],
        shuffle=False,
    )
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
    steps_to_eval_end = eval_range.stop - train_range.stop
    forecast_offset = eval_range.start - train_range.stop
    forecast_horizon = steps_to_eval_end

    log.debug(f"Total series length: {len(df)}")
    log.debug(f"Training data range: [{train_range.start}, {train_range.stop-1}], count={train_range.stop - train_range.start}")
    log.debug(f"Evaluation data range: [{eval_range.start}, {eval_range.stop-1}], count={eval_range.stop - eval_range.start}")

    model, preprocessor, history_df_for_saving = _fit_direct_model(
        parsed_data,
        feature_types,
        train_range,
        look_back,
        epochs,
        normalization,
        forecast_horizon,
    )

    # Make predictions for evaluation
    if not list(eval_range):
        log.warning("Empty evaluation range, returning empty metrics")
        return {}, model, preprocessor, [], [], _estimate_std_dev([], train_target_values), history_df_for_saving

    log.info(f"Generating predictions for evaluation range...")
    state_for_eval = {
        "keras_model": model,
        "preprocessor": preprocessor,
        "look_back": look_back,
        "forecast_horizon": forecast_horizon,
        "forecast_strategy": DIRECT_FORECAST_STRATEGY,
        "history_df": history_df_for_saving,
        "normalization": normalization,
    }
    full_forecast, _ = _forecast_from_state(state_for_eval, forecast_horizon)
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
        "forecast_horizon": forecast_horizon,
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
    tf.keras.utils.set_random_seed(42)
    _set_random_seed(42)

    log.info(f"LSTM run started with args: {_summarize_args_for_log(args)}")

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
        forecast_strategy = state.get("forecast_strategy", LEGACY_FORECAST_STRATEGY)
        train_size = te - ts + 1
        desired_forecast_horizon = (
            _resolve_production_forecast_horizon(state=state)
            if forecast_strategy == DIRECT_FORECAST_STRATEGY
            else 0
        )
        effective_forecast_strategy = forecast_strategy
        fit_forecast_horizon = desired_forecast_horizon
        if (
            forecast_strategy == DIRECT_FORECAST_STRATEGY
            and desired_forecast_horizon > 0
            and not _can_fit_direct_forecast(train_size, look_back, desired_forecast_horizon)
        ):
            raise ValueError(
                "[VALIDATION] Production LSTM cannot preserve direct multi-horizon forecasting: "
                f"train_size={train_size}, look_back={look_back}, "
                f"required_horizon={desired_forecast_horizon}."
            )

        _validate_fit_args(
            parsed_data,
            ts,
            te,
            look_back,
            fit_forecast_horizon,
            effective_forecast_strategy,
        )

        feature_types = state.get("feature_types") or infer_parsed_feature_types(
            parsed_data,
            target_key=parsed_data.target_key,
            feature_keys=parsed_data.feature_keys,
        )
        _validate_lstm_feature_types(parsed_data, feature_types)
        _validate_lstm_numeric_columns(parsed_data, feature_types)
        if effective_forecast_strategy == DIRECT_FORECAST_STRATEGY and fit_forecast_horizon > 0:
            model, preprocessor, history_df = _fit_direct_model(
                parsed_data,
                feature_types,
                range(ts, te + 1),
                look_back,
                epochs,
                normalization,
                fit_forecast_horizon,
            )
        else:
            model, preprocessor, history_df = _fit_autoregressive_model(
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
            fsync_file(temp_keras_model_path)
            os.replace(temp_keras_model_path, keras_model_path)
            fsync_parent_directory(keras_model_path)
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
            "feature_types": feature_types,
            "keras_model_path": keras_model_path,
            "forecast_horizon": (
                fit_forecast_horizon
                if effective_forecast_strategy == DIRECT_FORECAST_STRATEGY and fit_forecast_horizon > 0
                else 0
            ),
            "forecast_strategy": effective_forecast_strategy,
            "production_forecast_horizon": desired_forecast_horizon,
            "std_dev_residuals": std_dev_residuals,
        }
        output_strategy, output_notes = _metadata_for_strategy(effective_forecast_strategy)
        attach_method_metadata(next_state, METHOD_NAME, output_strategy, output_notes)
        dump_model(next_state, save_path)

        out: Dict[str, Any] = {
            "mode": "lstm_fit_only",
            "look_back": look_back,
            "epochs": epochs,
            "normalization": normalization,
            "train_end_index": te,
            "target_key": parsed_data.target_key,
            "feature_keys": parsed_data.feature_keys,
            "feature_types": feature_types,
            "saved_model": save_path,
        }
        attach_method_metadata(out, METHOD_NAME, output_strategy, output_notes)
        if fit_forecast_horizon > 0:
            out["forecast_horizon"] = fit_forecast_horizon
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
        output_strategy, output_notes = _metadata_for_strategy(str(state.get("forecast_strategy", LEGACY_FORECAST_STRATEGY)))
        attach_method_metadata(out, METHOD_NAME, output_strategy, output_notes)
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
    feature_keys_raw = getattr(args, "lstm_features_json", None) or getattr(args, "lstm_features", None)
    feature_keys_list = _parse_feature_key_list(feature_keys_raw)
    log.debug(f"Received target_key: {target_key}, feature_keys_raw: {feature_keys_raw}, parsed_list: {feature_keys_list}")

    try:
        feature_types_str = getattr(args, "feature_types", None) or "{}"
        feature_types_override = json.loads(feature_types_str)
        log.debug(f"Received feature_types override: {feature_types_override}")

        parsed_data = load_data_from_json(
            args.data, target_key=target_key, feature_keys=feature_keys_list
        )
        months = parsed_data.timestamps or [str(i) for i in range(parsed_data.n_obs)]

        inferred_feature_types = infer_parsed_feature_types(
            parsed_data,
            target_key=parsed_data.target_key,
            feature_keys=parsed_data.feature_keys,
        )

        # If feature_types is provided, use it as an override over inferred defaults.
        if feature_types_override:
            feature_types = {**inferred_feature_types, **dict(feature_types_override)}
            log.info("Using provided feature_types override.")
        else:
            log.info("No feature_types provided, inferring from DataFrame.")
            feature_types = inferred_feature_types

        log.info(f"Resolved feature_types: {feature_types}")

        _validate_lstm_feature_types(parsed_data, feature_types)
        _validate_lstm_numeric_columns(parsed_data, feature_types)

        ts, te = int(args.train_start), int(args.train_end)
        es, ee = int(args.eval_start), int(args.eval_end)

        look_back_arg = getattr(args, "look_back", None)
        epochs_arg = getattr(args, "epochs", None)

        # Auto-select look_back from the shared rule; epochs defaults to a fixed 20.
        train_size = te - ts + 1
        look_back = resolve_lstm_look_back(train_size, look_back_arg)
        if look_back_arg is None:
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
            "forecast_horizon": metrics.get("forecast_horizon"),
            "train_range": {"start_index": ts, "end_index": te},
            "evaluate_range": {"start_index": es, "end_index": ee},
            "metrics": metrics,
            "eval_y_true": y_true,
            "predictions": predictions_with_std,
        }
        attach_method_metadata(out, METHOD_NAME, DIRECT_FORECAST_STRATEGY, DIRECT_IMPLEMENTATION_NOTES)

        if save_path:
            log.info(f"Saving model to {save_path}")
            keras_model_path = os.path.splitext(save_path)[0] + "_keras_model.h5"
            temp_keras_model_path = os.path.splitext(save_path)[0] + f"_keras_model.tmp-{os.getpid()}.h5"
            log.debug(f"Saving Keras model to temporary path {temp_keras_model_path}")
            try:
                model.save(temp_keras_model_path)
                fsync_file(temp_keras_model_path)
                os.replace(temp_keras_model_path, keras_model_path)
                fsync_parent_directory(keras_model_path)
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
                "feature_types": feature_types,
                "keras_model_path": keras_model_path,
                "forecast_horizon": int(metrics.get("forecast_horizon", len(range(es, ee + 1)))),
                "forecast_strategy": DIRECT_FORECAST_STRATEGY,
                "production_forecast_horizon": _resolve_production_forecast_horizon(
                    eval_forecast_horizon=int(metrics.get("forecast_horizon", 0) or 0)
                ),
                "std_dev_residuals": std_dev_residuals,
            }
            attach_method_metadata(state, METHOD_NAME, DIRECT_FORECAST_STRATEGY, DIRECT_IMPLEMENTATION_NOTES)
            dump_model(state, save_path)
            log.info(f"Model saved successfully to {save_path} (Keras: {keras_model_path})")
            out["saved_model"] = save_path

        # Add captured logs to the final output
        out["logs"] = get_log_contents()

        log.info("Run finished successfully.")
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))

    except (ValueError, TypeError, json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        if str(e).startswith("[VALIDATION]") or str(e).startswith("[DATA]"):
            raise
        log.error(f"An error occurred during data processing or training: {e}", exc_info=True)
        raise ValueError(f"[DATA] Failed to parse or process --data: {e}")


def _validate_fit_args(
    parsed_data: ParsedData,
    ts: int,
    te: int,
    look_back: int,
    forecast_horizon: int = 0,
    forecast_strategy: str = DIRECT_FORECAST_STRATEGY,
) -> None:
    n = parsed_data.n_obs

    if not (0 <= ts <= te < n):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], must be within [0, {n-1}]")
    if look_back < 1:
        raise ValueError("[VALIDATION] look_back must be >= 1.")
    if te - ts + 1 <= look_back:
        raise ValueError("[VALIDATION] Training range must be larger than look_back.")
    if forecast_strategy == DIRECT_FORECAST_STRATEGY and forecast_horizon > 0:
        if te - ts + 1 < look_back + forecast_horizon:
            raise ValueError(
                "[VALIDATION] Training range is too short for the saved LSTM direct forecast horizon."
            )


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
    validate_minimum_evaluation_points(es, ee)
    if look_back < 1:
        log.error(f"look_back {look_back} must be >= 1")
        raise ValueError("[VALIDATION] look_back must be >= 1.")
    if te - ts + 1 <= look_back:
        log.error(f"Training range size {te - ts + 1} must be larger than look_back {look_back}")
        raise ValueError("[VALIDATION] Training range must be larger than look_back.")
    forecast_horizon = ee - te
    if te - ts + 1 < look_back + forecast_horizon:
        log.error(
            "Training range size %s is too short for look_back=%s and forecast_horizon=%s",
            te - ts + 1,
            look_back,
            forecast_horizon,
        )
        raise ValueError(
            "[VALIDATION] Training range must contain at least look_back + forecast horizon points."
        )

    # Additional helpful validations
    train_size = te - ts + 1
    eval_size = ee - es + 1
    log.info(
        "Validation passed: train_size=%s, eval_size=%s, look_back=%s, forecast_horizon=%s",
        train_size,
        eval_size,
        look_back,
        forecast_horizon,
    )
