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
from forecast_utils import (
    DEMAND_PREDICTION_DOMAIN,
    apply_prediction_domain,
    resolve_prediction_domain,
)
from logger import log, get_log_contents
from log_safety import summarize_cli_args
from metrics import format_optional_metric, mae, mape, r2, rmse
from model_info import attach_method_metadata, ensure_state_metadata
from serialization import dump_model, load_model
from dynamic_config import derive_lstm_config
from uncertainty import (
    fallback_profile,
    format_predictions,
    profile_for_steps,
    residual_profile_by_horizon,
)

MODEL_TYPE = "lstm"
DIRECT_FORECAST_STRATEGY = "direct_multi_horizon"
LEGACY_FORECAST_STRATEGY = "autoregressive"
DEFAULT_PRODUCTION_FORECAST_HORIZON = 6
STANDARD_STABILITY_SEEDS = (42, 314, 2718)
METHOD_NAME = "LSTM 历史多特征销量预测"
DIRECT_IMPLEMENTATION_NOTES = [
    "教科书 LSTM 定义的是带遗忘门、输入门和输出门的循环神经网络单元。",
    "本系统仅用训练区间拟合数值缩放器和类别 One-Hot 编码器，再采用单层 LSTM 加线性 Dense 多步输出；look-back 在 2 到 12 之间，隐藏单元、批大小和训练预算由样本量、预测跨度及编码后维度共同确定。",
    "为了服务未来多期销量预测，新训练模型采用直接多步策略，一次输出完整预测 horizon，减少递归预测的误差累积；输入只包含预测起点之前最后 look_back 期的历史目标与历史特征，不读取未来特征值。",
    "样本足够时使用带 horizon 隔离带的时间顺序验证集监控 val_loss，并以 3 个固定随机种子重复训练记录内部验证稳定性；选择轮数的临时预处理器和网络容量都只依据验证起点之前可见的数据，最终模型沿用种子 42 的选定轮数，随后在全部训练窗口重新推导最终容量并拟合。小样本明确标记为教学演示并使用有限训练预算，不以训练 loss 冒充泛化早停。",
    "Dense 输出层本身不带非负约束；本系统将逆缩放后的最终销量点预测按 max(0, y_hat) 截断，并以截断值计算评估残差和指标。该业务后处理不是 LSTM 单元定义的一部分。",
    "std_dev 优先来自时间顺序验证窗口中逆缩放、未做非负截断的逐输出 horizon 残差；2.5%–97.5% 经验残差分位范围和生产用 99% 上侧压力偏移取各 horizon 的 actual-prediction 经验残差分位数，因而保留稳定偏差。该范围不声明 95% 覆盖保证；教学演示模式使用明确标记的一阶差分 sqrt(h) 正态近似回退。",
]
AUTOREGRESSIVE_IMPLEMENTATION_NOTES = [
    "教科书 LSTM 定义的是带遗忘门、输入门和输出门的循环神经网络单元。",
    "本系统采用两层 LSTM 加 Dense 输出层，隐藏单元数随数据维度和样本量动态选择。",
    "该一步递归路径仅用于兼容旧模型产物：后续步骤把预测目标滚入窗口，其他特征保持为最后观测值；当前新训练流程不会自动回退到此策略。",
    "训练中的 EarlyStopping 监控训练损失而非验证损失（教学数据过小，不再切分验证集），其作用是训练损失进入平台期即停止，并非教科书意义的验证集早停。",
    "Dense 输出层本身不带非负约束；兼容旧产物时也只把最终逆缩放销量点预测按 max(0, y_hat) 截断。Boosting 的内部残差学习会显式关闭该截断，以保留有符号残差。",
    "兼容路径只读取产物中已经保存的不确定性信息；若无法覆盖请求跨度则使用明确标记的训练历史一阶差分 sqrt(h) 回退，不读取当前预测期真实值。该回退范围使用 coverage_claim=none，不声明 95% 覆盖保证。",
]

# Type alias for input flexibility
ArrayLike = Union[List[float], np.ndarray]


# ---------- utilities ----------
def _metadata_for_strategy(strategy: str) -> Tuple[str, List[str]]:
    if strategy == DIRECT_FORECAST_STRATEGY:
        return DIRECT_FORECAST_STRATEGY, DIRECT_IMPLEMENTATION_NOTES
    return LEGACY_FORECAST_STRATEGY, AUTOREGRESSIVE_IMPLEMENTATION_NOTES


def _format_predictions(
    preds: List[float],
    std_dev: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Format predictions with standard deviation for JSON output.
    """
    profile = list(std_dev) if isinstance(std_dev, (list, tuple, np.ndarray)) else [float(std_dev)] * len(preds)
    return format_predictions(preds, profile, metadata or {"source": "empirical", "fallback": False})


def _parse_feature_key_list(raw_value: Optional[str]) -> List[str]:
    if raw_value is None or not str(raw_value).strip():
        raise ValueError(
            "[VALIDATION] LSTM requires an explicit feature list; "
            "use [] for a target-only LSTM."
        )

    try:
        parsed = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [item.strip() for item in str(raw_value).split(',') if item.strip()]


def _load_keras_artifact(keras_model_path: Optional[str]) -> Any:
    """Load the Keras sidecar while preserving the model-error protocol."""
    from tensorflow.keras.models import load_model as keras_load_model

    if not keras_model_path or not os.path.exists(keras_model_path):
        raise ValueError(f"[MODEL] Keras model file not found at {keras_model_path}")

    try:
        return keras_load_model(keras_model_path, compile=False)
    except (EOFError, KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"[MODEL] Failed to load Keras model from '{keras_model_path}': {exc}"
        ) from exc


def _load_state(model_path: str) -> Dict[str, Any]:
    """
    Load and validate a saved LSTM model state.
    """
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid LSTM model file.")

    saved_strategy = str(state.get("forecast_strategy", LEGACY_FORECAST_STRATEGY))
    forecast_strategy, notes = _metadata_for_strategy(saved_strategy)
    ensure_state_metadata(state, METHOD_NAME, forecast_strategy, notes)
    if saved_strategy != DIRECT_FORECAST_STRATEGY:
        raise ValueError("[MODEL] Obsolete autoregressive LSTM artifacts must be retrained.")
    if "normalization" not in state:
        raise ValueError("[MODEL] Saved LSTM model is missing normalization metadata.")
    raw_feature_keys = state.get("feature_keys")
    if not isinstance(raw_feature_keys, list) or any(
        not isinstance(feature_name, str) or not feature_name.strip()
        for feature_name in raw_feature_keys
    ):
        raise ValueError("[MODEL] Saved LSTM model is missing an explicit feature list.")
    state["feature_keys"] = [feature_name.strip() for feature_name in raw_feature_keys]

    # Reject obsolete metadata before paying the cost of loading the sidecar.
    keras_model_path = state.get("keras_model_path")
    state["keras_model"] = _load_keras_artifact(keras_model_path)

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


def _training_metadata(
    derived_config: Dict[str, Any],
    feature_keys: List[str],
) -> Dict[str, Any]:
    fields = (
        "regime",
        "look_back",
        "units",
        "parameter_count",
        "parameter_budget",
        "batch_size",
        "max_epochs",
        "epochs_ran",
        "windows",
        "validation_windows",
        "purge_windows",
        "selection_train_windows",
    )
    seed_stability = derived_config.get("seed_stability")
    seed_runs = (
        int(seed_stability.get("run_count", 0))
        if isinstance(seed_stability, dict)
        else 1
    )
    return {
        **{
            field: derived_config.get(field)
            for field in fields
            if derived_config.get(field) is not None
        },
        "feature_keys": list(feature_keys),
        "feature_count": int(derived_config.get("feature_count", len(feature_keys) + 1)),
        "raw_feature_count": len(feature_keys) + 1,
        "random_seed": 42,
        "seed_runs": seed_runs,
        "seed_stability": seed_stability,
        "stability_note": (
            "single_seed_teaching_demo"
            if derived_config.get("regime") == "teaching_demo"
            else "three_seed_internal_validation_summary"
        ),
    }


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


def _forecast_direct_from_state(state: Dict[str, Any], steps: int) -> Tuple[List[float], List[float]]:
    keras_model = state["keras_model"]
    preprocessor = state["preprocessor"]
    look_back = state["look_back"]
    history_df = state["history_df"]
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
    profile, _ = profile_for_steps(
        state.get("uncertainty_profile"),
        steps,
        state.get("reference_series", []),
        state.get("uncertainty_summary"),
    )
    return unscaled_predictions[:steps].flatten().tolist(), profile


def _forecast_autoregressive_from_state(state: Dict[str, Any], steps: int) -> Tuple[List[float], List[float]]:
    keras_model = state["keras_model"]
    preprocessor = state["preprocessor"]
    look_back = state["look_back"]
    history_df = state["history_df"]
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

    profile, _ = profile_for_steps(
        state.get("uncertainty_profile"),
        steps,
        state.get("reference_series", []),
        state.get("uncertainty_summary"),
    )
    return unscaled_predictions.flatten().tolist(), profile


def _forecast_from_state(
    state: Dict[str, Any],
    steps: int,
    prediction_domain: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    strategy = state.get("forecast_strategy", LEGACY_FORECAST_STRATEGY)
    if strategy == DIRECT_FORECAST_STRATEGY:
        predictions, std_dev = _forecast_direct_from_state(state, steps)
    else:
        predictions, std_dev = _forecast_autoregressive_from_state(state, steps)
    output_domain = prediction_domain or resolve_prediction_domain(state)
    return apply_prediction_domain(predictions, output_domain).tolist(), std_dev


# ---------- predict ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
    allow_negative_predictions: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Forecast using a saved LSTM model.
    """
    log.debug(f"LSTM predict called with steps={steps}, model_path={model_path}")
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")
    state = state or _load_state(model_path)

    # If state is passed directly, the keras model might not be loaded yet
    if "keras_model" not in state:
        keras_model_path = state.get("keras_model_path")
        state["keras_model"] = _load_keras_artifact(keras_model_path)
        log.debug(f"Loaded Keras model from {keras_model_path}")

    prediction_domain = resolve_prediction_domain(
        state,
        allow_negative_predictions=allow_negative_predictions,
    )
    return _forecast_from_state(state, steps, prediction_domain)


def _fit_direct_model(
    parsed_data: ParsedData,
    feature_types: Dict[str, str],
    train_range: range,
    normalization: str,
    forecast_horizon: int,
):
    from tensorflow.keras.layers import LSTM as KerasLSTM
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf

    df = parsed_data.df
    preprocessor, scaled_series = _build_preprocessor(parsed_data, feature_types, train_range, normalization)
    n_features_out = scaled_series.shape[1]
    config = derive_lstm_config(len(train_range), forecast_horizon, n_features_out)
    look_back = int(config["look_back"])

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

    log.info(
        "Building direct LSTM model: regime=%s, units=%s, horizon=%s, input_shape=(%s, %s)",
        config["regime"],
        config["units"],
        forecast_horizon,
        look_back,
        n_features_out,
    )

    def build_model(
        input_feature_count: int = n_features_out,
        units: int = int(config["units"]),
    ):
        network = Sequential([
            Input(shape=(look_back, input_feature_count)),
            KerasLSTM(units),
            Dense(forecast_horizon),
        ])
        network.compile(
            loss="mean_squared_error",
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        )
        return network

    validation_residuals = None
    epochs_ran = int(config["max_epochs"])
    if config["regime"] == "standard":
        validation_count = int(config["validation_windows"])
        selection_count = int(config["selection_train_windows"])
        validation_start = len(trainX) - validation_count
        selection_fit_points = selection_count + look_back + forecast_horizon - 1
        selection_fit_range = range(
            train_range.start,
            train_range.start + selection_fit_points,
        )
        validation_preprocessor, validation_scaled_series = _build_preprocessor(
            parsed_data,
            feature_types,
            selection_fit_range,
            normalization,
        )
        validation_train_data = validation_scaled_series[train_range]
        validationX, validationY = _create_direct_dataset(
            validation_train_data,
            look_back,
            forecast_horizon,
        )
        if len(validationX) != len(trainX):
            raise ValueError("[MODEL] LSTM validation preprocessing changed the window count.")
        selection_config = derive_lstm_config(
            len(train_range),
            forecast_horizon,
            int(validationX.shape[2]),
        )
        schedule_keys = (
            "look_back",
            "windows",
            "validation_windows",
            "purge_windows",
            "selection_train_windows",
            "batch_size",
            "max_epochs",
            "patience",
        )
        if any(selection_config[key] != config[key] for key in schedule_keys):
            raise ValueError("[MODEL] LSTM validation preprocessing changed the training schedule.")
        config["validation_preprocessor_fit_points"] = selection_fit_points
        config["validation_feature_count"] = int(validationX.shape[2])
        config["selection_units"] = int(selection_config["units"])
        config["selection_parameter_count"] = int(selection_config["parameter_count"])
        validation_actuals = _inverse_transform_target_values(
            validation_preprocessor,
            validationY[validation_start:],
        ).reshape(validationY[validation_start:].shape)
        seed_runs = []
        for seed in STANDARD_STABILITY_SEEDS:
            tf.keras.utils.set_random_seed(seed)
            _set_random_seed(seed)
            selection_model = build_model(
                int(validationX.shape[2]),
                int(selection_config["units"]),
            )
            history = selection_model.fit(
                validationX[:selection_count],
                validationY[:selection_count],
                validation_data=(validationX[validation_start:], validationY[validation_start:]),
                epochs=int(selection_config["max_epochs"]),
                batch_size=int(selection_config["batch_size"]),
                verbose=0,
                callbacks=[EarlyStopping(
                    monitor="val_loss",
                    patience=int(selection_config["patience"]),
                    restore_best_weights=True,
                )],
                shuffle=False,
            )
            validation_losses = history.history.get("val_loss", [])
            if not validation_losses:
                raise ValueError("[MODEL] LSTM validation training produced no val_loss history.")
            selected_epochs = int(np.argmin(validation_losses)) + 1
            validation_predictions_scaled = selection_model.predict(
                validationX[validation_start:],
                verbose=0,
            )
            validation_predictions = _inverse_transform_target_values(
                validation_preprocessor,
                validation_predictions_scaled,
            ).reshape(validation_predictions_scaled.shape)
            run_residuals = validation_actuals - validation_predictions
            run_summary = {
                "seed": int(seed),
                "epochs_ran": selected_epochs,
                "best_validation_loss": float(np.min(validation_losses)),
                "validation_rmse": float(np.sqrt(np.mean(np.square(run_residuals)))),
                "validation_mae": float(np.mean(np.abs(run_residuals))),
            }
            seed_runs.append(run_summary)
            if seed == 42:
                epochs_ran = selected_epochs
                validation_residuals = run_residuals

        rmse_values = np.asarray(
            [run["validation_rmse"] for run in seed_runs],
            dtype=float,
        )
        mae_values = np.asarray(
            [run["validation_mae"] for run in seed_runs],
            dtype=float,
        )
        config["seed_stability"] = {
            "scope": "internal_time_validation",
            "run_count": len(seed_runs),
            "seeds": [int(run["seed"]) for run in seed_runs],
            "runs": seed_runs,
            "validation_rmse_mean": float(np.mean(rmse_values)),
            "validation_rmse_std": float(np.std(rmse_values)),
            "validation_mae_mean": float(np.mean(mae_values)),
            "validation_mae_std": float(np.std(mae_values)),
        }
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        _set_random_seed(42)

    model = build_model()
    model.fit(
        trainX,
        trainY,
        epochs=epochs_ran,
        batch_size=int(config["batch_size"]),
        verbose=0,
        shuffle=False,
    )
    config["epochs_ran"] = epochs_ran
    log.info("Model training completed")

    history_df_for_saving = df.iloc[train_range.stop - look_back : train_range.stop].copy()
    log.debug(f"Saved history for future predictions: {len(history_df_for_saving)} rows")

    reference_series = df[parsed_data.target_key].values[train_range]
    if validation_residuals is None:
        uncertainty_profile, uncertainty_summary = fallback_profile(
            reference_series,
            forecast_horizon,
        )
    else:
        uncertainty_profile, uncertainty_summary = residual_profile_by_horizon(
            validation_residuals,
            reference_series,
            forecast_horizon,
        )

    return (
        model,
        preprocessor,
        history_df_for_saving,
        config,
        uncertainty_profile,
        uncertainty_summary,
    )


def _fit_autoregressive_model(
    parsed_data: ParsedData,
    feature_types: Dict[str, str],
    train_range: range,
    look_back: int,
    epochs: int,
    normalization: str,
):
    from tensorflow.keras.layers import LSTM as KerasLSTM
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Input
    from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    df = parsed_data.df
    preprocessor, scaled_series = _build_preprocessor(parsed_data, feature_types, train_range, normalization)
    n_features_out = scaled_series.shape[1]

    train_data = scaled_series[train_range]
    log.info(f"Creating autoregressive LSTM dataset with look_back={look_back}...")
    trainX, trainY = _create_autoregressive_dataset(train_data, look_back)
    if trainX.size == 0:
        raise ValueError("[VALIDATION] Training range is too short for autoregressive LSTM.")
    log.info(f"Training dataset created: X shape={trainX.shape}, Y shape={trainY.shape}")

    base_units = max(16, min(64, n_features_out * 4, max(16, len(trainX) * 4)))
    secondary_units = max(8, base_units // 2)
    dense_units = max(8, secondary_units)
    log.info(
        "Building autoregressive LSTM model: units=(%s,%s), dense=%s, input_shape=(%s, %s)",
        base_units,
        secondary_units,
        dense_units,
        look_back,
        n_features_out,
    )

    model = Sequential([
        Input(shape=(look_back, n_features_out)),
        KerasLSTM(base_units, return_sequences=True),
        KerasLSTM(secondary_units),
        Dense(dense_units, activation="relu"),
        Dense(1),
    ])

    initial_learning_rate = 0.001
    decay_rate = 0.98
    optimizer = Adam(learning_rate=initial_learning_rate, clipnorm=1.0)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

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
    normalization: str = 'minmax',
    timestamps: Optional[List[str]] = None,
    prediction_domain: str = DEMAND_PREDICTION_DOMAIN,
) -> Tuple[
    Dict[str, Any], Any, Any, List[float], List[float], float, pd.DataFrame,
    Dict[str, Any], List[float], Dict[str, Any],
]:
    """
    Trains and evaluates the LSTM model with typed features.
    """
    log.info(f"Starting LSTM train_and_evaluate: normalization='{normalization}', train_range={train_range}, eval_range={eval_range}")

    df = parsed_data.df
    target_key = parsed_data.target_key
    train_target_values = df[target_key].values[train_range]
    steps_to_eval_end = eval_range.stop - train_range.stop
    forecast_offset = eval_range.start - train_range.stop
    forecast_horizon = steps_to_eval_end

    log.debug(f"Total series length: {len(df)}")
    log.debug(f"Training data range: [{train_range.start}, {train_range.stop-1}], count={train_range.stop - train_range.start}")
    log.debug(f"Evaluation data range: [{eval_range.start}, {eval_range.stop-1}], count={eval_range.stop - eval_range.start}")

    (
        model,
        preprocessor,
        history_df_for_saving,
        derived_config,
        uncertainty_profile,
        uncertainty_summary,
    ) = _fit_direct_model(
        parsed_data,
        feature_types,
        train_range,
        normalization,
        forecast_horizon,
    )
    look_back = int(derived_config["look_back"])

    # Make predictions for evaluation
    if not list(eval_range):
        log.warning("Empty evaluation range, returning empty metrics")
        return (
            {}, model, preprocessor, [], [], float(uncertainty_profile[0]), history_df_for_saving,
            derived_config, uncertainty_profile, uncertainty_summary,
        )

    log.info("Generating predictions for evaluation range...")
    state_for_eval = {
        "keras_model": model,
        "preprocessor": preprocessor,
        "look_back": look_back,
        "forecast_horizon": forecast_horizon,
        "forecast_strategy": DIRECT_FORECAST_STRATEGY,
        "history_df": history_df_for_saving,
        "normalization": normalization,
        "prediction_domain": prediction_domain,
        "uncertainty_profile": uncertainty_profile,
        "uncertainty_summary": uncertainty_summary,
        "reference_series": train_target_values.tolist(),
    }
    full_forecast, _ = _forecast_from_state(
        state_for_eval,
        forecast_horizon,
        prediction_domain,
    )
    y_pred = full_forecast[forecast_offset: forecast_offset + len(eval_range)]
    if len(y_pred) != len(eval_range):
        raise ValueError("[VALIDATION] 无法为完整评估区间生成 LSTM 预测。")

    y_true = df[target_key].values[eval_range]
    log.info(f"Generated {len(y_pred)} predictions after inverse transform")

    residuals = y_true - y_pred
    std_dev_residuals = float(uncertainty_profile[0])
    log.info(f"Calculated residuals: mean={np.mean(residuals):.4f}, std_dev={std_dev_residuals:.4f}")

    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "forecast_horizon": forecast_horizon,
        "training_regime": derived_config["regime"],
        "evaluated_points": len(y_true),
        "evaluate_indices": list(eval_range),
        "evaluate_months": [timestamps[i] for i in eval_range] if timestamps else [],
    }

    log.info(
        f"Evaluation metrics computed: RMSE={metrics['rmse']:.4f}, "
        f"MAE={metrics['mae']:.4f}, MAPE={format_optional_metric(metrics['mape'])}, "
        f"R²={metrics['r2']:.4f}"
    )

    return (
        metrics,
        model,
        preprocessor,
        y_true.tolist(),
        list(y_pred),
        std_dev_residuals,
        history_df_for_saving,
        derived_config,
        uncertainty_profile,
        uncertainty_summary,
    )


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

    log.info(f"LSTM run started with args: {summarize_cli_args(args)}")

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
        prediction_domain = resolve_prediction_domain(
            state,
            allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
        )
        parsed_data = load_data_from_json(
            args.data,
            target_key=state.get("target_key"),
            feature_keys=state.get("feature_keys"),
        )

        ts, te = int(args.train_start), int(args.train_end)
        normalization = state.get("normalization", 'minmax')
        explicit_fit_forecast_steps = int(getattr(args, "fit_forecast_steps", 0) or 0)
        desired_forecast_horizon = (
            explicit_fit_forecast_steps
            if explicit_fit_forecast_steps > 0
            else _resolve_production_forecast_horizon(state=state)
        )
        fit_forecast_horizon = desired_forecast_horizon
        if not (0 <= ts <= te < parsed_data.n_obs):
            raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}].")

        feature_types = state.get("feature_types") or infer_parsed_feature_types(
            parsed_data,
            target_key=parsed_data.target_key,
            feature_keys=parsed_data.feature_keys,
        )
        _validate_lstm_feature_types(parsed_data, feature_types)
        _validate_lstm_numeric_columns(parsed_data, feature_types)
        (
            model,
            preprocessor,
            history_df,
            derived_config,
            uncertainty_profile,
            uncertainty_summary,
        ) = _fit_direct_model(
            parsed_data,
            feature_types,
            range(ts, te + 1),
            normalization,
            fit_forecast_horizon,
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

        std_dev_residuals = float(uncertainty_profile[0])
        next_state = {
            "model_type": MODEL_TYPE,
            "look_back": int(derived_config["look_back"]),
            "normalization": normalization,
            "train_end_index": te,
            "preprocessor": preprocessor,
            "history_df": history_df,
            "target_key": parsed_data.target_key,
            "feature_keys": parsed_data.feature_keys,
            "feature_types": feature_types,
            "keras_model_path": keras_model_path,
            "forecast_horizon": fit_forecast_horizon,
            "forecast_strategy": DIRECT_FORECAST_STRATEGY,
            "production_forecast_horizon": desired_forecast_horizon,
            "std_dev_residuals": std_dev_residuals,
            "derived_config": derived_config,
            "uncertainty_profile": uncertainty_profile,
            "uncertainty_summary": uncertainty_summary,
            "reference_series": parsed_data.df[parsed_data.target_key].values[ts : te + 1].tolist(),
            "prediction_domain": prediction_domain,
        }
        output_strategy, output_notes = _metadata_for_strategy(DIRECT_FORECAST_STRATEGY)
        attach_method_metadata(next_state, METHOD_NAME, output_strategy, output_notes)
        dump_model(next_state, save_path)

        out: Dict[str, Any] = {
            "mode": "lstm_fit_only",
            "normalization": normalization,
            "train_end_index": te,
            "target_key": parsed_data.target_key,
            "feature_keys": parsed_data.feature_keys,
            "feature_types": feature_types,
            "saved_model": save_path,
            "forecast_horizon": fit_forecast_horizon,
            "training_regime": derived_config["regime"],
            "training_metadata": _training_metadata(
                derived_config,
                parsed_data.feature_keys,
            ),
            "uncertainty_summary": uncertainty_summary,
        }
        fit_predict_steps = int(getattr(args, "fit_predict_steps", 0) or 0)
        if fit_predict_steps > 0:
            preds, resolved_profile = predict(
                save_path,
                fit_predict_steps,
                state=next_state,
                allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
            )
            _, resolved_summary = profile_for_steps(
                next_state["uncertainty_profile"],
                fit_predict_steps,
                next_state["reference_series"],
                next_state["uncertainty_summary"],
            )
            out["forecast_steps"] = fit_predict_steps
            out["predictions"] = _format_predictions(preds, resolved_profile, resolved_summary)
        attach_method_metadata(out, METHOD_NAME, output_strategy, output_notes)
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
        prepared = state.get("prepared_predictions")
        if (
            not bool(getattr(args, "allow_negative_predictions", False))
            and isinstance(prepared, list)
            and len(prepared) >= steps
        ):
            predictions_with_std = [dict(point) for point in prepared[:steps]]
        else:
            preds, profile = predict(
                load_path,
                steps,
                state=state,
                allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
            )
            _, resolved_uncertainty_summary = profile_for_steps(
                state.get("uncertainty_profile"),
                steps,
                state.get("reference_series") or [],
                state.get("uncertainty_summary") or {},
            )
            predictions_with_std = _format_predictions(preds, profile, resolved_uncertainty_summary)

        out: Dict[str, Any] = {
            "mode": "lstm_loaded_predict",
            "train_end_index": state["train_end_index"],
            "target_key": state.get("target_key"),
            "feature_keys": state.get("feature_keys"),
            "forecast_steps": steps,
            "predictions": predictions_with_std,
            "uncertainty_summary": state.get("uncertainty_summary"),
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

        normalization_arg = getattr(args, "normalization", None)
        normalization = normalization_arg if normalization_arg is not None else 'minmax'
        log.debug(f"Training params: normalization='{normalization}'")

        _validate_training_args(parsed_data, ts, te, es, ee)
        prediction_domain = resolve_prediction_domain(
            allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
        )

        log.info("Starting model training and evaluation...")
        (
            metrics,
            model,
            preprocessor,
            y_true,
            y_pred,
            std_dev_residuals,
            history_df,
            derived_config,
            uncertainty_profile,
            uncertainty_summary,
        ) = train_and_evaluate(
            parsed_data,
            feature_types,
            range(ts, te + 1),
            range(es, ee + 1),
            normalization,
            months,
            prediction_domain,
        )
        log.info(f"Training finished. Metrics: {metrics}")

        forecast_offset = es - te - 1
        evaluation_profile = uncertainty_profile[
            forecast_offset : forecast_offset + len(y_pred)
        ]
        predictions_with_std = _format_predictions(
            y_pred,
            evaluation_profile,
            {**uncertainty_summary, "horizon_start": forecast_offset + 1},
        )

        out: Dict[str, Any] = {
            "mode": "lstm_train_eval",
            "data_points": parsed_data.n_obs,
            "target_key": parsed_data.target_key,
            "feature_keys": parsed_data.feature_keys,
            "feature_types": feature_types,
            "normalization": normalization,
            "forecast_horizon": metrics.get("forecast_horizon"),
            "training_regime": derived_config["regime"],
            "training_metadata": _training_metadata(
                derived_config,
                parsed_data.feature_keys,
            ),
            "train_range": {"start_index": ts, "end_index": te},
            "evaluate_range": {"start_index": es, "end_index": ee},
            "metrics": metrics,
            "eval_y_true": y_true,
            "predictions": predictions_with_std,
            "uncertainty_summary": uncertainty_summary,
        }
        attach_method_metadata(out, METHOD_NAME, DIRECT_FORECAST_STRATEGY, DIRECT_IMPLEMENTATION_NOTES)

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
                "look_back": int(derived_config["look_back"]),
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
                "derived_config": derived_config,
                "uncertainty_profile": uncertainty_profile,
                "uncertainty_summary": uncertainty_summary,
                "reference_series": parsed_data.df[parsed_data.target_key].values[ts : te + 1].tolist(),
                "prediction_domain": prediction_domain,
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
    """Validate only public fit boundaries; hidden look-back is re-derived later."""
    n = parsed_data.n_obs

    if not (0 <= ts <= te < n):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], must be within [0, {n-1}]")
    if forecast_strategy == DIRECT_FORECAST_STRATEGY and forecast_horizon < 1:
        raise ValueError("[VALIDATION] LSTM direct forecast horizon must be at least 1.")


def _validate_training_args(
    parsed_data: ParsedData, ts: int, te: int, es: int, ee: int
) -> None:
    """
    Validate training and evaluation parameters for LSTM.
    """
    log.debug(f"Validating training arguments: ts={ts}, te={te}, es={es}, ee={ee}")
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
    forecast_horizon = ee - te

    # Additional helpful validations
    train_size = te - ts + 1
    eval_size = ee - es + 1
    log.info(
        "Validation passed: train_size=%s, eval_size=%s, forecast_horizon=%s",
        train_size,
        eval_size,
        forecast_horizon,
    )
