# py/src/models/lstm.py
# -*- coding: utf-8 -*-
"""
LSTM (Long Short-Term Memory) model for time series forecasting.

Textbook implementation: Single-variable, autoregressive forecasting with PyTorch.
Supports zscore normalization, sliding window sequences, and multi-step prediction.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import shared utilities
from data_utils import parse_data_from_json
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Type alias for input flexibility
ArrayLike = Union[List[float], np.ndarray]

MODEL_TYPE = "lstm"


# ---------- LSTM Model ----------
class LSTMModel(nn.Module):
    """
    Standard LSTM for single-variable forecasting.

    Architecture based on customer requirements:
    - 3 LSTM layers: Increases model depth to learn complex time series patterns
    - 288 hidden units per layer: Provides sufficient capacity to capture time series complexity
    - 1 output feature: The predicted target value (e.g., sales quantity or amount)
    - Fully connected layer: Integrates LSTM output for final prediction
    """

    def __init__(
        self, input_size: int = 1, hidden_size: int = 288, num_layers: int = 3
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features (default 1 for univariate)
            hidden_size: Number of hidden units per layer (default 288 per customer spec)
            num_layers: Number of LSTM layers (default 3 per customer spec)
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output: 1 target value

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last timestep output
        return out


# ---------- Utilities ----------
def _determine_target_key(data_json_str: str) -> Tuple[str, Optional[str], list]:
    """
    Determine the target key and time key from JSON data string.
    Returns parsed records to avoid duplicate JSON parsing.

    Args:
        data_json_str: JSON string containing the data

    Returns:
        Tuple of (target_key, time_key, records) where:
        - target_key is either "y" or "sales"
        - time_key is "month" if present, else None
        - records is the parsed JSON list

    Raises:
        ValueError: If data format is invalid or target key cannot be determined
    """
    try:
        records = json.loads(data_json_str)
        if not isinstance(records, list) or not records:
            raise ValueError("[DATA] --data must be a non-empty JSON array.")

        first_record = records[0]
        if "y" in first_record:
            target_key = "y"
        elif "sales" in first_record:
            target_key = "sales"
        else:
            raise ValueError(
                "[DATA] Cannot determine target key. Expecting 'y' or 'sales'."
            )

        time_key = "month" if "month" in first_record else None
        return target_key, time_key, records

    except json.JSONDecodeError as e:
        raise ValueError(f"[DATA] Invalid JSON format: {e}")
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"[DATA] Failed to parse data structure: {e}")


def _create_sequences(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for multivariate data.

    Args:
        data: Time series data with shape (n_samples, n_features)
        lookback: Number of timesteps to look back

    Returns:
        Tuple of (X, y) where X has shape (n_sequences, lookback, n_features)
        and y has shape (n_sequences,)

    Raises:
        ValueError: If lookback is invalid or data too short
    """
    if lookback <= 0:
        raise ValueError("[VALIDATION] lookback must be > 0")
    if len(data) <= lookback:
        raise ValueError(f"[VALIDATION] Data length ({len(data)}) must be > lookback ({lookback})")
    if data.ndim != 2:
        raise ValueError(f"[VALIDATION] Data must be 2D, got shape {data.shape}")

    # Check for NaN or inf values
    if not np.isfinite(data).all():
        raise ValueError("[DATA] Data contains NaN or inf values")

    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback, 0])  # Target is first column
    return np.array(X), np.array(y)


def _load_state(model_path: str) -> Tuple[Dict[str, Any], LSTMModel, Union[StandardScaler, MinMaxScaler]]:
    """
    Load saved LSTM model state and scaler.

    Args:
        model_path: Path to saved model (.pkl file)

    Returns:
        Tuple of (state_dict, model, scaler)

    Raises:
        FileNotFoundError: If model file or PyTorch weights file not found
        ValueError: If model state is corrupted or invalid
    """
    try:
        state = load_model(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"[MODEL] Model file not found: {model_path}")
    except Exception as e:
        raise ValueError(f"[MODEL] Failed to load model state: {e}")

    # Validate state structure
    required_keys = ["model_path", "input_size", "scaler"]
    missing = [k for k in required_keys if k not in state]
    if missing:
        raise ValueError(f"[MODEL] Model state missing keys: {missing}")

    input_size = state.get("input_size", 1)
    model = LSTMModel(input_size=input_size)

    # Load PyTorch model weights with exception handling
    pytorch_model_path = state["model_path"]
    if not os.path.exists(pytorch_model_path):
        raise FileNotFoundError(f"[MODEL] PyTorch weights file not found: {pytorch_model_path}")

    try:
        model.load_state_dict(torch.load(pytorch_model_path))
    except Exception as e:
        raise ValueError(f"[MODEL] Failed to load PyTorch weights: {e}")

    # Reconstruct scaler
    scaler_type = state["scaler"]["type"]
    if scaler_type == "zscore":
        scaler = StandardScaler()
        scaler.mean_ = np.array(state["scaler"]["mean"])
        scaler.scale_ = np.array(state["scaler"]["scale"])
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
        scaler.min_ = np.array(state["scaler"]["min"])
        scaler.scale_ = np.array(state["scaler"]["scale"])
        scaler.data_min_ = np.array(state["scaler"]["data_min"])
        scaler.data_max_ = np.array(state["scaler"]["data_max"])
    else:
        raise ValueError(f"[MODEL] Unknown scaler type: {scaler_type}")

    return state, model, scaler


# ---------- Train and Evaluate ----------
def train_and_evaluate(
    data: np.ndarray,
    lookback: int = 10,
    epochs: int = 20,
    normalization: str = "zscore",
    batch_size: int = 6,
    lr: float = 0.01,
    train_test_split: float = 0.8,
) -> Tuple[LSTMModel, Union[StandardScaler, MinMaxScaler], Dict[str, float], np.ndarray, float]:
    """
    Train and evaluate LSTM for multivariate data.

    Training parameters based on customer requirements:
    - Loss function: MSE (Mean Squared Error) to measure prediction error
    - Optimizer: Adam (adaptive learning rate optimization)
    - Max epochs: 20 (determines training iterations)
    - Batch size: 6 (samples per batch, affects training stability and speed)
    - Initial learning rate: 0.01
    - LR decay rate: 0.99 (gradually reduces learning rate for fine-tuning)

    Args:
        data: Time series data with shape (n_samples, n_features)
        lookback: Number of timesteps to look back
        epochs: Number of training epochs (default 20 per customer spec)
        normalization: Normalization type ('zscore' or 'minmax')
        batch_size: Batch size for training (default 6 per customer spec)
        lr: Initial learning rate (default 0.01 per customer spec)
        train_test_split: Train/test split ratio (default 0.8)

    Returns:
        Tuple of (model, scaler, metrics, predictions, std_dev)
    """
    if data.shape[1] < 1:
        raise ValueError("[VALIDATION] Data must have at least one feature.")
    if len(data) <= lookback:
        raise ValueError("[VALIDATION] Data too short.")
    if not 0.0 < train_test_split < 1.0:
        raise ValueError("[VALIDATION] train_test_split must be between 0 and 1")

    # Normalize
    if normalization == "zscore":
        scaler = StandardScaler()
    elif normalization == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("[VALIDATION] Unsupported normalization.")
    data_scaled = scaler.fit_transform(data)

    # Split
    train_size = int(len(data) * train_test_split)
    train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

    # Sequences
    X_train, y_train = _create_sequences(train_data, lookback)
    X_test, y_test = _create_sequences(test_data, lookback)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "[VALIDATION] No sequences for training or testing. Increase data size or decrease lookback."
        )

    # Model - Customer requirements: 3 layers, 288 hidden units, MSE loss, Adam optimizer
    input_size = data.shape[1]
    model = LSTMModel(input_size=input_size, hidden_size=288, num_layers=3)
    criterion = nn.MSELoss()  # Customer: MSE loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Customer: Adam optimizer
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.99  # Customer: LR decay rate 0.99
    )

    # Training
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()  # Apply LR decay per epoch

    # Evaluate
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_test_tensor).numpy().flatten()
        y_true = y_test_tensor.numpy().flatten()

    # Inverse transform
    preds_full = np.zeros((len(preds), data.shape[1]))
    preds_full[:, 0] = preds
    y_true_full = np.zeros((len(y_true), data.shape[1]))
    y_true_full[:, 0] = y_true
    preds_inv = scaler.inverse_transform(preds_full)[:, 0]
    y_true_inv = scaler.inverse_transform(y_true_full)[:, 0]

    # Calculate metrics - pass numpy arrays directly
    metrics = {
        "rmse": rmse(y_true_inv, preds_inv),
        "mae": mae(y_true_inv, preds_inv),
        "mape": mape(y_true_inv, preds_inv),
        "r2": r2(y_true_inv, preds_inv),
    }

    # Calculate real standard deviation of residuals
    residuals = y_true_inv - preds_inv
    std_dev = float(np.std(residuals))

    return model, scaler, metrics, preds_inv, std_dev


# ---------- Predict ----------
def predict(
    model_path: str, steps: int, data: Optional[str] = None
) -> List[Dict[str, float]]:
    """
    Predict future steps using saved LSTM model.

    Args:
        model_path: Path to saved model file
        steps: Number of steps to predict
        data: Optional JSON data string with recent observations

    Returns:
        List of prediction dicts with 'prediction' and 'std_dev' keys

    Raises:
        ValueError: If steps <= 0 or data is invalid
        FileNotFoundError: If model file not found
    """
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0")

    state, model, scaler = _load_state(model_path)
    lookback = state["lookback"]
    input_size = state["input_size"]
    features = state.get("features", [])
    std_dev = state.get("std_dev", 1.0)  # Get stored std_dev or default to 1.0

    # Recent data
    if data:
        try:
            records = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"[DATA] Invalid JSON format: {e}")

        if not isinstance(records, list) or not records:
            raise ValueError("[DATA] Data must be a non-empty list")

        target_key = state.get("target_feature", "y")
        if not features:
            recent_raw = np.array(
                [float(r[target_key]) for r in records[-lookback:]]
            ).reshape(-1, 1)
        else:
            cols = [target_key] + features
            recent_raw = np.array(
                [[float(r[key]) for key in cols] for r in records[-lookback:]]
            )
    else:
        recent_raw = np.array(state["recent_data"])

    if len(recent_raw) < lookback:
        raise ValueError(f"[DATA] Insufficient data. Need at least {lookback} observations, got {len(recent_raw)}")

    recent_scaled = scaler.transform(recent_raw)
    current = recent_scaled.copy()
    predictions = []

    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            seq = torch.tensor(current[-lookback:], dtype=torch.float32).unsqueeze(0)
            pred_scaled = model(seq).item()
            pred_full = np.zeros((1, input_size))
            pred_full[0, 0] = pred_scaled
            pred_inv = scaler.inverse_transform(pred_full)[0, 0]
            predictions.append(float(pred_inv))
            # Update: append predicted target, keep features constant
            new_row = pred_full.flatten()
            current = np.vstack([current, new_row])

    # Use actual std_dev from training
    return [{"prediction": p, "std_dev": std_dev} for p in predictions]


# ---------- Run ----------
def run(args) -> None:
    """
    Run LSTM training or prediction.

    Args:
        args: Namespace with command-line arguments

    Raises:
        ValueError: For invalid arguments or data
        FileNotFoundError: If model file not found in prediction mode
    """
    pretty = bool(getattr(args, "pretty", False))

    # Prediction mode
    if getattr(args, "load_model", None):
        steps = int(getattr(args, "forecast_steps", 0))
        if steps <= 0:
            raise ValueError("[VALIDATION] forecast_steps must be > 0")

        preds = predict(args.load_model, steps, getattr(args, "data", None))
        out = {
            "mode": "lstm_loaded_predict",
            "forecast_steps": steps,
            "predictions": preds,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # Training mode
    if not hasattr(args, "data") or not args.data:
        raise ValueError("[VALIDATION] --data required.")

    normalization = getattr(args, "lstm_normalization", "zscore")
    target_key = getattr(args, "lstm_target_feature", "sales")
    features = getattr(args, "lstm_features", [])

    # Parse data with exception handling - avoid duplicate parsing
    try:
        records = json.loads(args.data)
    except json.JSONDecodeError as e:
        raise ValueError(f"[DATA] Invalid JSON format: {e}")

    if not isinstance(records, list) or not records:
        raise ValueError("[DATA] --data must be non-empty array.")

    first_record = records[0]
    if target_key not in first_record:
        raise ValueError(f"[DATA] Target '{target_key}' not found.")

    cols = [target_key] + features
    for f in cols:
        if f not in first_record:
            raise ValueError(f"[DATA] Feature '{f}' not found.")

    # Build data matrix with enumerate for efficiency
    data_matrix = []
    for idx, r in enumerate(records):
        # Check key existence for each record
        missing_keys = [key for key in cols if key not in r]
        if missing_keys:
            raise ValueError(
                f"[DATA] Record {idx} missing keys: {missing_keys}"
            )
        row = [float(r[key]) for key in cols]
        data_matrix.append(row)
    data = np.array(data_matrix)

    lookback = int(getattr(args, "lookback", 10))
    epochs = int(getattr(args, "epochs", 20))  # Customer: 20 epochs
    batch_size = int(getattr(args, "batch_size", 6))  # Customer: 6
    lr = float(getattr(args, "lr", 0.01))  # Customer: 0.01
    train_test_split = float(getattr(args, "train_test_split", 0.8))

    model, scaler, metrics, predictions, std_dev = train_and_evaluate(
        data, lookback, epochs, normalization, batch_size=batch_size, lr=lr, train_test_split=train_test_split
    )

    # Save model - FIX: Use user-specified path properly
    save_path = getattr(args, "save_model", "/tmp/lstm_model.pkl")

    # Generate PyTorch model path based on save_path
    save_path_obj = Path(save_path)
    pytorch_model_path = str(save_path_obj.parent / (save_path_obj.stem + ".pth"))

    # Build scaler state
    scaler_state = {}
    if isinstance(scaler, StandardScaler):
        scaler_state = {
            "type": "zscore",
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }
    elif isinstance(scaler, MinMaxScaler):
        scaler_state = {
            "type": "minmax",
            "min": scaler.min_.tolist(),
            "scale": scaler.scale_.tolist(),
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
        }

    state = {
        "model_type": MODEL_TYPE,
        "lookback": lookback,
        "input_size": data.shape[1],
        "target_feature": target_key,
        "features": features,
        "recent_data": data[-lookback:].tolist(),
        "scaler": scaler_state,
        "model_path": pytorch_model_path,  # FIX: Use proper path
        "std_dev": std_dev,  # Save std_dev for predictions
    }

    # Save PyTorch model weights
    torch.save(model.state_dict(), pytorch_model_path)
    # Save full state
    dump_model(state, save_path)

    out = {
        "mode": "lstm_trained",
        "normalization": normalization,
        "target_feature": target_key,
        "features": features,
        "lookback": lookback,
        "epochs": epochs,
        "metrics": metrics,
        "predictions": [{"prediction": float(p), "std_dev": std_dev} for p in predictions],
        "saved_model": save_path,
        "saved_pytorch_model": pytorch_model_path,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
