# py/src/models/lstm.py
# -*- coding: utf-8 -*-
"""
LSTM (Long Short-Term Memory) model for time series forecasting.

Textbook implementation: Single-variable, autoregressive forecasting with PyTorch.
Supports zscore normalization, sliding window sequences, and multi-step prediction.
"""

import json
from typing import Any, Dict, List, Optional

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

MODEL_TYPE = "lstm"


# ---------- LSTM Model ----------
class LSTMModel(nn.Module):
    """Standard LSTM for single-variable forecasting."""

    def __init__(
        self, input_size: int = 1, hidden_size: int = 288, num_layers: int = 3
    ):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ---------- Utilities ----------
def _create_sequences(data: np.ndarray, lookback: int):
    """Create sliding window sequences for multivariate data."""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback, 0])  # Target is first column
    return np.array(X), np.array(y)


def _load_state(model_path: str):
    """Load saved model."""
    state = load_model(model_path)
    input_size = state.get("input_size", 1)
    model = LSTMModel(input_size=input_size)
    model.load_state_dict(torch.load(state["model_path"]))
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
    return state, model, scaler


# ---------- Train and Evaluate ----------
def train_and_evaluate(
    data: np.ndarray,
    lookback: int = 10,
    epochs: int = 20,
    normalization: str = "zscore",
    batch_size: int = 6,
    lr: float = 0.01,
):
    """Train and evaluate LSTM for multivariate data."""
    if data.shape[1] < 1:
        raise ValueError("[VALIDATION] Data must have at least one feature.")
    if len(data) <= lookback:
        raise ValueError("[VALIDATION] Data too short.")

    # Normalize
    if normalization == "zscore":
        scaler = StandardScaler()
    elif normalization == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("[VALIDATION] Unsupported normalization.")
    data_scaled = scaler.fit_transform(data)

    # Split
    train_size = int(len(data) * 0.8)
    train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

    # Sequences
    X_train, y_train = _create_sequences(train_data, lookback)
    X_test, y_test = _create_sequences(test_data, lookback)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "[VALIDATION] No sequences for training or testing. Increase data size or decrease lookback."
        )

    # Model
    input_size = data.shape[1]
    model = LSTMModel(input_size=input_size, hidden_size=288, num_layers=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.99
    )  # Customer: LR decay 0.99

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

    metrics = {
        "rmse": rmse(y_true_inv.tolist(), preds_inv.tolist()),
        "mae": mae(y_true_inv.tolist(), preds_inv.tolist()),
        "mape": mape(y_true_inv.tolist(), preds_inv.tolist()),
        "r2": r2(y_true_inv.tolist(), preds_inv.tolist()),
    }

    return model, scaler, metrics, preds_inv.tolist()


# ---------- Predict ----------
def predict(model_path: str, steps: int, data: Optional[str] = None):
    """Predict future steps."""
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0")

    state, model, scaler = _load_state(model_path)
    lookback = state["lookback"]
    input_size = state["input_size"]
    features = state.get("features", [])

    # Recent data
    if data:
        records = json.loads(data)
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
        raise ValueError("[DATA] Insufficient data.")

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
            predictions.append(pred_inv)
            # Update: append predicted target, keep features constant
            new_row = pred_full.flatten()
            current = np.vstack([current, new_row])

    return [{"prediction": p, "std_dev": 1.0} for p in predictions]


# ---------- Run ----------
def run(args) -> None:
    """Run LSTM training or prediction."""
    pretty = bool(getattr(args, "pretty", False))

    if getattr(args, "load_model", None):
        steps = int(getattr(args, "forecast_steps", 0))
        preds = predict(args.load_model, steps, getattr(args, "data", None))
        out = {
            "mode": "lstm_loaded_predict",
            "forecast_steps": steps,
            "predictions": preds,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # Training
    if not hasattr(args, "data") or not args.data:
        raise ValueError("[VALIDATION] --data required.")

    normalization = getattr(args, "lstm_normalization", "zscore")
    target_key = getattr(args, "lstm_target_feature", "sales")
    features = getattr(args, "lstm_features", [])

    # Parse data
    records = json.loads(args.data)
    if not isinstance(records, list) or not records:
        raise ValueError("[DATA] --data must be non-empty array.")

    first_record = records[0]
    if target_key not in first_record:
        raise ValueError(f"[DATA] Target '{target_key}' not found.")

    cols = [target_key] + features
    for f in cols:
        if f not in first_record:
            raise ValueError(f"[DATA] Feature '{f}' not found.")

    data_matrix = []
    for r in records:
        # Check key existence for each record
        missing_keys = [key for key in cols if key not in r]
        if missing_keys:
            raise ValueError(
                f"[DATA] Record {records.index(r)} missing keys: {missing_keys}"
            )
        row = [float(r[key]) for key in cols]
        data_matrix.append(row)
    data = np.array(data_matrix)

    lookback = int(getattr(args, "lookback", 10))
    epochs = int(getattr(args, "epochs", 20))  # Customer: 20 epochs

    batch_size = int(getattr(args, "batch_size", 6))  # Customer: 6
    lr = float(getattr(args, "lr", 0.01))  # Customer: 0.01

    model, scaler, metrics, predictions = train_and_evaluate(
        data, lookback, epochs, normalization, batch_size=batch_size, lr=lr
    )

    # Save
    save_path = getattr(args, "save_model", "/tmp/lstm_model.pkl")
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
        "model_path": "/tmp/lstm_model.pth",
    }
    torch.save(model.state_dict(), state["model_path"])
    dump_model(state, save_path)

    out = {
        "mode": "lstm_trained",
        "normalization": normalization,
        "target_feature": target_key,
        "features": features,
        "lookback": lookback,
        "epochs": epochs,
        "metrics": metrics,
        "predictions": [{"prediction": p, "std_dev": 1.0} for p in predictions],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
