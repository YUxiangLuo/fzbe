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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

MODEL_TYPE = "lstm"


# ---------- LSTM Model ----------
class LSTMModel(nn.Module):
    """Standard LSTM for single-variable forecasting."""

    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ---------- Utilities ----------
def _create_sequences(data: np.ndarray, lookback: int):
    """Create sliding window sequences."""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


def _load_state(model_path: str):
    """Load saved model."""
    state = load_model(model_path)
    model = LSTMModel()
    model.load_state_dict(torch.load(state["model_path"]))
    scaler = StandardScaler()
    scaler.mean_ = np.array(state["scaler"]["mean"])
    scaler.scale_ = np.array(state["scaler"]["scale"])
    return state, model, scaler


# ---------- Train and Evaluate ----------
def train_and_evaluate(sales: List[float], lookback: int = 10, epochs: int = 50):
    """Train and evaluate LSTM (textbook-style)."""
    # Prepare data
    data = np.array(sales).reshape(-1, 1)
    if len(data) <= lookback:
        raise ValueError("[VALIDATION] Data too short.")

    # Normalize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Split
    train_size = int(len(data) * 0.8)
    train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

    # Sequences
    X_train, y_train = _create_sequences(train_data.flatten(), lookback)
    X_test, y_test = _create_sequences(test_data.flatten(), lookback)

    # Model
    model = LSTMModel(hidden_size=50)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=16, shuffle=True
    )

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_test).numpy().flatten()
        y_true = y_test.numpy().flatten()

    # Inverse transform
    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

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

    # Recent data
    if data:
        parsed = parse_data_from_json(data, "y")
        recent = np.array(parsed.y[-lookback:]).reshape(-1, 1)
    else:
        recent = np.array(state["recent_data"]).reshape(-1, 1)

    if len(recent) < lookback:
        raise ValueError("[DATA] Insufficient data.")

    recent_scaled = scaler.transform(recent)
    current = recent_scaled.copy().tolist()
    predictions = []

    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            seq = (
                torch.tensor(current[-lookback:], dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            pred_scaled = model(seq).item()
            pred_inv = scaler.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred_inv)
            current.append([pred_scaled])

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

    target_key = "y"
    parsed = parse_data_from_json(args.data, target_key)
    sales = parsed.y.tolist()

    lookback = int(getattr(args, "lookback", 10))
    epochs = int(getattr(args, "epochs", 50))

    model, scaler, metrics, predictions = train_and_evaluate(sales, lookback, epochs)

    # Save
    save_path = getattr(args, "save_model", "/tmp/lstm_model.pkl")
    state = {
        "model_type": MODEL_TYPE,
        "lookback": lookback,
        "recent_data": sales[-lookback:],
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "model_path": "/tmp/lstm_model.pth",
    }
    torch.save(model.state_dict(), state["model_path"])
    dump_model(state, save_path)

    out = {
        "mode": "lstm_trained",
        "lookback": lookback,
        "epochs": epochs,
        "metrics": metrics,
        "predictions": [{"prediction": p, "std_dev": 1.0} for p in predictions],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
