# py/src/models/lstm.py
# -*- coding: utf-8 -*-
"""
LSTM (Long Short-Term Memory) model for time series forecasting.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data_utils import load_data_from_json
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model

MODEL_TYPE = "lstm"
ArrayLike = Union[List[float], np.ndarray]


# ---------- LSTM Model ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 288, num_layers: int = 3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def _validate_ranges(n: int, ts: int, te: int, es: int, ee: int) -> None:
    if not (0 <= ts <= te < n):
        raise ValueError(f"Invalid train range: [{ts},{te}]")
    if not (0 <= es <= ee < n):
        raise ValueError(f"Invalid eval range: [{es},{ee}]")
    if es <= te:
        raise ValueError("eval_start must be > train_end.")


# ---------- Utilities ----------
def _create_sequences(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    if len(data) <= lookback:
        raise ValueError(f"Data length ({len(data)}) must be > lookback ({lookback})")
    
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)


def _load_state(model_path: str) -> Tuple[Dict[str, Any], LSTMModel, Union[StandardScaler, MinMaxScaler]]:
    state = load_model(model_path)
    model = LSTMModel(input_size=state.get("input_size", 1))
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
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    return state, model, scaler


# ---------- Train and Evaluate ----------
def train_and_evaluate(
    data: np.ndarray, lookback: int, epochs: int, normalization: str,
    batch_size: int, lr: float, train_range: range, eval_range: range
) -> Tuple[LSTMModel, Union[StandardScaler, MinMaxScaler], Dict[str, float], List[float], List[float], float]:
    
    scaler = StandardScaler() if normalization == "zscore" else MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    train_data = data_scaled[train_range.start : train_range.stop]
    eval_data = data_scaled[eval_range.start : eval_range.stop]

    X_train, y_train = _create_sequences(train_data, lookback)
    X_eval, y_eval = _create_sequences(eval_data, lookback)

    model = LSTMModel(input_size=data.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    ), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_eval, dtype=torch.float32)).numpy().flatten()

    preds_full = np.zeros((len(preds), data.shape[1])); preds_full[:, 0] = preds
    y_true_full = np.zeros((len(y_eval), data.shape[1])); y_true_full[:, 0] = y_eval
    preds_inv = scaler.inverse_transform(preds_full)[:, 0]
    y_true_inv = scaler.inverse_transform(y_true_full)[:, 0]

    metrics = {"rmse": rmse(y_true_inv, preds_inv), "mae": mae(y_true_inv, preds_inv),
               "mape": mape(y_true_inv, preds_inv), "r2": r2(y_true_inv, preds_inv)}
    std_dev = float(np.std(y_true_inv - preds_inv))

    return model, scaler, metrics, y_true_inv.tolist(), preds_inv.tolist(), std_dev


# ---------- Predict ----------
def predict(model_path: str, steps: int, data: Optional[str] = None) -> List[Dict[str, float]]:
    if steps <= 0:
        raise ValueError("steps must be > 0")

    state, model, scaler = _load_state(model_path)
    lookback = state["lookback"]
    std_dev = state.get("std_dev", 1.0)

    if data:
        parsed_data = load_data_from_json(data, target_key=state.get("target_feature", "y"),
                                          feature_keys=state.get("features", []))
        recent_raw = parsed_data.y_featured[-lookback:]
    else:
        recent_raw = np.array(state["recent_data"])

    if len(recent_raw) < lookback:
        raise ValueError(f"Insufficient data: need {lookback}, got {len(recent_raw)}")

    current = scaler.transform(recent_raw)
    predictions = []
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            seq = torch.tensor(current[-lookback:], dtype=torch.float32).unsqueeze(0)
            pred_scaled = model(seq).item()
            
            pred_full = np.zeros((1, current.shape[1])); pred_full[0, 0] = pred_scaled
            predictions.append(float(scaler.inverse_transform(pred_full)[0, 0]))
            
            current = np.vstack([current, pred_full])

    return [{"prediction": p, "std_dev": std_dev} for p in predictions]


# ---------- Run ----------
def run(args) -> None:
    pretty = bool(getattr(args, "pretty", False))

    if getattr(args, "load_model", None):
        steps = int(getattr(args, "forecast_steps", 0))
        preds = predict(args.load_model, steps, getattr(args, "data", None))
        out = {"mode": "lstm_loaded_predict", "forecast_steps": steps, "predictions": preds}
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    if not hasattr(args, "data") or not args.data:
        raise ValueError("--data required for training.")

    target_key = getattr(args, "lstm_target_feature", "sales")
    features = getattr(args, "lstm_features", [])
    parsed_data = load_data_from_json(args.data, target_key=target_key, feature_keys=features)
    data = parsed_data.y_featured

    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)
    _validate_ranges(len(data), ts, te, es, ee)

    lookback = int(getattr(args, "lookback", 10))
    epochs = int(getattr(args, "epochs", 20))
    batch_size = int(getattr(args, "batch_size", 6))
    lr = float(getattr(args, "lr", 0.01))
    normalization = getattr(args, "lstm_normalization", "zscore")

    model, scaler, metrics, y_true, y_pred, std_dev = train_and_evaluate(
        data, lookback, epochs, normalization, batch_size, lr,
        range(ts, te + 1), range(es, ee + 1)
    )

    save_path = Path(getattr(args, "save_model", "/tmp/lstm_model.pkl"))
    pytorch_model_path = save_path.with_suffix(".pth")
    torch.save(model.state_dict(), pytorch_model_path)

    scaler_state = {}
    if isinstance(scaler, StandardScaler):
        scaler_state = {"type": "zscore", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    elif isinstance(scaler, MinMaxScaler):
        scaler_state = {"type": "minmax", "min": scaler.min_.tolist(), "scale": scaler.scale_.tolist()}

    state = {
        "model_type": MODEL_TYPE, "lookback": lookback, "input_size": data.shape[1],
        "target_feature": target_key, "features": features,
        "recent_data": data[max(0, te - lookback + 1) : te + 1].tolist(),
        "scaler": scaler_state, "model_path": str(pytorch_model_path),
        "std_dev": std_dev, "train_end_index": te,
    }
    dump_model(state, str(save_path))

    out = {
        "mode": "lstm_trained", "metrics": metrics,
        "eval_y_true": y_true, "eval_predictions": y_pred,
        "saved_model": str(save_path),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))