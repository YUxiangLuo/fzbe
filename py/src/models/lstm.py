# src/models/lstm.py
# -*- coding: utf-8 -*-

import json, math, sys
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ---------- metrics ----------
def rmse(y_true, y_pred):
    if not y_true:
        return float("nan")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def mae(y_true, y_pred):
    if not y_true:
        return float("nan")
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def mape(y_true, y_pred):
    if not y_true:
        return float("nan")
    eps = 1e-12
    return (
        100.0
        * sum(
            abs((a - b) / (a if abs(a) > eps else eps)) for a, b in zip(y_true, y_pred)
        )
        / len(y_true)
    )


def r2_score(y_true, y_pred):
    if not y_true:
        return float("nan")
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean_y) ** 2 for a in y_true)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


# ---------- utils ----------
def _parse_feature_types(s: Optional[str]) -> Dict[str, str]:
    if not s:
        return {}
    try:
        ft = json.loads(s)
        if not isinstance(ft, dict):
            raise ValueError
        return {str(k): str(v).lower() for k, v in ft.items()}
    except Exception:
        raise ValueError("--feature_types must be a JSON object")


def _split_features(records: List[Dict[str, Any]]) -> List[str]:
    keys = None
    for r in records:
        if "x" not in r or not isinstance(r["x"], dict):
            raise ValueError("each record must contain 'x' dict")
        ks = sorted(r["x"].keys())
        if keys is None:
            keys = ks
        elif ks != keys:
            raise ValueError("all records must share the same x keys and order")
    return keys or []


def _fit_numeric_scaler(values: np.ndarray, method: str):
    # returns (params, transform_fn, inverse_fn)
    method = method.lower()
    if method == "none":
        return (None, lambda a: a, lambda a: a)
    if method == "zscore":
        mu = float(values.mean()) if values.size > 0 else 0.0
        sd = float(values.std(ddof=0)) if values.size > 0 else 1.0
        if sd == 0.0:
            sd = 1.0
        return (("z", mu, sd), lambda a: (a - mu) / sd, lambda a: a * sd + mu)
    if method == "minmax":
        mn = float(values.min()) if values.size > 0 else 0.0
        mx = float(values.max()) if values.size > 0 else 1.0
        if mx == mn:
            mx = mn + 1.0
        return (
            ("m", mn, mx),
            lambda a: (a - mn) / (mx - mn),
            lambda a: a * (mx - mn) + mn,
        )
    raise ValueError("--norm must be one of none|zscore|minmax")


class SeqDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X_seq = X_seq
        self.y_seq = y_seq

    def __len__(self):
        return self.X_seq.shape[0]

    def __getitem__(self, i):
        return self.X_seq[i], self.y_seq[i]


class LSTMReg(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B,T,D)
        o, _ = self.lstm(x)
        h = o[:, -1, :]
        y = self.head(h).squeeze(-1)
        return y


def _prepare_arrays(
    records: List[Dict[str, Any]],
    feature_types: Dict[str, str],
    norm: str,
    train_slice: slice,
    lookback: int,
):
    n = len(records)
    # y 允许为 None/缺失（预测时可以不提供），此处训练/评估会用到
    y_list = [r.get("y", None) for r in records]
    # 用 0 占位以便缩放函数运行；真实训练/评估时再用索引取需要的 y
    y = np.array([float(v) if v is not None else 0.0 for v in y_list], dtype=float)

    feat_keys = _split_features(records)
    numeric_keys, cat_keys = [], []
    for k in feat_keys:
        t = feature_types.get(k, "numeric")
        if t in ("numeric", "binary"):
            numeric_keys.append(k)
        elif t == "categorical":
            cat_keys.append(k)
        else:
            raise ValueError(f"unknown feature type for '{k}': {t}")

    # 构建数值/分类原始矩阵
    X_num = (
        np.stack([[float(r["x"][k]) for k in numeric_keys] for r in records], axis=0)
        if numeric_keys
        else np.zeros((n, 0), dtype=float)
    )
    X_cat_raw = [[r["x"][k] for k in cat_keys] for r in records] if cat_keys else []

    # 训练段词表
    cat_maps = {}
    if cat_keys:
        train_rows = range(train_slice.start, train_slice.stop)
        for j, k in enumerate(cat_keys):
            vals = [X_cat_raw[i][j] for i in train_rows]
            vocab = {
                v: i + 1 for i, v in enumerate(sorted(set(vals), key=lambda x: str(x)))
            }  # 0: UNK
            cat_maps[k] = vocab
        X_cat = np.zeros((n, len(cat_keys)), dtype=np.int64)
        for i in range(n):
            for j, k in enumerate(cat_keys):
                X_cat[i, j] = cat_maps[k].get(X_cat_raw[i][j], 0)
    else:
        X_cat = np.zeros((n, 0), dtype=np.int64)

    # 缩放器（基于训练段）
    # 注意：y 仅在训练/评估需要取真实值位置；这里用训练段 y 统计重拟合缩放器
    y_train_vals = np.array(
        [float(records[i]["y"]) for i in range(train_slice.start, train_slice.stop)],
        dtype=float,
    )
    y_params, y_tf, y_inv = _fit_numeric_scaler(y_train_vals, norm)
    num_params = []
    X_num_tf = np.empty_like(X_num)
    for j in range(X_num.shape[1]):
        p, tf, _inv = _fit_numeric_scaler(X_num[train_slice, j], norm)
        num_params.append(p)
        X_num_tf[:, j] = tf(X_num[:, j])
    y_tf_full = y_tf(y)

    # 分类 one-hot
    if X_cat.shape[1] > 0:
        oh_cols = []
        for j, k in enumerate(cat_keys):
            vocab_size = max(cat_maps[k].values(), default=0) + 1
            oh = np.zeros((n, vocab_size), dtype=float)
            oh[np.arange(n), X_cat[:, j]] = 1.0
            oh_cols.append(oh)
        X_cat_oh = np.concatenate(oh_cols, axis=1)
    else:
        X_cat_oh = np.zeros((n, 0), dtype=float)

    X_all = (
        np.concatenate([X_num_tf, X_cat_oh], axis=1)
        if X_cat_oh.shape[1] > 0
        else X_num_tf
    )
    D = X_all.shape[1]

    def make_sequences(start: int, end: int):
        X_seq, y_seq = [], []
        for t in range(start + lookback, end + 1):  # 闭区间：end+1
            # 只有当 y 有真实值时才加入样本
            if records[t].get("y", None) is None:
                continue
            X_seq.append(X_all[t - lookback : t, :])
            y_seq.append(y_tf_full[t])
        if not X_seq:
            return np.zeros((0, lookback, D), dtype=float), np.zeros((0,), dtype=float)
        return np.stack(X_seq, axis=0), np.array(y_seq, dtype=float)

    return {
        "feat_keys": feat_keys,
        "numeric_keys": numeric_keys,
        "categorical_keys": cat_keys,
        "cat_maps": cat_maps,
        "num_params": num_params,
        "y_params": y_params,
        "y_inv": y_inv,
        "X_all": X_all,
        "y_all_tf": y_tf_full,
        "make_sequences": make_sequences,
    }


def _inverse_y(y_scaled: List[float], y_params):
    kind = y_params[0] if y_params else None
    if kind == "z":
        _, mu, sd = y_params
        return [v * sd + mu for v in y_scaled]
    if kind == "m":
        _, mn, mx = y_params
        return [v * (mx - mn) + mn for v in y_scaled]
    return y_scaled


def _reconstruct_window_from_data(records: List[Dict[str, Any]], meta: Dict[str, Any]) -> np.ndarray:
    """
    Builds a prediction-ready window from raw data using scaler params from a trained model.
    """
    model_lookback = int(meta["lookback"])
    input_dim = int(meta["input_dim"])
    
    if not records or len(records) < model_lookback:
        raise ValueError(f"Not enough records to form a lookback window of size {model_lookback}")

    num_keys = meta["numeric_keys"]
    cat_keys = meta["categorical_keys"]
    n = len(records)

    X_num = np.stack([[float(records[i]["x"][k]) for k in num_keys] for i in range(n)], axis=0) if num_keys else np.zeros((n, 0))

    # Apply saved scaling parameters
    X_num_tf = np.empty_like(X_num, dtype=float)
    for j, (kind, a, b) in enumerate(meta["num_params"]):
        if kind == "z": X_num_tf[:, j] = (X_num[:, j] - a) / b
        elif kind == "m": X_num_tf[:, j] = (X_num[:, j] - a) / (b - a)
        else: X_num_tf[:, j] = X_num[:, j]

    # Apply saved one-hot encoding
    if cat_keys:
        oh_cols = []
        for k in cat_keys:
            vocab = meta["cat_maps"].get(k, {})
            idx = [vocab.get(records[i]["x"].get(k), 0) for i in range(n)]
            vocab_size = (max(vocab.values()) if vocab else 0) + 1
            oh = np.zeros((n, vocab_size), dtype=float)
            oh[np.arange(n), idx] = 1.0
            oh_cols.append(oh)
        X_cat_oh = np.concatenate(oh_cols, axis=1)
    else:
        X_cat_oh = np.zeros((n, 0), dtype=float)

    X_all = np.concatenate([X_num_tf, X_cat_oh], axis=1) if X_cat_oh.shape[1] > 0 else X_num_tf
    
    if X_all.shape[1] != input_dim:
        raise ValueError(f"Input dimension mismatch: provided data has {X_all.shape[1]} features, model expects {input_dim}")

    return X_all[-model_lookback:, :].copy()




# ---------- predict (for ensemble) ----------
def predict(model_path: str, steps: int, data: Optional[str] = None) -> List[float]:
    """
    纯预测函数，用于 ensemble 调用。
    加载已训练的 LSTM 模型并预测 steps 步。
    
    Args:
        model_path: 模型文件路径
        steps: 预测步数
        data: 可选的 JSON 数据字符串（如果模型没有嵌入 last_window）
    
    Returns:
        预测值列表
    """
    state = torch.load(model_path, map_location="cpu")
    meta = state["meta"]
    model = LSTMReg(
        meta["input_dim"], meta["hidden"], meta["num_layers"], meta["dropout"]
    )
    model.load_state_dict(state["model"])
    model.eval()

    model_lookback = int(meta["lookback"])
    input_dim = int(meta["input_dim"])

    # 获取窗口（优先使用嵌入的 last_window）
    window = None
    if "last_window" in meta:
        window = np.array(meta["last_window"], dtype=float)
    elif data:
        # 从提供的数据构造窗口
        try:
            arr = json.loads(data)
            records = arr if isinstance(arr, list) else []
            if not records or len(records) < model_lookback:
                raise ValueError("not enough records to form lookback window")

            num_keys = meta["numeric_keys"]
            cat_keys = meta["categorical_keys"]
            n = len(records)
            
            X_num = (
                np.stack(
                    [[float(records[i]["x"][k]) for k in num_keys] for i in range(n)],
                    axis=0,
                )
                if num_keys
                else np.zeros((n, 0), dtype=float)
            )
            
            # 数值缩放
            X_num_tf = np.empty_like(X_num, dtype=float)
            for j, (kind, a, b) in enumerate(meta["num_params"]):
                if kind == "z":
                    X_num_tf[:, j] = (X_num[:, j] - a) / b
                elif kind == "m":
                    X_num_tf[:, j] = (X_num[:, j] - a) / (b - a)
                else:
                    X_num_tf[:, j] = X_num[:, j]
            
            # 分类 one-hot
            if cat_keys:
                oh_cols = []
                for k, vocab in meta["cat_maps"].items():
                    idx = [vocab.get(records[i]["x"][k], 0) for i in range(n)]
                    vocab_size = (max(vocab.values()) if vocab else 0) + 1
                    oh = np.zeros((n, vocab_size), dtype=float)
                    oh[np.arange(n), idx] = 1.0
                    oh_cols.append(oh)
                X_cat_oh = np.concatenate(oh_cols, axis=1)
            else:
                X_cat_oh = np.zeros((n, 0), dtype=float)
            
            X_all = (
                np.concatenate([X_num_tf, X_cat_oh], axis=1)
                if X_cat_oh.shape[1] > 0
                else X_num_tf
            )
            window = X_all[-model_lookback:, :].copy()
        except Exception as e:
            raise ValueError(f"Failed to construct window from data: {e}")
    else:
        raise ValueError("Model has no embedded last_window and no data provided")

    # 递归预测
    preds_scaled = []
    for _ in range(steps):
        x = torch.tensor(window[None, ...], dtype=torch.float32)
        with torch.no_grad():
            y_hat = model(x).cpu().numpy().ravel()[0]
        preds_scaled.append(float(y_hat))
        window = np.vstack([window[1:], window[-1]])

    # 反缩放
    return _inverse_y(preds_scaled, meta["y_params"])


# ---------- public entry ----------
def run(args) -> None:
    pretty = bool(getattr(args, "pretty", False))
    load_path = getattr(args, "load_model", None)
    save_path = getattr(args, "save_model", None)
    norm = getattr(args, "norm", "zscore").lower()

    if norm not in ("none", "zscore", "minmax"):
        print("--norm must be one of none|zscore|minmax", file=sys.stderr)
        sys.exit(2)

    # ------------ 加载/预测模式 ------------
    if load_path:
        state = torch.load(load_path, map_location="cpu")
        meta = state["meta"]
        model = LSTMReg(
            meta["input_dim"], meta["hidden"], meta["num_layers"], meta["dropout"]
        )
        model.load_state_dict(state["model"])
        model.eval()

        # 决定 steps
        steps = None
        fs = getattr(args, "forecast_steps", None)
        if fs is not None:
            steps = int(fs)
            if steps <= 0:
                raise ValueError("--forecast_steps must be > 0")
        
        if steps is None and hasattr(args, "eval_start") and hasattr(args, "eval_end") and args.eval_start is not None and args.eval_end is not None:
            try:
                es, ee = int(args.eval_start), int(args.eval_end)
                if ee >= es:
                    steps = ee - es + 1
            except Exception:
                pass

        if steps is None:
            raise ValueError("When --load_model, you must provide either --forecast_steps or both --eval_start and --eval_end.")


        out = {"mode": "lstm_loaded_predict", "config": meta, "forecast_steps": steps}

        model_lookback = int(meta["lookback"])
        input_dim = int(meta["input_dim"])

        # 路径 A：如果提供了 --data，则用它来构造最新的预测窗口
        window = None
        if hasattr(args, "data") and args.data:
            try:
                records = json.loads(args.data)
                if not isinstance(records, list): raise ValueError("JSON data must be an array.")
                window = _reconstruct_window_from_data(records, meta)
            except Exception as e:
                raise ValueError(f"Failed to construct window from --data: {e}")

        # 路径 B：如果不提供 --data，则使用模型中嵌入的 last_window
        if window is None:
            if "last_window" not in meta:
                raise ValueError("Model has no embedded last_window, you must provide --data for prediction.")
            window = np.array(meta["last_window"], dtype=float)
            if window.shape != (model_lookback, input_dim):
                raise ValueError("Embedded last_window has an incorrect shape. You may need to provide --data.")


        # 递归预测 steps 步（特征占位：重复最后一行）
        preds_scaled = []
        for _ in range(steps):
            x = torch.tensor(window[None, ...], dtype=torch.float32)
            with torch.no_grad():
                y_hat = model(x).cpu().numpy().ravel()[0]
            preds_scaled.append(float(y_hat))
            window = np.vstack([window[1:], window[-1]])

        # 反缩放 y
        y_params = meta["y_params"]
        if y_params and y_params[0] == "z":
            _, mu, sd = y_params
            preds = [v * sd + mu for v in preds_scaled]
        elif y_params and y_params[0] == "m":
            _, mn, mx = y_params
            preds = [v * (mx - mn) + mn for v in preds_scaled]
        else:
            preds = preds_scaled
        out["predictions"] = preds

        # 若有 data 且 eval_start/end，则顺带评估（闭区间）
        if (
            hasattr(args, "data")
            and args.data
            and hasattr(args, "eval_start")
            and hasattr(args, "eval_end")
        ):
            try:
                arr = json.loads(args.data)
                records = arr if isinstance(arr, list) else []
                es, ee = int(args.eval_start), int(args.eval_end)
                y_true = (
                    [float(records[i]["y"]) for i in range(es, ee + 1)]
                    if records and ee >= es
                    else []
                )
                if len(y_true) == len(preds):
                    out["metrics"] = {
                        "rmse": rmse(y_true, preds),
                        "mae": mae(y_true, preds),
                        "mape": mape(y_true, preds),
                        "r2": r2_score(y_true, preds),
                    }
            except Exception:
                pass

        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    # ------------ 训练+评估模式 ------------
    try:
        arr = json.loads(args.data)
        if not isinstance(arr, list):
            raise ValueError
        records = arr
    except Exception:
        print("Failed to parse --data as JSON array", file=sys.stderr)
        sys.exit(2)

    ft = _parse_feature_types(getattr(args, "feature_types", None))
    ts, te = int(args.train_start), int(args.train_end)  # [ts, te] 闭区间
    es, ee = int(args.eval_start), int(args.eval_end)  # [es, ee] 闭区间
    n = len(records)
    
    # 范围校验（闭区间）
    if not (0 <= ts <= te < n):
        print(f"Invalid train range: [{ts},{te}] within [0,{n-1}]", file=sys.stderr)
        sys.exit(2)
    if not (0 <= es <= ee < n):
        print(f"Invalid eval range: [{es},{ee}] within [0,{n-1}]", file=sys.stderr)
        sys.exit(2)
    if ee < es:
        print("eval_end must be >= eval_start", file=sys.stderr)
        sys.exit(2)
    if es <= te:
        print("eval must start after train_end to avoid leakage", file=sys.stderr)
        sys.exit(2)

    train_len = te - ts + 1  # 闭区间长度
    eval_len = ee - es + 1  # 闭区间长度
    auto = {}

    # lookback 安全默认与自适应
    lb_in = getattr(args, "lookback", None)
    if lb_in is None:
        lookback = max(2, min(12, int(max(2, round(train_len * 0.2)))))  # ~20% of train
        auto["lookback"] = {
            "input": None,
            "final": lookback,
            "reason": "heuristic: ~20% of train, clipped to [2,12]",
        }
    else:
        lookback = int(lb_in)
    if eval_len < lookback + 1:
        lookback = max(1, eval_len - 1)
        auto["lookback"] = {
            "input": lb_in,
            "final": lookback,
            "reason": f"ensure at least 1 eval sample (eval_len={eval_len})",
        }
    if train_len < lookback + 1:
        lookback = max(1, min(lookback, train_len - 1))
        auto["lookback"] = {
            "input": lb_in,
            "final": lookback,
            "reason": f"ensure at least 1 train sample (train_len={train_len})",
        }
    if eval_len < lookback + 1:
        print(
            f"no evaluation samples even after auto-adjust: eval_len={eval_len}, lookback={lookback}",
            file=sys.stderr,
        )
        sys.exit(2)
    if train_len < lookback + 1:
        print(
            f"train segment too short after auto-adjust: train_len={train_len}, lookback={lookback}",
            file=sys.stderr,
        )
        sys.exit(2)

    # 批量大小
    bs_in = getattr(args, "batch_size", None)
    if bs_in is None:
        batch_size = max(8, min(64, train_len - lookback))
        auto["batch_size"] = {
            "input": None,
            "final": batch_size,
            "reason": "bounded by train samples, clipped to [8,64]",
        }
    else:
        batch_size = max(1, int(bs_in))
        if batch_size > max(1, train_len - lookback):
            auto["batch_size"] = {
                "input": bs_in,
                "final": max(1, train_len - lookback),
                "reason": "too large for dataset",
            }
            batch_size = max(1, train_len - lookback)

    # 训练 epoch
    ep_in = getattr(args, "epochs", None)
    if ep_in is None:
        epochs = 50 if train_len >= 24 else 30
        auto["epochs"] = {
            "input": None,
            "final": epochs,
            "reason": "small-data heuristic",
        }
    else:
        epochs = max(5, int(ep_in))

    # 学习率
    lr_in = getattr(args, "lr", None)
    if lr_in is None:
        lr = 1e-3
        auto["lr"] = {"input": None, "final": lr, "reason": "default safe lr"}
    else:
        lr = float(lr_in)

    # 随机种子
    seed = int(getattr(args, "seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 组装数据（注意：slice 的 stop 需要 +1 以适配闭区间）
    prep = _prepare_arrays(records, ft, norm, slice(ts, te + 1), lookback)
    make_seq = prep["make_sequences"]
    Xtr, ytr = make_seq(ts, te)
    Xev, yev = make_seq(es, ee)
    if Xev.shape[0] == 0:
        print("no evaluation samples; adjust lookback/eval range", file=sys.stderr)
        sys.exit(2)

    input_dim = Xtr.shape[-1]
    model = LSTMReg(
        input_dim,
        hidden=int(getattr(args, "hidden", 64)),
        num_layers=int(getattr(args, "layers", 1)),
        dropout=float(getattr(args, "dropout", 0.0)),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_ds = SeqDataset(
        torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 训练：早停 + 梯度裁剪
    patience = 10
    best_loss = float("inf")
    pat = 0
    best_state = None
    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += float(loss.detach().cpu())

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            pat = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 评估（滚动一步）
    model.eval()
    preds_scaled = []
    window = prep["X_all"][es - lookback : es, :].copy()
    for t in range(es, ee + 1):  # 闭区间
        x = torch.tensor(window[None, ...], dtype=torch.float32)
        with torch.no_grad():
            y_hat = model(x).cpu().numpy().ravel()[0]
        preds_scaled.append(float(y_hat))
        window = np.vstack([window[1:], prep["X_all"][t, :]])

    preds = _inverse_y(preds_scaled, prep["y_params"])
    y_true = [float(records[i]["y"]) for i in range(es, ee + 1)]  # 闭区间

    out = {
        "mode": "lstm_train_eval",
        "norm": norm,
        "lookback": lookback,
        "evaluated_points": len(preds),
        "auto_adjustments": auto,
        "metrics": {
            "rmse": rmse(y_true, preds),
            "mae": mae(y_true, preds),
            "mape": mape(y_true, preds),
            "r2": r2_score(y_true, preds),
        },
        "notes": ["All indices use closed interval semantics: [start, end]."],
    }

    # === 嵌入最后窗口（已缩放/已编码），便于无 data 预测 ===
    last_window = prep["X_all"][te - lookback + 1 : te + 1, :].astype(
        float
    )  # (lookback, input_dim) 闭区间切片

    if save_path:
        meta = {
            "input_dim": int(input_dim),
            "hidden": int(getattr(args, "hidden", 64)),
            "num_layers": int(getattr(args, "layers", 1)),
            "dropout": float(getattr(args, "dropout", 0.0)),
            "lookback": lookback,
            "norm": norm,
            "feat_keys": prep["feat_keys"],
            "numeric_keys": prep["numeric_keys"],
            "categorical_keys": prep["categorical_keys"],
            "cat_maps": prep["cat_maps"],
            "num_params": prep["num_params"],
            "y_params": prep["y_params"],
            "last_window": last_window.tolist(),  # 关键：保存最后窗口
        }
        torch.save({"model": model.state_dict(), "meta": meta}, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))