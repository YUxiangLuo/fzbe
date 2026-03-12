#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual Boosting Ensemble for Heterogeneous Time Series Models.

Stage 0 fits the original target. Subsequent stages fit the residuals left by the
current ensemble. Stage selection is performed on a held-out validation window,
while each selected stage is retrained on the full train_range for final eval and
artifact persistence.

Uses subprocess to call base models for consistency with API behavior.
"""
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from data_utils import ParsedData, infer_feature_types, load_data_from_json
from ensemble_utils import (
    build_ensemble_subprocess_data,
    get_member_model_path,
    parse_base_model_files_arg,
    persist_member_model_artifact,
    remove_member_model_artifact,
)
from metrics import mae, mape, r2, rmse
from serialization import dump_model, load_model
from subprocess_utils import (
    extract_predictions_from_result,
    run_base_model_fit_subprocess,
    run_base_model_predict_subprocess,
    run_base_model_subprocess,
)

MODEL_TYPE = "boosting"
SUPPORTED_BASE_MODELS = ["ma", "es", "arima", "lstm"]


# ---------- utilities ----------
def _format_predictions(preds: List[float], std_dev: float = 0.0) -> List[Dict[str, float]]:
    return [{"prediction": float(p), "std_dev": float(std_dev)} for p in preds]


def _compute_phase1_train_size(train_size: int) -> int:
    if train_size <= 15:
        return max(train_size - 3, int(train_size * 0.85))
    return max(3, int(train_size * 0.8))


def _load_state(model_path: str) -> Dict[str, Any]:
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid Boosting model file.")
    return state


def _is_residual_boosting_state(state: Dict[str, Any]) -> bool:
    return state.get("boosting_variant") == "residual"


def _get_stage_coefficients(model_chain: List[str], state: Dict[str, Any]) -> List[float]:
    explicit_coefficients = state.get("stage_coefficients")
    if isinstance(explicit_coefficients, list) and len(explicit_coefficients) == len(model_chain):
        return [float(coef) for coef in explicit_coefficients]

    learning_rate = float(state.get("learning_rate", 0.3))
    return [1.0 if idx == 0 else learning_rate for idx in range(len(model_chain))]


def _iter_stage_models(state: Dict[str, Any]):
    model_chain = state.get("model_chain")
    base_model_states = state.get("base_model_states")
    if not isinstance(model_chain, list) or len(model_chain) == 0:
        raise ValueError("[VALIDATION] Saved Boosting model is missing a valid model_chain.")
    if not isinstance(base_model_states, dict):
        raise ValueError("[VALIDATION] Saved Boosting model is missing base_model_states.")

    stage_coefficients = _get_stage_coefficients(model_chain, state)
    usage_counts: Dict[str, int] = {}

    for stage_idx, model_name in enumerate(model_chain):
        model_info = base_model_states.get(model_name)
        if not isinstance(model_info, dict):
            raise ValueError(f"[VALIDATION] Missing saved state for base model: {model_name}")

        models = model_info.get("models")
        if not isinstance(models, list):
            raise ValueError(f"[VALIDATION] Missing saved rounds for base model: {model_name}")

        usage_idx = usage_counts.get(model_name, 0)
        if usage_idx >= len(models):
            raise ValueError(f"[VALIDATION] Missing saved round {usage_idx} for base model: {model_name}")

        usage_counts[model_name] = usage_idx + 1
        yield stage_idx, model_name, float(stage_coefficients[stage_idx]), models[usage_idx]


def _build_residual_parsed_data(
    parsed_data: ParsedData,
    train_range: range,
    train_target_values: np.ndarray,
) -> ParsedData:
    df = parsed_data.df.copy().astype({parsed_data.target_key: float})
    target_col_idx = df.columns.get_loc(parsed_data.target_key)
    df.iloc[list(train_range), target_col_idx] = train_target_values
    return ParsedData(
        df=df,
        target_key=parsed_data.target_key,
        time_key=parsed_data.time_key,
        feature_keys=parsed_data.feature_keys,
    )


def _normalize_fitted_values(values: Any, expected_length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size < expected_length:
        arr = np.concatenate([np.zeros(expected_length - arr.size), arr])
    elif arr.size > expected_length:
        arr = arr[-expected_length:]
    return arr


def _compute_ma_in_sample_fitted(train_target: np.ndarray, window: int) -> np.ndarray:
    fitted = np.zeros(len(train_target), dtype=float)
    for idx in range(1, len(train_target)):
        history_start = max(0, idx - window)
        history = train_target[history_start:idx]
        fitted[idx] = float(np.mean(history)) if history.size > 0 else 0.0
    return fitted


def _compute_lstm_in_sample_fitted(
    model_file: str,
    parsed_data: ParsedData,
    train_range: range,
) -> np.ndarray:
    from models import lstm as lstm_model

    state = lstm_model._load_state(model_file)
    train_df = parsed_data.df.iloc[list(train_range)]
    preprocessor = state["preprocessor"]
    scaled_train = preprocessor.transform(train_df)
    if isinstance(scaled_train, csr_matrix):
        scaled_train = scaled_train.toarray()

    look_back = int(state["look_back"])
    fitted = np.zeros(len(train_df), dtype=float)
    if len(train_df) <= look_back:
        return fitted

    forecast_strategy = state.get("forecast_strategy", lstm_model.LEGACY_FORECAST_STRATEGY)
    if forecast_strategy == lstm_model.DIRECT_FORECAST_STRATEGY and int(state.get("forecast_horizon", 0) or 0) > 0:
        windows = np.asarray(
            [scaled_train[idx - look_back : idx, :] for idx in range(look_back, len(train_df))],
            dtype=float,
        )
        pred_scaled = state["keras_model"].predict(windows, verbose=0)
        if pred_scaled.ndim == 1:
            pred_scaled = pred_scaled.reshape(-1, 1)
        next_step_scaled = pred_scaled[:, 0]
    else:
        trainX, _ = lstm_model._create_autoregressive_dataset(scaled_train, look_back)
        if len(trainX) == 0:
            return fitted
        next_step_scaled = state["keras_model"].predict(trainX, verbose=0).reshape(-1)

    target_scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    n_numeric_features = target_scaler.n_features_in_
    dummy_array = np.zeros((len(next_step_scaled), n_numeric_features))
    dummy_array[:, 0] = next_step_scaled
    unscaled_predictions = target_scaler.inverse_transform(dummy_array)[:, 0]
    fitted[look_back: look_back + len(unscaled_predictions)] = unscaled_predictions
    return fitted


def _compute_in_sample_fitted_predictions(
    model_name: str,
    model_file: str,
    parsed_data: ParsedData,
    train_range: range,
) -> np.ndarray:
    train_target = np.asarray(parsed_data.y[train_range], dtype=float)

    if model_name == "ma":
        from models import ma as ma_model

        state = ma_model._load_state(model_file)
        return _compute_ma_in_sample_fitted(train_target, int(state["window"]))

    if model_name == "es":
        from models import es as es_model

        state = es_model._load_state(model_file)
        fitted = getattr(state["model"], "fittedvalues", [])
        return _normalize_fitted_values(fitted, len(train_target))

    if model_name == "arima":
        from models import arima as arima_model

        state = arima_model._load_state(model_file)
        model = state["model"]
        if hasattr(model, "predict_in_sample"):
            fitted = model.predict_in_sample()
        elif hasattr(model, "arima_res_") and hasattr(model.arima_res_, "fittedvalues"):
            fitted = model.arima_res_.fittedvalues
        else:
            fitted = []
        return _normalize_fitted_values(fitted, len(train_target))

    if model_name == "lstm":
        return _compute_lstm_in_sample_fitted(model_file, parsed_data, train_range)

    raise ValueError(f"[VALIDATION] Unsupported boosting base model: {model_name}")


def _predict_eval_segment(
    model_name: str,
    model_file: str,
    train_range: range,
    eval_range: range,
    timeout: int,
) -> np.ndarray:
    forecast_steps = eval_range.stop - train_range.stop
    forecast_offset = eval_range.start - train_range.stop
    result = run_base_model_predict_subprocess(
        model_type=model_name,
        model_file=model_file,
        forecast_steps=forecast_steps,
        timeout=timeout,
    )
    preds = np.asarray(extract_predictions_from_result(result), dtype=float)
    eval_preds = preds[forecast_offset: forecast_offset + len(eval_range)]
    if len(eval_preds) != len(eval_range):
        raise ValueError("[VALIDATION] Unable to generate full eval_range predictions for boosting stage.")
    return eval_preds


def _make_temp_model_path(prefix: str) -> str:
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".pkl")
    os.close(fd)
    os.remove(temp_path)
    return temp_path


def _build_round_records(
    round_parsed_data: ParsedData,
    model_name: str,
    model_params: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return build_ensemble_subprocess_data(round_parsed_data, [model_name], {model_name: model_params})


def _get_model_timeout(model_name: str, params: Dict[str, Any]) -> int:
    if model_name == "lstm":
        return max(120, int(params.get("epochs", 20)) * 10)
    return 120


def _train_stage_model(
    model_name: str,
    round_numeric_records: List[Dict[str, Any]],
    round_lstm_records: List[Dict[str, Any]],
    train_range: range,
    eval_range: range,
    params: Dict[str, Any],
    timeout: int,
) -> Tuple[str, np.ndarray]:
    temp_model_file = _make_temp_model_path(f"boosting_stage_{model_name}_")
    params_with_save = params.copy()
    params_with_save["save_model"] = temp_model_file

    result = run_base_model_subprocess(
        model_type=model_name,
        data=round_lstm_records if model_name == "lstm" else round_numeric_records,
        train_start=train_range.start,
        train_end=train_range.stop - 1,
        eval_start=eval_range.start,
        eval_end=eval_range.stop - 1,
        model_params=params_with_save,
        timeout=timeout,
    )
    eval_predictions = np.asarray(extract_predictions_from_result(result), dtype=float)
    return temp_model_file, eval_predictions


def _validate_fit_args(parsed_data: ParsedData, ts: int, te: int, state: Dict[str, Any]) -> None:
    n = parsed_data.n_obs
    if not (0 <= ts <= te < n):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], n_obs={n}")

    list(_iter_stage_models(state))


def _validate_training_args(
    parsed_data: ParsedData,
    ts: int,
    te: int,
    es: int,
    ee: int,
    models_to_use: List[str],
    model_params: Dict[str, Any],
) -> None:
    n = parsed_data.n_obs

    if not models_to_use:
        raise ValueError("[VALIDATION] models_to_use cannot be empty.")

    for name in models_to_use:
        if name not in SUPPORTED_BASE_MODELS:
            raise ValueError(f"[VALIDATION] Unknown model: {name}")

    if not (0 <= ts <= te < n):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], n_obs={n}")
    if not (0 <= es <= ee < n):
        raise ValueError(f"[VALIDATION] Invalid eval range: [{es},{ee}], n_obs={n}")
    if es <= te:
        raise ValueError("[VALIDATION] eval_start must be > train_end.")

    for model_name in models_to_use:
        if model_name not in model_params:
            raise ValueError(f"[VALIDATION] Missing parameters for model: {model_name}")

        params = model_params[model_name]
        if model_name == "arima":
            if "d" not in params:
                raise ValueError("[VALIDATION] ARIMA requires 'd' parameter.")
            if params["d"] < 0:
                raise ValueError("[VALIDATION] ARIMA d must be >= 0.")
        elif model_name == "es":
            if "alpha" not in params:
                raise ValueError("[VALIDATION] ES requires 'alpha' parameter.")
            if not (0 < params["alpha"] <= 1):
                raise ValueError("[VALIDATION] ES alpha must be in (0, 1].")
        elif model_name == "ma":
            if "window" not in params:
                raise ValueError("[VALIDATION] MA requires 'window' parameter.")
            if params["window"] < 1:
                raise ValueError("[VALIDATION] MA window must be >= 1.")
        elif model_name == "lstm":
            if "look_back" not in params:
                raise ValueError("[VALIDATION] LSTM requires 'look_back' parameter.")
            if "epochs" not in params:
                raise ValueError("[VALIDATION] LSTM requires 'epochs' parameter.")
            if params["look_back"] < 1:
                raise ValueError("[VALIDATION] LSTM look_back must be >= 1.")
            if params["epochs"] < 1:
                raise ValueError("[VALIDATION] LSTM epochs must be >= 1.")
            if te - ts + 1 <= params["look_back"]:
                raise ValueError("[VALIDATION] Training range must be larger than LSTM look_back.")
            train_size = te - ts + 1
            boost_train_size = _compute_phase1_train_size(train_size)
            if boost_train_size <= params["look_back"]:
                raise ValueError(
                    f"[VALIDATION] LSTM look_back ({params['look_back']}) is too large for "
                    f"the boosting internal training split ({boost_train_size} points). "
                    f"Please reduce look_back or increase training data."
                )


# ---------- predict ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], float]:
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    state = state or _load_state(model_path)
    std_dev_residuals = float(state.get("std_dev_residuals", 0.0))

    final_predictions: Optional[np.ndarray] = None
    for _stage_idx, model_name, coefficient, model_state in _iter_stage_models(state):
        model_file = model_state["model_file"]
        result = run_base_model_predict_subprocess(
            model_type=model_name,
            model_file=model_file,
            forecast_steps=steps,
            timeout=60,
        )
        stage_predictions = np.asarray(extract_predictions_from_result(result), dtype=float)
        if final_predictions is None:
            final_predictions = coefficient * stage_predictions
        else:
            final_predictions += coefficient * stage_predictions

    if final_predictions is None:
        raise ValueError("[VALIDATION] Saved Boosting model is empty.")

    return final_predictions.tolist(), std_dev_residuals


# ---------- train and evaluate ----------
def train_and_evaluate(
    parsed_data: ParsedData,
    train_range: range,
    eval_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Any],
    learning_rate: float,
    max_rounds: int,
    base_model_files: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], List[str], List[float], Dict[str, Any], List[float], List[float], float, Dict[str, Any]]:
    train_len = len(train_range)
    if train_len < 8:
        raise ValueError("[VALIDATION] train_range too small for boosting (需要至少8个点，推荐≥12点)")

    boost_split = _compute_phase1_train_size(train_len)
    boost_train_range = range(train_range.start, train_range.start + boost_split)
    boost_val_range = range(train_range.start + boost_split, train_range.stop)

    print(f"[Boosting] 数据划分:", file=sys.stderr)
    print(f"  Boost Train Range: {boost_train_range} ({len(boost_train_range)} points)", file=sys.stderr)
    print(f"  Boost Val Range: {boost_val_range} ({len(boost_val_range)} points)", file=sys.stderr)
    print(f"  Eval Range: {eval_range} ({len(eval_range)} points)", file=sys.stderr)
    print(f"  Learning Rate: {learning_rate}", file=sys.stderr)
    print(f"  Max Rounds: {max_rounds}", file=sys.stderr)

    y_true_val = np.asarray(parsed_data.y[boost_val_range], dtype=float)
    residual_train = np.asarray(parsed_data.y[boost_train_range], dtype=float).copy()
    residual_val = y_true_val.copy()

    model_chain: List[str] = []
    stage_coefficients: List[float] = []
    search_summary: Dict[str, Any] = {
        "algorithm": "residual_boosting",
        "learning_rate": learning_rate,
        "stages": [],
    }

    for round_idx in range(max_rounds):
        coefficient = 1.0 if round_idx == 0 else learning_rate
        current_rmse = float("inf") if round_idx == 0 else rmse(np.zeros_like(residual_val), residual_val)
        round_parsed_data = _build_residual_parsed_data(parsed_data, boost_train_range, residual_train)

        stage_results: List[Dict[str, Any]] = []
        for model_name in models_to_use:
            params = model_params[model_name]
            timeout = _get_model_timeout(model_name, params)
            round_numeric_records, round_lstm_records = _build_round_records(
                round_parsed_data,
                model_name,
                params,
            )
            temp_model_file, stage_predictions = _train_stage_model(
                model_name,
                round_numeric_records,
                round_lstm_records,
                boost_train_range,
                boost_val_range,
                params,
                timeout,
            )
            fitted_train_predictions = _compute_in_sample_fitted_predictions(
                model_name,
                temp_model_file,
                round_parsed_data,
                boost_train_range,
            )
            next_residual_val = residual_val - coefficient * stage_predictions
            next_rmse = rmse(np.zeros_like(next_residual_val), next_residual_val)
            stage_results.append({
                "model_name": model_name,
                "coefficient": coefficient,
                "rmse": next_rmse,
                "temp_model_file": temp_model_file,
                "fitted_train_predictions": fitted_train_predictions,
                "next_residual_val": next_residual_val,
            })
            print(f"  Round {round_idx + 1} candidate {model_name}: residual RMSE={next_rmse:.4f}", file=sys.stderr)

        best_result = min(stage_results, key=lambda result: result["rmse"])
        improvement = float("inf") if current_rmse == float("inf") else current_rmse - best_result["rmse"]
        stage_summary = {
            "round": round_idx + 1,
            "objective": "validation_residual_rmse",
            "coefficient": coefficient,
            "previous_rmse": None if current_rmse == float("inf") else current_rmse,
            "candidates": {result["model_name"]: result["rmse"] for result in stage_results},
            "winner": best_result["model_name"],
            "winner_rmse": best_result["rmse"],
            "improvement": None if improvement == float("inf") else improvement,
        }

        if best_result["rmse"] < current_rmse - 1e-12:
            model_chain.append(best_result["model_name"])
            stage_coefficients.append(coefficient)
            residual_train = residual_train - coefficient * best_result["fitted_train_predictions"]
            residual_val = best_result["next_residual_val"]
            stage_summary["action"] = "selected"
            print(
                f"  ✓ Selected {best_result['model_name']} (residual RMSE: {best_result['rmse']:.4f})",
                file=sys.stderr,
            )
        else:
            stage_summary["action"] = "early_stop"
            print(
                f"  🛑 Early stopping: no candidate improves residual RMSE {current_rmse:.4f}",
                file=sys.stderr,
            )
            search_summary["stages"].append(stage_summary)
            for result in stage_results:
                remove_member_model_artifact(result["model_name"], result["temp_model_file"])
            break

        search_summary["stages"].append(stage_summary)
        for result in stage_results:
            remove_member_model_artifact(result["model_name"], result["temp_model_file"])

    if not model_chain:
        y_true_eval = np.asarray(parsed_data.y[eval_range], dtype=float)
        timestamps = parsed_data.timestamps
        metrics = {
            "rmse": float("inf"),
            "mae": float("inf"),
            "mape": float("inf"),
            "r2": float("-inf"),
            "evaluate_indices": list(eval_range),
            "evaluate_months": [timestamps[i] for i in eval_range] if timestamps else [],
        }
        return metrics, [], [], {}, y_true_eval.tolist(), [0.0] * len(y_true_eval), 0.0, search_summary

    final_eval_preds = np.zeros(len(eval_range), dtype=float)
    residual_full_train = np.asarray(parsed_data.y[train_range], dtype=float).copy()
    y_true_eval = np.asarray(parsed_data.y[eval_range], dtype=float)
    timestamps = parsed_data.timestamps
    base_model_states: Dict[str, Any] = {}

    for stage_idx, model_name in enumerate(model_chain):
        coefficient = stage_coefficients[stage_idx]
        params = model_params[model_name]
        timeout = _get_model_timeout(model_name, params)
        round_parsed_data = _build_residual_parsed_data(parsed_data, train_range, residual_full_train)
        artifact_origin = "temporary_training"

        if stage_idx == 0 and base_model_files and model_name in base_model_files and os.path.exists(base_model_files[model_name]):
            stage_model_file = base_model_files[model_name]
            stage_eval_predictions = _predict_eval_segment(
                model_name,
                stage_model_file,
                train_range,
                eval_range,
                timeout,
            )
            artifact_origin = "standalone_base"
        else:
            round_numeric_records, round_lstm_records = _build_round_records(
                round_parsed_data,
                model_name,
                params,
            )
            stage_model_file, stage_eval_predictions = _train_stage_model(
                model_name,
                round_numeric_records,
                round_lstm_records,
                train_range,
                eval_range,
                params,
                timeout,
            )

        fitted_train_predictions = _compute_in_sample_fitted_predictions(
            model_name,
            stage_model_file,
            round_parsed_data,
            train_range,
        )
        final_eval_preds += coefficient * stage_eval_predictions
        residual_full_train = residual_full_train - coefficient * fitted_train_predictions

        base_model_states.setdefault(model_name, {"models": []})["models"].append({
            "model_type": model_name,
            "model_file": stage_model_file,
            "params": params,
            "round": stage_idx,
            "coefficient": coefficient,
            "artifact_origin": artifact_origin,
        })

        print(
            f"  Stage {stage_idx + 1}: {model_name} coefficient={coefficient:.4f}, eval RMSE={rmse(y_true_eval, final_eval_preds):.4f}",
            file=sys.stderr,
        )

    final_residuals = y_true_eval - final_eval_preds
    std_dev_residuals = float(np.std(final_residuals, ddof=1)) if len(final_residuals) > 1 else 0.0
    metrics = {
        "rmse": rmse(y_true_eval, final_eval_preds),
        "mae": mae(y_true_eval, final_eval_preds),
        "mape": mape(y_true_eval, final_eval_preds),
        "r2": r2(y_true_eval, final_eval_preds),
        "evaluate_indices": list(eval_range),
        "evaluate_months": [timestamps[i] for i in eval_range] if timestamps else [],
    }

    print(f"  ✓ Final RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}", file=sys.stderr)
    print(f"  ✓ Model chain: {' → '.join(model_chain)}", file=sys.stderr)

    return (
        metrics,
        model_chain,
        stage_coefficients,
        base_model_states,
        y_true_eval.tolist(),
        final_eval_preds.tolist(),
        std_dev_residuals,
        search_summary,
    )


# ---------- public entry ----------
def run(args) -> None:
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
        parsed_data = load_data_from_json(args.data)
        ts, te = int(args.train_start), int(args.train_end)
        train_range = range(ts, te + 1)
        _validate_fit_args(parsed_data, ts, te, state)

        next_base_model_states: Dict[str, Any] = {}
        if _is_residual_boosting_state(state):
            residual_train = np.asarray(parsed_data.y[train_range], dtype=float).copy()

            for stage_idx, model_name, coefficient, previous_model_state in _iter_stage_models(state):
                params = previous_model_state.get("params", {})
                timeout = _get_model_timeout(model_name, params)
                round_parsed_data = _build_residual_parsed_data(parsed_data, train_range, residual_train)
                round_numeric_records, round_lstm_records = _build_round_records(
                    round_parsed_data,
                    model_name,
                    params,
                )
                member_save_path = get_member_model_path(save_path, model_name, stage_idx)
                run_base_model_fit_subprocess(
                    model_type=model_name,
                    data=round_lstm_records if model_name == "lstm" else round_numeric_records,
                    train_start=ts,
                    train_end=te,
                    source_model_file=previous_model_state["model_file"],
                    save_model_file=member_save_path,
                    timeout=timeout,
                )
                fitted_train_predictions = _compute_in_sample_fitted_predictions(
                    model_name,
                    member_save_path,
                    round_parsed_data,
                    train_range,
                )
                residual_train = residual_train - coefficient * fitted_train_predictions
                next_base_model_states.setdefault(model_name, {"models": []})["models"].append({
                    "model_type": model_name,
                    "model_file": member_save_path,
                    "params": params,
                    "round": stage_idx,
                    "coefficient": coefficient,
                })
        else:
            stage_model_params: Dict[str, Dict[str, Any]] = {}
            for _stage_idx, model_name, _coefficient, previous_model_state in _iter_stage_models(state):
                stage_model_params.setdefault(model_name, previous_model_state.get("params", {}))

            base_models_in_order = list(stage_model_params.keys())
            numeric_data_dicts, lstm_data_dicts = build_ensemble_subprocess_data(
                parsed_data,
                base_models_in_order,
                stage_model_params,
            )

            for stage_idx, model_name, coefficient, previous_model_state in _iter_stage_models(state):
                params = previous_model_state.get("params", {})
                timeout = _get_model_timeout(model_name, params)
                member_save_path = get_member_model_path(save_path, model_name, stage_idx)
                run_base_model_fit_subprocess(
                    model_type=model_name,
                    data=lstm_data_dicts if model_name == "lstm" else numeric_data_dicts,
                    train_start=ts,
                    train_end=te,
                    source_model_file=previous_model_state["model_file"],
                    save_model_file=member_save_path,
                    timeout=timeout,
                )
                next_base_model_states.setdefault(model_name, {"models": []})["models"].append({
                    "model_type": model_name,
                    "model_file": member_save_path,
                    "params": params,
                    "round": stage_idx,
                    "coefficient": coefficient,
                })

        next_state = {
            "model_type": MODEL_TYPE,
            "boosting_variant": state.get("boosting_variant", "legacy"),
            "model_chain": state["model_chain"],
            "stage_coefficients": _get_stage_coefficients(state["model_chain"], state),
            "base_model_states": next_base_model_states,
            "learning_rate": float(state.get("learning_rate", 0.3)),
            "std_dev_residuals": float(state.get("std_dev_residuals", 0.0)),
        }
        dump_model(next_state, save_path)

        out = {
            "mode": "boosting_fit_only",
            "model_chain": state["model_chain"],
            "saved_model": save_path,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    if load_path:
        steps = int(getattr(args, "forecast_steps", 0))
        preds, std_dev = predict(load_path, steps)
        out = {
            "mode": "boosting_loaded_predict",
            "forecast_steps": steps,
            "predictions": _format_predictions(preds, std_dev),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    if not hasattr(args, "data") or not args.data:
        raise ValueError("[VALIDATION] --data is required for training mode.")

    parsed_data = load_data_from_json(args.data)
    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)
    train_range = range(ts, te + 1)
    eval_range = range(es, ee + 1)
    models_to_use = [model.strip() for model in args.models.split(',') if model.strip()]

    def get_arg(name, default, cast_type=None):
        val = getattr(args, name.replace("-", "_"), None)
        if val is None:
            return default
        if cast_type:
            return cast_type(val)
        return val

    learning_rate = get_arg("learning_rate", 0.3, float)
    max_rounds = get_arg("max_rounds", len(models_to_use), int)
    if not (0 < learning_rate <= 1):
        raise ValueError("[VALIDATION] learning_rate must be in (0, 1]")
    if max_rounds < 1:
        raise ValueError("[VALIDATION] max_rounds must be >= 1")

    train_size = te - ts + 1
    boost_train_size = max(3, int(train_size * 0.85)) if train_size <= 15 else int(train_size * 0.8)
    auto_window = max(3, min(12, boost_train_size // 4))
    auto_look_back = max(3, min(12, boost_train_size // 4))
    if boost_train_size <= 12:
        auto_max_pq = 1
    elif boost_train_size <= 20:
        auto_max_pq = 2
    else:
        auto_max_pq = max(1, min(5, boost_train_size // 6))
    auto_epochs = 20 if train_size <= 15 else 30

    model_params: Dict[str, Dict[str, Any]] = {}
    base_model_files = parse_base_model_files_arg(get_arg("base_model_files", None))

    if "arima" in models_to_use:
        model_params["arima"] = {
            "d": get_arg("d", 0, int),
            "max_p": get_arg("max_p", auto_max_pq, int),
            "max_q": get_arg("max_q", auto_max_pq, int),
        }

    if "es" in models_to_use:
        model_params["es"] = {"alpha": get_arg("alpha", 0.5, float)}

    if "ma" in models_to_use:
        model_params["ma"] = {"window": get_arg("window", auto_window, int)}

    if "lstm" in models_to_use:
        feature_types_str = get_arg("feature_types", None)
        lstm_features_str = get_arg("lstm_features", None)
        target_feature = get_arg("lstm_target_feature", parsed_data.target_key)
        lstm_feature_names = (
            [f.strip() for f in lstm_features_str.split(",") if f.strip()]
            if lstm_features_str
            else parsed_data.feature_keys
        )

        if feature_types_str:
            feature_types = json.loads(feature_types_str)
        else:
            feature_types = infer_feature_types(
                parsed_data=parsed_data,
                target_key=target_feature,
                feature_keys=lstm_feature_names,
            )

        if lstm_feature_names:
            allowed_keys = set(lstm_feature_names + [target_feature])
            feature_types = {k: v for k, v in feature_types.items() if k in allowed_keys}
        feature_types[target_feature] = "numeric"

        lstm_params = {
            "look_back": get_arg("look_back", auto_look_back, int),
            "epochs": get_arg("epochs", auto_epochs, int),
            "feature_types": feature_types,
            "normalization": get_arg("normalization", "minmax"),
            "lstm_target_feature": target_feature,
        }
        if lstm_features_str:
            lstm_params["lstm_features"] = lstm_features_str
        model_params["lstm"] = lstm_params

    _validate_training_args(parsed_data, ts, te, es, ee, models_to_use, model_params)

    (
        metrics,
        model_chain,
        stage_coefficients,
        base_model_states,
        y_true,
        y_pred,
        std_dev,
        summary,
    ) = train_and_evaluate(
        parsed_data,
        train_range,
        eval_range,
        models_to_use,
        model_params,
        learning_rate,
        max_rounds,
        base_model_files,
    )

    out = {
        "mode": "boosting_residual_learning",
        "models": models_to_use,
        "model_chain": model_chain,
        "stage_coefficients": stage_coefficients,
        "learning_rate": learning_rate,
        "metrics": metrics,
        "search_summary": summary,
        "eval_y_true": y_true,
        "predictions": _format_predictions(y_pred, std_dev),
    }

    if save_path:
        for model_name, model_info in base_model_states.items():
            for idx, model_state in enumerate(model_info["models"]):
                source_file = model_state["model_file"]
                perm_file = get_member_model_path(save_path, model_name, int(model_state.get("round", idx)))
                persist_member_model_artifact(
                    model_name,
                    source_file,
                    perm_file,
                    move_source=model_state.get("artifact_origin") == "temporary_training",
                )
                model_state["model_file"] = perm_file
                model_state.pop("artifact_origin", None)

        state = {
            "model_type": MODEL_TYPE,
            "boosting_variant": "residual",
            "model_chain": model_chain,
            "stage_coefficients": stage_coefficients,
            "base_model_states": base_model_states,
            "learning_rate": learning_rate,
            "std_dev_residuals": std_dev,
        }
        dump_model(state, save_path)
        out["saved_model"] = save_path

    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
