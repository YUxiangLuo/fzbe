#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-based training runner for the student guided training UI.

Each invocation executes exactly one training stage, persists intermediate
state to a pickle file, and prints a compact JSON payload for the backend.
"""

import argparse
import json
import os
import pickle
import random
import shutil
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data_utils import infer_feature_types, load_data_from_json
from durable_io import fsync_file, fsync_parent_directory
from guided_session_lock import guided_session_lock
from logger import get_log_contents, log
from metrics import mae, mape, r2, rmse
from model_defaults import resolve_lstm_look_back
from model_info import attach_method_metadata
from model_validation import validate_minimum_evaluation_points
from serialization import dump_model

arima = None
es = None
lstm = None
ma = None


def _load_model_module(model_type: str) -> None:
    global arima, es, lstm, ma

    if model_type == "ma" and ma is None:
        from models import ma as ma_module
        ma = ma_module
        return
    if model_type == "es" and es is None:
        from models import es as es_module
        es = es_module
        return
    if model_type == "arima" and arima is None:
        from models import arima as arima_module
        arima = arima_module
        return
    if model_type == "lstm" and lstm is None:
        from models import lstm as lstm_module
        lstm = lstm_module
        return


def _format_known_error(exc: Exception):
    message = str(exc).strip()
    if message.startswith("[VALIDATION]"):
        return f"[VALIDATION] {message[13:].strip()}".strip()
    if message.startswith("[DATA]"):
        return f"[DATA] {message[7:].strip()}".strip()
    if message.startswith("[CONFLICT]"):
        return f"[CONFLICT] {message[10:].strip()}".strip()
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_state(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    if os.path.getsize(path) == 0:
        return {}
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _save_state(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp-{os.getpid()}"
    with open(temp_path, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    fsync_parent_directory(path)


def _month_labels(timestamps: List[str], range_value: range) -> List[str]:
    return [timestamps[index] for index in range_value] if timestamps else []


def _parse_common_ranges(args) -> Tuple[int, int, int, int]:
    return int(args.train_start), int(args.train_end), int(args.eval_start), int(args.eval_end)


def _state_ranges(state: Dict[str, Any]) -> Tuple[range, range]:
    train = state["train_range"]
    evaluate = state["evaluate_range"]
    return (
        range(int(train["start_index"]), int(train["end_index"]) + 1),
        range(int(evaluate["start_index"]), int(evaluate["end_index"]) + 1),
    )


def _run_ma(step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if step_id == "prepare_data":
        parsed_data = load_data_from_json(args.data)
        y = parsed_data.y
        timestamps = parsed_data.timestamps or [str(i) for i in range(parsed_data.n_obs)]
        ts, te, es_idx, ee = _parse_common_ranges(args)
        window_arg = getattr(args, "window", None)
        if window_arg is not None:
            window = int(window_arg)
        else:
            train_size = te - ts + 1
            window = max(3, min(12, train_size // 4))

        ma._validate_training_args(y, ts, te, es_idx, ee, window)
        state.clear()
        state.update({
            "model_type": "ma",
            "y": y,
            "timestamps": timestamps,
            "window": window,
            "train_range": {"start_index": ts, "end_index": te},
            "evaluate_range": {"start_index": es_idx, "end_index": ee},
            "save_model": args.save_model,
            "data_points": parsed_data.n_obs,
        })
        return {
            "data_points": parsed_data.n_obs,
            "train_size": te - ts + 1,
            "evaluate_size": ee - es_idx + 1,
            "window": window,
        }

    if step_id == "build_windows":
        train_range, eval_range = _state_ranges(state)
        y = np.asarray(state["y"])
        window = int(state["window"])
        history_start = max(train_range.start, train_range.stop - window)
        forecast_steps = eval_range.stop - train_range.stop
        forecast_offset = eval_range.start - train_range.stop
        history = y[history_start:train_range.stop].tolist()
        state.update({
            "history_start": history_start,
            "forecast_steps": forecast_steps,
            "forecast_offset": forecast_offset,
            "history": history,
        })
        return {
            "history_start_index": history_start,
            "history_end_index": train_range.stop - 1,
            "window_values": history,
            "forecast_steps": forecast_steps,
        }

    if step_id == "forecast":
        train_range, eval_range = _state_ranges(state)
        y = np.asarray(state["y"])
        window = int(state["window"])
        all_forecasts = ma._forecast_future(state["history"], window, int(state["forecast_steps"]))
        preds = all_forecasts[
            int(state["forecast_offset"]): int(state["forecast_offset"]) + len(list(eval_range))
        ]
        if len(preds) != len(list(eval_range)):
            raise ValueError("[VALIDATION] 无法为全部评估区间生成移动平均预测。")

        y_true = y[list(eval_range)]
        y_pred = np.asarray(preds)
        state.update({
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        })
        return {
            "predicted_points": len(preds),
            "preview": [
                {"actual": float(actual), "predicted": float(predicted)}
                for actual, predicted in list(zip(y_true, y_pred))[:3]
            ],
        }

    if step_id == "estimate_residuals":
        train_range, _ = _state_ranges(state)
        y = np.asarray(state["y"])
        y_true = np.asarray(state["y_true"])
        y_pred = np.asarray(state["y_pred"])
        residuals = y_true - y_pred
        train_history = y[train_range]
        std_dev_residuals = ma._estimate_std_dev(residuals, train_history)
        state.update({
            "residuals": residuals.tolist(),
            "std_dev_residuals": std_dev_residuals,
        })
        return {
            "residual_points": len(residuals),
            "mean_residual": float(np.mean(residuals)) if len(residuals) else 0.0,
            "std_dev_residuals": float(std_dev_residuals),
        }

    if step_id == "evaluate_metrics":
        _, eval_range = _state_ranges(state)
        timestamps = state["timestamps"]
        y_true = np.asarray(state["y_true"])
        y_pred = np.asarray(state["y_pred"])
        metrics = {
            "window": int(state["window"]),
            "evaluated_points": len(y_true),
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "mape": mape(y_true, y_pred),
            "r2": r2(y_true, y_pred),
            "evaluate_indices": list(eval_range),
            "evaluate_months": _month_labels(timestamps, eval_range),
        }
        predictions_with_std = ma._format_predictions(y_pred.tolist(), float(state["std_dev_residuals"]))
        state.update({
            "metrics": metrics,
            "predictions": predictions_with_std,
        })
        return {"metrics": metrics}

    if step_id == "save_artifact":
        train_range, eval_range = _state_ranges(state)
        timestamps = state["timestamps"]
        y_true = np.asarray(state["y_true"])
        metrics = state["metrics"]
        predictions_with_std = state["predictions"]
        final_result: Dict[str, Any] = {
            "mode": "fixed_window_only",
            "data_points": int(state["data_points"]),
            "train_range": {
                "start_index": train_range.start,
                "end_index": train_range.stop - 1,
                "months": _month_labels(timestamps, train_range),
            },
            "evaluate_range": {
                "start_index": eval_range.start,
                "end_index": eval_range.stop - 1,
                "months": _month_labels(timestamps, eval_range),
            },
            "metrics": metrics,
            "eval_y_true": y_true.tolist(),
            "predictions": predictions_with_std,
        }
        attach_method_metadata(final_result, ma.METHOD_NAME, ma.FORECAST_STRATEGY, ma.IMPLEMENTATION_NOTES)

        if state.get("save_model"):
            history_start = max(train_range.start, train_range.stop - int(state["window"]))
            model_state = {
                "model_type": ma.MODEL_TYPE,
                "window": int(state["window"]),
                "train_end_index": train_range.stop - 1,
                "history": np.asarray(state["y"])[history_start:train_range.stop].tolist(),
                "std_dev_residuals": float(state["std_dev_residuals"]),
            }
            attach_method_metadata(model_state, ma.METHOD_NAME, ma.FORECAST_STRATEGY, ma.IMPLEMENTATION_NOTES)
            dump_model(model_state, state["save_model"])
            final_result["saved_model"] = state["save_model"]

        final_result["logs"] = get_log_contents()
        return {"output": {"metrics": metrics}, "final_result": final_result}

    raise ValueError(f"[VALIDATION] Unknown MA guided step: {step_id}")


def _run_es(step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if step_id == "prepare_data":
        parsed_data = load_data_from_json(args.data)
        y = parsed_data.y
        timestamps = parsed_data.timestamps or [str(i) for i in range(parsed_data.n_obs)]
        ts, te, es_idx, ee = _parse_common_ranges(args)
        alpha_arg = getattr(args, "alpha", None)
        alpha = float(alpha_arg) if alpha_arg is not None else 0.5
        es._validate_training_args(y, ts, te, es_idx, ee, alpha)
        state.clear()
        state.update({
            "model_type": "es",
            "y": y,
            "timestamps": timestamps,
            "alpha": alpha,
            "train_range": {"start_index": ts, "end_index": te},
            "evaluate_range": {"start_index": es_idx, "end_index": ee},
            "save_model": args.save_model,
            "data_points": parsed_data.n_obs,
        })
        return {
            "data_points": parsed_data.n_obs,
            "train_size": te - ts + 1,
            "evaluate_size": ee - es_idx + 1,
            "alpha": alpha,
        }

    if step_id == "fit_model":
        train_range, _ = _state_ranges(state)
        y_train = np.asarray(state["y"])[train_range]
        model = es.SimpleExpSmoothing(y_train, initialization_method="estimated").fit(
            smoothing_level=float(state["alpha"]),
            optimized=False,
        )
        state["model"] = model
        initial_level = float(model.params.get("initial_level", y_train[0] if len(y_train) else 0.0))
        fitted_levels = np.asarray(model.level, dtype=float)
        final_level = float(fitted_levels[-1]) if len(fitted_levels) else initial_level
        return {
            "alpha": float(state["alpha"]),
            "fitted_points": len(y_train),
            "initial_level": initial_level,
            "final_level": final_level,
        }

    if step_id == "forecast":
        train_range, eval_range = _state_ranges(state)
        forecast_steps = eval_range.stop - train_range.stop
        forecast_offset = eval_range.start - train_range.stop
        all_predictions = np.asarray(state["model"].forecast(forecast_steps), dtype=float)
        y_pred = all_predictions[forecast_offset:forecast_offset + len(eval_range)].tolist()
        if len(y_pred) != len(eval_range):
            raise ValueError("[VALIDATION] 无法为全部评估区间生成指数平滑预测。")
        state["y_pred"] = y_pred
        return {
            "predicted_points": len(y_pred),
            "forecast_steps": forecast_steps,
            "forecast_offset": forecast_offset,
            "preview": [float(value) for value in y_pred[:3]],
        }

    if step_id == "estimate_residuals":
        train_range, eval_range = _state_ranges(state)
        y = np.asarray(state["y"])
        y_true = y[eval_range].tolist()
        y_pred = state["y_pred"]
        residuals = np.asarray(y_true) - np.asarray(y_pred)
        std_dev_residuals = es._estimate_std_dev(residuals, y[train_range])
        state.update({
            "y_true": y_true,
            "residuals": residuals.tolist(),
            "std_dev_residuals": std_dev_residuals,
        })
        return {
            "residual_points": len(residuals),
            "mean_residual": float(np.mean(residuals)) if len(residuals) else 0.0,
            "std_dev_residuals": float(std_dev_residuals),
        }

    if step_id == "evaluate_metrics":
        _, eval_range = _state_ranges(state)
        y_true = state["y_true"]
        y_pred = state["y_pred"]
        timestamps = state["timestamps"]
        metrics = {
            "alpha": float(state["alpha"]),
            "evaluated_points": len(y_true),
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "mape": mape(y_true, y_pred),
            "r2": r2(y_true, y_pred),
            "evaluate_indices": list(eval_range),
            "evaluate_months": _month_labels(timestamps, eval_range),
        }
        predictions_with_std = es._format_predictions(y_pred, float(state["std_dev_residuals"]))
        state.update({
            "metrics": metrics,
            "predictions": predictions_with_std,
        })
        return {"metrics": metrics}

    if step_id == "save_artifact":
        train_range, eval_range = _state_ranges(state)
        timestamps = state["timestamps"]
        y_true = state["y_true"]
        metrics = state["metrics"]
        predictions_with_std = state["predictions"]
        final_result: Dict[str, Any] = {
            "mode": "fixed_alpha_only",
            "data_points": int(state["data_points"]),
            "alpha": float(state["alpha"]),
            "train_range": {
                "start_index": train_range.start,
                "end_index": train_range.stop - 1,
                "months": _month_labels(timestamps, train_range),
            },
            "evaluate_range": {
                "start_index": eval_range.start,
                "end_index": eval_range.stop - 1,
                "months": _month_labels(timestamps, eval_range),
            },
            "metrics": metrics,
            "eval_y_true": y_true,
            "predictions": predictions_with_std,
        }
        attach_method_metadata(final_result, es.METHOD_NAME, es.FORECAST_STRATEGY, es.IMPLEMENTATION_NOTES)

        if state.get("save_model"):
            model_state = {
                "model_type": es.MODEL_TYPE,
                "alpha": float(state["alpha"]),
                "train_end_index": train_range.stop - 1,
                "model": state["model"],
                "std_dev_residuals": float(state["std_dev_residuals"]),
            }
            attach_method_metadata(model_state, es.METHOD_NAME, es.FORECAST_STRATEGY, es.IMPLEMENTATION_NOTES)
            dump_model(model_state, state["save_model"])
            final_result["saved_model"] = state["save_model"]

        final_result["logs"] = get_log_contents()
        return {"output": {"metrics": metrics}, "final_result": final_result}

    raise ValueError(f"[VALIDATION] Unknown ES guided step: {step_id}")


def _resolve_arima_limits(train_size: int, max_p_arg: Any, max_q_arg: Any) -> Tuple[int, int]:
    if max_p_arg is not None and max_q_arg is not None:
        return int(max_p_arg), int(max_q_arg)
    if train_size <= 12:
        auto_max = 1
    elif train_size <= 20:
        auto_max = 2
    else:
        auto_max = max(1, min(5, train_size // 6))
    max_p = int(max_p_arg) if max_p_arg is not None else auto_max
    max_q = int(max_q_arg) if max_q_arg is not None else auto_max
    return max_p, max_q


def _run_arima(step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if step_id == "prepare_data":
        parsed_data = load_data_from_json(args.data)
        y = parsed_data.y
        timestamps = parsed_data.timestamps or [str(i) for i in range(parsed_data.n_obs)]
        ts, te, es_idx, ee = _parse_common_ranges(args)
        d = int(args.d)
        train_size = te - ts + 1
        max_p, max_q = _resolve_arima_limits(train_size, getattr(args, "max_p", None), getattr(args, "max_q", None))
        arima._validate_arima_search_args(d, max_p, max_q)

        if not (0 <= ts <= te < len(y)):
            raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], must be within [0, {len(y)-1}]")
        if not (0 <= es_idx <= ee < len(y)):
            raise ValueError(f"[VALIDATION] Invalid eval range: [{es_idx},{ee}], must be within [0, {len(y)-1}]")
        if es_idx <= te:
            raise ValueError(f"[VALIDATION] Evaluation start ({es_idx}) must be after training end ({te}).")
        validate_minimum_evaluation_points(es_idx, ee)
        effective_train_size = arima._validate_arima_train_size(train_size, d, "Training range")

        state.clear()
        state.update({
            "model_type": "arima",
            "y": y,
            "timestamps": timestamps,
            "d": d,
            "max_p": max_p,
            "max_q": max_q,
            "train_range": {"start_index": ts, "end_index": te},
            "evaluate_range": {"start_index": es_idx, "end_index": ee},
            "save_model": args.save_model,
            "data_points": parsed_data.n_obs,
        })
        return {
            "data_points": parsed_data.n_obs,
            "train_size": train_size,
            "effective_train_size": effective_train_size,
            "d": d,
            "max_p": max_p,
            "max_q": max_q,
        }

    if step_id == "aic_search":
        train_range, _ = _state_ranges(state)
        y_train = np.asarray(state["y"])[train_range]
        aic_model = arima._run_auto_arima_search(
            y_train=y_train,
            d=int(state["d"]),
            max_p=int(state["max_p"]),
            max_q=int(state["max_q"]),
            criterion="aic",
        )
        state["aic_model"] = aic_model
        return {
            "order": {"p": aic_model.order[0], "d": aic_model.order[1], "q": aic_model.order[2]},
            "aic": float(aic_model.aic()),
            "bic": float(aic_model.bic()),
        }

    if step_id == "bic_search":
        train_range, _ = _state_ranges(state)
        y_train = np.asarray(state["y"])[train_range]
        bic_model = arima._run_auto_arima_search(
            y_train=y_train,
            d=int(state["d"]),
            max_p=int(state["max_p"]),
            max_q=int(state["max_q"]),
            criterion="bic",
        )
        state["bic_model"] = bic_model
        return {
            "order": {"p": bic_model.order[0], "d": bic_model.order[1], "q": bic_model.order[2]},
            "aic": float(bic_model.aic()),
            "bic": float(bic_model.bic()),
        }

    if step_id == "select_model":
        train_range, _ = _state_ranges(state)
        aic_model = state["aic_model"]
        bic_model = state["bic_model"]
        aic_order = aic_model.order
        bic_order = bic_model.order
        aic_aic = aic_model.aic()
        aic_bic = aic_model.bic()
        bic_aic = bic_model.aic()
        bic_bic = bic_model.bic()
        train_size = len(np.asarray(state["y"])[train_range])
        selection_criterion = "aic_optimized"
        best_model = aic_model

        if train_size < 30:
            if bic_bic < aic_bic:
                best_model = bic_model
                selection_criterion = "bic_preferred_small_sample"
            elif bic_bic == aic_bic and bic_aic <= aic_aic:
                best_model = bic_model
                selection_criterion = "bic_tie_better_or_equal_aic"
        else:
            if bic_aic < aic_aic:
                best_model = bic_model
                selection_criterion = "bic_optimized_better_aic"
            elif bic_aic == aic_aic and bic_bic < aic_bic:
                best_model = bic_model
                selection_criterion = "bic_tie_aic_better_bic"

        best_order = {"p": best_model.order[0], "d": best_model.order[1], "q": best_model.order[2]}
        search_summary = {
            "aic_optimized_model": {
                "order": {"p": aic_order[0], "d": aic_order[1], "q": aic_order[2]},
                "aic": float(aic_aic),
                "bic": float(aic_bic),
            },
            "bic_optimized_model": {
                "order": {"p": bic_order[0], "d": bic_order[1], "q": bic_order[2]},
                "aic": float(bic_aic),
                "bic": float(bic_bic),
            },
            "selected_model": "aic_optimized" if best_model == aic_model else "bic_optimized",
            "selection_criterion": selection_criterion,
        }
        state["best_model"] = best_model
        state["best_order"] = best_order
        state["search_summary"] = search_summary
        return {"best_order": best_order, "selection_criterion": selection_criterion}

    if step_id == "forecast":
        train_range, eval_range = _state_ranges(state)
        y = np.asarray(state["y"])
        y_eval = y[eval_range]
        best_model = state["best_model"]
        forecast_steps = eval_range.stop - train_range.stop
        forecast_offset = eval_range.start - train_range.stop
        all_predictions, all_confidence_intervals = best_model.predict(
            n_periods=forecast_steps,
            return_conf_int=True,
        )
        y_pred = np.asarray(all_predictions, dtype=float)[
            forecast_offset:forecast_offset + len(eval_range)
        ]
        conf_int = np.asarray(all_confidence_intervals, dtype=float)[
            forecast_offset:forecast_offset + len(eval_range)
        ]
        if len(y_pred) != len(eval_range) or len(conf_int) != len(eval_range):
            raise ValueError("[VALIDATION] 无法为全部评估区间生成 ARIMA 预测。")
        y_true = y_eval.tolist()
        converged = bool(best_model.arima_res_.mle_retvals.get("converged", False))
        state.update({
            "y_true": y_true,
            "y_pred": y_pred.tolist(),
            "confidence_intervals": conf_int.tolist(),
            "converged": converged,
        })
        return {
            "predicted_points": len(y_true),
            "forecast_steps": forecast_steps,
            "forecast_offset": forecast_offset,
            "preview": [
                {
                    "actual": float(actual),
                    "predicted": float(predicted),
                    "lower": float(interval[0]),
                    "upper": float(interval[1]),
                }
                for actual, predicted, interval in list(zip(y_true, y_pred, conf_int))[:3]
            ],
        }

    if step_id == "evaluate_metrics":
        _, eval_range = _state_ranges(state)
        y_eval = np.asarray(state["y_true"])
        y_pred = np.asarray(state["y_pred"])
        best_model = state["best_model"]
        timestamps = state["timestamps"]
        metrics = {
            "rmse": rmse(y_eval, y_pred),
            "mae": mae(y_eval, y_pred),
            "mape": mape(y_eval, y_pred),
            "r2": r2(y_eval, y_pred),
            "aic": float(best_model.aic()),
            "bic": float(best_model.bic()),
            "converged": bool(state["converged"]),
            "evaluated_points": len(y_eval),
            "evaluate_indices": list(eval_range),
            "evaluate_months": _month_labels(timestamps, eval_range),
        }
        predictions_with_std = arima._format_predictions(
            y_pred.tolist(),
            state["confidence_intervals"],
        )
        state.update({
            "metrics": metrics,
            "predictions": predictions_with_std,
        })
        return {"metrics": metrics, "best_order": state["best_order"]}

    if step_id == "save_artifact":
        train_range, eval_range = _state_ranges(state)
        timestamps = state["timestamps"]
        y_true = state["y_true"]
        best_model = state["best_model"]
        metrics = state["metrics"]
        predictions_with_std = state["predictions"]
        final_result: Dict[str, Any] = {
            "mode": "arima_search_pq_fixed_d",
            "data_points": int(state["data_points"]),
            "d": int(state["d"]),
            "best_order": state["best_order"],
            "train_range": {
                "start_index": train_range.start,
                "end_index": train_range.stop - 1,
                "months": _month_labels(timestamps, train_range),
            },
            "evaluate_range": {
                "start_index": eval_range.start,
                "end_index": eval_range.stop - 1,
                "months": _month_labels(timestamps, eval_range),
            },
            "metrics": metrics,
            "eval_y_true": y_true,
            "predictions": predictions_with_std,
            "search_summary": state["search_summary"],
        }
        attach_method_metadata(final_result, arima.METHOD_NAME, arima.FORECAST_STRATEGY, arima.IMPLEMENTATION_NOTES)

        if state.get("save_model"):
            model_state = {
                "model_type": arima.MODEL_TYPE,
                "model": best_model,
                "order": state["best_order"],
                "d": int(state["d"]),
            }
            attach_method_metadata(model_state, arima.METHOD_NAME, arima.FORECAST_STRATEGY, arima.IMPLEMENTATION_NOTES)
            dump_model(model_state, state["save_model"])
            final_result["saved_model"] = state["save_model"]

        final_result["logs"] = get_log_contents()
        return {"output": {"metrics": metrics, "best_order": state["best_order"]}, "final_result": final_result}

    raise ValueError(f"[VALIDATION] Unknown ARIMA guided step: {step_id}")


def _set_lstm_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_lstm_model(config: Dict[str, Any]):
    # 直接复用 src/models/lstm.py 的架构构建器，确保 guided 与非 guided 路径不再分叉。
    from models import lstm as lstm_module

    return lstm_module.build_lstm_network(
        look_back=int(config["look_back"]),
        n_features_out=int(config["n_features_out"]),
        recurrent_units=int(config["recurrent_units"]),
        dense_units=int(config["dense_units"]),
        output_units=int(config["forecast_horizon"]),
        learning_rate=float(config.get("initial_learning_rate", lstm_module.LSTM_INITIAL_LEARNING_RATE)),
    )


def _run_lstm(step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if step_id == "prepare_data":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        feature_keys = lstm._parse_feature_key_list(getattr(args, "lstm_features", None))
        target_key = getattr(args, "lstm_target_feature", None)
        feature_types_override = json.loads(getattr(args, "feature_types", None) or "{}")
        parsed_data = load_data_from_json(args.data, target_key=target_key, feature_keys=feature_keys)
        months = parsed_data.timestamps or [str(i) for i in range(parsed_data.n_obs)]
        inferred_feature_types = infer_feature_types(
            parsed_data,
            target_key=parsed_data.target_key,
            feature_keys=parsed_data.feature_keys,
        )
        feature_types = {**inferred_feature_types, **dict(feature_types_override)} if feature_types_override else inferred_feature_types
        lstm._validate_lstm_feature_types(parsed_data, feature_types)
        lstm._validate_lstm_numeric_columns(parsed_data, feature_types)

        ts, te, es_idx, ee = _parse_common_ranges(args)
        train_size = te - ts + 1
        look_back_arg = getattr(args, "look_back", None)
        look_back = resolve_lstm_look_back(train_size, look_back_arg)
        epochs_arg = getattr(args, "epochs", None)
        epochs = int(epochs_arg) if epochs_arg is not None else 20
        normalization = getattr(args, "normalization", None) or "minmax"
        lstm._validate_training_args(parsed_data, ts, te, es_idx, ee, look_back)

        state.clear()
        state.update({
            "model_type": "lstm",
            "parsed_data": parsed_data,
            "feature_types": feature_types,
            "months": months,
            "look_back": look_back,
            "epochs": epochs,
            "normalization": normalization,
            "train_range": {"start_index": ts, "end_index": te},
            "evaluate_range": {"start_index": es_idx, "end_index": ee},
            "save_model": args.save_model,
            "data_points": parsed_data.n_obs,
        })
        return {
            "data_points": parsed_data.n_obs,
            "target_key": parsed_data.target_key,
            "feature_keys": parsed_data.feature_keys,
            "feature_types": feature_types,
            "look_back": look_back,
            "epochs": epochs,
            "normalization": normalization,
        }

    if step_id == "preprocess":
        train_range, eval_range = _state_ranges(state)
        parsed_data = state["parsed_data"]
        forecast_horizon = eval_range.stop - train_range.stop
        preprocessor, scaled_series = lstm._build_preprocessor(
            parsed_data,
            state["feature_types"],
            train_range,
            state["normalization"],
        )
        train_data = scaled_series[train_range]
        train_x, train_y = lstm._create_direct_dataset(train_data, int(state["look_back"]), forecast_horizon)
        if train_x.size == 0:
            raise ValueError("[VALIDATION] Training range is too short for the requested LSTM look_back and forecast horizon.")
        history_df = parsed_data.df.iloc[train_range.stop - int(state["look_back"]): train_range.stop].copy()
        state.update({
            "preprocessor": preprocessor,
            "train_x": train_x,
            "train_y": train_y,
            "history_df": history_df,
            "forecast_horizon": forecast_horizon,
            "n_features_out": int(scaled_series.shape[1]),
        })
        return {
            "input_shape": list(train_x.shape),
            "target_shape": list(train_y.shape),
            "encoded_feature_count": int(scaled_series.shape[1]),
            "forecast_horizon": forecast_horizon,
        }

    if step_id == "build_model":
        train_x = state["train_x"]
        n_features_out = int(state["n_features_out"])
        # 复用 src/models/lstm.py 的单元数选择与超参，保持与非 guided 路径一致。
        recurrent_units, dense_units = lstm.select_lstm_units(len(train_x))
        batch_size = max(1, min(8, len(train_x)))
        config = {
            "look_back": int(state["look_back"]),
            "n_features_out": n_features_out,
            "recurrent_units": recurrent_units,
            "dense_units": dense_units,
            "forecast_horizon": int(state["forecast_horizon"]),
            "batch_size": batch_size,
            "initial_learning_rate": lstm.LSTM_INITIAL_LEARNING_RATE,
            "decay_rate": lstm.LSTM_LR_DECAY_RATE,
        }
        state["model_config"] = config
        return config

    if step_id == "train_model":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

        tf.get_logger().setLevel("ERROR")
        tf.keras.utils.set_random_seed(42)
        _set_lstm_seed(42)

        config = state["model_config"]
        model = _build_lstm_model(config)
        epochs = int(state["epochs"])
        initial_learning_rate = float(config["initial_learning_rate"])
        decay_rate = float(config["decay_rate"])

        def lr_schedule(epoch):
            return initial_learning_rate * (decay_rate ** epoch)

        early_stopping = EarlyStopping(
            monitor="loss",
            patience=max(2, min(6, epochs // 4 or 2)),
            restore_best_weights=True,
        )
        history = model.fit(
            state["train_x"],
            state["train_y"],
            epochs=epochs,
            batch_size=int(config["batch_size"]),
            verbose=0,
            callbacks=[
                LearningRateScheduler(lr_schedule, verbose=0),
                early_stopping,
            ],
            shuffle=False,
        )

        keras_model_path = os.path.join(args.artifact_dir, "lstm_trained_keras_model.h5")
        temp_keras_model_path = os.path.join(args.artifact_dir, f"lstm_trained_keras_model.tmp-{os.getpid()}.h5")
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

        loss_history = [float(value) for value in history.history.get("loss", [])]
        state.update({
            "trained_keras_model_path": keras_model_path,
            "training_loss": loss_history,
            "epochs_ran": len(loss_history),
            "early_stopped": bool(getattr(early_stopping, "stopped_epoch", 0)),
        })
        return {
            "epochs_requested": epochs,
            "epochs_ran": len(loss_history),
            "batch_size": int(config["batch_size"]),
            "final_loss": loss_history[-1] if loss_history else None,
            "early_stopped": bool(getattr(early_stopping, "stopped_epoch", 0)),
        }

    if step_id == "forecast":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        tf.get_logger().setLevel("ERROR")
        forecast_horizon = int(state["forecast_horizon"])
        train_range, eval_range = _state_ranges(state)
        parsed_data = state["parsed_data"]
        model = load_model(state["trained_keras_model_path"], compile=False)
        forecast_offset = eval_range.start - train_range.stop
        state_for_eval = {
            "keras_model": model,
            "preprocessor": state["preprocessor"],
            "look_back": int(state["look_back"]),
            "forecast_horizon": forecast_horizon,
            "forecast_strategy": lstm.DIRECT_FORECAST_STRATEGY,
            "history_df": state["history_df"],
            "normalization": state["normalization"],
        }
        full_forecast, _ = lstm._forecast_from_state(state_for_eval, forecast_horizon)
        y_pred = full_forecast[forecast_offset: forecast_offset + len(eval_range)]
        if len(y_pred) != len(eval_range):
            raise ValueError("[VALIDATION] 无法为完整评估区间生成 LSTM 预测。")

        y_true = parsed_data.df[parsed_data.target_key].values[eval_range]
        state.update({
            "y_true": y_true.tolist(),
            "y_pred": np.asarray(y_pred).tolist(),
        })
        return {
            "predicted_points": len(y_pred),
            "forecast_horizon": forecast_horizon,
            "preview": [
                {"actual": float(actual), "predicted": float(predicted)}
                for actual, predicted in list(zip(y_true, y_pred))[:3]
            ],
        }

    if step_id == "evaluate_metrics":
        train_range, eval_range = _state_ranges(state)
        parsed_data = state["parsed_data"]
        forecast_horizon = int(state["forecast_horizon"])
        y_true = np.asarray(state["y_true"], dtype=float)
        y_pred = np.asarray(state["y_pred"], dtype=float)
        residuals = y_true - y_pred
        train_target_values = parsed_data.df[parsed_data.target_key].values[train_range]
        std_dev_residuals = lstm._estimate_std_dev(residuals, train_target_values)
        epochs = int(state["epochs"])
        metrics = {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "mape": mape(y_true, y_pred),
            "r2": r2(y_true, y_pred),
            "look_back": int(state["look_back"]),
            "epochs": epochs,
            "forecast_horizon": forecast_horizon,
            "evaluated_points": len(y_true),
            "evaluate_indices": list(eval_range),
            "evaluate_months": _month_labels(state["months"], eval_range),
        }
        predictions_with_std = lstm._format_predictions(y_pred, std_dev_residuals)
        state.update({
            "residuals": residuals.tolist(),
            "std_dev_residuals": std_dev_residuals,
            "metrics": metrics,
            "predictions": predictions_with_std,
        })
        return {"metrics": metrics}

    if step_id == "save_artifact":
        train_range, eval_range = _state_ranges(state)
        parsed_data = state["parsed_data"]
        epochs = int(state["epochs"])
        forecast_horizon = int(state["forecast_horizon"])
        metrics = state["metrics"]
        y_true = state["y_true"]
        predictions_with_std = state["predictions"]
        final_result: Dict[str, Any] = {
            "mode": "lstm_train_eval",
            "data_points": int(state["data_points"]),
            "target_key": parsed_data.target_key,
            "feature_types": state["feature_types"],
            "look_back": int(state["look_back"]),
            "epochs": epochs,
            "normalization": state["normalization"],
            "forecast_horizon": forecast_horizon,
            "train_range": {
                "start_index": train_range.start,
                "end_index": train_range.stop - 1,
                "months": _month_labels(state["months"], train_range),
            },
            "evaluate_range": {
                "start_index": eval_range.start,
                "end_index": eval_range.stop - 1,
                "months": _month_labels(state["months"], eval_range),
            },
            "metrics": metrics,
            "eval_y_true": y_true,
            "predictions": predictions_with_std,
        }
        attach_method_metadata(final_result, lstm.METHOD_NAME, lstm.DIRECT_FORECAST_STRATEGY, lstm.DIRECT_IMPLEMENTATION_NOTES)

        if state.get("save_model"):
            save_path = state["save_model"]
            keras_model_path = os.path.splitext(save_path)[0] + "_keras_model.h5"
            temp_keras_model_path = os.path.splitext(save_path)[0] + f"_keras_model.tmp-{os.getpid()}.h5"
            try:
                shutil.copy2(state["trained_keras_model_path"], temp_keras_model_path)
                fsync_file(temp_keras_model_path)
                os.replace(temp_keras_model_path, keras_model_path)
                fsync_parent_directory(keras_model_path)
            finally:
                if os.path.exists(temp_keras_model_path):
                    try:
                        os.remove(temp_keras_model_path)
                    except OSError:
                        pass

            model_state = {
                "model_type": lstm.MODEL_TYPE,
                "look_back": int(state["look_back"]),
                "epochs": epochs,
                "normalization": state["normalization"],
                "train_end_index": train_range.stop - 1,
                "preprocessor": state["preprocessor"],
                "history_df": state["history_df"],
                "target_key": parsed_data.target_key,
                "feature_keys": parsed_data.feature_keys,
                "feature_types": state["feature_types"],
                "keras_model_path": keras_model_path,
                "forecast_horizon": forecast_horizon,
                "forecast_strategy": lstm.DIRECT_FORECAST_STRATEGY,
                "production_forecast_horizon": lstm._resolve_production_forecast_horizon(
                    eval_forecast_horizon=forecast_horizon
                ),
                "std_dev_residuals": float(state["std_dev_residuals"]),
            }
            attach_method_metadata(model_state, lstm.METHOD_NAME, lstm.DIRECT_FORECAST_STRATEGY, lstm.DIRECT_IMPLEMENTATION_NOTES)
            dump_model(model_state, save_path)
            final_result["saved_model"] = save_path

        final_result["logs"] = get_log_contents()
        return {"output": {"metrics": metrics}, "final_result": final_result}

    raise ValueError(f"[VALIDATION] Unknown LSTM guided step: {step_id}")


def run_step(model_type: str, step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    _load_model_module(model_type)

    if model_type == "ma":
        return _run_ma(step_id, args, state)
    if model_type == "es":
        return _run_es(step_id, args, state)
    if model_type == "arima":
        return _run_arima(step_id, args, state)
    if model_type == "lstm":
        return _run_lstm(step_id, args, state)
    raise ValueError(f"[VALIDATION] Unsupported guided model type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one guided model training stage.")
    parser.add_argument("model_type", type=str)
    parser.add_argument("step_id", type=str)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--data", type=str)
    parser.add_argument("--data-stdin", action="store_true")
    parser.add_argument("--train-start", type=int)
    parser.add_argument("--train-end", type=int)
    parser.add_argument("--eval-start", type=int)
    parser.add_argument("--eval-end", type=int)
    parser.add_argument("--save-model", type=str)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--window", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--d", type=int)
    parser.add_argument("--max-p", type=int)
    parser.add_argument("--max-q", type=int)
    parser.add_argument("--look-back", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--feature-types", type=str)
    parser.add_argument("--lstm-features", type=str)
    parser.add_argument("--lstm-target-feature", type=str)
    parser.add_argument("--normalization", type=str, choices=["minmax", "zscore"])
    args, _ = parser.parse_known_args()

    if args.data_stdin:
        if args.data is not None:
            raise ValueError("[VALIDATION] Use either --data or --data-stdin, not both.")
        args.data = sys.stdin.read()
        if not args.data:
            raise ValueError("[DATA] stdin did not contain model input data.")

    with guided_session_lock(args.artifact_dir):
        state = _load_state(args.state_file)
        completed_payloads = state.get("_completed_step_payloads", {})
        cached_payload = completed_payloads.get(args.step_id) if isinstance(completed_payloads, dict) else None
        if isinstance(cached_payload, dict):
            print(json.dumps(cached_payload, ensure_ascii=False, indent=2 if args.pretty else None))
            return

        step_result = run_step(args.model_type, args.step_id, args, state)

        final_result = None
        if isinstance(step_result, dict) and "final_result" in step_result:
            final_result = step_result["final_result"]
            output = step_result.get("output", {})
        else:
            output = step_result

        payload = {
            "step_id": args.step_id,
            "output": _jsonable(output),
        }
        if final_result is not None:
            payload["final_result"] = _jsonable(final_result)

        state.setdefault("_completed_step_payloads", {})[args.step_id] = payload
        _save_state(args.state_file, state)
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        formatted_error = _format_known_error(exc)
        if formatted_error is None:
            raise
        print(formatted_error, file=sys.stderr, flush=True)
        sys.exit(1)
