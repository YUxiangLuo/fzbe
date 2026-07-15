#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-based training runner for the student guided training UI.

Each invocation executes exactly one training stage, persists intermediate
state to a pickle file, and prints a compact JSON payload for the backend.
"""

import argparse
import hashlib
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
from error_protocol import format_known_error
from forecast_utils import DEMAND_PREDICTION_DOMAIN, normalize_demand_forecast
from logger import get_log_contents
from metrics import mae, mape, r2, rmse
from model_info import attach_method_metadata
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


def _sha256_file(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_receipt(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp-{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _load_existing_receipt(path: str, attempt_id: str, step_id: str) -> Dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("attempt_id") != attempt_id or payload.get("step_id") != step_id:
        return None
    candidate_path = payload.get("candidate_state_path")
    if not isinstance(candidate_path, str) or _sha256_file(candidate_path) != payload.get("candidate_state_sha256"):
        return None
    return payload


def _month_labels(timestamps: List[str], range_value: range) -> List[str]:
    return [timestamps[index] for index in range_value] if timestamps else []


def _parse_common_ranges(args) -> Tuple[int, int, int, int]:
    return int(args.train_start), int(args.train_end), int(args.eval_start), int(args.eval_end)


def _read_data_json(args) -> str:
    data_file = getattr(args, "data_file", None)
    if data_file:
        with open(data_file, "r", encoding="utf-8") as handle:
            return handle.read()
    if args.data:
        return args.data
    raise ValueError("[DATA] 训练数据快照缺失。")


def _state_ranges(state: Dict[str, Any]) -> Tuple[range, range]:
    train = state["train_range"]
    evaluate = state["evaluate_range"]
    return (
        range(int(train["start_index"]), int(train["end_index"]) + 1),
        range(int(evaluate["start_index"]), int(evaluate["end_index"]) + 1),
    )


def _run_ma(step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if step_id == "prepare_data":
        parsed_data = load_data_from_json(_read_data_json(args))
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
        y_pred = normalize_demand_forecast(preds)
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
        train_range, eval_range = _state_ranges(state)
        y = np.asarray(state["y"])
        train_history = y[train_range]
        uncertainty_profile, uncertainty_summary = ma.moving_average_profile(
            train_history,
            int(state["window"]),
            max(12, eval_range.stop - train_range.stop),
        )
        std_dev_residuals = float(uncertainty_profile[0])
        state.update({
            "std_dev_residuals": std_dev_residuals,
            "uncertainty_profile": uncertainty_profile,
            "uncertainty_summary": uncertainty_summary,
        })
        return {
            "forecast_horizon": len(uncertainty_profile),
            "std_dev_residuals": float(std_dev_residuals),
            "uncertainty_summary": uncertainty_summary,
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
        train_range, eval_range = _state_ranges(state)
        forecast_offset = eval_range.start - train_range.stop
        evaluation_profile = state["uncertainty_profile"][
            forecast_offset : forecast_offset + len(y_pred)
        ]
        predictions_with_std = ma._format_predictions(
            y_pred.tolist(),
            evaluation_profile,
            {
                **state["uncertainty_summary"],
                "horizon_start": forecast_offset + 1,
            },
        )
        state.update({
            "metrics": metrics,
            "predictions": predictions_with_std,
            "uncertainty_summary": state["uncertainty_summary"],
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
                "uncertainty_profile": state["uncertainty_profile"][:12],
                "uncertainty_summary": state["uncertainty_summary"],
                "reference_series": np.asarray(state["y"])[train_range].tolist(),
                "prediction_domain": DEMAND_PREDICTION_DOMAIN,
            }
            attach_method_metadata(model_state, ma.METHOD_NAME, ma.FORECAST_STRATEGY, ma.IMPLEMENTATION_NOTES)
            dump_model(model_state, state["save_model"])
            final_result["saved_model"] = state["save_model"]

        final_result["logs"] = get_log_contents()
        return {"output": {"metrics": metrics}, "final_result": final_result}

    raise ValueError(f"[VALIDATION] Unknown MA guided step: {step_id}")


def _run_es(step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if step_id == "prepare_data":
        parsed_data = load_data_from_json(_read_data_json(args))
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
        model = es._fit_textbook_ses(y_train, float(state["alpha"]))
        state["model"] = model
        return {
            "alpha": float(state["alpha"]),
            "fitted_points": len(y_train),
            "level": float(model.params.get("initial_level", y_train[0] if len(y_train) else 0.0)),
        }

    if step_id == "forecast":
        train_range, eval_range = _state_ranges(state)
        forecast_steps = eval_range.stop - train_range.stop
        forecast_offset = eval_range.start - train_range.stop
        all_predictions = np.asarray(state["model"].forecast(forecast_steps))
        y_pred = all_predictions[
            forecast_offset : forecast_offset + len(eval_range)
        ]
        if len(y_pred) != len(eval_range):
            raise ValueError(
                "[VALIDATION] ES forecast did not cover the configured evaluation range"
            )
        y_pred = normalize_demand_forecast(y_pred).tolist()
        state["y_pred"] = y_pred
        return {
            "predicted_points": len(y_pred),
            "preview": [float(value) for value in y_pred[:3]],
        }

    if step_id == "estimate_residuals":
        train_range, eval_range = _state_ranges(state)
        y = np.asarray(state["y"])
        y_true = y[eval_range].tolist()
        y_pred = state["y_pred"]
        residuals = np.asarray(y_true) - np.asarray(y_pred)
        std_dev_residuals = es._training_residual_scale(state["model"], y[train_range])
        profile, uncertainty_summary = es.ses_profile(
            std_dev_residuals,
            float(state["alpha"]),
            max(12, eval_range.stop - train_range.stop),
            es._training_residuals(state["model"], y[train_range]),
        )
        state.update({
            "y_true": y_true,
            "residuals": residuals.tolist(),
            "std_dev_residuals": std_dev_residuals,
            "uncertainty_profile": profile,
            "uncertainty_summary": uncertainty_summary,
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
        train_range, eval_range = _state_ranges(state)
        forecast_offset = eval_range.start - train_range.stop
        evaluation_profile = state["uncertainty_profile"][
            forecast_offset : forecast_offset + len(y_pred)
        ]
        predictions_with_std = es._format_predictions(
            y_pred,
            evaluation_profile,
            {
                **state["uncertainty_summary"],
                "horizon_start": forecast_offset + 1,
            },
        )
        state.update({
            "metrics": metrics,
            "predictions": predictions_with_std,
            "initial_level": float(state["model"].params["initial_level"]),
            "uncertainty_summary": state["uncertainty_summary"],
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
                "initial_level": float(state["model"].params["initial_level"]),
                "uncertainty_profile": state["uncertainty_profile"][:12],
                "uncertainty_summary": state["uncertainty_summary"],
                "reference_series": np.asarray(state["y"])[train_range].tolist(),
                "prediction_domain": DEMAND_PREDICTION_DOMAIN,
            }
            attach_method_metadata(model_state, es.METHOD_NAME, es.FORECAST_STRATEGY, es.IMPLEMENTATION_NOTES)
            dump_model(model_state, state["save_model"])
            final_result["saved_model"] = state["save_model"]

        final_result["logs"] = get_log_contents()
        return {"output": {"metrics": metrics}, "final_result": final_result}

    raise ValueError(f"[VALIDATION] Unknown ES guided step: {step_id}")


def _resolve_arima_limits(train_size: int, d: int) -> Tuple[int, int]:
    config = arima.derive_arima_search_limit(train_size, d)
    return int(config["max_p"]), int(config["max_q"])


def _run_arima(step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if step_id == "prepare_data":
        parsed_data = load_data_from_json(_read_data_json(args))
        y = parsed_data.y
        timestamps = parsed_data.timestamps or [str(i) for i in range(parsed_data.n_obs)]
        ts, te, es_idx, ee = _parse_common_ranges(args)
        d = int(args.d)
        train_size = te - ts + 1
        max_p, max_q = _resolve_arima_limits(train_size, d)

        if d < 0:
            raise ValueError(f"[VALIDATION] d must be >= 0, got {d}")
        if max_p < 0 or max_q < 0:
            raise ValueError("[VALIDATION] max_p and max_q must be >= 0")

        if not (0 <= ts <= te < len(y)):
            raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], must be within [0, {len(y)-1}]")
        if not (0 <= es_idx <= ee < len(y)):
            raise ValueError(f"[VALIDATION] Invalid eval range: [{es_idx},{ee}], must be within [0, {len(y)-1}]")
        if es_idx <= te:
            raise ValueError(f"[VALIDATION] Evaluation start ({es_idx}) must be after training end ({te}).")
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
        best_model = arima._run_auto_arima_search(
            y_train=y_train,
            d=int(state["d"]),
            max_p=int(state["max_p"]),
            max_q=int(state["max_q"]),
            criterion="aicc",
        )
        best_model, convergence_retried = arima._ensure_converged(
            best_model,
            y_train,
            "Selected guided ARIMA model",
        )
        state["best_model"] = best_model
        state["convergence_retried"] = convergence_retried
        return {
            "criterion": "aicc",
            "order": {"p": best_model.order[0], "d": best_model.order[1], "q": best_model.order[2]},
            "aicc": float(best_model.aicc()),
            "convergence_retried": convergence_retried,
        }

    if step_id == "bic_search":
        train_range, _ = _state_ranges(state)
        diagnostics = arima._residual_diagnostics(
            state["best_model"],
            len(train_range) - int(state["d"]),
        )
        state["diagnostics"] = diagnostics
        return diagnostics

    if step_id == "select_model":
        train_range, _ = _state_ranges(state)
        best_model = state["best_model"]
        best_order = {"p": best_model.order[0], "d": best_model.order[1], "q": best_model.order[2]}
        search_summary = {
            "selection_criterion": "aicc",
            "effective_train_size": len(train_range) - int(state["d"]),
            "max_p": int(state["max_p"]),
            "max_q": int(state["max_q"]),
            "selected_order": best_order,
            "model_spec": arima._model_spec(best_model),
            "aic": float(best_model.aic()),
            "aicc": float(best_model.aicc()),
            "bic": float(best_model.bic()),
            "convergence_retried": bool(state["convergence_retried"]),
            "diagnostics": state["diagnostics"],
        }
        state["best_order"] = best_order
        state["search_summary"] = search_summary
        return {"best_order": best_order, "selection_criterion": "aicc"}

    if step_id == "forecast":
        train_range, eval_range = _state_ranges(state)
        y = np.asarray(state["y"])
        y_eval = y[eval_range]
        best_model = state["best_model"]
        forecast_steps = eval_range.stop - train_range.stop
        forecast_offset = eval_range.start - train_range.stop
        all_predictions, all_conf_int = best_model.predict(
            n_periods=forecast_steps,
            return_conf_int=True,
        )
        y_pred = np.asarray(all_predictions)[
            forecast_offset : forecast_offset + len(eval_range)
        ]
        conf_int = np.asarray(all_conf_int)[
            forecast_offset : forecast_offset + len(eval_range)
        ]
        if len(y_pred) != len(eval_range):
            raise ValueError(
                "[VALIDATION] ARIMA forecast did not cover the configured evaluation range"
            )
        y_pred = normalize_demand_forecast(y_pred)
        y_true = y_eval.tolist()
        converged = arima._is_converged(best_model)
        state.update({
            "y_true": y_true,
            "y_pred": y_pred.tolist(),
            "confidence_intervals": conf_int.tolist(),
            "converged": converged,
        })
        return {
            "predicted_points": len(y_true),
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
            "aicc": float(best_model.aicc()),
            "bic": float(best_model.bic()),
            "converged": bool(state["converged"]),
            "diagnostics": state["diagnostics"],
            "evaluated_points": len(y_eval),
            "evaluate_indices": list(eval_range),
            "evaluate_months": _month_labels(timestamps, eval_range),
        }
        calibration_mean_error = float(np.mean(y_eval - y_pred))
        calibration_count = int(len(y_eval))
        predictions_with_std = arima._format_predictions(
            y_pred.tolist(),
            state["confidence_intervals"],
            calibration_mean_error,
            calibration_count,
        )
        state.update({
            "metrics": metrics,
            "predictions": predictions_with_std,
            "calibration_mean_error": calibration_mean_error,
            "calibration_count": calibration_count,
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
            "diagnostics": state["diagnostics"],
        }
        attach_method_metadata(final_result, arima.METHOD_NAME, arima.FORECAST_STRATEGY, arima.IMPLEMENTATION_NOTES)

        if state.get("save_model"):
            model_state = {
                "model_type": arima.MODEL_TYPE,
                "model": best_model,
                "order": state["best_order"],
                "d": int(state["d"]),
                "model_spec": arima._model_spec(best_model),
                "search_summary": state["search_summary"],
                "diagnostics": state["diagnostics"],
                "prediction_domain": DEMAND_PREDICTION_DOMAIN,
                "calibration_mean_error": state["calibration_mean_error"],
                "calibration_count": state["calibration_count"],
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
    from tensorflow.keras.layers import LSTM as KerasLSTM
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(int(config["look_back"]), int(config["feature_count"]))),
        KerasLSTM(int(config["units"])),
        Dense(int(config["forecast_horizon"])),
    ])
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    return model


def _run_lstm(step_id: str, args, state: Dict[str, Any]) -> Dict[str, Any]:
    if step_id == "prepare_data":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        feature_keys = lstm._parse_feature_key_list(getattr(args, "lstm_features", None))
        target_key = getattr(args, "lstm_target_feature", None)
        feature_types_override = json.loads(getattr(args, "feature_types", None) or "{}")
        parsed_data = load_data_from_json(_read_data_json(args), target_key=target_key, feature_keys=feature_keys)
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
        normalization = getattr(args, "normalization", None) or "minmax"
        lstm._validate_training_args(parsed_data, ts, te, es_idx, ee)

        state.clear()
        state.update({
            "model_type": "lstm",
            "parsed_data": parsed_data,
            "feature_types": feature_types,
            "months": months,
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
        config = lstm.derive_lstm_config(
            len(train_range),
            forecast_horizon,
            int(scaled_series.shape[1]),
        )
        train_data = scaled_series[train_range]
        train_x, train_y = lstm._create_direct_dataset(
            train_data,
            int(config["look_back"]),
            forecast_horizon,
        )
        if train_x.size == 0:
            raise ValueError("[VALIDATION] Training range is too short for the requested LSTM look_back and forecast horizon.")
        history_df = parsed_data.df.iloc[train_range.stop - int(config["look_back"]): train_range.stop].copy()
        state.update({
            "preprocessor": preprocessor,
            "train_x": train_x,
            "train_y": train_y,
            "history_df": history_df,
            "forecast_horizon": forecast_horizon,
            "n_features_out": int(scaled_series.shape[1]),
            "derived_config": config,
        })
        return {
            "input_shape": list(train_x.shape),
            "target_shape": list(train_y.shape),
            "forecast_horizon": forecast_horizon,
            "training_regime": config["regime"],
            "look_back": config["look_back"],
            "window_count": config["windows"],
            "validation_windows": config["validation_windows"],
            "purge_windows": config["purge_windows"],
        }

    if step_id == "build_model":
        config = state["derived_config"]
        return {
            "training_regime": config["regime"],
            "window_count": config["windows"],
            "uses_validation": config["regime"] == "standard",
            "units": config["units"],
            "parameter_count": config["parameter_count"],
            "parameter_budget": config["parameter_budget"],
            "batch_size": config["batch_size"],
            "max_epochs": config["max_epochs"],
            "patience": config["patience"],
        }

    if step_id == "train_model":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        tf.keras.utils.set_random_seed(42)
        _set_lstm_seed(42)

        train_range, _ = _state_ranges(state)
        (
            model,
            preprocessor,
            history_df,
            config,
            uncertainty_profile,
            uncertainty_summary,
        ) = lstm._fit_direct_model(
            state["parsed_data"],
            state["feature_types"],
            train_range,
            state["normalization"],
            int(state["forecast_horizon"]),
        )

        keras_model_path = os.path.join(args.artifact_dir, "lstm_trained_keras_model.h5")
        temp_keras_model_path = os.path.join(args.artifact_dir, f"lstm_trained_keras_model.tmp-{os.getpid()}.h5")
        try:
            model.save(temp_keras_model_path)
            os.replace(temp_keras_model_path, keras_model_path)
        finally:
            if os.path.exists(temp_keras_model_path):
                try:
                    os.remove(temp_keras_model_path)
                except OSError:
                    pass

        state.update({
            "trained_keras_model_path": keras_model_path,
            "preprocessor": preprocessor,
            "history_df": history_df,
            "derived_config": config,
            "uncertainty_profile": uncertainty_profile,
            "uncertainty_summary": uncertainty_summary,
        })
        return {
            "training_regime": config["regime"],
            "used_validation": config["regime"] == "standard",
            "epochs_ran": config["epochs_ran"],
            "uncertainty_summary": uncertainty_summary,
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
            "look_back": int(state["derived_config"]["look_back"]),
            "forecast_horizon": forecast_horizon,
            "forecast_strategy": lstm.DIRECT_FORECAST_STRATEGY,
            "history_df": state["history_df"],
            "normalization": state["normalization"],
            "prediction_domain": DEMAND_PREDICTION_DOMAIN,
            "uncertainty_profile": state["uncertainty_profile"],
            "uncertainty_summary": state["uncertainty_summary"],
            "reference_series": parsed_data.df[parsed_data.target_key].values[train_range].tolist(),
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
        std_dev_residuals = float(state["uncertainty_profile"][0])
        metrics = {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "mape": mape(y_true, y_pred),
            "r2": r2(y_true, y_pred),
            "forecast_horizon": forecast_horizon,
            "training_regime": state["derived_config"]["regime"],
            "evaluated_points": len(y_true),
            "evaluate_indices": list(eval_range),
            "evaluate_months": _month_labels(state["months"], eval_range),
        }
        forecast_offset = eval_range.start - train_range.stop
        evaluation_profile = state["uncertainty_profile"][
            forecast_offset : forecast_offset + len(y_pred)
        ]
        predictions_with_std = lstm._format_predictions(
            y_pred.tolist(),
            evaluation_profile,
            {
                **state["uncertainty_summary"],
                "horizon_start": forecast_offset + 1,
            },
        )
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
        forecast_horizon = int(state["forecast_horizon"])
        metrics = state["metrics"]
        y_true = state["y_true"]
        predictions_with_std = state["predictions"]
        final_result: Dict[str, Any] = {
            "mode": "lstm_train_eval",
            "data_points": int(state["data_points"]),
            "target_key": parsed_data.target_key,
            "feature_types": state["feature_types"],
            "normalization": state["normalization"],
            "forecast_horizon": forecast_horizon,
            "training_regime": state["derived_config"]["regime"],
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
            "uncertainty_summary": state["uncertainty_summary"],
        }
        attach_method_metadata(final_result, lstm.METHOD_NAME, lstm.DIRECT_FORECAST_STRATEGY, lstm.DIRECT_IMPLEMENTATION_NOTES)

        if state.get("save_model"):
            save_path = state["save_model"]
            keras_model_path = os.path.splitext(save_path)[0] + "_keras_model.h5"
            temp_keras_model_path = os.path.splitext(save_path)[0] + f"_keras_model.tmp-{os.getpid()}.h5"
            try:
                shutil.copy2(state["trained_keras_model_path"], temp_keras_model_path)
                os.replace(temp_keras_model_path, keras_model_path)
            finally:
                if os.path.exists(temp_keras_model_path):
                    try:
                        os.remove(temp_keras_model_path)
                    except OSError:
                        pass

            model_state = {
                "model_type": lstm.MODEL_TYPE,
                "look_back": int(state["derived_config"]["look_back"]),
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
                "derived_config": state["derived_config"],
                "uncertainty_profile": state["uncertainty_profile"],
                "uncertainty_summary": state["uncertainty_summary"],
                "reference_series": parsed_data.df[parsed_data.target_key].values[train_range].tolist(),
                "prediction_domain": DEMAND_PREDICTION_DOMAIN,
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
    parser.add_argument("--input-state-file")
    parser.add_argument("--output-state-file", required=True)
    parser.add_argument("--receipt-file", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--data", type=str)
    parser.add_argument("--data-file", type=str)
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
    args = parser.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)
    existing_receipt = _load_existing_receipt(
        args.receipt_file,
        args.attempt_id,
        args.step_id,
    )
    if existing_receipt is not None:
        print(json.dumps(existing_receipt, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    state = _load_state(args.input_state_file)
    step_result = run_step(args.model_type, args.step_id, args, state)

    final_result = None
    if isinstance(step_result, dict) and "final_result" in step_result:
        final_result = step_result["final_result"]
        output = step_result.get("output", {})
    else:
        output = step_result

    _save_state(args.output_state_file, state)
    payload = {
        "protocol_version": 3,
        "attempt_id": args.attempt_id,
        "step_id": args.step_id,
        "output": _jsonable(output),
        "input_state_path": args.input_state_file,
        "input_state_sha256": _sha256_file(args.input_state_file),
        "candidate_state_path": args.output_state_file,
        "candidate_state_sha256": _sha256_file(args.output_state_file),
    }
    if final_result is not None:
        payload["final_result"] = _jsonable(final_result)

    _write_receipt(args.receipt_file, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        formatted_error = format_known_error(exc)
        if formatted_error is None:
            raise
        print(formatted_error, file=sys.stderr, flush=True)
        sys.exit(1)
