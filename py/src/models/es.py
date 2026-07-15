# py/src/models/es.py
# -*- coding: utf-8 -*-
"""
Exponential Smoothing (ES) time series forecasting model.
"""
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing

from data_utils import load_data_from_json
from forecast_utils import (
    DEMAND_PREDICTION_DOMAIN,
    apply_prediction_domain,
    resolve_prediction_domain,
)
from logger import log, get_log_contents
from log_safety import summarize_cli_args
from metrics import mae, mape, r2, rmse
from model_info import attach_method_metadata, ensure_state_metadata
from serialization import dump_model, load_model
from uncertainty import format_predictions, profile_for_steps, ses_profile

MODEL_TYPE = "es"
METHOD_NAME = "一次指数平滑"
FORECAST_STRATEGY = "flat_simple_exponential_smoothing"
IMPLEMENTATION_NOTES = [
    "本系统实现教科书中的一次指数平滑，平滑系数 alpha 满足 0 < alpha <= 1。",
    "alpha 由用户指定且不会被优化；初始水平 l0 在固定 alpha 下通过最小化训练期一步预测 SSE 自动估计。",
    "一次指数平滑的多步未来销量预测为水平外推；趋势和季节性需要 Holt 或 Holt-Winters 扩展，本系统未在该模型中启用。",
    "在非负销量输入下，一次指数平滑的凸递推通常自然非负；本系统仍将最终销量点预测按 max(0, y_hat) 截断，并用截断后的值计算残差与指标。该后处理不属于教科书 SES 递推式。",
    "std_dev 使用 ETS(A,N,N) 的逐步预测方差 sigma²[1+alpha²(h-1)]；sigma 来自训练期一步预测残差。95% 区间和生产用 99% 上侧误差则由带相同步长增长因子的 actual-prediction 经验残差分位数给出，样本不足时明确标记正态近似回退。",
]

# Type alias for input flexibility
ArrayLike = Union[List[float], np.ndarray]


def _format_predictions(
    preds: List[float],
    std_dev: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Format predictions with a horizon-aware uncertainty profile."""
    profile = list(std_dev) if isinstance(std_dev, (list, tuple, np.ndarray)) else [float(std_dev)] * len(preds)
    return format_predictions(preds, profile, metadata or {"source": "empirical", "fallback": False})


def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved ES model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid ES model file.")
    return ensure_state_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)


def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
    allow_negative_predictions: bool = False,
) -> Tuple[List[float], List[float]]:
    """Forecast future values using a saved ES model."""
    log.debug(f"ES predict called with steps={steps}, model_path={model_path}")
    state = state or _load_state(model_path)
    model = state["model"]
    alpha = float(state.get("alpha", 0.5))

    log.info(f"Generating {steps}-step forecast with alpha={alpha}")
    prediction_domain = resolve_prediction_domain(
        state,
        allow_negative_predictions=allow_negative_predictions,
    )
    predictions = apply_prediction_domain(model.forecast(steps), prediction_domain)
    uncertainty_profile, _ = profile_for_steps(
        state.get("uncertainty_profile"),
        steps,
        state.get("reference_series", []),
        state.get("uncertainty_summary"),
    )
    log.info(f"Generated {len(predictions)} predictions")

    return predictions.tolist(), uncertainty_profile


def _reference_scale(reference_series: ArrayLike) -> float:
    """Return a training-history fallback scale when residuals are unavailable."""
    reference_array = np.asarray(reference_series, dtype=float)
    if reference_array.size > 1:
        return float(np.std(reference_array, ddof=1))

    baseline = abs(float(np.mean(reference_array))) if reference_array.size else 0.0
    return float(max(baseline * 0.05, 1.0))


def _fit_textbook_ses(y_train: ArrayLike, alpha: float) -> Any:
    """Estimate the initial level by SSE while keeping user-selected alpha fixed."""
    train_array = np.asarray(y_train, dtype=float)
    if train_array.size == 0:
        raise ValueError("[VALIDATION] ES training range is empty.")

    return SimpleExpSmoothing(
        train_array,
        initialization_method="estimated",
    ).fit(
        smoothing_level=alpha,
        optimized=True,
    )


def _training_residuals(model: Any, y_train: ArrayLike) -> np.ndarray:
    values = np.asarray(y_train, dtype=float)
    fitted = np.asarray(model.fittedvalues, dtype=float)
    residuals = values - fitted
    return residuals[np.isfinite(residuals)]


def _training_residual_scale(model: Any, y_train: ArrayLike) -> float:
    values = np.asarray(y_train, dtype=float)
    finite = _training_residuals(model, values)
    if finite.size > 1:
        return float(np.std(finite, ddof=1))
    return _reference_scale(values)


def train_and_evaluate(
    y: ArrayLike,
    timestamps: List[str],
    train_range: range,
    eval_range: range,
    alpha: float,
    prediction_domain: str = DEMAND_PREDICTION_DOMAIN,
) -> Tuple[Dict[str, Any], Any, List[float], List[float], float]:
    """Train and evaluate the Exponential Smoothing model."""
    log.info(f"Starting ES train_and_evaluate: alpha={alpha}, train_range={train_range}, eval_range={eval_range}")
    series = np.asarray(y)
    y_train = series[train_range]
    y_eval = series[eval_range]

    log.debug(f"Total series length: {len(series)}")
    log.debug(f"Training data range: [{train_range.start}, {train_range.stop-1}], count={len(y_train)}")
    log.debug(f"Evaluation data range: [{eval_range.start}, {eval_range.stop-1}], count={len(y_eval)}")
    log.debug(f"Training data: min={np.min(y_train):.4f}, max={np.max(y_train):.4f}, mean={np.mean(y_train):.4f}")

    log.info(f"Fitting SimpleExpSmoothing with alpha={alpha}")
    model = _fit_textbook_ses(y_train, alpha)
    log.debug("Model fitted successfully")

    forecast_steps = eval_range.stop - train_range.stop
    forecast_offset = eval_range.start - train_range.stop
    log.info(
        f"Forecasting {forecast_steps} steps through evaluation gap "
        f"(offset={forecast_offset}, evaluated_points={len(eval_range)})"
    )
    all_predictions = np.asarray(model.forecast(forecast_steps))
    y_pred = all_predictions[
        forecast_offset : forecast_offset + len(eval_range)
    ]
    if len(y_pred) != len(eval_range):
        raise ValueError(
            "[VALIDATION] ES forecast did not cover the configured evaluation range"
        )
    y_pred = apply_prediction_domain(y_pred, prediction_domain)
    y_true = y_eval.tolist()

    residuals = np.array(y_true) - np.array(y_pred)
    std_dev_residuals = _training_residual_scale(model, y_train)

    log.info(f"Calculated residuals: mean={np.mean(residuals):.4f}, std_dev={std_dev_residuals:.4f}")

    metrics = {
        "alpha": alpha,
        "evaluated_points": len(y_true),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "evaluate_indices": list(eval_range),
        "evaluate_months": [timestamps[i] for i in eval_range] if timestamps else [],
    }

    log.info(f"Evaluation metrics computed: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}, R²={metrics['r2']:.4f}")

    return metrics, model, y_true, y_pred.tolist(), std_dev_residuals


def _validate_training_args(y: ArrayLike, ts: int, te: int, es: int, ee: int, alpha: float) -> None:
    """Validate training and evaluation parameters."""
    log.debug(f"Validating training arguments: ts={ts}, te={te}, es={es}, ee={ee}, alpha={alpha}")
    n_total = len(y)
    log.debug(f"Total data points: {n_total}")

    if not (0 <= ts <= te < n_total):
        log.error(f"Invalid training range: [{ts}, {te}], data size: {n_total}")
        raise ValueError(f"[VALIDATION] Invalid training range [{ts}, {te}], must be within [0, {n_total-1}]")
    if not (es >= 0 and ee < n_total and es <= ee):
        log.error(f"Invalid evaluation range: [{es}, {ee}], data size: {n_total}")
        raise ValueError(f"[VALIDATION] Invalid evaluation range [{es}, {ee}], must be within [0, {n_total-1}]")
    if es <= te:
        log.error(f"Evaluation start {es} must be after training end {te}")
        raise ValueError(f"[VALIDATION] Evaluation start ({es}) must be after training end ({te}).")
    if not (0.0 < alpha <= 1.0):
        log.error(f"Alpha {alpha} must be in (0, 1]")
        raise ValueError(f"[VALIDATION] Alpha must be in (0, 1], got {alpha}")

    # Additional helpful validations
    train_size = te - ts + 1
    eval_size = ee - es + 1
    log.info(f"Validation passed: train_size={train_size}, eval_size={eval_size}, alpha={alpha}")


def _validate_fit_args(y: ArrayLike, ts: int, te: int, alpha: float) -> None:
    n_total = len(y)

    if not (0 <= ts <= te < n_total):
        raise ValueError(f"[VALIDATION] Invalid training range [{ts}, {te}], must be within [0, {n_total-1}]")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"[VALIDATION] Alpha must be in (0, 1], got {alpha}")


def run(args) -> None:
    """Main entry point for ES model training and prediction."""
    log.info(f"ES run started with args: {summarize_cli_args(args)}")
    pretty: bool = bool(getattr(args, "pretty", False))
    save_path = getattr(args, "save_model", None)
    load_path = getattr(args, "load_model", None)

    if getattr(args, "fit_only", False):
        if not hasattr(args, "data") or not args.data:
            raise ValueError("[VALIDATION] --data is required for fit-only mode.")
        if not save_path:
            raise ValueError("[VALIDATION] --save-model is required for fit-only mode.")

        state = _load_state(load_path) if load_path else {}
        prediction_domain = resolve_prediction_domain(
            state,
            allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
        )
        parsed_data = load_data_from_json(args.data)
        y = parsed_data.y

        ts, te = int(args.train_start), int(args.train_end)
        if state and state.get("alpha") is None:
            raise ValueError("[VALIDATION] Saved ES model is missing alpha.")
        alpha = float(state["alpha"] if state else getattr(args, "alpha", 0.5))
        _validate_fit_args(y, ts, te, alpha)

        y_fit = np.asarray(y)[range(ts, te + 1)]
        model = _fit_textbook_ses(y_fit, alpha)
        std_dev_residuals = _training_residual_scale(model, y_fit)
        profile, uncertainty_summary = ses_profile(
            std_dev_residuals,
            alpha,
            12,
            _training_residuals(model, y_fit),
        )

        next_state = {
            "model_type": MODEL_TYPE,
            "alpha": alpha,
            "train_end_index": te,
            "model": model,
            "std_dev_residuals": std_dev_residuals,
            "initial_level": float(model.params["initial_level"]),
            "uncertainty_profile": profile,
            "uncertainty_summary": uncertainty_summary,
            "reference_series": y_fit.tolist(),
            "prediction_domain": prediction_domain,
        }
        attach_method_metadata(next_state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(next_state, save_path)

        out: Dict[str, Any] = {
            "mode": "es_fit_only",
            "alpha": alpha,
            "train_end_index": te,
            "saved_model": save_path,
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
        attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        out["logs"] = get_log_contents()
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    if load_path:
        log.info(f"Entering prediction mode with model: {load_path}")
        try:
            state = _load_state(load_path)
            steps = int(getattr(args, "forecast_steps", 0))
            if steps <= 0:
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
                "mode": "es_loaded_predict",
                "alpha": float(state["alpha"]),
                "train_end_index": int(state["train_end_index"]),
                "forecast_steps": steps,
                "predictions": predictions_with_std,
            }
            attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
            log.info("Prediction finished successfully")
            out["logs"] = get_log_contents()
            print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
            return
        except Exception as e:
            log.error(f"An error occurred during prediction: {e}", exc_info=True)
            raise

    # Training Mode
    log.info("Entering training mode.")
    if not hasattr(args, "data") or not args.data:
        log.error("--data is required for training.")
        raise ValueError("[VALIDATION] --data is required for training.")

    try:
        parsed_data = load_data_from_json(args.data)
        y = parsed_data.y
        timestamps = parsed_data.timestamps or [str(i) for i in range(parsed_data.n_obs)]
        log.info(f"Data loaded successfully. Number of observations: {parsed_data.n_obs}")

        ts, te = int(args.train_start), int(args.train_end)
        es, ee = int(args.eval_start), int(args.eval_end)

        alpha_arg = getattr(args, "alpha", None)
        alpha = float(alpha_arg) if alpha_arg is not None else 0.5
        log.debug(f"Training params: alpha={alpha}")

        _validate_training_args(y, ts, te, es, ee, alpha)
        prediction_domain = resolve_prediction_domain(
            allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
        )

        log.info("Starting model training and evaluation...")
        metrics, model, y_true, y_pred, std_dev_residuals = train_and_evaluate(
            y,
            timestamps,
            range(ts, te + 1),
            range(es, ee + 1),
            alpha,
            prediction_domain,
        )
        log.info(f"Training finished. Metrics: {metrics}")

        full_profile, uncertainty_summary = ses_profile(
            std_dev_residuals,
            alpha,
            max(12, ee - te),
            _training_residuals(model, np.asarray(y)[range(ts, te + 1)]),
        )
        forecast_offset = es - te - 1
        evaluation_profile = full_profile[forecast_offset : forecast_offset + len(y_pred)]
        predictions_with_std = _format_predictions(
            y_pred,
            evaluation_profile,
            {**uncertainty_summary, "horizon_start": forecast_offset + 1},
        )

        out: Dict[str, Any] = {
            "mode": "fixed_alpha_only",
            "data_points": len(y),
            "alpha": alpha,
            "train_range": {"start_index": ts, "end_index": te, "months": timestamps[ts : te + 1]},
            "evaluate_range": {"start_index": es, "end_index": ee, "months": timestamps[es : ee + 1]},
            "metrics": metrics,
            "eval_y_true": y_true,
            "predictions": predictions_with_std,
            "initial_level": float(model.params["initial_level"]),
            "uncertainty_summary": uncertainty_summary,
        }
        attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)

        if save_path:
            log.info(f"Saving model to {save_path}")
            state = {
                "model_type": MODEL_TYPE, "alpha": alpha, "train_end_index": te,
                "model": model, "std_dev_residuals": std_dev_residuals,
                "initial_level": float(model.params["initial_level"]),
                "uncertainty_profile": full_profile[:12],
                "uncertainty_summary": uncertainty_summary,
                "reference_series": np.asarray(y)[range(ts, te + 1)].tolist(),
                "prediction_domain": prediction_domain,
            }
            attach_method_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
            dump_model(state, save_path)
            log.info(f"Model saved successfully to {save_path}")
            out["saved_model"] = save_path

        out["logs"] = get_log_contents()
        log.info("Run finished successfully.")
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))

    except ValueError:
        raise
    except Exception as e:
        log.error(f"An error occurred during data processing or training: {e}", exc_info=True)
        raise ValueError(f"[DATA] Failed to parse or process --data: {e}")
