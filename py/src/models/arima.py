#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA model with automatic order selection for p and q.
User specifies d, and the model searches for p and q using AICc.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox

from data_utils import load_data_from_json
from forecast_utils import (
    DEMAND_PREDICTION_DOMAIN,
    SIGNED_PREDICTION_DOMAIN,
    apply_prediction_domain,
    resolve_prediction_domain,
)
from logger import log, get_log_contents
from log_safety import summarize_cli_args
from metrics import mae, mape, r2, rmse
from model_info import attach_method_metadata, ensure_state_metadata
from serialization import dump_model, load_model
from dynamic_config import derive_arima_search_limit
from uncertainty import NORMAL_95, NORMAL_99

MODEL_TYPE = "arima"
METHOD_NAME = "ARIMA 自动定阶"
FORECAST_STRATEGY = "arima_multi_step_forecast"
ARIMA_SEARCH_MAXITER = 100
ARIMA_RETRY_MAXITER = 200
MODEL_INTERVAL_KIND = "model_prediction_interval"
NONNEGATIVE_INTERVAL_KIND = "censored_nonnegative_model_prediction_interval"
IMPLEMENTATION_NOTES = [
    "教科书 ARIMA 通常通过平稳性检验、ACF/PACF 和信息准则定阶。",
    "本系统由用户根据 ADF 结果指定差分阶数 d；模型固定为非季节 ARIMA，并执行一次以 AICc 为目标的 stepwise 搜索来选择 p、q 和是否包含截距。",
    "p/q 搜索上限按差分后的有效样本量自动收紧；少于 6 个有效样本时拒绝拟合，不用不可靠结果冒充完整 ARIMA。",
    "AICc stepwise 搜索中的每个候选拟合使用 maxiter=100；选中模型必须收敛，首次结果未收敛时按完整规格以 maxiter=200 重拟合一次。",
    "预测未来销量时使用拟合后的 ARIMA 模型直接生成多步 forecast；原始点预测可能为负，本系统将用于残差、指标和输出的销量点预测按 max(0, y_hat) 截断。该约束是业务后处理，不属于教科书 ARIMA。",
    "std_dev 和 99% 上侧误差来自未截断 ARIMA 预测分布；非负销量域会把 95% 区间分位数同步映射为 max(0, q)，有符号残差域则保留原始区间。",
    "Ljung-Box 仅检查输出中注明的单个非季节滞后；兼容字段 residual_white_noise=true 只表示该检验未发现残差自相关，不代表已经证明残差为白噪声。",
    "独立评估段只用于报告泛化指标，不写回模型产物，也不作为部署偏差校准；部署区间完全来自训练后 ARIMA 预测分布，因而不会把最终评估标签带入未来预测。",
]


# ---------- utilities ----------
def _format_predictions(
    preds: List[float],
    conf_intervals: List[List[float]],
    prediction_domain: str = DEMAND_PREDICTION_DOMAIN,
) -> List[Dict[str, Any]]:
    """
    Format predictions with 95% prediction intervals for JSON output.
    ``std_dev`` remains the scale of the original Gaussian ARIMA distribution.
    Demand-domain interval quantiles are mapped through ``max(0, q)`` so they
    remain coherent with the displayed nonnegative point forecast, while
    signed residual forecasts preserve the original interval.
    """
    if prediction_domain not in {
        DEMAND_PREDICTION_DOMAIN,
        SIGNED_PREDICTION_DOMAIN,
    }:
        raise ValueError(f"[VALIDATION] Unknown prediction domain: {prediction_domain}")

    points: List[Dict[str, Any]] = []
    for prediction, interval in zip(preds, conf_intervals):
        numeric_prediction = float(
            apply_prediction_domain([prediction], prediction_domain)[0]
        )
        raw_lower = float(interval[0])
        raw_upper = float(interval[1])
        std_dev = max(0.0, (raw_upper - raw_lower) / (2.0 * NORMAL_95))
        raw_distribution_mean = (raw_lower + raw_upper) / 2.0
        if prediction_domain == DEMAND_PREDICTION_DOMAIN:
            lower = max(0.0, raw_lower)
            upper = max(0.0, raw_upper)
            lower = min(lower, numeric_prediction)
            upper = max(upper, numeric_prediction)
            interval_kind = NONNEGATIVE_INTERVAL_KIND
        else:
            lower = raw_lower
            upper = raw_upper
            interval_kind = MODEL_INTERVAL_KIND
        points.append({
            "prediction": numeric_prediction,
            "std_dev": std_dev,
            "uncertainty_source": "model",
            "calibration_source": "arima_forecast_distribution",
            # Independent evaluation labels are report-only. ARIMA deployment
            # intervals are already supplied by the fitted forecast distribution.
            "calibration_mean_error": None,
            "calibration_count": None,
            "interval_lower": lower,
            "interval_upper": upper,
            "interval_level": 0.95,
            "interval_kind": interval_kind,
            "upper_error_p99": max(
                0.0,
                raw_distribution_mean + NORMAL_99 * std_dev - numeric_prediction,
            ),
        })
    return points


def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved ARIMA model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid ARIMA model file.")
    return ensure_state_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)


def _validate_arima_train_size(train_size: int, d: int, context: str) -> int:
    try:
        return int(derive_arima_search_limit(train_size, d)["effective_train_size"])
    except ValueError as exc:
        raise ValueError(f"{exc} Context: {context}.") from exc


def _run_auto_arima_search(
    y_train: np.ndarray,
    d: int,
    max_p: int,
    max_q: int,
    criterion: str = "aicc",
) -> Any:
    if d < 0:
        raise ValueError(f"[VALIDATION] ARIMA d must be >= 0, got {d}")
    if max_p < 0 or max_q < 0:
        raise ValueError("[VALIDATION] ARIMA max_p and max_q must be >= 0")
    try:
        return pm.auto_arima(
            y_train,
            d=d,
            start_p=0,
            start_q=0,
            max_p=max_p,
            max_q=max_q,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion=criterion,
            with_intercept="auto",
            maxiter=ARIMA_SEARCH_MAXITER,
            trace=False,
        )
    except Exception as exc:
        raise ValueError(
            f"[VALIDATION] ARIMA {criterion.upper()} search failed for the current training range: {exc}"
        ) from exc


def _model_spec(model: Any) -> Dict[str, Any]:
    return {
        "order": [int(value) for value in model.order],
        "seasonal_order": [int(value) for value in getattr(model, "seasonal_order", (0, 0, 0, 0))],
        "with_intercept": bool(getattr(model, "with_intercept", False)),
        "trend": getattr(model, "trend", None),
    }


def _is_converged(model: Any) -> bool:
    return bool(getattr(model, "arima_res_", None) is not None and model.arima_res_.mle_retvals.get("converged", False))


def _fit_arima_spec(
    y_train: np.ndarray,
    spec: Dict[str, Any],
    context: str,
    maxiter: int = ARIMA_SEARCH_MAXITER,
) -> Any:
    try:
        order = tuple(int(value) for value in spec["order"])
        seasonal_order = tuple(int(value) for value in spec.get("seasonal_order", (0, 0, 0, 0)))
        fitted_model = pm.ARIMA(
            order=order,
            seasonal_order=seasonal_order,
            with_intercept=bool(spec.get("with_intercept", False)),
            trend=spec.get("trend"),
            suppress_warnings=True,
            maxiter=maxiter,
        )
        fitted_model.fit(y_train)
        return fitted_model
    except Exception as exc:
        raise ValueError(
            f"[VALIDATION] {context} failed to fit ARIMA specification {spec}: {exc}"
        ) from exc


def _fit_arima_order(y_train: np.ndarray, order: Tuple[int, int, int], context: str) -> Any:
    return _fit_arima_spec(
        y_train,
        {"order": list(order), "seasonal_order": [0, 0, 0, 0], "with_intercept": False, "trend": None},
        context,
    )


def _is_deterministic_boundary_case(model: Any, y_train: np.ndarray) -> bool:
    """Recognize a zero-innovation random walk/drift likelihood boundary."""
    order = tuple(int(value) for value in model.order)
    if order[0] != 0 or order[2] != 0:
        return False
    effective = np.diff(np.asarray(y_train, dtype=float), n=order[1]) if order[1] else np.asarray(y_train, dtype=float)
    if effective.size < 2 or not np.isfinite(effective).all():
        return False
    scale = max(1.0, float(np.max(np.abs(effective))))
    return bool(float(np.ptp(effective)) <= np.sqrt(np.finfo(float).eps) * scale)


def _ensure_converged(model: Any, y_train: np.ndarray, context: str) -> Tuple[Any, bool]:
    if _is_converged(model):
        return model, False
    retried = _fit_arima_spec(
        y_train,
        _model_spec(model),
        context,
        maxiter=ARIMA_RETRY_MAXITER,
    )
    if not _is_converged(retried):
        if _is_deterministic_boundary_case(retried, y_train):
            log.warning(
                "%s reached the zero-innovation likelihood boundary; accepting the "
                "finite ARIMA(0,d,0) forecast as a deterministic teaching case.",
                context,
            )
            return retried, True
        raise ValueError(
            f"[MODEL] {context} did not converge after retrying the selected ARIMA specification."
        )
    return retried, True


def _residual_diagnostics(model: Any, effective_train_size: int) -> Dict[str, Any]:
    order = tuple(int(value) for value in model.order)
    lag = min(10, effective_train_size // 5)
    model_df = order[0] + order[2]
    if lag <= model_df or lag < 1:
        return {
            "ljung_box_lag": None,
            "ljung_box_pvalue": None,
            "residual_white_noise": None,
            "ljung_box_conclusion": "test_unavailable",
            "warning": "effective sample is too short for a degrees-of-freedom adjusted Ljung-Box test",
        }
    residuals = np.asarray(model.arima_res_.resid, dtype=float)
    result = acorr_ljungbox(residuals, lags=[lag], model_df=model_df, return_df=True)
    pvalue = float(result["lb_pvalue"].iloc[-1])
    if not np.isfinite(pvalue):
        return {
            "ljung_box_lag": lag,
            "ljung_box_pvalue": None,
            "residual_white_noise": None,
            "ljung_box_conclusion": "test_unavailable",
            "warning": "residual variance is degenerate, so Ljung-Box is undefined",
        }
    autocorrelation_detected = pvalue < 0.05
    return {
        "ljung_box_lag": lag,
        "ljung_box_pvalue": pvalue,
        # Compatibility shorthand. A non-rejection at one tested lag does not
        # prove that the complete residual process is white noise.
        "residual_white_noise": not autocorrelation_detected,
        "ljung_box_conclusion": (
            "residual_autocorrelation_detected_at_tested_lag"
            if autocorrelation_detected
            else "residual_autocorrelation_not_detected_at_tested_lag"
        ),
        "warning": (
            "residual autocorrelation remains at the 5% level"
            if autocorrelation_detected
            else None
        ),
    }


# ---------- predict ----------
def predict(
    model_path: str,
    steps: int,
    state: Optional[Dict[str, Any]] = None,
    allow_negative_predictions: bool = False,
) -> Tuple[List[float], List[List[float]]]:
    """
    Forecast using a saved ARIMA model.

    Returns:
        predictions: List of forecasted values
        conf_int: List of [lower, upper] prediction interval bounds
    """
    log.debug(f"ARIMA predict called with steps={steps}, model_path={model_path}")
    if steps <= 0:
        raise ValueError("[VALIDATION] steps must be > 0 for forecasting.")

    state = state or _load_state(model_path)
    model = state["model"]
    order = state.get("order", {})

    log.info(f"Generating {steps}-step forecast with order (p={order.get('p')}, d={order.get('d')}, q={order.get('q')})")
    y_pred, conf_int = model.predict(n_periods=steps, return_conf_int=True)
    prediction_domain = resolve_prediction_domain(
        state,
        allow_negative_predictions=allow_negative_predictions,
    )
    y_pred = apply_prediction_domain(y_pred, prediction_domain)
    log.info(f"Generated {len(y_pred)} predictions with prediction intervals")

    return y_pred.tolist(), conf_int.tolist()


# ---------- train and evaluate ----------
def train_and_evaluate(
    y: np.ndarray,
    timestamps: List[str],
    train_range: range,
    eval_range: range,
    d: int,
    max_p: int = 5,
    max_q: int = 5,
    prediction_domain: str = DEMAND_PREDICTION_DOMAIN,
) -> Tuple[Dict[str, Any], Any, Dict[str, int], Dict[str, Any], List[float], List[float], List[List[float]]]:
    """
    Train a fixed-d non-seasonal ARIMA selected by AICc stepwise search.

    Args:
        y: Full time series
        timestamps: Timestamp labels
        train_range: Range of indices for training
        eval_range: Range of indices for evaluation
        d: Differencing order (fixed by user)
        max_p: Maximum AR order to search
        max_q: Maximum MA order to search

    Returns:
        metrics: Evaluation metrics dict
        best_model: The selected ARIMA model
        best_order: Dict with p, d, q
        search_summary: Summary of AICc model search
        y_true: True evaluation values
        y_pred: Predicted evaluation values
        conf_int: Prediction intervals
    """
    log.info(f"Starting ARIMA train_and_evaluate: d={d}, max_p={max_p}, max_q={max_q}, train_range={train_range}, eval_range={eval_range}")

    if d < 0:
        raise ValueError(f"[VALIDATION] d must be >= 0, got {d}")
    if max_p < 0 or max_q < 0:
        raise ValueError("[VALIDATION] max_p and max_q must be >= 0")

    y_train = y[train_range]
    y_eval = y[eval_range]

    log.debug(f"Total series length: {len(y)}")
    log.debug(f"Training data range: [{train_range.start}, {train_range.stop-1}], count={len(y_train)}")
    log.debug(f"Evaluation data range: [{eval_range.start}, {eval_range.stop-1}], count={len(y_eval)}")
    log.debug(f"Training data: min={np.min(y_train):.4f}, max={np.max(y_train):.4f}, mean={np.mean(y_train):.4f}")

    if len(y_train) == 0:
        raise ValueError("[VALIDATION] Training range is empty")
    if len(y_eval) == 0:
        raise ValueError("[VALIDATION] Evaluation range is empty")

    effective_train_size = _validate_arima_train_size(
        train_size=len(y_train),
        d=d,
        context="Training range",
    )

    # Warn about effective sample size reduction due to differencing
    if d > 0:
        log.info(f"After d={d} differencing, effective training size: {len(y_train)} → {effective_train_size}")
        if effective_train_size < 10:
            log.warning(f"Very small effective sample size ({effective_train_size}) after differencing. "
                       f"Consider using d=0 or collecting more data for reliable parameter estimation.")
        elif effective_train_size < 20:
            log.warning(f"Small effective sample size ({effective_train_size}) after differencing. "
                       f"Model may have limited predictive power. Consider simpler models.")

    log.info(f"Running auto_arima with AICc criterion: searching p in [0, {max_p}], q in [0, {max_q}], d={d}")
    best_model = _run_auto_arima_search(
        y_train=y_train,
        d=d,
        max_p=max_p,
        max_q=max_q,
        criterion="aicc",
    )
    best_model, convergence_retried = _ensure_converged(
        best_model,
        y_train,
        "Selected ARIMA model",
    )
    log.debug(f"AICc-optimized model found: order={best_model.order}, spec={_model_spec(best_model)}")

    best_order = {"p": best_model.order[0], "d": best_model.order[1], "q": best_model.order[2]}

    # Calculate and log parameter-to-data ratio for diagnostic purposes
    n_params = best_order['p'] + best_order['q']  # Number of AR and MA parameters
    effective_sample_size = len(y_train) - d
    if n_params > 0:
        param_ratio = (n_params / effective_sample_size) * 100
        log.info(f"Final selected order: (p={best_order['p']}, d={best_order['d']}, q={best_order['q']})")
        log.info(f"Model complexity: {n_params} parameters / {effective_sample_size} effective samples = {param_ratio:.1f}%")
        if param_ratio > 30:
            log.warning(f"High parameter-to-sample ratio ({param_ratio:.1f}%). Model may be at risk of overfitting.")
    else:
        log.info(f"Final selected order: (p={best_order['p']}, d={best_order['d']}, q={best_order['q']}) - Random walk model")

    diagnostics = _residual_diagnostics(best_model, effective_train_size)
    search_summary = {
        "selection_criterion": "aicc",
        "effective_train_size": effective_train_size,
        "max_p": max_p,
        "max_q": max_q,
        "search_maxiter": ARIMA_SEARCH_MAXITER,
        "retry_maxiter": ARIMA_RETRY_MAXITER,
        "selected_order": best_order,
        "model_spec": _model_spec(best_model),
        "aic": float(best_model.aic()),
        "aicc": float(best_model.aicc()),
        "bic": float(best_model.bic()),
        "convergence_retried": convergence_retried,
        "convergence_boundary_case": not _is_converged(best_model),
        "diagnostics": diagnostics,
    }

    # --- Evaluate the best model ---
    forecast_steps = eval_range.stop - train_range.stop
    forecast_offset = eval_range.start - train_range.stop
    log.info(
        f"Forecasting {forecast_steps} steps through evaluation gap "
        f"(offset={forecast_offset}, evaluated_points={len(eval_range)})"
    )
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
    y_pred = apply_prediction_domain(y_pred, prediction_domain)
    y_true = y_eval.tolist()

    converged = _is_converged(best_model)

    residuals = np.array(y_true) - y_pred
    log.info(f"Calculated residuals: mean={np.mean(residuals):.4f}, std_dev={np.std(residuals):.4f}")

    metrics = {
        "rmse": rmse(y_eval, y_pred),
        "mae": mae(y_eval, y_pred),
        "mape": mape(y_eval, y_pred),
        "r2": r2(y_eval, y_pred),
        "aic": float(best_model.aic()),
        "aicc": float(best_model.aicc()),
        "bic": float(best_model.bic()),
        "converged": converged,
        "convergence_boundary_case": not converged and _is_deterministic_boundary_case(best_model, y_train),
        "diagnostics": diagnostics,
        "evaluated_points": len(y_eval),
        "evaluate_indices": list(eval_range),
        "evaluate_months": [timestamps[i] for i in eval_range] if timestamps else [],
    }

    log.info(f"Evaluation metrics computed: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}, R²={metrics['r2']:.4f}, AIC={metrics['aic']:.4f}, BIC={metrics['bic']:.4f}")

    return metrics, best_model, best_order, search_summary, y_true, y_pred.tolist(), conf_int.tolist()


def _validate_fit_args(y: np.ndarray, ts: int, te: int) -> None:
    if not (0 <= ts <= te < len(y)):
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], must be within [0, {len(y)-1}]")


# ---------- public entry ----------
def run(args) -> None:
    """
    Main entry point for ARIMA model training and prediction.
    """
    log.info(f"ARIMA run started with args: {summarize_cli_args(args)}")
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
        parsed_data = load_data_from_json(args.data)
        y = parsed_data.y
        ts, te = int(args.train_start), int(args.train_end)
        _validate_fit_args(y, ts, te)

        model_spec = state.get("model_spec")
        if not isinstance(model_spec, dict):
            raise ValueError("[MODEL] Saved ARIMA model is missing its complete specification.")
        order = tuple(int(value) for value in model_spec.get("order", []))
        if len(order) != 3:
            raise ValueError("[MODEL] Saved ARIMA order is invalid.")
        train_size = te - ts + 1
        _validate_arima_train_size(
            train_size=train_size,
            d=order[1],
            context="ARIMA fit-only training range",
        )
        fitted_model = _fit_arima_spec(
            y_train=y[ts : te + 1],
            spec=model_spec,
            context="ARIMA fit-only",
        )
        fitted_model, convergence_retried = _ensure_converged(
            fitted_model,
            y[ts : te + 1],
            "ARIMA fit-only",
        )
        diagnostics = _residual_diagnostics(fitted_model, train_size - order[1])

        next_state = {
            "model_type": MODEL_TYPE,
            "model": fitted_model,
            "order": {"p": order[0], "d": order[1], "q": order[2]},
            "d": order[1],
            "model_spec": _model_spec(fitted_model),
            "search_summary": state.get("search_summary"),
            "diagnostics": diagnostics,
            "prediction_domain": prediction_domain,
        }
        attach_method_metadata(next_state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(next_state, save_path)

        out: Dict[str, Any] = {
            "mode": "arima_fit_only",
            "order": next_state["order"],
            "train_end_index": te,
            "saved_model": save_path,
            "convergence_retried": convergence_retried,
            "convergence_boundary_case": not _is_converged(fitted_model),
            "diagnostics": diagnostics,
        }
        fit_predict_steps = int(getattr(args, "fit_predict_steps", 0) or 0)
        if fit_predict_steps > 0:
            preds, confidence_intervals = predict(
                save_path,
                fit_predict_steps,
                state=next_state,
                allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
            )
            out["forecast_steps"] = fit_predict_steps
            out["predictions"] = _format_predictions(
                preds,
                confidence_intervals,
                prediction_domain,
            )
        attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        out["logs"] = get_log_contents()
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    if load_path:
        log.info(f"Entering prediction mode with model: {load_path}")
        state = _load_state(load_path)
        allow_negative_predictions = bool(
            getattr(args, "allow_negative_predictions", False)
        )
        prediction_domain = resolve_prediction_domain(
            state,
            allow_negative_predictions=allow_negative_predictions,
        )
        steps = int(getattr(args, "forecast_steps", 0))
        if steps <= 0:
            log.error("forecast_steps must be > 0 for prediction mode.")
            raise ValueError("[VALIDATION] --forecast_steps must be > 0 for prediction mode.")

        log.info(f"Starting prediction with {steps} forecast steps")
        prepared = state.get("prepared_predictions")
        if (
            not allow_negative_predictions
            and isinstance(prepared, list)
            and len(prepared) >= steps
        ):
            predictions_with_std = [dict(point) for point in prepared[:steps]]
            if prediction_domain == DEMAND_PREDICTION_DOMAIN:
                for point in predictions_with_std:
                    point["prediction"] = max(0.0, float(point["prediction"]))
                    if "interval_lower" in point and "interval_upper" in point:
                        point["interval_lower"] = min(
                            point["prediction"],
                            max(0.0, float(point["interval_lower"])),
                        )
                        point["interval_upper"] = max(
                            point["prediction"],
                            max(0.0, float(point["interval_upper"])),
                        )
                        point["interval_kind"] = NONNEGATIVE_INTERVAL_KIND
        else:
            preds, conf_int = predict(
                load_path,
                steps,
                state=state,
                allow_negative_predictions=allow_negative_predictions,
            )
            predictions_with_std = _format_predictions(
                preds,
                conf_int,
                prediction_domain,
            )

        out: Dict[str, Any] = {
            "mode": "arima_loaded_predict",
            "order": state.get("order", {}),
            "forecast_steps": steps,
            "predictions": predictions_with_std,
        }
        attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        log.info("Prediction finished successfully")
        out["logs"] = get_log_contents()
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

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
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        log.error(f"Failed to parse --data: {e}", exc_info=True)
        raise ValueError(f"[DATA] Failed to parse --data: {e}")

    ts, te = int(args.train_start), int(args.train_end)
    es, ee = int(args.eval_start), int(args.eval_end)
    d = int(args.d)

    search_config = derive_arima_search_limit(te - ts + 1, d)
    max_p = int(search_config["max_p"])
    max_q = int(search_config["max_q"])
    log.debug(f"Training params: d={d}, max_p={max_p}, max_q={max_q}")

    if not (0 <= ts <= te < len(y)):
        log.error(f"Invalid training range: [{ts}, {te}], data size: {len(y)}")
        raise ValueError(f"[VALIDATION] Invalid train range: [{ts},{te}], must be within [0, {len(y)-1}]")
    if not (0 <= es <= ee < len(y)):
        log.error(f"Invalid evaluation range: [{es}, {ee}], data size: {len(y)}")
        raise ValueError(f"[VALIDATION] Invalid eval range: [{es},{ee}], must be within [0, {len(y)-1}]")
    if es <= te:
        log.error(f"Evaluation start {es} must be after training end {te}")
        raise ValueError(f"[VALIDATION] Evaluation start ({es}) must be after training end ({te}).")

    # Additional helpful validations
    train_size = te - ts + 1
    eval_size = ee - es + 1
    log.info(f"Validation passed: train_size={train_size}, eval_size={eval_size}, d={d}, max_p={max_p}, max_q={max_q}")

    log.info("Starting model training and evaluation...")
    prediction_domain = resolve_prediction_domain(
        allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
    )
    metrics, best_model, best_order, search_summary, y_true, y_pred, conf_int = train_and_evaluate(
        y,
        timestamps,
        range(ts, te + 1),
        range(es, ee + 1),
        d,
        max_p,
        max_q,
        prediction_domain,
    )
    log.info(f"Training finished. Best order: {best_order}, Metrics: {metrics}")
    log.debug(f"Search summary: {search_summary}")

    predictions_with_std = _format_predictions(
        y_pred,
        conf_int,
        prediction_domain,
    )

    out: Dict[str, Any] = {
        "mode": "arima_search_pq_fixed_d",
        "data_points": len(y),
        "d": d,
        "best_order": best_order,
        "train_range": {"start_index": ts, "end_index": te, "months": timestamps[ts : te + 1]},
        "evaluate_range": {"start_index": es, "end_index": ee, "months": timestamps[es : ee + 1]},
        "metrics": metrics,
        "eval_y_true": y_true,
        "predictions": predictions_with_std,
        "search_summary": search_summary,
        "diagnostics": search_summary["diagnostics"],
    }
    attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)

    if save_path:
        log.info(f"Saving model to {save_path}")
        log.debug(f"Saving model with order (p={best_order['p']}, d={best_order['d']}, q={best_order['q']})")
        state = {
            "model_type": MODEL_TYPE,
            "model": best_model,
            "order": best_order,
            "d": d,
            "model_spec": _model_spec(best_model),
            "search_summary": search_summary,
            "diagnostics": search_summary["diagnostics"],
            "prediction_domain": prediction_domain,
        }
        attach_method_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(state, save_path)
        log.info(f"Model saved successfully to {save_path}")
        out["saved_model"] = save_path

    out["logs"] = get_log_contents()
    log.info("Run finished successfully.")
    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
