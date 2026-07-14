#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA model with automatic order selection for p and q.
User specifies d, and the model searches for optimal p and q using AIC and BIC.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pmdarima as pm

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

MODEL_TYPE = "arima"
METHOD_NAME = "ARIMA 自动定阶"
FORECAST_STRATEGY = "arima_multi_step_forecast"
IMPLEMENTATION_NOTES = [
    "教科书 ARIMA 通常通过平稳性检验、ACF/PACF 和信息准则定阶。",
    "本系统由用户根据 ADF 结果指定差分阶数 d；模型固定为非季节 ARIMA，并分别执行以 AIC、BIC 为目标的 stepwise 搜索来选择 p 和 q，而不是穷举所有组合。",
    "p/q 搜索上限随训练样本量自动收紧；训练样本少于 30 时优先比较两个搜索胜出模型的 BIC，否则优先比较 AIC。",
    "预测未来销量时使用拟合后的 ARIMA 模型直接生成多步 forecast；原始点预测可能为负，本系统将用于残差、指标和输出的销量点预测按 max(0, y_hat) 截断。该约束是业务后处理，不属于教科书 ARIMA。",
    "默认 95% 预测区间仍由未截断的 ARIMA 预测分布计算，本系统只取区间宽度除以 3.92 近似换算为正态标准差，因此区间本身并不是截断正态区间。",
]


# ---------- utilities ----------
def _format_predictions(preds: List[float], conf_intervals: List[List[float]]) -> List[Dict[str, float]]:
    """
    Format predictions with confidence intervals for JSON output.
    Converts 95% confidence interval to approximate std_dev.
    """
    return [
        {"prediction": float(p), "std_dev": float((ci[1] - ci[0]) / 3.92)}
        for p, ci in zip(preds, conf_intervals)
    ]


def _load_state(model_path: str) -> Dict[str, Any]:
    """Load and validate a saved ARIMA model state."""
    state = load_model(model_path)
    if not isinstance(state, dict) or state.get("model_type") != MODEL_TYPE:
        raise ValueError("[VALIDATION] Invalid ARIMA model file.")
    return ensure_state_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)


def _validate_arima_train_size(train_size: int, d: int, context: str) -> int:
    if train_size < 2:
        raise ValueError(f"[VALIDATION] {context} must contain at least 2 points for ARIMA fitting.")

    effective_train_size = train_size - d
    if d > 0 and effective_train_size < 2:
        raise ValueError(
            f"[VALIDATION] {context} must contain at least d + 2 points for ARIMA fitting "
            f"(got train_size={train_size}, d={d})."
        )
    return effective_train_size


def _run_auto_arima_search(
    y_train: np.ndarray,
    d: int,
    max_p: int,
    max_q: int,
    criterion: str,
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
            trace=False,
        )
    except Exception as exc:
        raise ValueError(
            f"[VALIDATION] ARIMA {criterion.upper()} search failed for the current training range: {exc}"
        ) from exc


def _fit_arima_order(y_train: np.ndarray, order: Tuple[int, int, int], context: str) -> Any:
    try:
        fitted_model = pm.ARIMA(order=order, seasonal=False, suppress_warnings=True)
        fitted_model.fit(y_train)
        return fitted_model
    except Exception as exc:
        raise ValueError(
            f"[VALIDATION] {context} failed to fit ARIMA order {order}: {exc}"
        ) from exc


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
        conf_int: List of [lower, upper] confidence interval bounds
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
    log.info(f"Generated {len(y_pred)} predictions with confidence intervals")

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
    Trains an ARIMA model by searching for the best p and q using both AIC and BIC,
    and evaluates the best resulting model.

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
        search_summary: Summary of AIC/BIC model search
        y_true: True evaluation values
        y_pred: Predicted evaluation values
        conf_int: Confidence intervals
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

    # --- Run auto_arima for both AIC and BIC ---
    log.info(f"Running auto_arima with AIC criterion: searching p in [0, {max_p}], q in [0, {max_q}], d={d}")
    aic_model = _run_auto_arima_search(
        y_train=y_train,
        d=d,
        max_p=max_p,
        max_q=max_q,
        criterion="aic",
    )
    log.debug(f"AIC-optimized model found: order={aic_model.order}")

    log.info(f"Running auto_arima with BIC criterion: searching p in [0, {max_p}], q in [0, {max_q}], d={d}")
    bic_model = _run_auto_arima_search(
        y_train=y_train,
        d=d,
        max_p=max_p,
        max_q=max_q,
        criterion="bic",
    )
    log.debug(f"BIC-optimized model found: order={bic_model.order}")

    # --- Compare and select the best model ---
    aic_order = aic_model.order
    bic_order = bic_model.order

    aic_aic = aic_model.aic()
    aic_bic = aic_model.bic()
    bic_aic = bic_model.aic()
    bic_bic = bic_model.bic()

    log.debug(f"AIC model: order={aic_order}, AIC={aic_aic:.4f}, BIC={aic_bic:.4f}")
    log.debug(f"BIC model: order={bic_order}, AIC={bic_aic:.4f}, BIC={bic_bic:.4f}")

    # Selection strategy: for small samples, prioritize BIC to prevent overfitting
    train_size = len(y_train)
    selection_criterion = "aic_optimized"
    best_model = aic_model

    if train_size < 30:
        # Small sample: prioritize BIC (stronger penalty against complexity)
        log.info(f"Using BIC-priority selection for small sample (n={train_size})")
        if bic_bic < aic_bic:
            best_model = bic_model
            selection_criterion = "bic_preferred_small_sample"
            log.info(f"Selected BIC-optimized model (better BIC, small sample): order={bic_order}")
        elif bic_bic == aic_bic and bic_aic <= aic_aic:
            best_model = bic_model
            selection_criterion = "bic_tie_better_or_equal_aic"
            log.info(f"Selected BIC-optimized model (tie BIC, AIC not worse): order={bic_order}")
        else:
            log.info(f"Selected AIC-optimized model (better BIC even in small sample): order={aic_order}")
    else:
        # Large sample: prioritize AIC (asymptotically efficient)
        log.info(f"Using AIC-priority selection for larger sample (n={train_size})")
        if bic_aic < aic_aic:
            best_model = bic_model
            selection_criterion = "bic_optimized_better_aic"
            log.info(f"Selected BIC-optimized model (better AIC): order={bic_order}")
        elif bic_aic == aic_aic and bic_bic < aic_bic:
            best_model = bic_model
            selection_criterion = "bic_tie_aic_better_bic"
            log.info(f"Selected BIC-optimized model (tie AIC, better BIC): order={bic_order}")
        else:
            log.info(f"Selected AIC-optimized model: order={aic_order}")

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

    # Search summary for transparency
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
        "selected_model": "aic_optimized" if best_model is aic_model else "bic_optimized",
        "selection_criterion": selection_criterion,
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

    # Check convergence
    converged = bool(best_model.arima_res_.mle_retvals.get("converged", False))
    if converged:
        log.info("Model converged successfully")
    else:
        log.warning("Model did not converge properly")

    residuals = np.array(y_true) - y_pred
    log.info(f"Calculated residuals: mean={np.mean(residuals):.4f}, std_dev={np.std(residuals):.4f}")

    metrics = {
        "rmse": rmse(y_eval, y_pred),
        "mae": mae(y_eval, y_pred),
        "mape": mape(y_eval, y_pred),
        "r2": r2(y_eval, y_pred),
        "aic": float(best_model.aic()),
        "bic": float(best_model.bic()),
        "converged": converged,
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

        order_info = state.get("order", {})
        order = (
            int(order_info.get("p", 0)),
            int(order_info.get("d", state.get("d", 0))),
            int(order_info.get("q", 0)),
        )
        train_size = te - ts + 1
        _validate_arima_train_size(
            train_size=train_size,
            d=order[1],
            context="ARIMA fit-only training range",
        )
        fitted_model = _fit_arima_order(
            y_train=y[ts : te + 1],
            order=order,
            context="ARIMA fit-only",
        )

        next_state = {
            "model_type": MODEL_TYPE,
            "model": fitted_model,
            "order": {"p": order[0], "d": order[1], "q": order[2]},
            "d": order[1],
            "prediction_domain": prediction_domain,
        }
        attach_method_metadata(next_state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(next_state, save_path)

        out: Dict[str, Any] = {
            "mode": "arima_fit_only",
            "order": next_state["order"],
            "train_end_index": te,
            "saved_model": save_path,
        }
        attach_method_metadata(out, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        out["logs"] = get_log_contents()
        print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        return

    if load_path:
        log.info(f"Entering prediction mode with model: {load_path}")
        state = _load_state(load_path)
        steps = int(getattr(args, "forecast_steps", 0))
        if steps <= 0:
            log.error("forecast_steps must be > 0 for prediction mode.")
            raise ValueError("[VALIDATION] --forecast_steps must be > 0 for prediction mode.")

        log.info(f"Starting prediction with {steps} forecast steps")
        preds, conf_int = predict(
            load_path,
            steps,
            state=state,
            allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
        )
        predictions_with_std = _format_predictions(preds, conf_int)

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

    max_p_arg = getattr(args, "max_p", None)
    max_q_arg = getattr(args, "max_q", None)

    if max_p_arg is not None and max_q_arg is not None:
        max_p = int(max_p_arg)
        max_q = int(max_q_arg)
        log.debug(f"Using user-specified max_p={max_p}, max_q={max_q}")
    else:
        # Auto-select max_p and max_q based on training data size
        # For small datasets, use conservative limits to prevent overfitting
        train_size = te - ts + 1
        if train_size <= 12:
            auto_max = 1  # ARIMA(1,d,1): max 2 parameters, avoid overfitting
            log.info(f"Small dataset detected (train_size={train_size}), using conservative max_p=max_q={auto_max}")
        elif train_size <= 20:
            auto_max = 2  # ARIMA(2,d,2): max 4 parameters
            log.info(f"Medium-small dataset detected (train_size={train_size}), using moderate max_p=max_q={auto_max}")
        else:
            auto_max = max(1, min(5, train_size // 6))  # Rule of thumb: params < train_size/6
            log.info(f"Using calculated max_p=max_q={auto_max} based on train_size={train_size}")

        max_p = int(max_p_arg) if max_p_arg is not None else auto_max
        max_q = int(max_q_arg) if max_q_arg is not None else auto_max
        if max_p_arg is None or max_q_arg is None:
            log.info(f"Auto-selected max_p={max_p}, max_q={max_q} for preventing overfitting")
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

    predictions_with_std = _format_predictions(y_pred, conf_int)

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
            "prediction_domain": prediction_domain,
        }
        attach_method_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(state, save_path)
        log.info(f"Model saved successfully to {save_path}")
        out["saved_model"] = save_path

    out["logs"] = get_log_contents()
    log.info("Run finished successfully.")
    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
