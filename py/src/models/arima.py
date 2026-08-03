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
METHOD_NAME = "ARIMA 逐步搜索定阶"
FORECAST_STRATEGY = "arima_multi_step_forecast"
IMPLEMENTATION_NOTES = [
    "教科书 ARIMA 通常通过平稳性检验、ACF/PACF 和信息准则定阶。",
    "本系统由用户根据 ADF 结果指定差分阶数 d；模型固定为非季节 ARIMA，并执行一次以 AICc 为目标的 stepwise 搜索来选择 p、q 和是否包含截距。",
    "p/q 搜索上限按差分后的有效样本量自动收紧；少于 6 个有效样本时拒绝拟合，不用不可靠结果冒充完整 ARIMA。",
    "候选模型必须收敛；AICc 排名靠前的规格会先按完整规格将最大迭代数加倍重拟合一次，仍不收敛时依次尝试下一名有效候选，并记录候选回退和 Ljung-Box 白噪声诊断。",
    "预测未来销量时使用拟合后的 ARIMA 模型直接生成多步 forecast；原始点预测可能为负，本系统将用于残差、指标和输出的销量点预测按 max(0, y_hat) 截断。该约束是业务后处理，不属于教科书 ARIMA。",
    "默认 95% 预测区间仍由未截断的 ARIMA 预测分布计算；std_dev 按该分布的精确 95% 正态临界值由区间宽度反推，99% 上侧误差压力诊断也由同一未截断分布计算。该诊断不参与客户安全库存公式，区间本身也不是截断正态区间。",
]


# ---------- utilities ----------
def _format_predictions(
    preds: List[float],
    conf_intervals: List[List[float]],
    calibration_mean_error: float | None = None,
    calibration_count: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Format predictions with 95% prediction intervals for JSON output.
    Converts interval width to its Gaussian std_dev while preserving the
    model's original, untruncated interval bounds.
    """
    has_mean = calibration_mean_error is not None
    has_count = calibration_count is not None
    if has_mean != has_count:
        raise ValueError("[MODEL] ARIMA bias calibration must provide both mean error and sample count.")
    if has_mean:
        try:
            numeric_mean_error = float(calibration_mean_error)
            raw_numeric_count = float(calibration_count)
            numeric_count = int(raw_numeric_count)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("[MODEL] ARIMA bias calibration is invalid.") from exc
        if (
            not np.isfinite(numeric_mean_error)
            or not np.isfinite(raw_numeric_count)
            or isinstance(calibration_count, bool)
            or numeric_count <= 0
            or raw_numeric_count != numeric_count
        ):
            raise ValueError("[MODEL] ARIMA bias calibration is invalid.")
    else:
        numeric_mean_error = None
        numeric_count = None

    points: List[Dict[str, Any]] = []
    for prediction, interval in zip(preds, conf_intervals):
        numeric_prediction = float(prediction)
        lower = float(interval[0])
        upper = float(interval[1])
        std_dev = max(0.0, (upper - lower) / (2.0 * NORMAL_95))
        raw_distribution_mean = (lower + upper) / 2.0
        points.append({
            "prediction": numeric_prediction,
            "std_dev": std_dev,
            "uncertainty_source": "model",
            "calibration_source": "arima_forecast_distribution",
            "calibration_mean_error": numeric_mean_error,
            "calibration_count": numeric_count,
            "bias_diagnostic_source": "outer_selection_assessment" if numeric_count is not None else None,
            "interval_lower": lower,
            "interval_upper": upper,
            "interval_level": 0.95,
            "interval_kind": "model_prediction_interval",
            "coverage_claim": "model_nominal",
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
    *,
    return_valid_fits: bool = False,
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
            maxiter=50,
            trace=False,
            return_valid_fits=return_valid_fits,
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
    maxiter: int = 100,
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
    retried = _fit_arima_spec(y_train, _model_spec(model), context, maxiter=100)
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


def _select_converged_auto_arima(
    y_train: np.ndarray,
    d: int,
    max_p: int,
    max_q: int,
    context: str,
    criterion: str = "aicc",
) -> Tuple[Any, bool, Dict[str, Any]]:
    """Select the best converged candidate instead of failing on rank one.

    ``pmdarima`` considers a fit valid when its information criterion is
    finite, even when the underlying optimizer reports non-convergence.  Ask
    for the full AICc-sorted candidate list, retry each candidate once using
    the existing convergence policy, and fall through to the next candidate
    when a higher-ranked specification still cannot converge.
    """
    raw_candidates = _run_auto_arima_search(
        y_train=y_train,
        d=d,
        max_p=max_p,
        max_q=max_q,
        criterion=criterion,
        return_valid_fits=True,
    )
    candidates = (
        list(raw_candidates)
        if isinstance(raw_candidates, (list, tuple))
        else [raw_candidates]
    )
    if not candidates:
        raise ValueError(
            f"[MODEL] {context} did not produce any finite {criterion.upper()} candidates."
        )

    skipped: List[Dict[str, Any]] = []
    convergence_retried = False
    for rank, candidate in enumerate(candidates, start=1):
        candidate_spec = _model_spec(candidate)
        was_converged = _is_converged(candidate)
        try:
            selected, retried = _ensure_converged(
                candidate,
                y_train,
                f"{context} candidate rank {rank}",
            )
        except ValueError:
            convergence_retried = convergence_retried or not was_converged
            skipped.append(
                {
                    "rank": rank,
                    "order": candidate_spec["order"],
                    "with_intercept": candidate_spec["with_intercept"],
                    "reason": "did_not_converge_after_retry",
                }
            )
            log.warning(
                "%s skipped non-converged %s candidate rank %s with order=%s",
                context,
                criterion.upper(),
                rank,
                candidate_spec["order"],
            )
            continue

        convergence_retried = convergence_retried or retried
        summary = {
            "candidate_count": len(candidates),
            "selected_rank": rank,
            "skipped_candidates": skipped,
            "fallback_used": rank > 1,
        }
        if rank > 1:
            log.warning(
                "%s selected converged %s candidate rank %s after skipping %s "
                "higher-ranked non-converged candidate(s)",
                context,
                criterion.upper(),
                rank,
                len(skipped),
            )
        return selected, convergence_retried, summary

    raise ValueError(
        f"[MODEL] {context} did not converge for any of the "
        f"{len(candidates)} finite {criterion.upper()} candidates."
    )


def _residual_diagnostics(model: Any, effective_train_size: int) -> Dict[str, Any]:
    order = tuple(int(value) for value in model.order)
    lag = min(10, effective_train_size // 5)
    model_df = order[0] + order[2]
    if lag <= model_df or lag < 1:
        return {
            "ljung_box_lag": None,
            "ljung_box_pvalue": None,
            "residual_white_noise": None,
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
            "warning": "residual variance is degenerate, so Ljung-Box is undefined",
        }
    return {
        "ljung_box_lag": lag,
        "ljung_box_pvalue": pvalue,
        "residual_white_noise": pvalue >= 0.05,
        "warning": None if pvalue >= 0.05 else "residual autocorrelation remains at the 5% level",
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
    (
        best_model,
        convergence_retried,
        candidate_selection,
    ) = _select_converged_auto_arima(
        y_train=y_train,
        d=d,
        max_p=max_p,
        max_q=max_q,
        context="Selected ARIMA model",
        criterion="aicc",
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
        "selected_order": best_order,
        "model_spec": _model_spec(best_model),
        "aic": float(best_model.aic()),
        "aicc": float(best_model.aicc()),
        "bic": float(best_model.bic()),
        "convergence_retried": convergence_retried,
        "convergence_boundary_case": not _is_converged(best_model),
        "candidate_selection": candidate_selection,
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
            "calibration_mean_error": state.get("calibration_mean_error"),
            "calibration_count": state.get("calibration_count"),
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
                next_state.get("calibration_mean_error"),
                next_state.get("calibration_count"),
            )
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
        prepared = state.get("prepared_predictions")
        if (
            not bool(getattr(args, "allow_negative_predictions", False))
            and isinstance(prepared, list)
            and len(prepared) >= steps
        ):
            predictions_with_std = [dict(point) for point in prepared[:steps]]
        else:
            preds, conf_int = predict(
                load_path,
                steps,
                state=state,
                allow_negative_predictions=bool(getattr(args, "allow_negative_predictions", False)),
            )
            predictions_with_std = _format_predictions(
                preds,
                conf_int,
                state.get("calibration_mean_error"),
                state.get("calibration_count"),
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

    calibration_mean_error = float(np.mean(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))
    calibration_count = len(y_true)
    predictions_with_std = _format_predictions(
        y_pred,
        conf_int,
        calibration_mean_error,
        calibration_count,
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
            "calibration_mean_error": calibration_mean_error,
            "calibration_count": calibration_count,
        }
        attach_method_metadata(state, METHOD_NAME, FORECAST_STRATEGY, IMPLEMENTATION_NOTES)
        dump_model(state, save_path)
        log.info(f"Model saved successfully to {save_path}")
        out["saved_model"] = save_path

    out["logs"] = get_log_contents()
    log.info("Run finished successfully.")
    print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
