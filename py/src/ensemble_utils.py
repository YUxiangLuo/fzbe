#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_utils import (
    ParsedData,
    build_numeric_subprocess_records,
    build_subprocess_records,
    infer_feature_types,
)
# Re-exported here because all ensemble modules share this boundary helper.
from forecast_utils import normalize_demand_forecast as normalize_demand_forecast
from serialization import dump_model, load_model
from subprocess_utils import (
    extract_predictions_from_result,
    run_base_model_fit_subprocess,
    run_base_model_subprocess,
)
from uncertainty import format_predictions as format_uncertainty_predictions
from dynamic_config import (
    derive_arima_search_limit,
    derive_lstm_config,
    plan_internal_holdout,
)

SUPPORTED_BASE_MODEL_TYPES = frozenset({"ma", "es", "arima", "lstm"})


def warn_ignored_boosting_controls(args: Any) -> Optional[str]:
    supplied = [
        flag
        for flag, attribute in (
            ("--learning-rate", "learning_rate"),
            ("--max-rounds", "max_rounds"),
        )
        if getattr(args, attribute, None) is not None
    ]
    if not supplied:
        return None

    message = (
        "[DEPRECATED] Ignored Boosting compatibility options: "
        f"{', '.join(supplied)}. Stage coefficients and round limits are derived from data."
    )
    print(message, file=sys.stderr, flush=True)
    return message


def validate_base_model_names(
    models_to_use: Any,
    context: str = "Ensemble",
) -> List[str]:
    if not isinstance(models_to_use, list):
        raise ValueError(f"[VALIDATION] {context} base models must be a list.")
    if any(not isinstance(model, str) or not model.strip() for model in models_to_use):
        raise ValueError(f"[VALIDATION] {context} contains an invalid base model name.")

    normalized = [model.strip() for model in models_to_use]
    if len(normalized) < 2:
        raise ValueError(f"[VALIDATION] {context} requires at least 2 base models.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"[VALIDATION] {context} base models must not contain duplicates.")

    unknown_models = [
        model for model in normalized if model not in SUPPORTED_BASE_MODEL_TYPES
    ]
    if unknown_models:
        raise ValueError(
            f"[VALIDATION] {context} contains unsupported base model: {unknown_models[0]}"
        )
    return normalized


def parse_base_model_names(raw_value: Any, context: str = "Ensemble") -> List[str]:
    if not isinstance(raw_value, str):
        raise ValueError(f"[VALIDATION] {context} requires --models.")
    return validate_base_model_names(
        [model.strip() for model in raw_value.split(",") if model.strip()],
        context,
    )


def parse_feature_name_list(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    return []


def require_explicit_lstm_feature_names(
    raw_value: Any,
    context: str = "LSTM",
) -> List[str]:
    """Parse an explicit LSTM feature selection without inventing defaults.

    An empty JSON/list value is meaningful: it selects a target-only LSTM.  A
    missing or blank value is ambiguous and must never be interpreted as every
    dataset column.
    """
    if raw_value is None or (
        isinstance(raw_value, str) and not raw_value.strip()
    ):
        raise ValueError(
            f"[VALIDATION] {context} requires an explicit LSTM feature list; "
            "use [] for a target-only LSTM."
        )
    return parse_feature_name_list(raw_value)


def load_lstm_source_state(model_file: str, context: str) -> Dict[str, Any]:
    source_model_state = load_model(model_file)
    if not isinstance(source_model_state, dict) or source_model_state.get("model_type") != "lstm":
        raise ValueError(f"[VALIDATION] {context} is invalid.")
    return source_model_state


def build_lstm_params_from_source_state(
    parsed_data: ParsedData,
    source_model_state: Dict[str, Any],
) -> Dict[str, Any]:
    raw_target_feature = source_model_state.get("target_key")
    if not isinstance(raw_target_feature, str) or not raw_target_feature.strip():
        raise ValueError("Saved LSTM artifact is missing an explicit target field.")
    target_feature = raw_target_feature.strip()
    if target_feature not in parsed_data.df.columns:
        raise ValueError(
            f"Saved LSTM artifact target field is not present in the dataset: {target_feature}"
        )

    normalization = source_model_state.get("normalization")
    if normalization not in {"minmax", "zscore"}:
        raise ValueError(
            "Saved LSTM artifact normalization must be 'minmax' or 'zscore'."
        )

    raw_feature_keys = source_model_state.get("feature_keys")
    if not isinstance(raw_feature_keys, list) or any(
        not isinstance(feature_name, str) or not feature_name.strip()
        for feature_name in raw_feature_keys
    ):
        raise ValueError("Saved LSTM artifact is missing an explicit feature list.")
    feature_keys = [
        feature_name.strip()
        for feature_name in raw_feature_keys
    ]
    if len(set(feature_keys)) != len(feature_keys):
        raise ValueError("Saved LSTM artifact feature list contains duplicates.")
    if target_feature in feature_keys:
        raise ValueError(
            "Saved LSTM artifact feature list must not repeat the target field."
        )
    missing_features = [
        feature_name
        for feature_name in feature_keys
        if feature_name not in parsed_data.df.columns
    ]
    if missing_features:
        raise ValueError(
            "Saved LSTM artifact feature fields are not present in the dataset: "
            + ", ".join(missing_features)
        )

    raw_feature_types = source_model_state.get("feature_types")
    if isinstance(raw_feature_types, dict) and raw_feature_types:
        feature_types = {
            str(key): value
            for key, value in raw_feature_types.items()
            if key == target_feature or key in feature_keys
        }
        invalid_types = [
            f"{feature_name}={feature_type!r}"
            for feature_name, feature_type in feature_types.items()
            if feature_type not in {"numeric", "categorical"}
        ]
        if invalid_types:
            raise ValueError(
                "Saved LSTM artifact has an invalid feature type: "
                + ", ".join(invalid_types)
            )
        missing_types = [
            feature_name
            for feature_name in [target_feature, *feature_keys]
            if feature_name not in feature_types
        ]
        if missing_types:
            raise ValueError(
                "Saved LSTM artifact is missing feature types for: "
                + ", ".join(missing_types)
            )
        if feature_types[target_feature] != "numeric":
            raise ValueError("Saved LSTM artifact target feature type must be numeric.")
    else:
        feature_types = infer_feature_types(
            parsed_data=parsed_data,
            target_key=target_feature,
            feature_keys=feature_keys,
        )
    feature_types[target_feature] = "numeric"

    params = {
        "feature_types": feature_types,
        "normalization": normalization,
        "lstm_target_feature": target_feature,
        "lstm_features": json.dumps(feature_keys, ensure_ascii=False),
    }
    return params


def get_lstm_source_forecast_strategy(source_model_state: Dict[str, Any]) -> str:
    from models import lstm as lstm_model

    if source_model_state.get("forecast_strategy"):
        return str(source_model_state["forecast_strategy"])
    if int(source_model_state.get("forecast_horizon", 0) or 0) > 0:
        return lstm_model.DIRECT_FORECAST_STRATEGY
    return lstm_model.LEGACY_FORECAST_STRATEGY


def resolve_lstm_reuse_production_horizon(
    source_model_state: Dict[str, Any],
    requested_steps: int = 0,
) -> int:
    from models import lstm as lstm_model

    forecast_strategy = get_lstm_source_forecast_strategy(source_model_state)
    if forecast_strategy != lstm_model.DIRECT_FORECAST_STRATEGY:
        return 0

    saved_eval_horizon = int(source_model_state.get("forecast_horizon", 0) or 0)
    current_required_horizon = lstm_model._resolve_production_forecast_horizon(
        state=source_model_state,
        eval_forecast_horizon=saved_eval_horizon,
    )
    return max(current_required_horizon, int(requested_steps or 0))


def prepare_lstm_source_model_for_requested_steps(
    source_model_file: str,
    source_model_state: Dict[str, Any],
    requested_steps: int,
) -> Tuple[str, Optional[str]]:
    current_required_horizon = resolve_lstm_reuse_production_horizon(source_model_state)
    requested_required_horizon = resolve_lstm_reuse_production_horizon(
        source_model_state,
        requested_steps=requested_steps,
    )
    if requested_required_horizon <= current_required_horizon:
        return source_model_file, None

    adjusted_state = dict(source_model_state)
    adjusted_state["production_forecast_horizon"] = requested_required_horizon
    fd, adjusted_source_model_file = tempfile.mkstemp(
        prefix=f"ensemble_lstm_source_{os.getpid()}_",
        suffix=".pkl",
    )
    os.close(fd)
    try:
        dump_model(adjusted_state, adjusted_source_model_file)
    except Exception:
        try:
            os.remove(adjusted_source_model_file)
        except OSError:
            pass
        raise
    return adjusted_source_model_file, adjusted_source_model_file


def format_predictions(
    preds: List[float],
    std_dev: Any = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    profile = (
        [float(value) for value in std_dev]
        if isinstance(std_dev, (list, tuple, np.ndarray))
        else [float(std_dev)] * len(preds)
    )
    return format_uncertainty_predictions(
        preds,
        profile,
        metadata or {"source": "empirical", "fallback": False},
    )


def estimate_lstm_encoded_feature_count(
    parsed_data: ParsedData,
    params: Dict[str, Any],
    train_start: int,
    prefix_size: int,
) -> int:
    target = str(params.get("lstm_target_feature") or parsed_data.target_key)
    raw_features = require_explicit_lstm_feature_names(
        params.get("lstm_features") if "lstm_features" in params else None,
        "LSTM holdout planning",
    )
    feature_types = params.get("feature_types") or {}
    prefix = parsed_data.df.iloc[train_start : train_start + prefix_size]
    count = 1
    for feature in raw_features:
        if feature == target or feature not in prefix.columns:
            continue
        if feature_types.get(feature) == "categorical":
            count += max(1, int(prefix[feature].nunique(dropna=False)))
        else:
            count += 1
    return count


def plan_ensemble_internal_holdout(
    parsed_data: ParsedData,
    train_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Dict[str, Any]],
    purpose: str,
) -> Dict[str, Any]:
    models = validate_base_model_names(models_to_use, purpose.title())
    for model_name in models:
        if model_name != "lstm":
            continue
        params = model_params.get(model_name, {})
        require_explicit_lstm_feature_names(
            params.get("lstm_features") if "lstm_features" in params else None,
            f"{purpose.title()} holdout planning",
        )

    return plan_internal_holdout(
        purpose,
        len(train_range),
        len(models),
        lambda prefix_size, validation_size: (
            is_ensemble_member_holdout_feasible(
                parsed_data,
                train_range.start,
                prefix_size,
                validation_size,
                models,
                model_params,
            )
        ),
    )


def is_ensemble_member_holdout_feasible(
    parsed_data: ParsedData,
    train_start: int,
    prefix_size: int,
    holdout_size: int,
    models_to_use: List[str],
    model_params: Dict[str, Dict[str, Any]],
) -> bool:
    """Check every selected member against one exact chronological OOF fold."""
    try:
        for model_name in models_to_use:
            params = model_params.get(model_name, {})
            if model_name == "ma" and prefix_size < int(params.get("window", 0)):
                return False
            if model_name == "es" and prefix_size < 2:
                return False
            if model_name == "arima":
                derive_arima_search_limit(prefix_size, int(params.get("d", 0)))
            if model_name == "lstm":
                feature_count = estimate_lstm_encoded_feature_count(
                    parsed_data,
                    params,
                    train_start,
                    prefix_size,
                )
                derive_lstm_config(prefix_size, holdout_size, feature_count)
        return True
    except (TypeError, ValueError):
        return False


def plan_ensemble_selection_calibration_split(
    parsed_data: ParsedData,
    train_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Dict[str, Any]],
    purpose: str,
) -> Dict[str, Any]:
    """Plan disjoint chronological selection and post-selection calibration sets.

    The original internal holdout remains the selection set used to estimate
    ensemble weights, stages, or meta-model parameters.  When the available
    history can support it, a later tail is withheld from ensemble selection and
    used to generate post-selection residuals before any full-history refit.  A
    final refit may later absorb that tail, so these empirical ranges do not claim
    nominal coverage.  If no feasible three-way split exists, the selection plan
    is retained and calibration is explicitly marked unavailable; selection
    residuals must never be relabelled as calibration residuals.
    """
    train_size = len(train_range)
    desired_calibration_size = max(2, (train_size + 4) // 5)
    maximum_calibration_size = max(0, train_size - 4)

    for calibration_size in range(
        min(desired_calibration_size, maximum_calibration_size),
        1,
        -1,
    ):
        pre_calibration_range = range(
            train_range.start,
            train_range.stop - calibration_size,
        )
        try:
            selection_plan = plan_ensemble_internal_holdout(
                parsed_data,
                pre_calibration_range,
                models_to_use,
                model_params,
                purpose,
            )
        except ValueError:
            continue

        selection_size = int(selection_plan["validation_size"])
        prefix_size = int(selection_plan["prefix_size"])
        stacking_oof: Dict[str, Any] = {}
        if purpose == "stacking":
            prefix_range = range(
                train_range.start,
                train_range.start + prefix_size,
            )
            selection_range = range(
                prefix_range.stop,
                prefix_range.stop + selection_size,
            )
            try:
                folds = plan_feasible_expanding_window_oof_folds(
                    parsed_data,
                    prefix_range,
                    selection_range,
                    models_to_use,
                    model_params,
                    min_folds=3,
                )
            except ValueError:
                continue
            stacking_oof = {
                "oof_strategy": "expanding_window_rolling_origin",
                "oof_fold_count": len(folds),
                "oof_folds": folds,
            }
        return {
            **selection_plan,
            **stacking_oof,
            "train_size": train_size,
            "prefix_size": prefix_size,
            "selection_size": selection_size,
            "validation_size": selection_size,
            "calibration_size": calibration_size,
            "desired_calibration_size": desired_calibration_size,
            "selection_start_offset": prefix_size,
            "calibration_start_offset": prefix_size + selection_size,
            "calibration_available": True,
            "calibration_strategy": "post_selection_holdout",
            "teaching_small_sample": bool(
                selection_plan["teaching_small_sample"]
                or calibration_size < desired_calibration_size
                or train_size < 20
            ),
        }

    selection_plan = plan_ensemble_internal_holdout(
        parsed_data,
        train_range,
        models_to_use,
        model_params,
        purpose,
    )
    selection_size = int(selection_plan["validation_size"])
    prefix_size = int(selection_plan["prefix_size"])
    stacking_oof = {}
    if purpose == "stacking":
        prefix_range = range(
            train_range.start,
            train_range.start + prefix_size,
        )
        selection_range = range(
            prefix_range.stop,
            prefix_range.stop + selection_size,
        )
        folds = plan_feasible_expanding_window_oof_folds(
            parsed_data,
            prefix_range,
            selection_range,
            models_to_use,
            model_params,
        )
        stacking_oof = {
            "oof_strategy": "expanding_window_rolling_origin",
            "oof_fold_count": len(folds),
            "oof_folds": folds,
        }
    return {
        **selection_plan,
        **stacking_oof,
        "selection_size": selection_size,
        "validation_size": selection_size,
        "calibration_size": 0,
        "desired_calibration_size": desired_calibration_size,
        "selection_start_offset": prefix_size,
        "calibration_start_offset": train_size,
        "calibration_available": False,
        "calibration_strategy": "fallback_no_post_selection_holdout",
        "teaching_small_sample": True,
    }


def plan_expanding_window_oof_folds(
    prefix_range: range,
    selection_range: range,
    max_folds: int = 3,
) -> List[Dict[str, int]]:
    """Partition a selection range into expanding-window rolling-origin folds."""
    if not prefix_range or not selection_range:
        raise ValueError("[VALIDATION] Rolling-origin OOF ranges must not be empty.")
    if prefix_range.stop != selection_range.start:
        raise ValueError(
            "[VALIDATION] Rolling-origin OOF selection must immediately follow its prefix."
        )
    fold_limit = int(max_folds)
    if fold_limit < 2:
        raise ValueError("[VALIDATION] Rolling-origin OOF requires at least two folds.")

    fold_count = min(fold_limit, len(selection_range))
    if fold_count < 2:
        raise ValueError(
            "[VALIDATION] Rolling-origin OOF requires at least two selection rows."
        )
    base_size, remainder = divmod(len(selection_range), fold_count)
    fold_sizes = [
        base_size + (1 if fold_index < remainder else 0)
        for fold_index in range(fold_count)
    ]

    folds: List[Dict[str, int]] = []
    holdout_start = selection_range.start
    for fold_index, fold_size in enumerate(fold_sizes):
        holdout_stop = holdout_start + fold_size
        folds.append(
            {
                "fold": fold_index + 1,
                "train_start": prefix_range.start,
                "train_end_exclusive": holdout_start,
                "holdout_start": holdout_start,
                "holdout_end_exclusive": holdout_stop,
            }
        )
        holdout_start = holdout_stop
    return folds


def plan_feasible_expanding_window_oof_folds(
    parsed_data: ParsedData,
    prefix_range: range,
    selection_range: range,
    models_to_use: List[str],
    model_params: Dict[str, Dict[str, Any]],
    max_folds: int = 3,
    min_folds: int = 2,
) -> List[Dict[str, int]]:
    """Use the most detailed OOF partition that every selected member can fit.

    A three-fold partition can create one-row holdouts when the selection set is
    small. For a direct multi-horizon LSTM, a one-row horizon may have a tighter
    parameter budget than a two-row horizon. Falling back to two chronological
    folds preserves out-of-fold coverage without rejecting an otherwise valid
    Stacking training range.
    """
    candidate_count = min(int(max_folds), len(selection_range))
    minimum_count = int(min_folds)
    if candidate_count < 2 or minimum_count < 2:
        # Reuse the canonical validation message for malformed ranges/options.
        return plan_expanding_window_oof_folds(
            prefix_range,
            selection_range,
            max_folds=max_folds,
        )

    for fold_count in range(candidate_count, minimum_count - 1, -1):
        folds = plan_expanding_window_oof_folds(
            prefix_range,
            selection_range,
            max_folds=fold_count,
        )
        if all(
            is_ensemble_member_holdout_feasible(
                parsed_data,
                int(fold["train_start"]),
                int(fold["train_end_exclusive"])
                - int(fold["train_start"]),
                int(fold["holdout_end_exclusive"])
                - int(fold["holdout_start"]),
                models_to_use,
                model_params,
            )
            for fold in folds
        ):
            return folds

    raise ValueError(
        "[VALIDATION] No rolling-origin OOF split can fit every selected "
        "Stacking member."
    )


def can_refit_source_artifact_for_internal_holdout(model_name: str) -> bool:
    """Whether fit-only can safely refit directly from the source artifact.

    MA and ES artifacts contain fixed user parameters, while LSTM fit-only mode
    carries over only user-facing feature/normalization choices and rebuilds all
    hidden configuration on the shorter prefix. An ARIMA artifact additionally
    contains p and q selected using the complete training range; reusing those
    orders would let the internal holdout influence its own prediction.  ARIMA
    therefore reruns the same fixed-d AICc search on the prefix.
    """
    return model_name in {"ma", "es", "lstm"}


def estimate_residual_std(
    calibration_residuals: Any,
    fallback_calibration_residuals: Any = None,
    reference_series: Any = None,
) -> float:
    """Estimate a finite sample scale from training-side calibration residuals.

    A one-point calibration segment cannot identify a sample standard deviation,
    so callers may provide a second chronological training-side residual series.
    The reference-series branch is the final small-data fallback. Independent
    evaluation labels must never be supplied to this helper.
    """
    for label, values in (
        ("primary calibration", calibration_residuals),
        ("fallback calibration", fallback_calibration_residuals),
        ("reference", reference_series),
    ):
        if values is None:
            continue
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size and not np.isfinite(array).all():
            raise ValueError(f"[DATA] {label} values contain NaN or infinity.")
        if array.size > 1:
            return float(np.std(array, ddof=1))

    reference = np.asarray(reference_series if reference_series is not None else [], dtype=float).reshape(-1)
    baseline = abs(float(np.mean(reference))) if reference.size else 0.0
    return float(max(baseline * 0.05, 1.0))


def parse_base_model_files_arg(raw_value: Any) -> Dict[str, str]:
    if raw_value is None or raw_value == "":
        return {}

    try:
        payload = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("[VALIDATION] --base-model-files must be valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("[VALIDATION] --base-model-files must be a JSON object.")

    parsed_files: Dict[str, str] = {}
    for model_name, model_file in payload.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "[VALIDATION] --base-model-files keys must be non-empty model names."
            )
        if not isinstance(model_file, str) or not model_file.strip():
            raise ValueError(
                f"[VALIDATION] --base-model-files entry for {model_name} must be a non-empty path."
            )
        parsed_files[model_name] = model_file
    return parsed_files


def _artifact_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"[VALIDATION] {label} must be an integer in the saved artifact.")
    return int(value)


def _artifact_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"[VALIDATION] {label} must be numeric in the saved artifact.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"[VALIDATION] {label} must be finite in the saved artifact.")
    return result


def resolve_base_model_artifact_configs(
    parsed_data: ParsedData,
    models_to_use: List[str],
    model_params: Dict[str, Dict[str, Any]],
    base_model_files: Dict[str, str],
    context: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Validate explicit source artifacts and make their fixed configuration authoritative.

    A selected model without a mapped artifact is trained from ``model_params``.
    Once an artifact is explicitly mapped, silently falling back to a fresh model
    would make the ensemble differ from the base-model run shown to the learner,
    so invalid paths and states are rejected here.

    ARIMA is deliberately special: only the user-fixed differencing order ``d``
    comes from the artifact. Search bounds and p/q are derived again on each
    internal prefix, avoiding holdout leakage from the full-data fit.
    """
    selected_models = set(models_to_use)
    canonical_params = {
        model_name: dict(params)
        for model_name, params in model_params.items()
    }
    validated_files: Dict[str, str] = {}

    for model_name, model_file in base_model_files.items():
        if model_name not in SUPPORTED_BASE_MODEL_TYPES:
            raise ValueError(
                f"[VALIDATION] {context} source artifact uses unsupported model: {model_name}."
            )
        if model_name not in selected_models:
            raise ValueError(
                f"[VALIDATION] {context} source artifact was supplied for unselected model: "
                f"{model_name}."
            )
        if not isinstance(model_file, str) or not model_file.strip() or not os.path.isfile(model_file):
            raise ValueError(
                f"[VALIDATION] {context} source artifact for {model_name} does not exist or is not a file."
            )

        try:
            source_state = load_model(model_file)
        except Exception as exc:
            raise ValueError(
                f"[VALIDATION] {context} source artifact for {model_name} cannot be read."
            ) from exc

        if not isinstance(source_state, dict) or source_state.get("model_type") != model_name:
            raise ValueError(
                f"[VALIDATION] {context} source artifact for {model_name} has a mismatched model type."
            )

        fallback_params = canonical_params.get(model_name)
        if fallback_params is None:
            raise ValueError(
                f"[VALIDATION] {context} is missing fallback parameters for {model_name}."
            )

        if model_name == "ma":
            if "window" not in source_state:
                raise ValueError(
                    f"[VALIDATION] {context} MA source artifact is missing window."
                )
            source_window = _artifact_integer(
                source_state["window"], f"{context} MA window"
            )
            if source_window < 2:
                raise ValueError(
                    f"[VALIDATION] {context} MA source artifact window must be >= 2."
                )
            fallback_params["window"] = source_window
        elif model_name == "es":
            if "alpha" not in source_state:
                raise ValueError(
                    f"[VALIDATION] {context} ES source artifact is missing alpha."
                )
            source_alpha = _artifact_float(
                source_state["alpha"], f"{context} ES alpha"
            )
            if not (0 < source_alpha <= 1):
                raise ValueError(
                    f"[VALIDATION] {context} ES source artifact alpha must be in (0, 1]."
                )
            fallback_params["alpha"] = source_alpha
        elif model_name == "arima":
            artifact_d = source_state.get("d")
            if artifact_d is None:
                order = source_state.get("order")
                artifact_d = order.get("d") if isinstance(order, dict) else None
            if artifact_d is None:
                raise ValueError(
                    f"[VALIDATION] {context} ARIMA source artifact is missing d."
                )
            source_d = _artifact_integer(
                artifact_d, f"{context} ARIMA d"
            )
            if source_d < 0:
                raise ValueError(
                    f"[VALIDATION] {context} ARIMA source artifact d must be >= 0."
                )
            canonical_params[model_name] = {
                "d": source_d,
            }
        else:
            try:
                canonical_params[model_name] = build_lstm_params_from_source_state(
                    parsed_data,
                    source_state,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"[VALIDATION] {context} LSTM source artifact has invalid "
                    f"training configuration: {exc}"
                ) from exc

        validated_files[model_name] = model_file

    return canonical_params, validated_files


def build_ensemble_subprocess_data(
    parsed_data: ParsedData,
    models_to_use: List[str],
    model_params: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    numeric_data_dicts, _ = build_numeric_subprocess_records(parsed_data)
    lstm_data_dicts = numeric_data_dicts
    if "lstm" in models_to_use:
        lstm_params = model_params.get("lstm", {})
        lstm_feature_keys = require_explicit_lstm_feature_names(
            lstm_params.get("lstm_features") if "lstm_features" in lstm_params else None,
            "Ensemble LSTM member",
        )
        lstm_data_dicts, _ = build_subprocess_records(parsed_data, feature_keys=lstm_feature_keys)

    return numeric_data_dicts, lstm_data_dicts


def get_member_model_path(save_path: str, model_name: str, round_idx: int | None = None) -> str:
    base_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    suffix = f"_round{round_idx}" if round_idx is not None else ""
    return os.path.join(base_dir, f"{base_name}_base_{model_name}{suffix}.pkl")


def remove_member_model_artifact(model_name: str, model_path: str) -> None:
    if os.path.exists(model_path):
        os.remove(model_path)

    if model_name == "lstm":
        keras_path = os.path.splitext(model_path)[0] + "_keras_model.h5"
        if os.path.exists(keras_path):
            os.remove(keras_path)


def persist_member_model_artifact(
    model_name: str,
    source_model_path: str,
    destination_model_path: str,
    *,
    move_source: bool,
) -> None:
    if os.path.normcase(os.path.realpath(source_model_path)) == os.path.normcase(os.path.realpath(destination_model_path)):
        return
    os.makedirs(os.path.dirname(destination_model_path), exist_ok=True)
    transfer = shutil.move if move_source else shutil.copy2
    transfer(source_model_path, destination_model_path)

    if model_name != "lstm":
        return

    lstm_state = load_model(destination_model_path)
    source_keras_path = lstm_state.get("keras_model_path")
    if not source_keras_path or not os.path.exists(source_keras_path):
        raise FileNotFoundError(f"LSTM Keras model file not found at {source_keras_path}")

    destination_keras_path = os.path.splitext(destination_model_path)[0] + "_keras_model.h5"
    transfer(source_keras_path, destination_keras_path)
    lstm_state["keras_model_path"] = destination_keras_path
    dump_model(lstm_state, destination_model_path)


def fit_and_predict_from_source_model(
    model_name: str,
    source_model_file: str,
    data: List[Dict[str, Any]],
    train_start: int,
    train_end: int,
    forecast_steps: int,
    timeout: int,
) -> List[float]:
    temp_model_file = os.path.join(
        tempfile.gettempdir(),
        f"ensemble_refit_{model_name}_{os.getpid()}.pkl",
    )

    try:
        fit_result = run_base_model_fit_subprocess(
            model_type=model_name,
            data=data,
            train_start=train_start,
            train_end=train_end,
            source_model_file=source_model_file,
            save_model_file=temp_model_file,
            fit_forecast_steps=forecast_steps if model_name == "lstm" else None,
            predict_steps=forecast_steps,
            timeout=timeout,
        )
        return extract_predictions_from_result(fit_result)
    finally:
        remove_member_model_artifact(model_name, temp_model_file)


def predict_member_on_chronological_holdout(
    *,
    model_name: str,
    params: Dict[str, Any],
    source_model_file: Optional[str],
    numeric_data: List[Dict[str, Any]],
    lstm_data: List[Dict[str, Any]],
    train_range: range,
    holdout_range: range,
    timeout: int,
) -> List[float]:
    """Fit one member on a prefix and predict the immediately following holdout."""
    if not train_range or not holdout_range:
        raise ValueError("[VALIDATION] Ensemble member holdout ranges must not be empty.")
    if train_range.stop != holdout_range.start:
        raise ValueError(
            "[VALIDATION] Ensemble member holdout must immediately follow its training prefix."
        )

    data = lstm_data if model_name == "lstm" else numeric_data
    if (
        source_model_file
        and can_refit_source_artifact_for_internal_holdout(model_name)
    ):
        adjusted_source_model_file = source_model_file
        temporary_source_model_file = None
        try:
            if model_name == "lstm":
                source_model_state = load_lstm_source_state(
                    source_model_file,
                    "Saved ensemble LSTM base model",
                )
                (
                    adjusted_source_model_file,
                    temporary_source_model_file,
                ) = prepare_lstm_source_model_for_requested_steps(
                    source_model_file,
                    source_model_state,
                    requested_steps=len(holdout_range),
                )
            predictions = fit_and_predict_from_source_model(
                model_name=model_name,
                source_model_file=adjusted_source_model_file,
                data=data,
                train_start=train_range.start,
                train_end=train_range.stop - 1,
                forecast_steps=len(holdout_range),
                timeout=timeout,
            )
        finally:
            if (
                temporary_source_model_file
                and os.path.exists(temporary_source_model_file)
            ):
                os.remove(temporary_source_model_file)
    else:
        result = run_base_model_subprocess(
            model_type=model_name,
            data=data,
            train_start=train_range.start,
            train_end=train_range.stop - 1,
            eval_start=holdout_range.start,
            eval_end=holdout_range.stop - 1,
            model_params=params.copy(),
            timeout=timeout,
        )
        predictions = extract_predictions_from_result(result)

    return normalize_demand_forecast(predictions).tolist()
