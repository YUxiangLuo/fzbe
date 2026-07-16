"""Independent checkpoint CLI drivers for guided and production parity tests."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

from .harness import PY_ROOT, run_command

# Guided checkpoint pickles intentionally retain the production subprocess's
# top-level module names (for example ``data_utils.ParsedData``). Keep targeted
# unittest invocations as robust as the profile runner, which adds this path.
SRC_ROOT = str(PY_ROOT / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

BASE_STEPS = {
    "ma": ["prepare_data", "build_windows", "forecast", "estimate_residuals", "evaluate_metrics", "save_artifact"],
    "es": ["prepare_data", "fit_model", "forecast", "estimate_residuals", "evaluate_metrics", "save_artifact"],
    "arima": ["prepare_data", "aic_search", "bic_search", "select_model", "forecast", "evaluate_metrics", "save_artifact"],
    "lstm": ["prepare_data", "preprocess", "build_model", "train_model", "forecast", "evaluate_metrics", "save_artifact"],
}
ENSEMBLE_STEPS = {
    "weighted_avg": ["prepare_data", "split_validation", "validation_predictions", "compute_weights", "eval_predictions", "blend_evaluate", "save_artifact"],
    "stacking": ["prepare_data", "split_levels", "level1_predictions", "fit_meta_model", "eval_predictions", "meta_evaluate", "save_artifact"],
}


def _checkpoint_step(entry: Path, model: str, step: str, workspace: Path, previous: Path | None, common: list[str], index: int, timeout: int) -> dict[str, Any]:
    output = workspace / f"state-{index:02d}-{step}.pkl"
    receipt = workspace / f"receipt-{index:02d}-{step}.json"
    argv = [sys.executable, entry, model, step]
    if previous:
        argv += ["--input-state-file", previous]
    argv += ["--output-state-file", output, "--receipt-file", receipt, "--attempt-id", f"deep-{model}-{index}", "--artifact-dir", workspace, *common]
    result = run_command(argv, timeout=timeout).payload
    if result["candidate_state_sha256"] != __import__("hashlib").sha256(output.read_bytes()).hexdigest():
        raise AssertionError("checkpoint receipt SHA does not match state")
    result["_state_path"] = output
    return result


def run_guided(model: str, data_file: Path, artifact: Path, workspace: Path, *, lstm_member: bool = False) -> dict[str, Any]:
    entry = PY_ROOT / ("guided_model.py" if model in BASE_STEPS else "guided_ensemble_model.py")
    common = ["--data-file", data_file, "--train-start", "0", "--train-end", "35", "--eval-start", "40", "--eval-end", "43", "--save-model", artifact]
    if model == "ma": common += ["--window", "3"]
    elif model == "es": common += ["--alpha", "0.4"]
    elif model == "arima": common += ["--d", "1"]
    elif model == "lstm": common += ["--lstm-target-feature", "y", "--lstm-features", "[]", "--feature-types", '{"y":"numeric"}', "--normalization", "minmax"]
    else:
        common += ["--models", "ma,lstm" if lstm_member else "ma,es", "--window", "3", "--alpha", "0.4"]
        if lstm_member:
            common += ["--lstm-target-feature", "y", "--lstm-features", "[]", "--feature-types", '{"y":"numeric"}', "--normalization", "minmax"]
    previous = None; final = None; index = 0
    steps = BASE_STEPS.get(model) or ENSEMBLE_STEPS.get(model)
    if steps:
        for step in steps:
            receipt = _checkpoint_step(entry, model, step, workspace, previous, common, index, 1200)
            previous = receipt["_state_path"]; final = receipt.get("final_result", final); index += 1
        return final

    # Boosting exposes data-derived rounds and a data-derived refit chain.
    for step in ("prepare_data", "split_residual_sets"):
        receipt = _checkpoint_step(entry, model, step, workspace, previous, common, index, 1800)
        previous = receipt["_state_path"]; index += 1
    with previous.open("rb") as handle: state = pickle.load(handle)
    for round_index in range(int(state["max_rounds"])):
        for action in ("evaluate_candidates", "select_winner"):
            step = f"round_{round_index + 1}_{action}"
            receipt = _checkpoint_step(entry, model, step, workspace, previous, common, index, 1800)
            previous = receipt["_state_path"]; index += 1
        output = receipt["output"]
        if output.get("action") == "early_stop" or output.get("stop_reason") == "round_budget": break
    receipt = _checkpoint_step(entry, model, "initialize_refit", workspace, previous, common, index, 1800)
    previous = receipt["_state_path"]; index += 1
    with previous.open("rb") as handle: state = pickle.load(handle)
    for stage_index, member in enumerate(state["selection_model_chain"], 1):
        receipt = _checkpoint_step(entry, model, f"refit_stage_{stage_index}_{member}", workspace, previous, common, index, 1800)
        previous = receipt["_state_path"]; index += 1
    for step in ("combine_evaluate", "save_artifact"):
        receipt = _checkpoint_step(entry, model, step, workspace, previous, common, index, 1800)
        previous = receipt["_state_path"]; final = receipt.get("final_result", final); index += 1
    return final


def run_production(model: str, data_file: Path, source_artifact: Path, artifact: Path, workspace: Path) -> dict[str, Any]:
    entry = PY_ROOT / "production_prepare_model.py"
    common = ["--data-file", data_file, "--source-model", source_artifact, "--save-model", artifact,
              "--train-start", "0", "--train-end", "47", "--forecast-steps", "4"]
    previous = None; index = 0
    initialized = _checkpoint_step(entry, model, "initialize", workspace, previous, common, index, 1800)
    previous = initialized["_state_path"]; index += 1
    for item in initialized["output"]["work_items"]:
        receipt = _checkpoint_step(entry, model, item["id"], workspace, previous, common, index, 1800)
        previous = receipt["_state_path"]; index += 1
        if receipt["output"].get("trim_remaining"): break
    finalized = _checkpoint_step(entry, model, "finalize", workspace, previous, common, index, 1800)
    return finalized["final_result"]
