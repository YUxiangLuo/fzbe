"""Business-facing algorithm contracts and manual-parameter inheritance tests.

These tests deliberately distinguish a reproducible, documented scenario from
the scientifically invalid claim that one forecasting algorithm is universally
best.  The air-conditioner scenario gives the multivariate LSTM information
that the univariate models cannot use: already-signed delivery commitments for
each future horizon.  The comparison is always made on a chronological holdout.
"""

from __future__ import annotations

from contextlib import redirect_stderr
import hashlib
import io
import json
import math
import pickle
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from typing import Any

import guided_ensemble_model
from data_utils import attach_training_provenance, load_data_from_json
from ensemble_utils import (
    parse_base_model_names,
    resolve_base_model_artifact_configs,
    warn_ignored_boosting_controls,
)
from py.deep_tests.support.harness import (
    IsolatedWorkspace,
    PY_ROOT,
    assert_prediction_contract,
    run_command,
)
from py.deep_tests.support.oracles import metrics as independent_metrics
from py.src.models import weighted_avg
from py.deep_tests.support.profiles import profile


AIRCON_SEED = 20_260_730
AIRCON_ROWS = 144
AIRCON_TRAIN_END = 131
AIRCON_EVAL_START = 132
AIRCON_EVAL_END = 143
AIRCON_HORIZON = AIRCON_EVAL_END - AIRCON_EVAL_START + 1
AIRCON_TARGET = "销售数量"
AIRCON_ORDER_FEATURES = [
    f"已签约未来{horizon}期订单" for horizon in range(1, AIRCON_HORIZON + 1)
]
BASE_MODELS = ("ma", "es", "arima", "lstm")
ENSEMBLE_MODELS = ("weighted_avg", "boosting", "stacking")
MANUAL_BASE_ARGS = {
    "ma": ("--window", "6"),
    "es": ("--alpha", "0.3"),
    "arima": ("--d", "1"),
}


def _air_conditioner_records() -> list[dict[str, Any]]:
    """Create deterministic air-conditioner demand with known leading orders.

    ``已签约未来N期订单`` is information available at the row's forecast origin,
    not a future row read by the model.  It represents delivery commitments
    already signed for a future month.  This makes the intended business
    advantage of a multivariate model explicit and auditable.
    """
    rng = random.Random(AIRCON_SEED)
    sales: list[float] = []
    for index in range(AIRCON_ROWS):
        seasonal = (
            140.0
            + 65.0 * math.sin(2.0 * math.pi * (index % 12) / 12.0)
            + 25.0 * math.cos(2.0 * math.pi * (index % 6) / 6.0)
        )
        campaign = 55.0 if ((index * 7 + 3) % 17) < 4 else -18.0
        weather = 38.0 * math.sin(2.0 * math.pi * index / 7.0 + 0.4)
        irregular = rng.uniform(-22.0, 22.0)
        sales.append(max(20.0, seasonal + campaign + weather + irregular))

    records: list[dict[str, Any]] = []
    for index, sale in enumerate(sales):
        record: dict[str, Any] = {
            "月份": f"{2020 + index // 12:04d}-{index % 12 + 1:02d}",
            "产品": "家用空调",
            AIRCON_TARGET: round(sale, 6),
        }
        for horizon, feature_name in enumerate(AIRCON_ORDER_FEATURES, start=1):
            future_index = min(AIRCON_ROWS - 1, index + horizon)
            record[feature_name] = round(sales[future_index], 6)
        records.append(record)
    return records


def _small_contract_records() -> list[dict[str, Any]]:
    return [
        {
            "月份": f"{2020 + index // 12:04d}-{index % 12 + 1:02d}",
            AIRCON_TARGET: 100.0 + 1.25 * index + 9.0 * math.sin(index / 3.0),
            "价格,指数": 1.0 + (index % 9) / 20.0,
            "渠道": "线上" if index % 2 == 0 else "门店",
        }
        for index in range(84)
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _write_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"expected a dictionary artifact at {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lstm_cli_args() -> list[str]:
    feature_types = {
        AIRCON_TARGET: "numeric",
        **{feature: "numeric" for feature in AIRCON_ORDER_FEATURES},
    }
    return [
        "--lstm-target-feature",
        AIRCON_TARGET,
        "--lstm-features",
        json.dumps(AIRCON_ORDER_FEATURES, ensure_ascii=False),
        "--feature-types",
        json.dumps(feature_types, ensure_ascii=False),
        "--normalization",
        "minmax",
    ]


def _train_base_model(
    model_name: str,
    data_file: Path,
    artifact: Path,
) -> dict[str, Any]:
    model_args = (
        _lstm_cli_args() if model_name == "lstm" else list(MANUAL_BASE_ARGS[model_name])
    )
    result = run_command(
        [
            sys.executable,
            PY_ROOT / "run_model.py",
            model_name,
            "--data-file",
            data_file,
            "--train-start",
            "0",
            "--train-end",
            str(AIRCON_TRAIN_END),
            "--eval-start",
            str(AIRCON_EVAL_START),
            "--eval-end",
            str(AIRCON_EVAL_END),
            *model_args,
            "--save-model",
            artifact,
        ],
        timeout=900 if model_name == "lstm" else 600,
    )
    assert result.payload is not None
    return result.payload


def _guided_prepare_args(
    workspace: Path,
    model_type: str,
    records: list[dict[str, Any]],
    base_model_files: dict[str, Path],
    *,
    train_end: int,
    eval_start: int,
    eval_end: int,
    models: tuple[str, ...] = BASE_MODELS,
) -> SimpleNamespace:
    """Build intentionally conflicting fallback args.

    The source artifacts must override these fallback values.  This simulates
    the defense-in-depth boundary after the backend has mapped active base runs.
    """
    artifact_dir = workspace / f"{model_type}-guided"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        model_type=model_type,
        data=json.dumps(records, ensure_ascii=False, allow_nan=False),
        data_file=None,
        train_start=0,
        train_end=train_end,
        eval_start=eval_start,
        eval_end=eval_end,
        models=",".join(models),
        window=2,
        alpha=0.91,
        d=0,
        feature_types=json.dumps(
            {AIRCON_TARGET: "numeric", "价格,指数": "numeric"},
            ensure_ascii=False,
        ),
        lstm_features=json.dumps(["价格,指数"], ensure_ascii=False),
        lstm_target_feature=AIRCON_TARGET,
        normalization="zscore",
        base_model_files=json.dumps(
            {name: str(path) for name, path in base_model_files.items()},
            ensure_ascii=False,
        ),
        save_model=str(workspace / f"{model_type}.pkl"),
        artifact_dir=str(artifact_dir),
    )


def _assert_manual_params(
    testcase: unittest.TestCase,
    model_params: dict[str, dict[str, Any]],
    *,
    window: int,
    alpha: float,
    d: int,
    normalization: str,
    features: list[str],
) -> None:
    testcase.assertEqual({"window": window}, model_params["ma"])
    testcase.assertEqual({"alpha": alpha}, model_params["es"])
    testcase.assertEqual({"d": d}, model_params["arima"])
    testcase.assertEqual(normalization, model_params["lstm"]["normalization"])
    testcase.assertEqual(AIRCON_TARGET, model_params["lstm"]["lstm_target_feature"])
    testcase.assertEqual(
        features,
        json.loads(model_params["lstm"]["lstm_features"]),
    )
    testcase.assertEqual(
        "numeric",
        model_params["lstm"]["feature_types"][AIRCON_TARGET],
    )


class AirConditionerForecastContractTests(unittest.TestCase):
    @profile("full")
    def test_lstm_beats_each_univariate_model_and_artifacts_drive_all_ensembles(self):
        """Exercise both requested examples with real model artifacts.

        The margin is intentionally broad.  It catches a lost-feature,
        misaligned-horizon, or wrong-artifact regression without freezing exact
        neural-network weights or claiming universal LSTM superiority.
        """
        records = _air_conditioner_records()
        with IsolatedWorkspace() as workspace:
            data_file = workspace / "air-conditioner-leading-orders.json"
            _write_json(data_file, records)
            artifacts = {
                model_name: workspace / f"{model_name}.pkl"
                for model_name in BASE_MODELS
            }
            payloads = {
                model_name: _train_base_model(
                    model_name,
                    data_file,
                    artifacts[model_name],
                )
                for model_name in BASE_MODELS
            }

            recomputed: dict[str, dict[str, float]] = {}
            for model_name, payload in payloads.items():
                assert_prediction_contract(payload, AIRCON_HORIZON)
                predictions = [
                    float(point["prediction"]) for point in payload["predictions"]
                ]
                actuals = [float(value) for value in payload["eval_y_true"]]
                self.assertEqual(
                    [
                        records[index][AIRCON_TARGET]
                        for index in range(
                            AIRCON_EVAL_START,
                            AIRCON_EVAL_END + 1,
                        )
                    ],
                    actuals,
                )
                recomputed[model_name] = independent_metrics(actuals, predictions)
                for metric_name in ("rmse", "mae", "mape", "r2"):
                    self.assertAlmostEqual(
                        recomputed[model_name][metric_name],
                        float(payload["metrics"][metric_name]),
                        places=8,
                        msg=f"{model_name}.{metric_name}",
                    )

            lstm_metrics = recomputed["lstm"]
            for metric_name in ("rmse", "mae", "mape"):
                best_univariate = min(
                    recomputed[model_name][metric_name]
                    for model_name in ("ma", "es", "arima")
                )
                self.assertLess(
                    lstm_metrics[metric_name],
                    0.65 * best_univariate,
                    msg=(
                        f"LSTM {metric_name}={lstm_metrics[metric_name]:.6f} "
                        f"must beat the best univariate value "
                        f"{best_univariate:.6f} by a stable margin"
                    ),
                )
            self.assertGreater(lstm_metrics["r2"], 0.5)
            self.assertEqual("standard", payloads["lstm"]["training_regime"])
            self.assertEqual(
                AIRCON_ORDER_FEATURES,
                payloads["lstm"]["training_metadata"]["feature_keys"],
            )

            ma_state = _read_pickle(artifacts["ma"])
            es_state = _read_pickle(artifacts["es"])
            arima_state = _read_pickle(artifacts["arima"])
            lstm_state = _read_pickle(artifacts["lstm"])
            self.assertEqual(6, ma_state["window"])
            self.assertEqual(0.3, es_state["alpha"])
            self.assertEqual(1, arima_state["d"])
            self.assertEqual("minmax", lstm_state["normalization"])
            self.assertEqual(AIRCON_ORDER_FEATURES, lstm_state["feature_keys"])

            source_paths = list(artifacts.values()) + [
                artifacts["lstm"].with_name(f"{artifacts['lstm'].stem}_keras_model.h5"),
            ]
            source_hashes = {path: _sha256(path) for path in source_paths}
            for ensemble_name in ENSEMBLE_MODELS:
                state: dict[str, Any] = {}
                output = guided_ensemble_model.run_step(
                    ensemble_name,
                    "prepare_data",
                    _guided_prepare_args(
                        workspace,
                        ensemble_name,
                        records,
                        artifacts,
                        train_end=AIRCON_TRAIN_END,
                        eval_start=AIRCON_EVAL_START,
                        eval_end=AIRCON_EVAL_END,
                    ),
                    state,
                )
                if ensemble_name == "boosting":
                    # Every Boosting stage after the zero-stage baseline learns a
                    # signed residual, so a saved demand-scale artifact is never
                    # replayed and the step must not claim otherwise.
                    self.assertNotIn("reused_artifacts", output)
                else:
                    self.assertEqual(set(BASE_MODELS), set(output["reused_artifacts"]))
                self.assertEqual(
                    {name: str(path) for name, path in artifacts.items()},
                    state["base_model_files"],
                )
                _assert_manual_params(
                    self,
                    state["model_params"],
                    window=6,
                    alpha=0.3,
                    d=1,
                    normalization="minmax",
                    features=AIRCON_ORDER_FEATURES,
                )

            self.assertEqual(
                source_hashes,
                {path: _sha256(path) for path in source_paths},
                "ensemble preparation must not rewrite active base artifacts",
            )


class EnsembleManualParameterContractTests(unittest.TestCase):
    @profile("fast")
    def test_deprecated_boosting_controls_are_explicitly_reported_as_ignored(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            message = warn_ignored_boosting_controls(
                SimpleNamespace(learning_rate=0.0, max_rounds=2),
            )

        self.assertIsNotNone(message)
        self.assertIn("--learning-rate, --max-rounds", message or "")
        self.assertIn("derived from data", message or "")
        self.assertEqual(f"{message}\n", stderr.getvalue())
        self.assertIsNone(warn_ignored_boosting_controls(SimpleNamespace()))

    def _valid_artifacts(
        self,
        workspace: Path,
        records: list[dict[str, Any]],
    ) -> dict[str, Path]:
        states = {
            "ma": {"model_type": "ma", "window": 9},
            "es": {"model_type": "es", "alpha": 0.27},
            "arima": {
                "model_type": "arima",
                "d": 2,
                "order": {"p": 0, "d": 2, "q": 1},
            },
            "lstm": {
                "model_type": "lstm",
                "normalization": "zscore",
                "target_key": AIRCON_TARGET,
                "feature_keys": ["价格,指数", "渠道"],
                "feature_types": {
                    AIRCON_TARGET: "numeric",
                    "价格,指数": "numeric",
                    "渠道": "categorical",
                },
                "forecast_strategy": "direct_multi_horizon",
                "forecast_horizon": 12,
                "look_back": 8,
            },
        }
        parsed_data = load_data_from_json(json.dumps(records, ensure_ascii=False))
        paths: dict[str, Path] = {}
        for model_name, state in states.items():
            path = workspace / f"{model_name}-source.pkl"
            attach_training_provenance(
                state,
                parsed_data,
                range(0, 72),
                feature_keys=["价格,指数", "渠道"] if model_name == "lstm" else [],
            )
            _write_pickle(path, state)
            paths[model_name] = path
        return paths

    @profile("fast")
    def test_each_guided_ensemble_uses_artifact_params_not_conflicting_fallbacks(self):
        records = _small_contract_records()
        with IsolatedWorkspace() as workspace:
            artifacts = self._valid_artifacts(workspace, records)
            before = {path: _sha256(path) for path in artifacts.values()}
            for ensemble_name in ENSEMBLE_MODELS:
                state: dict[str, Any] = {}
                output = guided_ensemble_model.run_step(
                    ensemble_name,
                    "prepare_data",
                    _guided_prepare_args(
                        workspace,
                        ensemble_name,
                        records,
                        artifacts,
                        train_end=71,
                        eval_start=72,
                        eval_end=83,
                    ),
                    state,
                )
                if ensemble_name == "boosting":
                    # Every Boosting stage after the zero-stage baseline learns a
                    # signed residual, so a saved demand-scale artifact is never
                    # replayed and the step must not claim otherwise.
                    self.assertNotIn("reused_artifacts", output)
                else:
                    self.assertEqual(set(BASE_MODELS), set(output["reused_artifacts"]))
                _assert_manual_params(
                    self,
                    state["model_params"],
                    window=9,
                    alpha=0.27,
                    d=2,
                    normalization="zscore",
                    features=["价格,指数", "渠道"],
                )
            self.assertEqual(
                before,
                {path: _sha256(path) for path in artifacts.values()},
            )

    @profile("fast")
    def test_provenance_mismatch_rejects_before_member_training_starts(self):
        records = _small_contract_records()
        parsed_data = load_data_from_json(json.dumps(records, ensure_ascii=False))
        with IsolatedWorkspace() as workspace:
            source_path = workspace / "wrong-prefix-ma.pkl"
            source_state = {"model_type": "ma", "window": 4}
            attach_training_provenance(
                source_state,
                parsed_data,
                range(1, 72),
                feature_keys=[],
            )
            _write_pickle(source_path, source_state)
            args = SimpleNamespace(
                data=json.dumps(records, ensure_ascii=False),
                train_start=0,
                train_end=71,
                eval_start=72,
                eval_end=83,
                models="ma,es",
                base_model_files=json.dumps({"ma": str(source_path)}),
                window=4,
                alpha=0.3,
                pretty=False,
                save_model=None,
                load_model=None,
                fit_only=False,
            )
            with (
                patch.object(weighted_avg, "train_and_evaluate") as training,
                patch.object(weighted_avg, "run_base_model_subprocess") as member_subprocess,
                self.assertRaisesRegex(ValueError, "training provenance"),
            ):
                weighted_avg.run(args)
            training.assert_not_called()
            member_subprocess.assert_not_called()

    @profile("fast")
    def test_invalid_or_ambiguous_source_artifacts_are_rejected_early(self):
        records = _small_contract_records()
        parsed_data = load_data_from_json(
            json.dumps(records, ensure_ascii=False, allow_nan=False)
        )
        fallback_params = {
            "ma": {"window": 4},
            "es": {"alpha": 0.4},
            "arima": {"d": 1},
            "lstm": {
                "normalization": "minmax",
                "lstm_target_feature": AIRCON_TARGET,
                "lstm_features": json.dumps(["价格,指数"], ensure_ascii=False),
                "feature_types": {
                    AIRCON_TARGET: "numeric",
                    "价格,指数": "numeric",
                },
            },
        }

        with IsolatedWorkspace() as workspace:
            cases: list[tuple[str, str, Any, str]] = [
                (
                    "boolean MA window",
                    "ma",
                    {"model_type": "ma", "window": True},
                    "must be an integer",
                ),
                (
                    "non-finite ES alpha",
                    "es",
                    {"model_type": "es", "alpha": float("nan")},
                    "must be finite",
                ),
                (
                    "missing ARIMA d",
                    "arima",
                    {"model_type": "arima", "order": {"p": 1, "q": 1}},
                    "is missing d",
                ),
                (
                    "missing LSTM features",
                    "lstm",
                    {
                        "model_type": "lstm",
                        "normalization": "minmax",
                        "target_key": AIRCON_TARGET,
                    },
                    "explicit feature list",
                ),
                (
                    "missing LSTM target",
                    "lstm",
                    {
                        "model_type": "lstm",
                        "normalization": "minmax",
                        "feature_keys": [],
                    },
                    "explicit target field",
                ),
                (
                    "invalid LSTM normalization",
                    "lstm",
                    {
                        "model_type": "lstm",
                        "normalization": "unit",
                        "target_key": AIRCON_TARGET,
                        "feature_keys": ["价格,指数"],
                        "feature_types": {
                            AIRCON_TARGET: "numeric",
                            "价格,指数": "numeric",
                        },
                    },
                    "normalization",
                ),
                (
                    "invalid LSTM feature type",
                    "lstm",
                    {
                        "model_type": "lstm",
                        "normalization": "minmax",
                        "target_key": AIRCON_TARGET,
                        "feature_keys": ["价格,指数"],
                        "feature_types": {
                            AIRCON_TARGET: "numeric",
                            "价格,指数": "ordinal",
                        },
                    },
                    "feature type",
                ),
                (
                    "duplicate LSTM features",
                    "lstm",
                    {
                        "model_type": "lstm",
                        "normalization": "minmax",
                        "target_key": AIRCON_TARGET,
                        "feature_keys": ["价格,指数", "价格,指数"],
                    },
                    "contains duplicates",
                ),
                (
                    "target repeated as LSTM input",
                    "lstm",
                    {
                        "model_type": "lstm",
                        "normalization": "minmax",
                        "target_key": AIRCON_TARGET,
                        "feature_keys": [AIRCON_TARGET],
                    },
                    "must not repeat the target",
                ),
                (
                    "unknown LSTM feature field",
                    "lstm",
                    {
                        "model_type": "lstm",
                        "normalization": "minmax",
                        "target_key": AIRCON_TARGET,
                        "feature_keys": ["不存在的字段"],
                    },
                    "not present in the dataset",
                ),
                (
                    "missing LSTM feature type",
                    "lstm",
                    {
                        "model_type": "lstm",
                        "normalization": "minmax",
                        "target_key": AIRCON_TARGET,
                        "feature_keys": ["价格,指数"],
                        "feature_types": {
                            AIRCON_TARGET: "numeric",
                        },
                    },
                    "missing feature types",
                ),
                (
                    "categorical LSTM target",
                    "lstm",
                    {
                        "model_type": "lstm",
                        "normalization": "minmax",
                        "target_key": AIRCON_TARGET,
                        "feature_keys": ["价格,指数"],
                        "feature_types": {
                            AIRCON_TARGET: "categorical",
                            "价格,指数": "numeric",
                        },
                    },
                    "target feature type must be numeric",
                ),
            ]
            for index, (label, model_name, source_state, message) in enumerate(cases):
                with self.subTest(label=label):
                    source_path = workspace / f"invalid-{index}.pkl"
                    _write_pickle(source_path, source_state)
                    with self.assertRaisesRegex(ValueError, message):
                        resolve_base_model_artifact_configs(
                            parsed_data,
                            list(BASE_MODELS),
                            fallback_params,
                            {model_name: str(source_path)},
                            "Contract test",
                            range(0, len(records)),
                        )

            corrupt_path = workspace / "corrupt.pkl"
            corrupt_path.write_bytes(b"not-a-model")
            with self.assertRaisesRegex(ValueError, "cannot be read"):
                resolve_base_model_artifact_configs(
                    parsed_data,
                    list(BASE_MODELS),
                    fallback_params,
                    {"ma": str(corrupt_path)},
                    "Contract test",
                    range(0, len(records)),
                )

            missing_path = workspace / "missing.pkl"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                resolve_base_model_artifact_configs(
                    parsed_data,
                    list(BASE_MODELS),
                    fallback_params,
                    {"ma": str(missing_path)},
                    "Contract test",
                    range(0, len(records)),
                )

            arima_path = workspace / "unselected-arima.pkl"
            _write_pickle(
                arima_path,
                {"model_type": "arima", "d": 1},
            )
            with self.assertRaisesRegex(ValueError, "unselected model"):
                resolve_base_model_artifact_configs(
                    parsed_data,
                    ["ma", "es"],
                    fallback_params,
                    {"arima": str(arima_path)},
                    "Contract test",
                    range(0, len(records)),
                )

            wrong_type_path = workspace / "wrong-type.pkl"
            _write_pickle(
                wrong_type_path,
                {"model_type": "es", "window": 4},
            )
            with self.assertRaisesRegex(ValueError, "mismatched model type"):
                resolve_base_model_artifact_configs(
                    parsed_data,
                    list(BASE_MODELS),
                    fallback_params,
                    {"ma": str(wrong_type_path)},
                    "Contract test",
                    range(0, len(records)),
                )

    @profile("fast")
    def test_model_selection_rejects_missing_duplicate_unknown_and_single_members(self):
        invalid_values = (None, "", "ma", "ma,ma", "ma,prophet")
        for value in invalid_values:
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError,
                    r"^\[VALIDATION\]",
                ),
            ):
                parse_base_model_names(value, "Contract test")

        self.assertEqual(
            ["ma", "lstm"],
            parse_base_model_names(" ma, lstm ", "Contract test"),
        )

    @profile("fast")
    def test_artifact_boundaries_are_revalidated_by_all_ensemble_planners(self):
        records = _small_contract_records()
        parsed_data = load_data_from_json(json.dumps(records, ensure_ascii=False))
        training_range = range(0, 72)
        with IsolatedWorkspace() as workspace:
            valid_es = workspace / "valid-es.pkl"
            invalid_ma = workspace / "invalid-ma.pkl"
            invalid_es = workspace / "invalid-es.pkl"
            valid_ma = workspace / "valid-ma.pkl"
            invalid_arima = workspace / "invalid-arima.pkl"
            invalid_lstm = workspace / "invalid-lstm.pkl"
            artifact_states = {
                valid_es: {"model_type": "es", "alpha": 0.27},
                invalid_ma: {"model_type": "ma", "window": 1},
                invalid_es: {"model_type": "es", "alpha": 0.0},
                valid_ma: {"model_type": "ma", "window": 4},
                invalid_arima: {"model_type": "arima", "d": -1},
                invalid_lstm: {
                    "model_type": "lstm",
                    "normalization": "unit",
                    "target_key": AIRCON_TARGET,
                    "feature_keys": ["价格,指数"],
                    "feature_types": {
                        AIRCON_TARGET: "numeric",
                        "价格,指数": "numeric",
                    },
                },
            }
            for artifact_path, artifact_state in artifact_states.items():
                provenance_features = (
                    artifact_state.get("feature_keys", [])
                    if artifact_state["model_type"] == "lstm"
                    else []
                )
                attach_training_provenance(
                    artifact_state,
                    parsed_data,
                    training_range,
                    feature_keys=provenance_features,
                )
                _write_pickle(artifact_path, artifact_state)

            invalid_maps = (
                (
                    {"ma": invalid_ma, "es": valid_es},
                    ("ma", "es"),
                    r"window.*>= 2",
                ),
                (
                    {"ma": valid_ma, "es": invalid_es},
                    ("ma", "es"),
                    r"alpha.*\(0, 1\]",
                ),
                (
                    {"ma": valid_ma, "arima": invalid_arima},
                    ("ma", "arima"),
                    r"ARIMA.*d.*>= 0",
                ),
                (
                    {"ma": valid_ma, "lstm": invalid_lstm},
                    ("ma", "lstm"),
                    r"LSTM.*normalization",
                ),
            )
            for ensemble_name in ENSEMBLE_MODELS:
                for artifacts, models, message in invalid_maps:
                    with (
                        self.subTest(
                            ensemble=ensemble_name,
                            artifacts=sorted(artifacts),
                        ),
                        self.assertRaisesRegex(ValueError, message),
                    ):
                        guided_ensemble_model.run_step(
                            ensemble_name,
                            "prepare_data",
                            _guided_prepare_args(
                                workspace,
                                ensemble_name,
                                records,
                                artifacts,
                                train_end=71,
                                eval_start=72,
                                eval_end=83,
                                models=models,
                            ),
                            {},
                        )
