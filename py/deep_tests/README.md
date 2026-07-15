# Seven-model Python deep tests

This is an independent standard-library `unittest` suite for `ma`, `es`, `arima`, `lstm`, `weighted_avg`, `stacking`, and `boosting`. It does not import fixtures or helpers from `py/test` and must not read frontend E2E data.

## One runner

Install with `./py/install.sh --test` (there are intentionally no extra test dependencies), then use only:

```bash
py/venv/bin/python py/deep_tests/run.py --profile fast
py/venv/bin/python py/deep_tests/run.py --profile full
py/venv/bin/python py/deep_tests/run.py --profile nightly
py/venv/bin/python py/deep_tests/run.py --profile resource
```

`fast` runs deterministic manifest, parser, formula, seeded-process, and dynamic-boundary cases. `full` includes fast plus matrix tests that perform real train/save/load/corruption and target-leakage checks for every one of the seven models. `nightly` adds seeded retraining stability, future-feature poison checks for the LSTM family, real LSTM-member ensembles, and all-seven standalone/guided/production checkpoint parity. `resource` runs Linux-only cold subprocess/cleanup cases. Use `--list` for the exact selected test IDs and counts.

If needed, the runner re-executes itself before discovery so `PYTHONHASHSEED=20250308` applies at interpreter startup. It also fixes TensorFlow deterministic mode and all common BLAS/TensorFlow thread counts to one. The subprocess harness uses a strict JSON decoder to reject actual `NaN`, `Infinity`, and `-Infinity` constants, recursively checks finite numbers, and prints argv plus bounded stdout/stderr diagnostics on failure.

## Datasets

Fixtures are canonical UTF-8 strict JSON and have a manifest containing generator revision, seed, SHA-256, row/column schema, chronological split, and an independent formula description. The seed directly drives the committed ARX process through a local RNG; a fast test proves same-seed identity and different-seed divergence.

```bash
py/venv/bin/python py/deep_tests/support/generate_datasets.py --check
# Intentional fixture maintenance only:
py/venv/bin/python py/deep_tests/support/generate_datasets.py --write
```

Never generate expected predictions from production model code. Formula goldens live in `support/oracles.py`; ARIMA/LSTM assertions use stable structure, tolerance, leakage, and repeatability properties rather than frozen order/weights/model bytes.

## Artifact safety and cleanup

Every trained artifact, Keras sidecar, member model, checkpoint, receipt, corruption copy, and mutated dataset is created under `TemporaryDirectory`. Tests snapshot the repository artifact set around these workspaces and fail on newly orphaned files. Pickle corruption cases use only artifacts created by this suite; pickle is not a safe untrusted-input format.
