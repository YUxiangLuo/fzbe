# Seven-model Python deep tests

This is an independent standard-library `unittest` suite for `ma`, `es`, `arima`, `lstm`, `weighted_avg`, `stacking`, and `boosting`. It does not import fixtures or helpers from `py/test` and must not read frontend E2E data.

## One runner

Install the exact tool version and locked model/test environment first:

```bash
python -m pip install "uv==$(cat py/.uv-version)"
./py/install.sh --test
```

Then use only:

```bash
bun run test:deep:fast
bun run test:deep:full
bun run test:deep:nightly
bun run test:deep:resource
```

The package commands delegate directly to `py/deep_tests/run.py`; use
`py/venv/bin/python py/deep_tests/run.py --profile <profile> --list` when exact
test IDs are needed.

`fast` runs deterministic manifest, parser, formula, seeded-process, dynamic-boundary, numerical-degeneracy, nonzero-origin, and chronological-split cases. It also covers four shared contracts that sit under every model: ADF stationarity structure and verdicts (`adf_fixed_d`), the demand-vs-signed output domain (`forecast_utils`), horizon-aware uncertainty calibration — fallback / SES / rolling-origin profiles plus prediction-interval and P99 formatting (`uncertainty`), and the log-redaction / stable-error-tag helpers (`log_safety`, `error_protocol`). `full` includes fast plus matrix tests that perform real train/save/load/corruption and target-leakage checks for every one of the seven models, classical-model d/window/alpha variants, and a real z-score mixed-feature LSTM. It also includes the ADF CLI cross-language contract (`python -m src.adf`, whose fields must satisfy the backend `parseAdfResultPayload` parser). `nightly` adds seeded retraining stability, future-feature poison checks for the LSTM family, real LSTM-member ensembles, three-classical-member ensembles, and all-seven standalone/guided/production checkpoint parity. `resource` runs Linux-only cold subprocess/cleanup cases. Use `--list` for the exact selected test IDs and counts.

## Algorithm situation matrix

The suite deliberately varies the algorithm as well as the data situation; a
single shared happy path is not considered sufficient coverage.

| Algorithm | Fast situations | Real-training situations |
| --- | --- | --- |
| MA | windows 2 / 3 / full-history, horizons 1 / 2 / 8, zero / signed / `1e12` scale, nonzero train origin, poisoned prefix and gap, invalid window/history/steps | nonzero-origin quadratic data, evaluation gap, exact independent recursive oracle, saved horizons 1 and 12 |
| ES | alpha `1e-8` / `0.35` / `1`, zero / signed / `1e12` scale, independent level recursion, nonzero origin and poisoned gap, invalid alpha values | alpha `0.05`, nonzero-origin quadratic data, flat multi-step contract, saved horizons 1 and 12 |
| ARIMA | negative/oversized differencing boundaries and fallback from a non-converged top-ranked AICc candidate | stochastic d=0, deterministic d=1, quadratic d=2, gap forecasts, AIC/AICc/BIC and Ljung-Box structure, confidence intervals, saved horizons 1 and 12 |
| LSTM | explicit empty feature list, JSON feature names containing commas, min-max and z-score, numeric/categorical order, unseen category, bad target/feature values, feature-budget and saved-horizon boundaries | z-score mixed numeric/categorical features, comma-containing name, unseen evaluation category, nonzero origin, save/load prefix parity, over-horizon rejection; separate target-only and business-feature cases remain in the all-seven/business tests |
| Weighted Average | zero-MSE epsilon path, identical columns, shrinkage, scale equivalence, nonzero-origin selection/calibration split, four-member replay | all-seven lifecycle with MA+ES; nightly MA+ES+ARIMA with artifact replay |
| Stacking | zero/collinear regularization, three-member NNLS, nonnegative constraint, corrupt meta state, exact rolling-origin OOF coverage, four-member replay | all-seven lifecycle with MA+ES; nightly MA+ES+ARIMA and MA+LSTM variants |
| Boosting | MA warm-up prefix, NaN padding/infinity rejection, signed residual line search, repeated model stages, corrupt stage chains, nonzero-origin selection/calibration split | baseline-only and nonempty-chain regressions; nightly MA+ES+ARIMA and MA+LSTM variants |

ARIMA candidate search treats a finite information criterion and optimizer
convergence as separate requirements. The highest-ranked candidate is retried
once with the persisted full specification; if it still does not converge, the
next AICc-ranked candidate is tried and the fallback rank is recorded in
`search_summary.candidate_selection`.

For the backend-facing algorithm contract (the TypeScript request boundary plus
the Python `full` profile), use:

```bash
bun run test:algorithms
```

`test_algorithm_business_contracts.py` contains the business-level checks. Its
air-conditioner case does **not** claim that LSTM is universally better. It
defines a chronological holdout where already-signed future delivery orders are
available at the forecast origin, verifies metrics with an independent formula,
and requires the multivariate LSTM to beat MA, ES, and ARIMA by a broad margin.
The same test then passes the four real base artifacts through the preparation
path of weighted average, Boosting, and Stacking and proves that MA window, ES
alpha, ARIMA differencing order, and LSTM normalization/target/features come
from those artifacts even when conflicting fallback values are supplied.

Uncertainty and ADF assertions follow the same "no golden from production code" rule as the model cases: interval offsets are checked against the documented `NORMAL_95` / `NORMAL_99` z-scores and the √h / ANN-variance growth laws, and stationarity is asserted as qualitative structure (a pure trend is non-stationary and collapses to a constant when differenced; a committed |φ|<1 AR process is stationary) rather than frozen statsmodels statistics.

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
