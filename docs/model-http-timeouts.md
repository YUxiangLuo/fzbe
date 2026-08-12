# Model HTTP timeout contract

`contracts/model-http-timeouts.json` is the single data-only source for static model and ADF timeout policy shared by the backend, browser, and nginx. Values are milliseconds and the contract is versioned with `schemaVersion` and `policyVersion`.

The backend validates the strict shape, positive safe integers, and these orderings when it imports configuration:

```text
model operation ceiling + default termination grace + response reserve <= nginx < browser
ADF execution + default termination grace + response reserve <= nginx < browser
```

`MODEL_OPERATION_DEADLINE_MS` and `MODEL_PROCESS_TERMINATION_GRACE_MS` remain deployment overrides. They may shorten the effective backend deadline or cleanup budget, but startup rejects combinations that exceed the static transport ceilings. They never regenerate browser or nginx settings.

Validate a backend change with:

```bash
bun run timeout:check
bun run test:check
```

The matching `fzfe` checkout deterministically generates its committed browser module and the two marked nginx `proxy_read_timeout` directives:

```bash
cd ../fzfe
TIMEOUT_BACKEND_DIR=../fzbe bun run timeout:generate
TIMEOUT_BACKEND_DIR=../fzbe bun run timeout:check
```

Cross-repository rollout is directional:

- For an increase, deploy compatible browser/nginx ceilings before increasing the backend ceiling.
- For a decrease, deploy the shorter backend ceiling before reducing browser/nginx ceilings.

Already-open browser bundles may retain the previous ceiling, so changes must preserve compatibility during the release-retention window.
