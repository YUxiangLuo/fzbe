#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/deployment-environment.sh
source "$project_root/scripts/deployment-environment.sh"

cd "$project_root"
./scripts/run-pinned-bun.sh --version
./scripts/run-pinned-bun.sh run python:lock:check

echo "===========database migrations============"
run_bun_with_service_environment "$project_root/.env" "$project_root" run scripts/migrate.ts

# The ordered ledger may already contain 009 from an earlier CASCADE rollout.
# Re-run its idempotent contract audit during staging so known incompatible
# data blocks the stop window. update.sh repeats it after quiescing the legacy
# writer to close writes that race this preliminary audit.
run_bun_with_service_environment \
  "$project_root/.env" \
  "$project_root" \
  run scripts/migrate-experiment-retry-lineage.ts

echo "==============custom actions=============="
