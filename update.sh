#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
# shellcheck source=scripts/deployment-environment.sh
source "$project_root/scripts/deployment-environment.sh"

git_common_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
exec 9>"$git_common_dir/fzbe-deploy.lock"
if ! flock -n 9; then
  echo "Deployment aborted: another backend deployment is already running." >&2
  exit 1
fi

service_name="fangzhen-be.service"
service_working_directory="$(systemctl show "$service_name" --property=WorkingDirectory --value)"
if [[ ! -d "$service_working_directory" ]]; then
  echo "Deployment aborted: $service_name WorkingDirectory does not exist: $service_working_directory" >&2
  exit 1
fi

project_realpath="$(realpath -e -- "$project_root")"
service_realpath="$(realpath -e -- "$service_working_directory")"
if [[ "$service_realpath" != "$project_realpath" ]]; then
  echo "Deployment aborted: $service_name runs $service_realpath, not $project_realpath." >&2
  exit 1
fi

if [[ ! -f "$project_root/.env" ]]; then
  echo "Deployment aborted: $project_root/.env does not exist." >&2
  exit 1
fi
service_environment_files="$(systemctl show "$service_name" --property=EnvironmentFiles --value)"
if ! service_environment_file="$(deployment_single_environment_file "$service_environment_files")"; then
  echo "Deployment aborted: $service_name must load only $project_root/.env." >&2
  exit 1
fi
if [[ ! -f "$service_environment_file" ]]; then
  echo "Deployment aborted: $service_name environment file does not exist: $service_environment_file" >&2
  exit 1
fi
service_environment_realpath="$(realpath -e -- "$service_environment_file")"
project_environment_realpath="$(realpath -e -- "$project_root/.env")"
if [[ "$service_environment_realpath" != "$project_environment_realpath" ]]; then
  echo "Deployment aborted: $service_name does not load $project_root/.env." >&2
  exit 1
fi
service_direct_environment="$(systemctl show "$service_name" --property=Environment --value)"
if deployment_environment_has_protected_override "$service_direct_environment"; then
  echo "Deployment aborted: $service_name has a deployment-sensitive Environment= override." >&2
  echo "Put DB_*, HOST, PORT, JWT*, and ALLOW_* values only in $project_root/.env." >&2
  exit 1
fi

if ! systemctl is-active --quiet "$service_name"; then
  echo "Deployment aborted: $service_name is not currently active." >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "Deployment aborted: curl is required for the post-start health check." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Deployment aborted: the backend worktree contains uncommitted changes." >&2
  exit 1
fi

upstream_ref="$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')"
remote_name="${upstream_ref%%/*}"
current_commit="$(git rev-parse HEAD)"

echo "===========FETCH BACKEND RELEASE============"
git fetch --prune "$remote_name"
target_commit="$(git rev-parse "$upstream_ref")"

if ! git merge-base --is-ancestor "$current_commit" "$target_commit"; then
  echo "Deployment aborted: $upstream_ref is not a fast-forward of $current_commit." >&2
  exit 1
fi

assert_live_checkout_unchanged() {
  if [[ "$(git rev-parse HEAD)" != "$current_commit" ]]; then
    echo "Deployment aborted: backend HEAD changed while the release was being prepared." >&2
    return 1
  fi
  if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    echo "Deployment aborted: backend worktree changed while the release was being prepared." >&2
    return 1
  fi
  assert_deployment_environment_unchanged
}

assert_deployment_environment_unchanged() {
  if ! deployment_environment_files_match \
    "$deployment_environment_snapshot" \
    "$project_root/.env"; then
    echo "Deployment aborted: $project_root/.env changed after the deployment snapshot was created." >&2
    echo "No activation or restart will use an environment that was not staged and migrated." >&2
    return 1
  fi
}

resolve_health_url() {
  # JavaScript template literals are evaluated by Bun, not by this shell.
  # shellcheck disable=SC2016
  run_bun_with_service_environment "$deployment_environment_snapshot" "$project_root" -e '
    const configuredHost = process.env.HOST?.trim() || "127.0.0.1";
    const host = configuredHost === "0.0.0.0"
      ? "127.0.0.1"
      : configuredHost === "::"
        ? "[::1]"
        : configuredHost.includes(":")
          ? `[${configuredHost}]`
          : configuredHost;
    const port = process.env.PORT?.trim() || "3001";
    process.stdout.write(`http://${host}:${port}/api/v1/health`);
  '
}

wait_for_service_health() {
  local health_url legacy_health_url
  assert_deployment_environment_unchanged
  health_url="$(resolve_health_url)"
  # The first release introducing /health must still be able to verify an
  # automatic rollback to its predecessor, whose public readiness route was
  # /runtime-info. Only permit that fallback after Git has returned to the
  # pre-deployment commit; it must never mask a failed target activation.
  legacy_health_url="${health_url%/health}/runtime-info"
  for _attempt in {1..30}; do
    if sudo systemctl is-active --quiet "$service_name" \
      && {
        curl --noproxy '*' --fail --silent --connect-timeout 1 --max-time 1 "$health_url" >/dev/null \
          || {
            [[ "$(git rev-parse HEAD)" == "$current_commit" ]] \
              && curl --noproxy '*' --fail --silent --connect-timeout 1 --max-time 1 "$legacy_health_url" >/dev/null
          }
      }; then
      return 0
    fi
    sleep 0.5
  done
  echo "Backend health check did not become ready: $health_url (legacy fallback: $legacy_health_url)" >&2
  return 1
}

staging_parent="$(mktemp -d "${TMPDIR:-/tmp}/fzbe-release.XXXXXX")"
release_dir="$staging_parent/release"
deployment_environment_snapshot="$release_dir/.env"
service_stopped=false
deployment_complete=false
python_environment_changed=false

cleanup() {
  if [[ -d "$release_dir" ]]; then
    git worktree remove --force "$release_dir" >/dev/null 2>&1 || true
  fi
  rm -rf -- "$staging_parent"
}

recover_and_cleanup() {
  status=$?
  trap - EXIT
  set +e

  if [[ "$service_stopped" == true && "$deployment_complete" != true ]]; then
    echo "Deployment activation failed; restoring backend commit $current_commit." >&2
    sudo systemctl stop "$service_name" >/dev/null 2>&1 || true

    recovery_status=0
    rollback_head="$(git rev-parse HEAD 2>/dev/null)"
    rollback_status="$(git status --porcelain --untracked-files=normal 2>/dev/null)"
    if [[ "$rollback_head" != "$current_commit" && "$rollback_head" != "$target_commit" ]] \
      || [[ -n "$rollback_status" ]]; then
      echo "Automatic rollback refused: the live checkout contains unexpected changes." >&2
      recovery_status=1
    else
      git reset --hard "$current_commit" || recovery_status=$?
    fi
    if [[ "$recovery_status" -eq 0 ]]; then
      bun install --frozen-lockfile || recovery_status=$?
    fi
    if [[ "$recovery_status" -eq 0 && "$python_environment_changed" == true ]]; then
      ./py/install.sh || recovery_status=$?
    fi
    if [[ "$recovery_status" -eq 0 ]]; then
      assert_deployment_environment_unchanged || recovery_status=$?
    fi
    if [[ "$recovery_status" -eq 0 ]]; then
      sudo systemctl start "$service_name" || recovery_status=$?
    fi
    if [[ "$recovery_status" -eq 0 ]]; then
      wait_for_service_health || recovery_status=$?
    fi

    if [[ "$recovery_status" -eq 0 ]]; then
      echo "Previous backend release restored and restarted." >&2
    else
      echo "Automatic rollback failed; $service_name requires manual recovery." >&2
    fi
  fi

  cleanup
  exit "$status"
}
trap recover_and_cleanup EXIT

git worktree add --detach "$release_dir" "$target_commit"
cp -p -- "$project_root/.env" "$deployment_environment_snapshot"
assert_deployment_environment_unchanged

if ! git diff --quiet "$current_commit" "$target_commit" -- py/requirements.txt py/install.sh; then
  python_environment_changed=true
fi

echo "===========PREPARE BACKEND RELEASE============"
(
  cd "$release_dir"
  bun install --frozen-lockfile
  if [[ "$python_environment_changed" == true ]]; then
    ./py/install.sh
  fi
  ./actions.sh
)

assert_live_checkout_unchanged

echo "============STOP $service_name==========="
service_stopped=true
sudo systemctl stop "$service_name"
assert_deployment_environment_unchanged

# The staging audit runs while the legacy writer is still live. It can create a
# later unlinked attempt or reject an earlier report after that audit completes,
# so close the race only after the writer is quiesced. This command is
# intentionally unguarded: under `set -e`, an audit failure reaches the EXIT
# recovery trap before the live checkout is advanced or the new service starts.
echo "============FINAL QUIESCED DATA RECONCILIATION============"
run_bun_with_service_environment \
  "$deployment_environment_snapshot" \
  "$release_dir" \
  run scripts/migrate-experiment-retry-lineage.ts
assert_deployment_environment_unchanged

# Migration 012 installs a generated-column unique index while the old writer
# is live, so canonical duplicates cannot race past staging. A legacy writer can
# still create a brand-new, unique phone with surrounding whitespace; converge
# those raw values after quiescing it and before activating the new writer.
run_bun_with_service_environment \
  "$deployment_environment_snapshot" \
  "$release_dir" \
  run scripts/migrate-user-phone-canonical-unique.ts
assert_deployment_environment_unchanged

echo "============ACTIVATE BACKEND RELEASE============"
git merge --ff-only "$target_commit"
bun install --frozen-lockfile
if [[ "$python_environment_changed" == true ]]; then
  ./py/install.sh
fi
assert_deployment_environment_unchanged

echo "============START $service_name==========="
sudo systemctl start "$service_name"
assert_deployment_environment_unchanged
wait_for_service_health

# Health confirms that the new writer has taken ownership.  From this point a
# post-activation reconciliation failure must not restore the legacy writer,
# which could recreate ambiguous retry lineage and destructive rejection zeros.
deployment_complete=true

echo "============POST-ACTIVATION DATA RECONCILIATION============"
post_activation_status=0
if ! run_bun_with_service_environment \
  "$deployment_environment_snapshot" \
  "$project_root" \
  run scripts/migrate-experiment-retry-lineage.ts; then
  echo "Retry-lineage reconciliation failed; the new healthy service remains active." >&2
  post_activation_status=1
fi
if ! run_bun_with_service_environment \
  "$deployment_environment_snapshot" \
  "$project_root" \
  run scripts/migrate-rejected-report-grades.ts; then
  echo "Rejected-report grade normalization failed; the new healthy service remains active." >&2
  post_activation_status=1
fi
if ! assert_deployment_environment_unchanged; then
  echo "The service remains active with the staged environment, but $project_root/.env now differs." >&2
  post_activation_status=1
fi
if [[ "$post_activation_status" -ne 0 ]]; then
  echo "Post-activation reconciliation requires operator repair and a deployment rerun." >&2
  exit 1
fi
