#!/usr/bin/env bash

# This file is sourced by deployment entry points that already enable
# `set -euo pipefail`. Keep deployment-sensitive values out of the caller's
# environment: systemd and these commands must read the same verified .env.

deployment_environment_has_protected_override() {
  local configured_environment="${1-}"
  local protected_pattern='(^|[[:space:]"])(DB_[[:alnum:]_]*|HOST|PORT|JWT[[:alnum:]_]*|ALLOW_[[:alnum:]_]*|NODE_ENV|BUN_ENV)='
  [[ "$configured_environment" =~ $protected_pattern ]]
}

deployment_single_environment_file() {
  local configured_files="${1-}"
  local -a fields=()
  read -r -a fields <<< "$configured_files"

  if [[ "${#fields[@]}" -eq 1 ]]; then
    printf '%s\n' "${fields[0]#-}"
    return 0
  fi

  # systemctl currently renders each EnvironmentFile as two fields:
  # /absolute/path (ignore_errors=no). Refuse additional files because a later
  # file could override the project .env after it has been validated here.
  if [[ "${#fields[@]}" -eq 2 && "${fields[1]}" =~ ^\(ignore_errors=(yes|no)\)$ ]]; then
    printf '%s\n' "${fields[0]#-}"
    return 0
  fi

  echo "Expected exactly one systemd EnvironmentFile, got: $configured_files" >&2
  return 1
}

deployment_environment_files_match() {
  if [[ "$#" -ne 2 ]]; then
    echo "deployment_environment_files_match requires SNAPSHOT CURRENT" >&2
    return 2
  fi

  [[ -f "$1" && -f "$2" ]] && cmp -s -- "$1" "$2"
}

run_bun_with_service_environment() {
  if [[ "$#" -lt 3 ]]; then
    echo "run_bun_with_service_environment requires ENV_FILE WORKING_DIRECTORY BUN_ARGS..." >&2
    return 2
  fi

  local environment_file="$1"
  local working_directory="$2"
  shift 2

  if [[ ! -f "$environment_file" ]]; then
    echo "Controlled environment file does not exist: $environment_file" >&2
    return 1
  fi
  if [[ ! -d "$working_directory" ]]; then
    echo "Controlled working directory does not exist: $working_directory" >&2
    return 1
  fi

  environment_file="$(realpath -e -- "$environment_file")"
  working_directory="$(realpath -e -- "$working_directory")"

  local bun_executable
  bun_executable="$(command -v bun)"
  bun_executable="$(realpath -e -- "$bun_executable")"
  local controlled_path
  controlled_path="$(dirname -- "$bun_executable"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

  # Bun's dotenv parser expands $VARIABLE references, while systemd's
  # EnvironmentFile parser deliberately does not. Run the command as a
  # transient service so migrations and the real service interpret the exact
  # same file syntax. --no-env-file also prevents Bun from loading worktree
  # .env.local or mode-specific files on top of the systemd environment.
  local -a systemd_arguments=(
    --quiet
    --wait
    --pipe
    --collect
    --service-type=exec
    "--uid=$(id -u)"
    "--gid=$(id -g)"
    "--working-directory=$working_directory"
    "--property=EnvironmentFile=$environment_file"
    "--setenv=HOME=${HOME:?HOME must be set for deployment commands}"
    "--setenv=PATH=$controlled_path"
  )
  if [[ -n "${TMPDIR-}" ]]; then
    systemd_arguments+=("--setenv=TMPDIR=$TMPDIR")
  fi
  if [[ -n "${LANG-}" ]]; then
    systemd_arguments+=("--setenv=LANG=$LANG")
  fi
  if [[ -n "${LC_ALL-}" ]]; then
    systemd_arguments+=("--setenv=LC_ALL=$LC_ALL")
  fi

  sudo systemd-run "${systemd_arguments[@]}" \
    "$bun_executable" \
    --no-env-file \
    --cwd="$working_directory" \
    "$@"
}
