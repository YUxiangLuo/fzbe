#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
version_file="$project_root/.bun-version"

if [[ ! -f "$version_file" ]]; then
  echo "Pinned Bun version file does not exist: $version_file" >&2
  exit 1
fi
expected_version="$(tr -d '[:space:]' < "$version_file")"
if [[ ! "$expected_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Invalid Bun version in $version_file: $expected_version" >&2
  exit 1
fi

bootstrap_bun="$(command -v bun 2>/dev/null || true)"
if [[ -z "$bootstrap_bun" && -n "${BUN_INSTALL-}" && -x "$BUN_INSTALL/bin/bun" ]]; then
  bootstrap_bun="$BUN_INSTALL/bin/bun"
fi
if [[ -z "$bootstrap_bun" && -n "${HOME-}" && -x "$HOME/.bun/bin/bun" ]]; then
  # A system service normally does not inherit the user's interactive PATH.
  bootstrap_bun="$HOME/.bun/bin/bun"
fi
if [[ -z "$bootstrap_bun" ]]; then
  echo "A bootstrap Bun installation is required to provision Bun $expected_version." >&2
  exit 1
fi

if [[ "$("$bootstrap_bun" --version 2>/dev/null)" == "$expected_version" ]]; then
  exec "$bootstrap_bun" "$@"
fi

# `bun x` stores the versioned bun package in Bun's persistent install cache. The
# updater invokes this runner for both the current and target checkout before
# downtime, so either version remains available during rollback.
exec "$bootstrap_bun" x "bun@$expected_version" "$@"
