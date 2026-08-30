#!/usr/bin/env bash
set -euo pipefail

py_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
version_file="$py_root/.uv-version"

if [[ ! -f "$version_file" ]]; then
  echo "Pinned uv version file does not exist: $version_file" >&2
  exit 1
fi
expected_version="$(tr -d '[:space:]' < "$version_file")"
if [[ ! "$expected_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Invalid uv version in $version_file: $expected_version" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "A bootstrap uv installation is required to provision uv $expected_version." >&2
  exit 1
fi

installed_version="$(uv --version 2>/dev/null | awk '{print $2}')"
if [[ "$installed_version" == "$expected_version" ]]; then
  exec uv "$@"
fi

# Do not replace the host bootstrap. A versioned tool environment lets the
# target and previous commits select their own uv during activation/rollback.
exec uv tool run --from "uv==$expected_version" uv "$@"
