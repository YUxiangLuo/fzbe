#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "------------------------------------------------------------"
echo "Virtualenv ready. Try:"
echo "source .venv/bin/activate"
echo "python -m adf_tool data/sample.json --pretty"
