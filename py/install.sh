#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR="venv"
if [ -d ".venv" ] && [ ! -d "$VENV_DIR" ]; then
  VENV_DIR=".venv"
fi

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements.txt

echo "------------------------------------------------------------"
echo "Python env ready: $PWD/$VENV_DIR"
echo "Use: source $VENV_DIR/bin/activate"
