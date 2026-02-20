#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d "venv" ]; then
  source venv/bin/activate
elif [ -d ".venv" ]; then
  # 兼容旧路径
  source .venv/bin/activate
else
  python3 -m venv venv
  source venv/bin/activate
fi

pip install --upgrade pip
pip install -r requirements.txt

echo "------------------------------------------------------------"
echo "Virtualenv ready. Try:"
if [ -d "venv" ]; then
  echo "source venv/bin/activate"
else
  echo "source .venv/bin/activate"
fi
echo "python -m adf_tool data/sample.json --pretty"
