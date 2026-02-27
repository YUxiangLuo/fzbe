#!/usr/bin/env bash
# 安装 Python 虚拟环境和所有依赖（模型训练 + ADF 检测）
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR="venv"

# 如果存在旧的 .venv 目录，迁移为 venv
if [ -d ".venv" ] && [ ! -d "$VENV_DIR" ]; then
  mv .venv "$VENV_DIR"
  echo "Migrated .venv -> $VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

echo "Installing dependencies..."
pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements.txt

# 清理 bytecode 缓存
find . -type d -name __pycache__ -not -path "./$VENV_DIR/*" -exec rm -rf {} + 2>/dev/null || true

echo "------------------------------------------------------------"
echo "Python env ready: $PWD/$VENV_DIR"
echo "Activate:  source $VENV_DIR/bin/activate"
echo "Verify:    python -m src.adf --help"
