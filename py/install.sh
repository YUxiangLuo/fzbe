#!/usr/bin/env bash
# 安装 Python 虚拟环境和所有依赖（模型训练 + ADF 检测）
# 使用 uv 管理 Python 版本和依赖安装
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_VERSION="3.13"
VENV_DIR="venv"

# 检查 uv 是否可用
if ! command -v uv &>/dev/null; then
  echo "Error: uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

# 如果存在旧的 .venv 目录，迁移为 venv
if [ -d ".venv" ] && [ ! -d "$VENV_DIR" ]; then
  mv .venv "$VENV_DIR"
  echo "Migrated .venv -> $VENV_DIR"
fi

# 检测 venv 是否损坏或 Python 版本不匹配
if [ -d "$VENV_DIR" ]; then
  if [ ! -x "$VENV_DIR/bin/python3" ]; then
    echo "Detected broken venv, recreating..."
    rm -rf "$VENV_DIR"
  elif ! "$VENV_DIR/bin/python3" -c "import sys; assert sys.version_info[:2] == (${PYTHON_VERSION/./, })" 2>/dev/null; then
    echo "Detected Python version mismatch, recreating venv for $PYTHON_VERSION..."
    rm -rf "$VENV_DIR"
  fi
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment (Python $PYTHON_VERSION)..."
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi

echo "Installing dependencies..."
uv pip install \
  --python "$VENV_DIR/bin/python3" \
  -r requirements.txt

# 清理 bytecode 缓存
find . -type d -name __pycache__ -not -path "./$VENV_DIR/*" -exec rm -rf {} + 2>/dev/null || true

echo "------------------------------------------------------------"
echo "Python env ready: $PWD/$VENV_DIR"
echo "Activate:  source $VENV_DIR/bin/activate"
echo "Verify:    python -m src.adf --help"
