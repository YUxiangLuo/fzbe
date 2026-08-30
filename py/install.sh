#!/usr/bin/env bash
# 安装 Python 虚拟环境和所有依赖（模型训练 + ADF 检测）
# 使用 uv 管理 Python 版本和依赖安装
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_VERSION="$(tr -d '[:space:]' < .python-version)"
UV_RUNNER="$PWD/run-pinned-uv.sh"
VENV_DIR="venv"
INSTALL_TESTS=false

if [[ ! "$PYTHON_VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
  echo "Error: invalid Python version in .python-version: $PYTHON_VERSION" >&2
  exit 1
fi

for arg in "$@"; do
  case "$arg" in
    --test) INSTALL_TESTS=true ;;
    *) echo "Error: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# Provision/check the commit-pinned uv without replacing the host bootstrap.
"$UV_RUNNER" --version

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
  "$UV_RUNNER" venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi

echo "Installing dependencies..."
sync_args=(sync --locked --no-install-project)
if [ "$INSTALL_TESTS" != true ]; then
  sync_args+=(--no-dev)
fi
UV_PROJECT_ENVIRONMENT="$PWD/$VENV_DIR" "$UV_RUNNER" "${sync_args[@]}"

if [ "$INSTALL_TESTS" = true ]; then
  echo "Installed the locked development tools used by Python verification."
fi

# 清理 bytecode 缓存
find . -type d -name __pycache__ -not -path "./$VENV_DIR/*" -exec rm -rf {} + 2>/dev/null || true

echo "------------------------------------------------------------"
echo "Python env ready: $PWD/$VENV_DIR"
echo "Activate:  source $VENV_DIR/bin/activate"
echo "Verify:    python -m src.adf --help"
if [ "$INSTALL_TESTS" = true ]; then
  echo "Deep test: $PWD/$VENV_DIR/bin/python deep_tests/run.py --profile fast"
fi
