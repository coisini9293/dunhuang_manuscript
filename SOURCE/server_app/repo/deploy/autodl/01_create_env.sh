#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${AUTODL_LOG_DIR:-$ROOT/outputs/autodl/logs}"
ENV_NAME="${AUTODL_ENV_NAME:-autohdr}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/01_create_env.log") 2>&1

command -v conda >/dev/null
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda run -n "$ENV_NAME" python --version
echo "ENV_READY=$ENV_NAME"
