#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${AUTODL_LOG_DIR:-$ROOT/outputs/autodl/logs}"
ENV_NAME="${AUTODL_ENV_NAME:-autohdr}"
SOURCE_DIR="${AUTOHDR_SOURCE_DIR:-$ROOT/third_party/AutoHDR_official}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/03_install_dependencies.log") 2>&1

test -f "$SOURCE_DIR/requirements.txt"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install -r "$SOURCE_DIR/requirements.txt"
conda run -n "$ENV_NAME" python -m pip check
echo "DEPENDENCIES_PASS"
