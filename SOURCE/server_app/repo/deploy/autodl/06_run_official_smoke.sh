#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${AUTODL_LOG_DIR:-$ROOT/outputs/autodl/logs}"
ENV_NAME="${AUTODL_ENV_NAME:-autohdr}"
SOURCE_DIR="${AUTOHDR_SOURCE_DIR:-$ROOT/third_party/AutoHDR_official}"
SMOKE_COMMAND="${AUTOHDR_SMOKE_COMMAND:-}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/06_run_official_smoke.log") 2>&1

[[ -n "$SMOKE_COMMAND" ]] || { echo "Set AUTOHDR_SMOKE_COMMAND to the reviewed, non-interactive upstream inference command." >&2; exit 2; }
nvidia-smi >/dev/null
(cd "$SOURCE_DIR" && conda run --no-capture-output -n "$ENV_NAME" bash -lc "$SMOKE_COMMAND")
echo "OFFICIAL_SMOKE_PASS"
