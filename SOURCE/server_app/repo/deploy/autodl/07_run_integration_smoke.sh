#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${AUTODL_LOG_DIR:-$ROOT/outputs/autodl/logs}"
ENV_NAME="${AUTODL_ENV_NAME:-autohdr}"
INTEGRATION_COMMAND="${AUTOHDR_INTEGRATION_COMMAND:-}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/07_run_integration_smoke.log") 2>&1

[[ -n "$INTEGRATION_COMMAND" ]] || { echo "Set AUTOHDR_INTEGRATION_COMMAND after implementing the official worker inference endpoints." >&2; exit 2; }
(cd "$ROOT" && conda run --no-capture-output -n "$ENV_NAME" bash -lc "$INTEGRATION_COMMAND")
echo "INTEGRATION_SMOKE_PASS"
