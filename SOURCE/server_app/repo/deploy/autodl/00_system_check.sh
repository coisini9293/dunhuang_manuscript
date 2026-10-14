#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${AUTODL_LOG_DIR:-$ROOT/outputs/autodl/logs}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/00_system_check.log") 2>&1

echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname)"
uname -a
df -h "$ROOT"
free -h
command -v git >/dev/null
command -v conda >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi
python --version
echo "SYSTEM_CHECK_PASS"
