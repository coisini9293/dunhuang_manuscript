#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${AUTODL_LOG_DIR:-$ROOT/outputs/autodl/logs}"
OUT="${AUTODL_DIAGNOSTICS_DIR:-$ROOT/outputs/autodl/diagnostics}"
ENV_NAME="${AUTODL_ENV_NAME:-autohdr}"
mkdir -p "$LOG_DIR" "$OUT"
exec > >(tee -a "$LOG_DIR/10_collect_diagnostics.log") 2>&1

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/date_utc.txt"
git -C "$ROOT" rev-parse HEAD > "$OUT/project_commit.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
conda run -n "$ENV_NAME" python -m pip freeze > "$OUT/pip-freeze.txt"
curl --fail --silent --show-error http://127.0.0.1:8010/health > "$OUT/worker-health.json"
curl --fail --silent --show-error http://127.0.0.1:8000/api/system/health > "$OUT/web-health.json"
cp -a "$LOG_DIR/." "$OUT/logs/"
(cd "$(dirname "$OUT")" && tar -czf autodl-diagnostics.tar.gz "$(basename "$OUT")")
echo "DIAGNOSTICS_PASS=$OUT"
