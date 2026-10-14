#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${AUTODL_LOG_DIR:-$ROOT/outputs/autodl/logs}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/09_healthcheck.log") 2>&1

worker="$(curl --fail --silent --show-error http://127.0.0.1:8010/health)"
web="$(curl --fail --silent --show-error http://127.0.0.1:8000/api/system/health)"
python - "$worker" "$web" <<'PY'
import json, sys
worker, web = map(json.loads, sys.argv[1:])
assert worker.get("status") == "ok", worker
assert worker.get("models_loaded") is True, worker
assert web.get("status") == "ok", web
assert web.get("autohdr_backend") == "official", web
assert web.get("is_mock") is False, web
PY
echo "HEALTHCHECK_PASS"
