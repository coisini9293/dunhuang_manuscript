#!/usr/bin/env bash
set -euo pipefail

BASE="${DUNHUANG_BASE:-/path/to/dunhuang-autohdr}"
PID_FILE="$BASE/runtime/pids/dunhuang-backend.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "BACKEND_STOPPED"
  exit 1
fi
pid="$(cat "$PID_FILE")"
if ! kill -0 "$pid" 2>/dev/null; then
  echo "BACKEND_STALE_PID PID=$pid"
  exit 1
fi
echo "BACKEND_RUNNING PID=$pid"
curl -fsS --max-time 5 http://127.0.0.1:8000/api/health
echo
