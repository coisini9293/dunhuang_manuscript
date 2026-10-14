#!/usr/bin/env bash
set -euo pipefail

BASE="${DUNHUANG_BASE:-/path/to/dunhuang-autohdr}"
PID_FILE="$BASE/runtime/pids/dunhuang-backend.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "BACKEND_NOT_RUNNING"
  exit 0
fi
pid="$(cat "$PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  if [[ "$command_line" != *"dunhuang_backend.app:app"* ]]; then
    echo "PID_MISMATCH: refusing to stop PID=$pid" >&2
    exit 1
  fi
  kill "$pid"
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
  echo "BACKEND_STOPPED PID=$pid"
fi
rm -f "$PID_FILE"
