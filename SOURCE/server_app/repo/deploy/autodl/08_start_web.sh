#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${AUTODL_LOG_DIR:-$ROOT/outputs/autodl/logs}"
PID_DIR="${AUTODL_PID_DIR:-$ROOT/outputs/autodl/pids}"
ENV_NAME="${AUTODL_ENV_NAME:-autohdr}"
mkdir -p "$LOG_DIR" "$PID_DIR"

start_one() {
  local name="$1" port="$2" command="$3" pid_file="$PID_DIR/$name.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name already running pid=$(cat "$pid_file")"
    return
  fi
  if command -v lsof >/dev/null && lsof -iTCP:"$port" -sTCP:LISTEN -t | grep -q .; then
    echo "Port $port is already in use" >&2
    exit 4
  fi
  nohup conda run --no-capture-output -n "$ENV_NAME" bash -lc "cd '$ROOT' && $command" >"$LOG_DIR/$name.log" 2>&1 &
  echo $! > "$pid_file"
}

start_one autohdr-worker 8010 "python -m services.autohdr_worker.app"
start_one gurou-web 8000 "AUTOHDR_BACKEND=official AUTOHDR_WORKER_URL=http://127.0.0.1:8010 ENABLE_LEGACY_DIFFHDR=true python -m demo.backend.main"
sleep 3
kill -0 "$(cat "$PID_DIR/autohdr-worker.pid")"
kill -0 "$(cat "$PID_DIR/gurou-web.pid")"
echo "WEB_START_PASS"
