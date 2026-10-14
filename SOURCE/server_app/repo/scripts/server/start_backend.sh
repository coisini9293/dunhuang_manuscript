#!/usr/bin/env bash
set -euo pipefail

BASE="${DUNHUANG_BASE:-/path/to/dunhuang-autohdr}"
REPO="$BASE/SOURCE/server_app/repo"
PYTHON="$BASE/RUNTIME/envs/dunhuang-backend/bin/python3"
PID_DIR="$BASE/RUNTIME/pids"
LOG_DIR="$BASE/RUNTIME/logs"
PID_FILE="$PID_DIR/dunhuang-backend.pid"
LOG_FILE="$LOG_DIR/dunhuang-backend.log"

mkdir -p "$PID_DIR" "$LOG_DIR" "$BASE/RAG/state" "$BASE/DATA/cases"
if [[ ! -x "$PYTHON" ]]; then
  echo "BACKEND_PYTHON_MISSING: $PYTHON" >&2
  exit 1
fi
if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "BACKEND_ALREADY_RUNNING PID=$pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi
if "$PYTHON" - <<'PY'
import socket
s = socket.socket()
try:
    raise SystemExit(0 if s.connect_ex(("127.0.0.1", 8000)) == 0 else 1)
finally:
    s.close()
PY
then
  echo "BACKEND_PORT_CONFLICT: 127.0.0.1:8000 is already occupied" >&2
  pgrep -af 'uvicorn|dunhuang_backend' >&2 || true
  exit 1
fi

cd "$REPO"
nohup env \
  DUNHUANG_BASE="$BASE" \
  RAG_ROOT="$BASE/RAG/rag_v2/current" \
  PYTHONDONTWRITEBYTECODE=1 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  "$PYTHON" -m uvicorn dunhuang_backend.app:app \
  --host 127.0.0.1 --port 8000 \
  >>"$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"
for _ in $(seq 1 20); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "BACKEND_START_FAILED; see $LOG_FILE" >&2
    exit 1
  fi
  if curl -fsS --max-time 3 http://127.0.0.1:8000/api/health >/dev/null; then
    echo "BACKEND_START_PASS PID=$pid LOG=$LOG_FILE"
    exit 0
  fi
  sleep 0.5
done
kill "$pid" 2>/dev/null || true
rm -f "$PID_FILE"
echo "BACKEND_HEALTH_TIMEOUT; see $LOG_FILE" >&2
exit 1
