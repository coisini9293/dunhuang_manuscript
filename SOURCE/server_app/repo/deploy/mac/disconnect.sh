#!/bin/bash
# Mac: disconnect SSH tunnel (only kill Gurou tunnel, not other SSH processes)
set -euo pipefail
PID_FILE="/tmp/gurou_tunnel.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No gurou tunnel PID file found"
    exit 0
fi

PID=$(cat "$PID_FILE")
if [ -z "$PID" ]; then
    echo "Empty PID file"
    rm -f "$PID_FILE"
    exit 0
fi

# Check if PID exists and is an SSH tunnel process
if kill -0 "$PID" 2>/dev/null; then
    CMD=$(ps -p "$PID" -o comm= 2>/dev/null || echo "")
    if echo "$CMD" | grep -q "ssh"; then
        kill "$PID"
        echo "Tunnel PID $PID stopped"
    else
        echo "PID $PID is not SSH ($CMD), not killing"
    fi
else
    echo "PID $PID already exited"
fi

rm -f "$PID_FILE"
echo "Disconnected"
