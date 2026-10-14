#!/bin/bash
# Mac: SSH tunnel to AutoDL server
set -euo pipefail
: "${AUTO_DL_HOST:?Set AUTO_DL_HOST env var}"
: "${AUTO_DL_PORT:?Set AUTO_DL_PORT env var}"
: "${AUTO_DL_USER:?Set AUTO_DL_USER env var}"
SSH_KEY="${SSH_KEY:-~/.ssh/id_rsa}"

echo "=== Gurou AutoDL SSH Tunnel ==="
echo "Host: $AUTO_DL_HOST  Port: $AUTO_DL_PORT  User: $AUTO_DL_USER"
echo "Local: http://127.0.0.1:8000"
echo ""

# Check local port
if lsof -i :8000 2>/dev/null | grep -q LISTEN; then
    echo "ERROR: Port 8000 already in use"
    lsof -i :8000
    exit 1
fi

# Connect
echo "Connecting..."
ssh -N -L 8000:127.0.0.1:8000 -p "$AUTO_DL_PORT" -i "$SSH_KEY" "$AUTO_DL_USER@$AUTO_DL_HOST" &
TUNNEL_PID=$!
echo "Tunnel PID: $TUNNEL_PID"
echo $TUNNEL_PID > /tmp/gurou_tunnel.pid

sleep 2
if kill -0 $TUNNEL_PID 2>/dev/null; then
    echo "Tunnel established. Open http://127.0.0.1:8000"
    open http://127.0.0.1:8000 2>/dev/null || echo "Open manually: http://127.0.0.1:8000"
else
    echo "ERROR: Tunnel failed"
    exit 1
fi
