#!/usr/bin/env bash
set -euo pipefail

base="${1:-http://127.0.0.1:8000}"
health="$(curl -fsS --max-time 5 "$base/api/health")"
rag="$(curl -fsS --max-time 30 "$base/api/rag/health")"
system="$(curl -fsS --max-time 5 "$base/api/system/status")"
printf '%s\n%s\n%s\n' "$health" "$rag" "$system"
echo "$rag" | grep -q '"loaded":true'
echo "$rag" | grep -q '"ngram_ready":true'
echo "$rag" | grep -q '"bm25_ready":true'
echo "BACKEND_HEALTH_PASS"
