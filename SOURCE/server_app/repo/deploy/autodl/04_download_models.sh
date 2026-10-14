#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${AUTODL_LOG_DIR:-$ROOT/outputs/autodl/logs}"
SOURCE_DIR="${AUTOHDR_SOURCE_DIR:-$ROOT/third_party/AutoHDR_official}"
MODEL_ARCHIVE="${AUTOHDR_MODEL_ARCHIVE:-}"
MODEL_ARCHIVE_SHA256="${AUTOHDR_MODEL_ARCHIVE_SHA256:-}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/04_download_models.log") 2>&1

[[ -n "$MODEL_ARCHIVE" && -n "$MODEL_ARCHIVE_SHA256" ]] || {
  echo "Set AUTOHDR_MODEL_ARCHIVE to a licensed local archive and AUTOHDR_MODEL_ARCHIVE_SHA256 to its full SHA-256." >&2
  exit 2
}
test -f "$MODEL_ARCHIVE"
echo "$MODEL_ARCHIVE_SHA256  $MODEL_ARCHIVE" | sha256sum --check --strict
mkdir -p "$SOURCE_DIR/ckpt"
case "$MODEL_ARCHIVE" in
  *.tar.gz|*.tgz) tar -xzf "$MODEL_ARCHIVE" -C "$SOURCE_DIR" ;;
  *.zip) unzip -o "$MODEL_ARCHIVE" -d "$SOURCE_DIR" ;;
  *) echo "Unsupported model archive: $MODEL_ARCHIVE" >&2; exit 2 ;;
esac
echo "MODEL_ARCHIVE_PASS"
