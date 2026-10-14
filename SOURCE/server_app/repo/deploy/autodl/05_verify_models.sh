#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${AUTODL_LOG_DIR:-$ROOT/outputs/autodl/logs}"
SOURCE_DIR="${AUTOHDR_SOURCE_DIR:-$ROOT/third_party/AutoHDR_official}"
MANIFEST="${AUTOHDR_MODEL_MANIFEST:-$SOURCE_DIR/ckpt/SHA256SUMS}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/05_verify_models.log") 2>&1

test -s "$MANIFEST"
(cd "$(dirname "$MANIFEST")" && sha256sum --check --strict "$(basename "$MANIFEST")")
for path in ckpt/damage_detect.py ckpt/damage_detect.pth ckpt/dic_31524.txt ckpt/best.pt ckpt/unet; do
  test -e "$SOURCE_DIR/$path" || { echo "Missing required model asset: $path" >&2; exit 3; }
done
if [[ ! -d "$SOURCE_DIR/ckpt/AutoHDR-Qwen2-7B" && ! -d "$SOURCE_DIR/ckpt/AutoHDR-Qwen2-1.5B" ]]; then
  echo "Missing AutoHDR Qwen model directory" >&2
  exit 3
fi
echo "MODEL_VERIFY_PASS"
