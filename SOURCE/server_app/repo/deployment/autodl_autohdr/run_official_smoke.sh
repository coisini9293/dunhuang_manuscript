#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <real-case-image>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM="$ROOT/third_party/AutoHDR"
INPUT="$(realpath "$1")"
OUT="$ROOT/unreviewed_official_raw/smoke"

[[ -f "$INPUT" ]] || { echo "FAIL: input missing: $INPUT" >&2; exit 3; }
[[ -d "$UPSTREAM" ]] || { echo "FAIL: AutoHDR source missing" >&2; exit 4; }
command -v nvidia-smi >/dev/null || { echo "FAIL: nvidia-smi missing" >&2; exit 5; }

required=(
  "$UPSTREAM/ckpt/damage_detect.pth"
  "$UPSTREAM/ckpt/best.pt"
  "$UPSTREAM/ckpt/unet"
  "$UPSTREAM/dist/det_model"
  "$UPSTREAM/dist/reg_model"
)
for path in "${required[@]}"; do
  [[ -e "$path" ]] || { echo "FAIL: required model missing: $path" >&2; exit 6; }
done
if [[ ! -d "$UPSTREAM/ckpt/AutoHDR-Qwen2-7B" ]]; then
  echo "FAIL: upstream script currently expects ckpt/AutoHDR-Qwen2-7B" >&2
  exit 7
fi

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

mkdir -p "$OUT"
sha256sum "$INPUT" > "$OUT/input.sha256"
backup=""
if [[ -f "$UPSTREAM/example.jpg" ]]; then
  backup="$(mktemp)"
  cp "$UPSTREAM/example.jpg" "$backup"
fi
restore_example() {
  if [[ -n "$backup" && -f "$backup" ]]; then
    cp "$backup" "$UPSTREAM/example.jpg"
    rm -f "$backup"
  fi
}
trap restore_example EXIT

cp "$INPUT" "$UPSTREAM/example.jpg"
(
  cd "$UPSTREAM"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python infer_pipeline.py 2>&1 | tee "$OUT/run.log"
)
cp -a "$UPSTREAM/results/." "$OUT/results/"
echo "OFFICIAL_SMOKE_RAW_OUTPUT_READY_REQUIRES_MANUAL_REVIEW"
