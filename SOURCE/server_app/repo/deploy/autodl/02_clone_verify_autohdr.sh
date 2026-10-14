#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${AUTODL_LOG_DIR:-$ROOT/outputs/autodl/logs}"
SOURCE_DIR="${AUTOHDR_SOURCE_DIR:-$ROOT/third_party/AutoHDR_official}"
REPO="https://github.com/SCUT-DLVCLab/AutoHDR.git"
PINNED_COMMIT="d92b223b3d9274b03ca6f1e4c62b09253f378c98"
mkdir -p "$LOG_DIR" "$(dirname "$SOURCE_DIR")"
exec > >(tee -a "$LOG_DIR/02_clone_verify_autohdr.log") 2>&1

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  [[ ! -e "$SOURCE_DIR" ]] || { echo "Refusing non-Git source: $SOURCE_DIR" >&2; exit 1; }
  git clone "$REPO" "$SOURCE_DIR"
fi
git -C "$SOURCE_DIR" remote get-url origin | grep -Fx "$REPO" >/dev/null
git -C "$SOURCE_DIR" fetch --depth 1 origin "$PINNED_COMMIT"
git -C "$SOURCE_DIR" checkout --detach "$PINNED_COMMIT"
test "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$PINNED_COMMIT"
test -f "$SOURCE_DIR/LICENSE"
echo "AUTOHDR_COMMIT_PASS=$PINNED_COMMIT"
