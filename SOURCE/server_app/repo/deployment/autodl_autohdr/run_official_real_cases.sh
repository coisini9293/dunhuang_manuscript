#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_MONOLITHIC_OFFICIAL_SMOKE:-0}" != "1" ]]; then
  echo "FAIL: set ALLOW_MONOLITHIC_OFFICIAL_SMOKE=1 after reading STEP_BY_STEP.md" >&2
  exit 2
fi
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <case_transfer_manifest.csv>" >&2
  exit 3
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$1"
RUNNER="$ROOT/deployment/autodl_autohdr/run_official_smoke.sh"
[[ -f "$MANIFEST" ]] || { echo "FAIL: manifest missing" >&2; exit 4; }

tail -n +2 "$MANIFEST" | while IFS=, read -r case_id source_path transfer_path sha256 status notes; do
  [[ "$status" == "ready_for_transfer" ]] || continue
  image="$ROOT/${transfer_path//\"/}"
  expected="${sha256//\"/}"
  actual="$(sha256sum "$image" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "FAIL: SHA-256 mismatch for $case_id" >&2
    exit 5
  }
  bash "$RUNNER" "$image"
  destination="$ROOT/unreviewed_official_raw/cases/${case_id//\"/}"
  mkdir -p "$destination"
  cp -a "$ROOT/unreviewed_official_raw/smoke/." "$destination/"
done

echo "OFFICIAL_REAL_CASE_RAW_OUTPUTS_READY_REQUIRES_MANUAL_REVIEW"
