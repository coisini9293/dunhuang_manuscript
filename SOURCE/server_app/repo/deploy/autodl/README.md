# AutoDL deployment sequence

Run `00_system_check.sh` through `10_collect_diagnostics.sh` in numeric order from the project checkout. Every script uses `set -euo pipefail`, appends logs below `outputs/autodl`, and returns non-zero if a prerequisite or validation fails.

The official AutoHDR source is pinned to commit `d92b223b3d9274b03ca6f1e4c62b09253f378c98`. The model archive is not redistributed by this project. Before step 04, provide a licensed local archive and its independently recorded full SHA-256:

```bash
export AUTOHDR_MODEL_ARCHIVE=/path/to/AutoHDR-models.tar.gz
export AUTOHDR_MODEL_ARCHIVE_SHA256=<64 lowercase hex characters>
```

Step 05 requires `ckpt/SHA256SUMS` covering every model file. Steps 06 and 07 intentionally fail until reviewed non-interactive smoke commands are supplied through `AUTOHDR_SMOKE_COMMAND` and `AUTOHDR_INTEGRATION_COMMAND`. This prevents an unimplemented Official path from being reported as successful.

Current repository limitation: `services/autohdr_worker` exposes health only. Implement and validate the Official inference endpoints before claiming `OFFICIAL_PASS`; until then the correct state is `OFFICIAL_STUB`, `WORKER_HEALTH_ONLY`, and `AUTODL_REQUIRED`.
