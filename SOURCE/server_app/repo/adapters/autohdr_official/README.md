# External Official AutoHDR adapter

This package is an independent interoperability layer for the unmodified
Official AutoHDR checkout pinned at:

`d92b223b3d9274b03ca6f1e4c62b09253f378c98`

It does not contain copied Official implementation files and must not be placed
inside the Official worktree. GPU workers import public functions/classes from
the configured, commit-verified checkout while using the separate immutable
runtime layout as their working directory for relative weight/executable
paths. `analysis_worker` stops after damage/OCR/Qwen
candidate export. `official_restore_worker` accepts only a validated,
server-generated repair plan and invokes the Official renderer and DiffHDR
pipeline for approved regions. `safe_compositor` is authoritative for the final
PNG and copies generated pixels only under the approved union mask.

The upstream README identifies the code and dataset as CC BY-NC-ND 4.0 for
non-commercial research. This adapter preserves the upstream checkout and
weights unchanged. Distribution or other use still requires an independent
license review.

No-GPU tests cover the schemas, hashes, stale/tampered plan rejection, mixed
approved/rejected/abstained decisions, RAG non-selection, and pixel-isolated
compositing. They do not claim that Official GPU inference has run.
