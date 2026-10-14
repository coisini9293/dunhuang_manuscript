# Dunhuang Manuscript Reconstruction

This repository contains the shareable source code for a Dunhuang manuscript
fragment analysis, retrieval, stitching, restoration, and AutoHDR integration
project.

## Included source areas

- `SOURCE/server_app/repo/`: backend, adapters, pipelines, retrieval modules,
  tests, deployment helpers, and web demo code.
- `SOURCE/scripts/paper_inpainting/`: paper-restoration utility scripts.
- `SOURCE/src/AutoHDR/`: project-facing AutoHDR code and integration utilities.
- `RAG/rag_v2/releases/f09207f/code/`: RAG v2 retrieval and service code.
- `RESTORATION/contracts/`: restoration geometry and human-review contracts.

## Deliberately excluded

Datasets, manuscript scans, OCR corpora, user or case data, generated results,
SQLite databases, search indexes, logs, model weights, CUDA/Python environments,
vendored third-party source trees without a confirmed redistribution license,
server-only configuration, and Git history are not included.

The excluded runtime assets must be obtained and reviewed separately. This
repository is therefore a source-code handoff, not a self-contained deployment
image.

## Main entry points

- Backend: `SOURCE/server_app/repo/dunhuang_backend/app.py`
- Full pipeline: `SOURCE/server_app/repo/pipelines/full_pipeline.py`
- Stitching: `SOURCE/server_app/repo/pipelines/true_stitching.py`
- RAG v2: `SOURCE/server_app/repo/retrieval/rag_v2/`
- AutoHDR integration: `SOURCE/server_app/repo/adapters/autohdr_official/`
- AutoHDR pipeline: `SOURCE/src/AutoHDR/infer_pipeline.py`

## Reproducibility and disclosure

Review dependencies, licenses, and data-access permissions before running or
redistributing this code. Paths and credentials from the original server
environment are intentionally not part of this repository.
