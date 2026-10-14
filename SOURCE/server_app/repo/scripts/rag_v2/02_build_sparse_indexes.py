from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc

from common import INDEX_VERSION, REPORT_ROOT, T0278_ARTIFACT, T0278_DB, now_iso, require_disk, write_json
from retrieval.rag_v2.bm25_index import build_bm25_index, prove_bm25_participation
from retrieval.rag_v2.ngram_index import build_ngram_indexes, prove_ngram_participation
from retrieval.rag_v2.storage import RagStorage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    require_disk()
    if not T0278_DB.exists():
        raise RuntimeError("Run 01_migrate_t0278.py first")
    storage = RagStorage(T0278_DB)
    if arguments.resume:
        meta = storage.get_meta()
        if meta.get("ngram_ready") == "true" and meta.get("bm25_ready") == "true":
            print("Sparse indexes already complete")
            return 0
    started = time.perf_counter()
    tracemalloc.start()
    ngram = build_ngram_indexes(storage)
    bm25 = build_bm25_index(storage)
    participation = {
        "ngram": prove_ngram_participation(storage),
        "bm25": prove_bm25_participation(storage),
    }
    if not all(participation["ngram"].values()) or not participation["bm25"]:
        raise RuntimeError(f"Index participation proof failed: {participation}")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    manifest_core = {
        "index_version": INDEX_VERSION,
        "built_at": now_iso(),
        "database": "metadata.sqlite3",
        "ngram": ngram,
        "bm25": bm25,
        "participation": participation,
    }
    manifest_hash = hashlib.sha256(
        json.dumps(manifest_core, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {**manifest_core, "index_manifest_hash": manifest_hash}
    write_json(T0278_ARTIFACT / "index_manifest.json", manifest)
    storage.set_meta("index_manifest_hash", manifest_hash)
    statistics = {
        **manifest,
        "build_seconds": round(time.perf_counter() - started, 4),
        "peak_memory_bytes": peak,
        "peak_memory_mib": round(peak / 1024**2, 3),
        "database_bytes": T0278_DB.stat().st_size,
        "counts": storage.counts(),
    }
    write_json(REPORT_ROOT / "03_sparse_index_statistics.json", statistics)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
