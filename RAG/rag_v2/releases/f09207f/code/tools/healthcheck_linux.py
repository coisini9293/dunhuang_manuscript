from __future__ import annotations
import json
import os
import sys
from pathlib import Path

package_root = Path(__file__).resolve().parents[2]
code_root = package_root / "code"
rag_root = Path(os.environ.get("RAG_ROOT", package_root / "rag_v2_runtime")).resolve()
sys.path.insert(0, str(code_root))
from retrieval.rag_v2.full_retriever import FullRagV2Retriever

retriever = FullRagV2Retriever(
    rag_root / "metadata.sqlite3",
    rag_root / "ngram",
    rag_root / "faiss",
    vector_enabled=False,
)
health = retriever.health()
expected = {
    "loaded": True,
    "corpus_scope": "full",
    "corpus_version": "CBETA Normal Text 2025 R3",
    "source_commit": "2ffcfdb5011e97637295c6589061b9779273ddda",
    "sutra_count": 4868,
    "record_count": 12557684,
    "short_chunk_count": 13928538,
    "medium_chunk_count": 1970083,
    "ngram_ready": True,
    "bm25_ready": True,
    "vector_enabled": False,
    "index_manifest_hash": "838507071ed263a9977ebef3c929ac66d0fcbc976ce78a5b6962f3fff1cf0ca8",
    "index_version": "rag-v2-full-1.0.0",
}
for key, value in expected.items():
    if health.get(key) != value:
        raise SystemExit(f"health mismatch {key}: {health.get(key)!r} != {value!r}")
print(json.dumps(health, ensure_ascii=False, indent=2))
print("PORTABLE_RAG_HEALTHCHECK_PASS")
