from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

from retrieval.rag_v2.hybrid_retriever import RagV2Retriever
from retrieval.rag_v2.full_retriever import FullRagV2Retriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "artifacts" / "rag_v2" / "t0278" / "metadata.sqlite3"
DEFAULT_VECTOR_DIR = PROJECT_ROOT / "artifacts" / "rag_v2" / "t0278" / "faiss"
FULL_ROOT = PROJECT_ROOT / "artifacts" / "rag_v2" / "full"
FULL_DB = FULL_ROOT / "metadata.sqlite3"
FULL_NGRAM = FULL_ROOT / "ngram"
FULL_VECTOR_DIR = FULL_ROOT / "faiss"
FULL_MANIFEST = FULL_ROOT / "manifests" / "index_manifest.json"
_retriever: RagV2Retriever | FullRagV2Retriever | None = None
_lock = Lock()


def get_rag_v2() -> RagV2Retriever | FullRagV2Retriever | None:
    global _retriever
    deploy_root_text = os.environ.get("RAG_V2_DEPLOY_ROOT", "")
    explicit_database = os.environ.get("RAG_V2_DB")
    if deploy_root_text:
        runtime_root = Path(deploy_root_text)
        database = runtime_root / "metadata.sqlite3"
        ngram_root = runtime_root / "ngram"
        vector_dir = runtime_root / "faiss"
        full_scope = True
    elif explicit_database:
        database = Path(explicit_database)
        ngram_root = Path(
            os.environ.get("RAG_V2_NGRAM_ROOT", str(database.parent / "ngram"))
        )
        vector_dir = Path(
            os.environ.get("RAG_V2_VECTOR_DIR", str(database.parent / "faiss"))
        )
        full_scope = os.environ.get("RAG_V2_SCOPE", "").lower() == "full"
    elif FULL_MANIFEST.exists() and FULL_DB.exists():
        database = FULL_DB
        ngram_root = FULL_NGRAM
        vector_dir = FULL_VECTOR_DIR
        full_scope = True
    else:
        database = DEFAULT_DB
        ngram_root = Path()
        vector_dir = DEFAULT_VECTOR_DIR
        full_scope = False
    if not database.exists():
        return None
    if _retriever is None:
        with _lock:
            if _retriever is None:
                vector_enabled = (
                    os.environ.get("RAG_V2_VECTOR_ENABLED", "false").lower() == "true"
                )
                if full_scope:
                    _retriever = FullRagV2Retriever(
                        database,
                        ngram_root,
                        vector_dir=vector_dir,
                        vector_enabled=vector_enabled,
                    )
                else:
                    _retriever = RagV2Retriever(
                        database,
                        vector_dir=vector_dir,
                        vector_enabled=vector_enabled,
                    )
    return _retriever


def reset_rag_v2() -> None:
    global _retriever
    with _lock:
        _retriever = None


def rag_v2_health() -> dict:
    retriever = get_rag_v2()
    if retriever is None:
        return {
            "loaded": False,
            "ngram_ready": False,
            "bm25_ready": False,
            "faiss_ready": False,
            "vector_enabled": False,
        }
    health = retriever.health()
    health.setdefault("corpus_scope", "t0278")
    return health
