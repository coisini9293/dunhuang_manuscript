from __future__ import annotations

from pathlib import Path


class VectorRetriever:
    """Optional FAISS side-retriever. It never produces missing text."""

    def __init__(self, index_dir: str | Path | None = None, enabled: bool = False):
        self.index_dir = Path(index_dir) if index_dir else None
        self.enabled = enabled
        self.ready = bool(
            self.index_dir
            and (self.index_dir / "index.faiss").exists()
            and (self.index_dir / "vector_manifest.json").exists()
        )

    def recall(self, query: str, limit: int = 250) -> dict[str, dict]:
        if not self.enabled or not self.ready or not query:
            return {}
        from .vector_runtime import search_faiss

        return search_faiss(self.index_dir, query, limit=limit)
