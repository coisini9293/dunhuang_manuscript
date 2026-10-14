from __future__ import annotations

import json
from pathlib import Path

from .full_sparse_retriever import FullSparseRetriever
from .full_storage import FullRagStorage
from .hybrid_retriever import RagV2Retriever
from .reranker import DEFAULT_WEIGHTS
from .vector_retriever import VectorRetriever


class FullRagV2Retriever(RagV2Retriever):
    supports_editions = True
    def __init__(
        self,
        database_path: str | Path,
        ngram_root: str | Path,
        vector_dir: str | Path | None = None,
        vector_enabled: bool = False,
        weights: dict[str, float] | None = None,
        use_ngram: bool = True,
        use_bm25: bool = True,
    ):
        self.database_path = Path(database_path)
        self.storage = FullRagStorage(self.database_path, readonly=True)
        self.sparse = FullSparseRetriever(self.database_path, ngram_root)
        self.vector = VectorRetriever(vector_dir, enabled=vector_enabled)
        self.vector_enabled = vector_enabled and self.vector.ready
        self.weights = weights or DEFAULT_WEIGHTS
        self.use_ngram = use_ngram
        self.use_bm25 = use_bm25
        self.meta = self.storage.get_meta()

    def health(self) -> dict:
        counts = self.storage.counts()
        return {
            "loaded": self.database_path.exists(),
            "corpus_scope": "full",
            "corpus_version": self.meta.get("corpus_version", ""),
            "source_commit": self.meta.get("source_commit", ""),
            "sutra_count": int(self.meta.get("sutra_count", "0")),
            "record_count": counts["records"],
            "short_chunk_count": counts["virtual_short_chunks"],
            "medium_chunk_count": counts["medium_chunks"],
            "ngram_ready": self.meta.get("ngram_ready") == "true",
            "bm25_ready": self.meta.get("bm25_ready") == "true",
            "faiss_ready": self.vector.ready,
            "vector_enabled": self.vector_enabled,
            "index_manifest_hash": self.meta.get("index_manifest_hash", ""),
            "index_version": self.meta.get("index_version", ""),
        }

    def _load_chunks(self, chunk_ids: set[str]) -> list[dict]:
        if not chunk_ids:
            return []
        rows: list[dict] = []
        ids = list(chunk_ids)
        with self.storage.connect() as db:
            for offset in range(0, len(ids), 500):
                batch = ids[offset : offset + 500]
                fetched = db.execute(
                    f"SELECT * FROM medium_chunks WHERE chunk_id IN "
                    f"({','.join('?' for _ in batch)})",
                    batch,
                ).fetchall()
                for row in fetched:
                    item = dict(row)
                    item["line_heads"] = json.loads(item["line_heads"])
                    rows.append(item)
        return rows

    def query(
        self,
        context_before: str,
        context_after: str,
        sutra_ids: list[str] | None = None,
        translators: list[str] | None = None,
        editions: list[str] | None = None,
        max_missing: int | None = 10,
        expected_missing_length: int | None = None,
        top_k: int = 5,
        normalization_profile: str = "cbeta_rag_v2",
    ):
        token = self.sparse.edition_filter.set(editions)
        try:
            result = super().query(
                context_before,
                context_after,
                sutra_ids=sutra_ids,
                translators=translators,
                max_missing=max_missing,
                expected_missing_length=expected_missing_length,
                top_k=top_k,
                normalization_profile=normalization_profile,
            )
            self._attach_long_evidence(result.candidates)
            return result
        finally:
            self.sparse.edition_filter.reset(token)

    def _attach_long_evidence(self, candidates) -> None:
        if not candidates:
            return
        with self.storage.connect() as db:
            for candidate in candidates:
                records = db.execute(
                    """
                    SELECT record_id,line_head,original_text,normalized_text
                    FROM records
                    WHERE sutra_id=? AND volume=? AND chapter_name=?
                    ORDER BY page,column_name,line
                    """,
                    (
                        candidate.sutra_id,
                        candidate.volume,
                        candidate.chapter_name,
                    ),
                ).fetchall()
                indexes = {
                    row["line_head"]: index for index, row in enumerate(records)
                }
                left = indexes.get(candidate.line_heads[0], 0)
                right = indexes.get(candidate.line_heads[-1], left)
                current = sum(
                    len(records[index]["normalized_text"])
                    for index in range(left, right + 1)
                )
                while current < 400 and (left > 0 or right + 1 < len(records)):
                    choices = []
                    if left > 0:
                        choices.append(
                            ("left", len(records[left - 1]["normalized_text"]))
                        )
                    if right + 1 < len(records):
                        choices.append(
                            ("right", len(records[right + 1]["normalized_text"]))
                        )
                    direction, added = min(choices, key=lambda item: item[1])
                    if current >= 320 and current + added > 512:
                        break
                    if direction == "left":
                        left -= 1
                    else:
                        right += 1
                    current += added
                selected = records[left : right + 1]
                candidate.evidence_original_text = "\n".join(
                    row["original_text"] for row in selected
                )
                candidate.evidence_normalized_text = "".join(
                    row["normalized_text"] for row in selected
                )
                candidate.evidence_line_heads = [
                    row["line_head"] for row in selected
                ]
                location = (
                    candidate.line_heads[0]
                    if len(candidate.line_heads) == 1
                    else f"{candidate.line_heads[0]}–{candidate.line_heads[-1]}"
                )
                candidate.citation = (
                    f"{candidate.sutra_id}《{candidate.canonical_title}》"
                    f"卷{candidate.volume} {location}"
                    f"（{candidate.translator}，{candidate.edition}）"
                )
