from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from .citation import citation_complete, citation_text, row_to_chunk
from .normalize import normalize_query, traditional_alias
from .reranker import DEFAULT_WEIGHTS, best_breakpoint, single_side_similarity
from .schema import QueryResult, RetrievalCandidate
from .sparse_retriever import SparseRetriever
from .storage import RagStorage
from .vector_retriever import VectorRetriever


class RagV2Retriever:
    def __init__(
        self,
        database_path: str | Path,
        vector_dir: str | Path | None = None,
        vector_enabled: bool = False,
        weights: dict[str, float] | None = None,
        use_ngram: bool = True,
        use_bm25: bool = True,
    ):
        self.database_path = Path(database_path)
        self.storage = RagStorage(self.database_path, readonly=True)
        self.sparse = SparseRetriever(self.database_path)
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
            "corpus_version": self.meta.get("corpus_version", ""),
            "sutra_count": int(self.meta.get("sutra_count", "0")),
            "record_count": counts["records"],
            "short_chunk_count": counts["short_chunks"],
            "medium_chunk_count": counts["medium_chunks"],
            "ngram_ready": self.meta.get("ngram_ready") == "true",
            "bm25_ready": self.meta.get("bm25_ready") == "true",
            "faiss_ready": self.vector.ready,
            "vector_enabled": self.vector_enabled,
            "index_manifest_hash": self.meta.get("index_manifest_hash", ""),
            "index_version": self.meta.get("index_version", ""),
        }

    def query(
        self,
        context_before: str,
        context_after: str,
        sutra_ids: list[str] | None = None,
        translators: list[str] | None = None,
        max_missing: int | None = 10,
        expected_missing_length: int | None = None,
        top_k: int = 5,
        normalization_profile: str = "cbeta_rag_v2",
    ) -> QueryResult:
        started = time.perf_counter()
        before = traditional_alias(normalize_query(context_before))
        after = traditional_alias(normalize_query(context_after))
        query_id = hashlib.sha256(
            json.dumps(
                [before, after, sutra_ids, translators, expected_missing_length],
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]
        if not before and not after:
            return QueryResult(query_id=query_id, decision_status="abstain")
        ngram = (
            self.sparse.ngram_recall(
                before, after, sutra_ids, translators, limit=100
            )
            if self.use_ngram
            else {}
        )
        bm25 = (
            self.sparse.bm25_recall(
                before, after, sutra_ids, translators, limit=100
            )
            if self.use_bm25
            else {}
        )
        vector_query = f"{before}[缺損]{after}" if before and after else before or after
        vector = self.vector.recall(vector_query, limit=100)
        chunk_ids = set(ngram) | set(bm25) | set(vector)
        # Each channel has already produced its required top-100 recall list.
        # Apply a cheap RRF pre-merge before expensive character alignment.
        if len(chunk_ids) > 80:
            def rrf_score(chunk_id: str) -> float:
                score = 0.0
                for channel, rank_key in (
                    (ngram, "ngram_rank"),
                    (bm25, "bm25_rank"),
                    (vector, "vector_rank"),
                ):
                    rank = int(channel.get(chunk_id, {}).get(rank_key, 0))
                    if rank:
                        score += 1.0 / (60 + rank)
                return score

            chunk_ids = set(
                sorted(chunk_ids, key=rrf_score, reverse=True)[:80]
            )
        candidates = self._rank_candidates(
            chunk_ids,
            before,
            after,
            ngram,
            bm25,
            vector,
            max_missing or 10,
            expected_missing_length,
        )
        candidates = candidates[:top_k]
        for rank, candidate in enumerate(candidates, 1):
            candidate.rank = rank
        if not candidates:
            status = "abstain"
        elif before and after:
            status = candidates[0].decision_status
        else:
            status = "single_side_localization"
        return QueryResult(
            query_id=query_id,
            decision_status=status,
            candidates=candidates,
            normalized_before=before,
            normalized_after=after,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            vector_enabled=self.vector_enabled,
            corpus_version=self.meta.get("corpus_version", ""),
            index_version=self.meta.get("index_version", ""),
        )

    def _load_chunks(self, chunk_ids: set[str]) -> list[dict]:
        if not chunk_ids:
            return []
        rows: list[dict] = []
        ids = list(chunk_ids)
        with self.storage.connect() as db:
            for offset in range(0, len(ids), 500):
                batch = ids[offset : offset + 500]
                query = f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' for _ in batch)})"
                rows.extend(row_to_chunk(row) for row in db.execute(query, batch).fetchall())
        return rows

    def _rank_candidates(
        self,
        chunk_ids: set[str],
        before: str,
        after: str,
        ngram: dict[str, dict],
        bm25: dict[str, dict],
        vector: dict[str, dict],
        max_missing: int,
        expected_missing_length: int | None,
    ) -> list[RetrievalCandidate]:
        ranked: list[RetrievalCandidate] = []
        chunks = self._load_chunks(chunk_ids)
        # Sparse channels use different chunk scales. Transfer scores between
        # overlapping short/medium chunks so fusion happens at one location.
        by_group: dict[tuple, list[dict]] = defaultdict(list)
        for item in chunks:
            by_group[
                (item["sutra_id"], item["volume"], item["chapter_name"])
            ].append(item)
        fused_features: dict[str, dict[str, float]] = {}
        for group_chunks in by_group.values():
            for chunk in group_chunks:
                overlaps = [
                    other
                    for other in group_chunks
                    if other["char_end"] > chunk["char_start"]
                    and other["char_start"] < chunk["char_end"]
                ]
                best_ngram = max(
                    overlaps,
                    key=lambda other: float(
                        ngram.get(other["chunk_id"], {}).get("ngram_score", 0.0)
                    ),
                    default=chunk,
                )
                best_bm25 = max(
                    overlaps,
                    key=lambda other: float(
                        bm25.get(other["chunk_id"], {}).get("bm25_score", 0.0)
                    ),
                    default=chunk,
                )
                best_vector = max(
                    overlaps,
                    key=lambda other: float(
                        vector.get(other["chunk_id"], {}).get(
                            "vector_similarity", 0.0
                        )
                    ),
                    default=chunk,
                )
                fused_features[chunk["chunk_id"]] = {
                    "ngram": float(
                        ngram.get(best_ngram["chunk_id"], {}).get("ngram_score", 0.0)
                    ),
                    "ngram_source": best_ngram["chunk_id"],
                    "bm25": float(
                        bm25.get(best_bm25["chunk_id"], {}).get("bm25_score", 0.0)
                    ),
                    "bm25_source": best_bm25["chunk_id"],
                    "vector": float(
                        vector.get(best_vector["chunk_id"], {}).get(
                            "vector_similarity", 0.0
                        )
                    ),
                    "vector_source": best_vector["chunk_id"],
                }
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            ngram_component = fused_features[chunk_id]["ngram"]
            bm25_component = fused_features[chunk_id]["bm25"]
            vector_component = fused_features[chunk_id]["vector"]
            ngram_details = ngram.get(
                fused_features[chunk_id]["ngram_source"], {}
            )
            bm25_details = bm25.get(fused_features[chunk_id]["bm25_source"], {})
            citation_component = 1.0 if citation_complete(chunk) else 0.0
            if before and after:
                match = best_breakpoint(
                    chunk["normalized_text"],
                    before,
                    after,
                    max_missing,
                    expected_missing_length,
                )
                if not match:
                    continue
                edit_component = (match.left_edit + match.right_edit) / 2
                continuity_component = match.continuity * match.gap_consistency
                final = (
                    self.weights["continuity"] * continuity_component
                    + self.weights["edit"] * edit_component
                    + self.weights["ngram"] * ngram_component
                    + self.weights["bm25"] * bm25_component
                    + self.weights["vector"] * vector_component
                    + self.weights["citation"] * citation_component
                )
                if final < 0.56 or min(match.left_edit, match.right_edit) < 0.58:
                    continue
                decision = (
                    "retrieved_exact"
                    if match.left_edit >= 0.99 and match.right_edit >= 0.99
                    else "retrieved_ambiguous"
                )
                absolute_start = chunk["char_start"] + match.gap_start
                absolute_end = chunk["char_start"] + match.gap_end
                gap = match.gap
                matched_before = match.matched_before
                matched_after = match.matched_after
            else:
                context = before or after
                localization, position = single_side_similarity(
                    chunk["normalized_text"], context
                )
                if localization < 0.72:
                    continue
                final = (
                    0.50 * localization
                    + self.weights["ngram"] * ngram_component
                    + self.weights["bm25"] * bm25_component
                    + self.weights["vector"] * vector_component
                    + self.weights["citation"] * citation_component
                )
                decision = "single_side_localization"
                absolute_start = chunk["char_start"] + max(position, 0)
                absolute_end = absolute_start
                gap = ""
                matched_before = context if before else ""
                matched_after = context if after else ""
                edit_component = localization
                continuity_component = localization
            components = {
                "ngram": round(ngram_component, 6),
                "ngram_rank": int(ngram_details.get("ngram_rank", 0)),
                "matched_grams": ngram_details.get("matched_grams", []),
                "bm25": round(bm25_component, 6),
                "bm25_raw": round(
                    float(bm25_details.get("bm25_raw_score", 0.0)), 6
                ),
                "bm25_rank": int(bm25_details.get("bm25_rank", 0)),
                "matched_terms": bm25_details.get("matched_terms", []),
                "vector": round(vector_component, 6),
                "edit": round(edit_component, 6),
                "continuity": round(continuity_component, 6),
                "citation": citation_component,
                "final": round(final, 6),
            }
            ranked.append(
                RetrievalCandidate(
                    rank=0,
                    candidate_text=gap,
                    decision_status=decision,
                    sutra_id=chunk["sutra_id"],
                    canonical_title=chunk["canonical_title"],
                    translator=chunk["translator"],
                    edition=chunk["edition"],
                    volume=chunk["volume"],
                    chapter_name=chunk["chapter_name"],
                    page_start=chunk["page_start"],
                    column_start=chunk["column_start"],
                    line_start=chunk["line_start"],
                    page_end=chunk["page_end"],
                    column_end=chunk["column_end"],
                    line_end=chunk["line_end"],
                    line_heads=chunk["line_heads"],
                    original_text=chunk["original_text"],
                    normalized_text=chunk["normalized_text"],
                    matched_before=matched_before,
                    inferred_gap=gap,
                    matched_after=matched_after,
                    retrieval_components=components,
                    source_version=chunk["source_version"],
                    source_commit=chunk["source_commit"],
                    text_hash=chunk["text_hash"],
                    chunk_id=chunk_id,
                    citation=citation_text(chunk),
                    absolute_gap_start=absolute_start,
                    absolute_gap_end=absolute_end,
                )
            )
        deduplicated: dict[tuple, RetrievalCandidate] = {}
        for candidate in ranked:
            key = (
                candidate.sutra_id,
                candidate.volume,
                candidate.chapter_name,
                candidate.absolute_gap_start,
                candidate.absolute_gap_end,
                candidate.candidate_text,
            )
            current = deduplicated.get(key)
            if (
                current is None
                or candidate.retrieval_components["final"]
                > current.retrieval_components["final"]
            ):
                deduplicated[key] = candidate
        return sorted(
            deduplicated.values(),
            key=lambda item: (
                item.retrieval_components["final"],
                item.retrieval_components["continuity"],
                item.retrieval_components["ngram"],
            ),
            reverse=True,
        )
