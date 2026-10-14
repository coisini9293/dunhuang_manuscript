from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AtomicRecord:
    record_id: str
    sutra_id: str
    canonical_title: str
    translator: str
    edition: str
    volume: int
    chapter_name: str
    page: int
    column: str
    line: int
    line_head: str
    original_text: str
    normalized_text: str
    simplified_alias: str
    normalization_profile: str
    normalization_version: str
    normalized_to_original_map: list[int]
    source_version: str
    source_commit: str
    source_file: str
    source_sha256: str
    text_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TextChunk:
    chunk_id: str
    scale: str
    sutra_id: str
    canonical_title: str
    translator: str
    edition: str
    volume: int
    chapter_name: str
    record_start_id: str
    record_end_id: str
    page_start: int
    column_start: str
    line_start: int
    page_end: int
    column_end: str
    line_end: int
    line_heads: list[str]
    original_text: str
    normalized_text: str
    simplified_alias: str
    char_start: int
    char_end: int
    source_version: str
    source_commit: str
    text_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalCandidate:
    rank: int
    candidate_text: str
    decision_status: str
    sutra_id: str
    canonical_title: str
    translator: str
    edition: str
    volume: int
    chapter_name: str
    page_start: int
    column_start: str
    line_start: int
    page_end: int
    column_end: str
    line_end: int
    line_heads: list[str]
    original_text: str
    normalized_text: str
    matched_before: str
    inferred_gap: str
    matched_after: str
    retrieval_components: dict[str, float | int | list[str]]
    source_version: str
    source_commit: str
    text_hash: str
    chunk_id: str
    citation: str
    absolute_gap_start: int = 0
    absolute_gap_end: int = 0
    evidence_original_text: str = ""
    evidence_normalized_text: str = ""
    evidence_line_heads: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueryResult:
    query_id: str
    decision_status: str
    candidates: list[RetrievalCandidate] = field(default_factory=list)
    normalized_before: str = ""
    normalized_after: str = ""
    latency_ms: float = 0.0
    vector_enabled: bool = False
    corpus_version: str = ""
    index_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "decision_status": self.decision_status,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "normalized_before": self.normalized_before,
            "normalized_after": self.normalized_after,
            "latency_ms": self.latency_ms,
            "vector_enabled": self.vector_enabled,
            "corpus_version": self.corpus_version,
            "index_version": self.index_version,
        }
