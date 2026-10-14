from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from integrations.decision_policy import decide_fusion
from retrieval.rag_v2.chunker import SHORT_SIZE, build_short_chunks
from retrieval.rag_v2.hybrid_retriever import RagV2Retriever
from retrieval.rag_v2.schema import AtomicRecord


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "artifacts" / "rag_v2" / "t0278" / "metadata.sqlite3"
MANIFEST = ROOT / "artifacts" / "rag_v2" / "t0278" / "index_manifest.json"
VECTOR_DIR = ROOT / "artifacts" / "rag_v2" / "t0278" / "faiss"


@pytest.fixture(scope="module")
def retriever() -> RagV2Retriever:
    assert DB.exists(), "build the T0278 RAG V2 database first"
    return RagV2Retriever(DB, vector_enabled=False)


def make_record(text: str) -> AtomicRecord:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return AtomicRecord(
        record_id="TTEST:001",
        sutra_id="TTEST",
        canonical_title="測試經",
        translator="測試",
        edition="T",
        volume=1,
        chapter_name="測試品",
        page=1,
        column="a",
        line=1,
        line_head="TTEST_p0001a01",
        original_text=text,
        normalized_text=text,
        simplified_alias=text,
        normalization_profile="cbeta_rag_v2",
        normalization_version="2.0.0",
        normalized_to_original_map=list(range(len(text))),
        source_version="test",
        source_commit="0" * 40,
        source_file="test.txt",
        source_sha256=digest,
        text_hash=digest,
    )


def test_tail_chunk_is_not_omitted_and_ids_hashes_are_valid() -> None:
    text = "天地玄黃宇宙洪荒日月盈昃辰宿列張寒來暑往秋收冬藏閏餘成歲律呂調陽雲騰致雨露結為霜金生麗水"
    chunks = build_short_chunks([make_record(text)])
    assert chunks[-1].normalized_text.endswith(text[-1])
    assert chunks[-1].char_end == len(text)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(len(chunk.text_hash) == 64 for chunk in chunks)
    assert all(len(chunk.normalized_text) <= SHORT_SIZE for chunk in chunks)


def test_sparse_channels_really_participate(retriever: RagV2Retriever) -> None:
    result = retriever.query(
        "成正覺其地金剛具足嚴淨",
        "以為",
        expected_missing_length=4,
        sutra_ids=["T0278"],
        top_k=5,
    )
    assert result.candidates
    components = result.candidates[0].retrieval_components
    assert components["bm25"] > 0 and components["bm25_rank"] > 0
    assert components["ngram"] > 0 and components["ngram_rank"] > 0
    gram_sizes = {int(item.split(":", 1)[0]) for item in components["matched_grams"]}
    assert {2, 3, 4}.issubset(gram_sizes)


def test_top_five_are_really_sorted_and_citations_complete(
    retriever: RagV2Retriever,
) -> None:
    result = retriever.query(
        "彼佛曾來",
        "此處是故此地",
        expected_missing_length=2,
        top_k=5,
    )
    assert len(result.candidates) == 5
    scores = [item.retrieval_components["final"] for item in result.candidates]
    assert scores == sorted(scores, reverse=True)
    for rank, candidate in enumerate(result.candidates, 1):
        assert candidate.rank == rank
        assert candidate.sutra_id == "T0278"
        assert candidate.volume > 0
        assert candidate.page_start > 0
        assert candidate.column_start in {"a", "b", "c"}
        assert candidate.line_start > 0
        assert "p0441" in candidate.citation


def test_repeated_text_preserves_multiple_citations(retriever: RagV2Retriever) -> None:
    result = retriever.query(
        "彼佛曾來",
        "此處是故此地",
        expected_missing_length=2,
        top_k=5,
    )
    texts = {candidate.candidate_text for candidate in result.candidates}
    citations = {candidate.citation for candidate in result.candidates}
    assert len(texts) == 1
    assert len(citations) >= 2


def test_single_side_localizes_without_inventing_text(
    retriever: RagV2Retriever,
) -> None:
    result = retriever.query("成正覺其地金剛具足嚴淨", "", top_k=5)
    assert result.decision_status == "single_side_localization"
    assert result.candidates
    assert all(candidate.candidate_text == "" for candidate in result.candidates)


def test_negative_query_abstains(retriever: RagV2Retriever) -> None:
    result = retriever.query(
        "甲乙丙丁戊己庚辛壬癸17",
        "天地玄黃宇宙洪荒日月17",
        expected_missing_length=4,
    )
    assert result.decision_status == "abstain"
    assert result.candidates == []


def test_single_cbeta_only_candidate_is_preserved(retriever: RagV2Retriever) -> None:
    result = retriever.query(
        "成正覺其地金剛具足嚴淨",
        "以為",
        expected_missing_length=4,
        sutra_ids=["T0278"],
        top_k=1,
    )
    candidate = result.candidates[0].to_dict()
    candidate["missing_chars"] = candidate["candidate_text"]
    fusion = decide_fusion("damage-1", [], [candidate])
    assert fusion.status == "cbeta_only"
    assert fusion.selected_text == "眾寶雜華"
    assert fusion.allow_restore is False
    assert fusion.cbeta_citation


def test_vector_feature_flag_off_and_real_index_on() -> None:
    assert (VECTOR_DIR / "index.faiss").exists()
    off = RagV2Retriever(DB, vector_dir=VECTOR_DIR, vector_enabled=False)
    assert off.vector_enabled is False
    on = RagV2Retriever(DB, vector_dir=VECTOR_DIR, vector_enabled=True)
    assert on.vector.ready is True
    assert on.vector_enabled is True
    result = on.query(
        "成正覺其地金剛具足嚴淨",
        "以為",
        expected_missing_length=4,
        top_k=1,
    )
    assert result.candidates[0].candidate_text == "眾寶雜華"
    assert result.candidates[0].retrieval_components["vector"] > 0


def test_index_manifest_integrity() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = manifest.pop("index_manifest_hash")
    actual = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert manifest["participation"]["bm25"] is True
    assert all(manifest["participation"]["ngram"].values())
