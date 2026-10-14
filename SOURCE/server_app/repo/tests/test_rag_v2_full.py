from __future__ import annotations

import json
import numpy as np

from retrieval.rag_v2.compact_ngram import (
    build_compact_ngram_indexes,
    lookup_positions,
)
from retrieval.rag_v2.full_sparse_retriever import FullSparseRetriever
from retrieval.rag_v2.full_storage import FullRagStorage


def _database(tmp_path):
    path = tmp_path / "metadata.sqlite3"
    storage = FullRagStorage(path)
    storage.initialize()
    text = "甲乙丙丁戊己庚辛壬癸"
    with storage.connect() as db:
        db.execute(
            """
            INSERT INTO groups(
                group_id,sutra_id,canonical_title,translator,edition,volume,
                chapter_name,global_start,global_end,short_id_start,short_count,
                normalized_text,text_hash
            ) VALUES(1,'T0001','測試經','測試譯者','T',1,'',0,10,0,1,?,?)
            """,
            (text, "0" * 64),
        )
        db.execute(
            """
            INSERT INTO medium_chunks(
                chunk_key,chunk_id,group_id,sutra_id,canonical_title,translator,
                edition,volume,chapter_name,record_start_id,record_end_id,
                page_start,column_start,line_start,page_end,column_end,line_end,
                line_heads,original_text,normalized_text,char_start,char_end,
                source_version,source_commit,text_hash
            ) VALUES(
                1,'chunk-1',1,'T0001','測試經','測試譯者','T',1,'',
                'T01_p0001a01','T01_p0001a01',1,'a',1,1,'a',1,?,
                ?,?,0,10,'test','commit',?
            )
            """,
            (
                json.dumps(["T01_p0001a01"], ensure_ascii=False),
                text,
                text,
                "1" * 64,
            ),
        )
        db.execute(
            "CREATE VIRTUAL TABLE bm25_fts USING fts5("
            "tokens, content='', tokenize='unicode61')"
        )
        db.execute(
            "INSERT INTO bm25_fts(rowid,tokens) VALUES(1,?)",
            (" ".join(list(text) + [text[i : i + 2] for i in range(len(text) - 1)]),),
        )
    return path


def test_compact_ngram_build_and_filtered_recall(tmp_path) -> None:
    database = _database(tmp_path)
    ngram_root = tmp_path / "ngram"
    stats = build_compact_ngram_indexes(database, ngram_root)
    assert stats["2"]["postings"] == 9
    resumed = build_compact_ngram_indexes(database, ngram_root, resume=True)
    assert resumed["2"]["postings"] == stats["2"]["postings"]
    assert lookup_positions(ngram_root, "丙丁").tolist() == [2]

    retriever = FullSparseRetriever(database, ngram_root)
    result = retriever.ngram_recall(
        "甲乙丙丁",
        "",
        sutra_ids=["T0001"],
        translators=["測試譯者"],
    )
    assert "chunk-1" in result
    assert result["chunk-1"]["ngram_score"] > 0
    assert {
        int(item.split(":", 1)[0])
        for item in result["chunk-1"]["matched_grams"]
    } == {2, 3, 4}
    bm25 = retriever.bm25_recall("甲乙丙丁", "")
    assert bm25["chunk-1"]["bm25_rank"] == 1
    assert bm25["chunk-1"]["bm25_score"] > 0
    assert retriever.ngram_recall(
        "甲乙丙丁",
        "",
        sutra_ids=["T9999"],
    ) == {}
    assert retriever.ngram_recall(
        "甲乙丙丁",
        "",
        editions=["X"],
    ) == {}
    assert retriever.bm25_recall(
        "甲乙丙丁",
        "",
        editions=["X"],
    ) == {}


def test_hash_postings_are_verified_against_canonical_text(
    tmp_path,
    monkeypatch,
) -> None:
    database = _database(tmp_path)
    ngram_root = tmp_path / "ngram"
    ngram_root.mkdir()
    monkeypatch.setattr(
        "retrieval.rag_v2.full_sparse_retriever.lookup_positions",
        lambda *_args, **_kwargs: np.asarray([5], dtype=np.uint32),
    )
    retriever = FullSparseRetriever(database, ngram_root)
    assert retriever.ngram_recall("甲乙", "") == {}
