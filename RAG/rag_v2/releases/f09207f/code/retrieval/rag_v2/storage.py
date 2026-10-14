from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from .schema import AtomicRecord, TextChunk


SCHEMA_VERSION = "2.0.0"


class RagStorage:
    def __init__(self, path: str | Path, readonly: bool = False):
        self.path = Path(path)
        self.readonly = readonly

    @contextmanager
    def connect(self):
        if self.readonly:
            uri = f"file:{self.path.as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            if not self.readonly:
                connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS records (
                    record_id TEXT PRIMARY KEY,
                    sutra_id TEXT NOT NULL,
                    canonical_title TEXT NOT NULL,
                    translator TEXT NOT NULL,
                    edition TEXT NOT NULL,
                    volume INTEGER NOT NULL,
                    chapter_name TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    column_name TEXT NOT NULL,
                    line INTEGER NOT NULL,
                    line_head TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    simplified_alias TEXT NOT NULL,
                    normalization_profile TEXT NOT NULL,
                    normalization_version TEXT NOT NULL,
                    normalized_to_original_map TEXT NOT NULL,
                    source_version TEXT NOT NULL,
                    source_commit TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    text_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    scale TEXT NOT NULL,
                    sutra_id TEXT NOT NULL,
                    canonical_title TEXT NOT NULL,
                    translator TEXT NOT NULL,
                    edition TEXT NOT NULL,
                    volume INTEGER NOT NULL,
                    chapter_name TEXT NOT NULL,
                    record_start_id TEXT NOT NULL,
                    record_end_id TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    column_start TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    column_end TEXT NOT NULL,
                    line_end INTEGER NOT NULL,
                    line_heads TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    simplified_alias TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    source_version TEXT NOT NULL,
                    source_commit TEXT NOT NULL,
                    text_hash TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_scale ON chunks(scale);
                CREATE INDEX IF NOT EXISTS idx_chunks_filter
                    ON chunks(sutra_id, translator, volume, scale);
                CREATE TABLE IF NOT EXISTS ngram_postings (
                    n INTEGER NOT NULL,
                    gram TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    positions TEXT NOT NULL,
                    PRIMARY KEY (n, gram, chunk_id),
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
                );
                CREATE INDEX IF NOT EXISTS idx_ngram_lookup
                    ON ngram_postings(n, gram);
                CREATE TABLE IF NOT EXISTS ngram_terms (
                    n INTEGER NOT NULL,
                    gram TEXT NOT NULL,
                    document_frequency INTEGER NOT NULL,
                    PRIMARY KEY (n, gram)
                );
                CREATE TABLE IF NOT EXISTS bm25_docs (
                    chunk_id TEXT PRIMARY KEY,
                    doc_len INTEGER NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
                );
                CREATE TABLE IF NOT EXISTS bm25_postings (
                    term TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    term_frequency INTEGER NOT NULL,
                    PRIMARY KEY(term, chunk_id),
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
                );
                CREATE INDEX IF NOT EXISTS idx_bm25_lookup ON bm25_postings(term);
                CREATE TABLE IF NOT EXISTS bm25_terms (
                    term TEXT PRIMARY KEY,
                    document_frequency INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vector_map (
                    vector_id INTEGER PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    text_hash TEXT NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
                );
                """
            )
            db.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (SCHEMA_VERSION,),
            )

    def clear_corpus(self) -> None:
        with self.connect() as db:
            for table in [
                "vector_map",
                "bm25_postings",
                "bm25_terms",
                "bm25_docs",
                "ngram_postings",
                "ngram_terms",
                "chunks",
                "records",
            ]:
                db.execute(f"DELETE FROM {table}")

    def insert_records(self, records: Iterable[AtomicRecord]) -> None:
        rows = []
        for record in records:
            payload = record.to_dict()
            payload["normalized_to_original_map"] = json.dumps(
                payload["normalized_to_original_map"], ensure_ascii=False
            )
            payload["column_name"] = payload.pop("column")
            rows.append(payload)
        if not rows:
            return
        columns = list(rows[0])
        sql = (
            f"INSERT OR REPLACE INTO records({','.join(columns)}) "
            f"VALUES({','.join('?' for _ in columns)})"
        )
        with self.connect() as db:
            db.executemany(sql, [[row[column] for column in columns] for row in rows])

    def insert_chunks(self, chunks: Iterable[TextChunk]) -> None:
        rows = []
        for chunk in chunks:
            payload = chunk.to_dict()
            payload["line_heads"] = json.dumps(payload["line_heads"], ensure_ascii=False)
            rows.append(payload)
        if not rows:
            return
        columns = list(rows[0])
        sql = (
            f"INSERT OR REPLACE INTO chunks({','.join(columns)}) "
            f"VALUES({','.join('?' for _ in columns)})"
        )
        with self.connect() as db:
            db.executemany(sql, [[row[column] for column in columns] for row in rows])

    def set_meta(self, key: str, value: object) -> None:
        encoded = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, encoded))

    def get_meta(self) -> dict[str, str]:
        with self.connect() as db:
            return {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM meta")}

    def counts(self) -> dict[str, int]:
        with self.connect() as db:
            result = {
                "records": db.execute("SELECT COUNT(*) FROM records").fetchone()[0],
                "short_chunks": db.execute(
                    "SELECT COUNT(*) FROM chunks WHERE scale='short'"
                ).fetchone()[0],
                "medium_chunks": db.execute(
                    "SELECT COUNT(*) FROM chunks WHERE scale='medium'"
                ).fetchone()[0],
                "ngram_postings": db.execute(
                    "SELECT COUNT(*) FROM ngram_postings"
                ).fetchone()[0],
                "bm25_postings": db.execute(
                    "SELECT COUNT(*) FROM bm25_postings"
                ).fetchone()[0],
                "vectors": db.execute("SELECT COUNT(*) FROM vector_map").fetchone()[0],
            }
        return result
