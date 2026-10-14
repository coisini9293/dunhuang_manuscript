from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


FULL_SCHEMA_VERSION = "2.0.0-compact-full"


class FullRagStorage:
    def __init__(self, path: str | Path, readonly: bool = False):
        self.path = Path(path)
        self.readonly = readonly

    @contextmanager
    def connect(self):
        if self.readonly:
            connection = sqlite3.connect(
                f"file:{self.path.as_posix()}?mode=ro",
                uri=True,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-262144")
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
                    record_key INTEGER PRIMARY KEY,
                    record_id TEXT NOT NULL UNIQUE,
                    group_id INTEGER NOT NULL,
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
                    norm_start INTEGER NOT NULL,
                    norm_end INTEGER NOT NULL,
                    source_file TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    text_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS groups (
                    group_id INTEGER PRIMARY KEY,
                    sutra_id TEXT NOT NULL,
                    canonical_title TEXT NOT NULL,
                    translator TEXT NOT NULL,
                    edition TEXT NOT NULL,
                    volume INTEGER NOT NULL,
                    chapter_name TEXT NOT NULL,
                    global_start INTEGER NOT NULL,
                    global_end INTEGER NOT NULL,
                    short_id_start INTEGER NOT NULL,
                    short_count INTEGER NOT NULL,
                    normalized_text TEXT NOT NULL,
                    text_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS medium_chunks (
                    chunk_key INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE,
                    group_id INTEGER NOT NULL,
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
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    source_version TEXT NOT NULL,
                    source_commit TEXT NOT NULL,
                    text_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vector_map (
                    vector_id INTEGER PRIMARY KEY,
                    chunk_key INTEGER NOT NULL UNIQUE,
                    chunk_id TEXT NOT NULL UNIQUE,
                    text_hash TEXT NOT NULL
                );
                """
            )

    def create_runtime_indexes(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_records_group
                    ON records(group_id, norm_start);
                CREATE INDEX IF NOT EXISTS idx_records_evidence
                    ON records(sutra_id, volume, chapter_name, page, column_name, line);
                CREATE INDEX IF NOT EXISTS idx_groups_global
                    ON groups(global_start, global_end);
                CREATE INDEX IF NOT EXISTS idx_groups_filter
                    ON groups(sutra_id, translator, edition, volume);
                CREATE INDEX IF NOT EXISTS idx_medium_group
                    ON medium_chunks(group_id, char_start);
                CREATE INDEX IF NOT EXISTS idx_medium_filter
                    ON medium_chunks(sutra_id, translator, edition, volume);
                """
            )
            db.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                (FULL_SCHEMA_VERSION,),
            )

    def set_meta(self, key: str, value: object) -> None:
        import json

        encoded = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                (key, encoded),
            )

    def get_meta(self) -> dict[str, str]:
        with self.connect() as db:
            return {
                row["key"]: row["value"]
                for row in db.execute("SELECT key,value FROM meta")
            }

    def counts(self) -> dict[str, int]:
        with self.connect() as db:
            return {
                "records": db.execute("SELECT COUNT(*) FROM records").fetchone()[0],
                "groups": db.execute("SELECT COUNT(*) FROM groups").fetchone()[0],
                "medium_chunks": db.execute(
                    "SELECT COUNT(*) FROM medium_chunks"
                ).fetchone()[0],
                "virtual_short_chunks": db.execute(
                    "SELECT COALESCE(SUM(short_count),0) FROM groups"
                ).fetchone()[0],
            }
