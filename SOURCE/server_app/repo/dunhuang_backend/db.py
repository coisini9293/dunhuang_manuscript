from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


SCHEMA_VERSION = 1


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def initialize(self) -> None:
        with self.transaction() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info(
                    schema_version INTEGER NOT NULL,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cases(
                    case_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    error_code TEXT,
                    error_detail TEXT
                );
                CREATE TABLE IF NOT EXISTS uploaded_files(
                    file_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    server_name TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    original_extension TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(case_id, sha256)
                );
                CREATE TABLE IF NOT EXISTS analysis_runs(
                    run_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    status TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    backend TEXT NOT NULL,
                    detail_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS damage_regions(
                    region_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    bbox_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    autohdr_candidates_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ocr_contexts(
                    context_id TEXT PRIMARY KEY,
                    region_id TEXT NOT NULL REFERENCES damage_regions(region_id),
                    context_before TEXT NOT NULL,
                    context_after TEXT NOT NULL,
                    ocr_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_queries(
                    query_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    region_id TEXT,
                    request_json TEXT NOT NULL,
                    decision_status TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    corpus_version TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_candidates(
                    query_id TEXT NOT NULL REFERENCES rag_queries(query_id),
                    rank INTEGER NOT NULL,
                    candidate_json TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    PRIMARY KEY(query_id, rank)
                );
                CREATE TABLE IF NOT EXISTS human_decisions(
                    decision_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    region_id TEXT NOT NULL REFERENCES damage_regions(region_id),
                    decision TEXT NOT NULL,
                    selected_text TEXT NOT NULL,
                    selected_source TEXT NOT NULL,
                    note TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS repair_plans(
                    plan_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    status TEXT NOT NULL,
                    allow_restore INTEGER NOT NULL CHECK(allow_restore IN (0,1)),
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS restore_jobs(
                    job_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    plan_id TEXT NOT NULL REFERENCES repair_plans(plan_id),
                    status TEXT NOT NULL,
                    detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outputs(
                    output_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    job_id TEXT NOT NULL REFERENCES restore_jobs(job_id),
                    relative_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events(
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    case_id TEXT,
                    actor_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    before_state TEXT,
                    after_state TEXT,
                    payload_sha256 TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    client_version TEXT NOT NULL,
                    detail_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_case
                    ON audit_events(case_id,event_id);
                CREATE INDEX IF NOT EXISTS idx_rag_case
                    ON rag_queries(case_id,created_at);
                """
            )
            current = db.execute(
                "SELECT MAX(schema_version) FROM schema_info"
            ).fetchone()[0]
            if current is None:
                db.execute(
                    "INSERT INTO schema_info VALUES(?,?)",
                    (SCHEMA_VERSION, utcnow()),
                )
            elif int(current) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported state schema {current}; expected {SCHEMA_VERSION}"
                )

    @staticmethod
    def audit(
        db: sqlite3.Connection,
        *,
        case_id: str | None,
        action: str,
        before_state: str | None,
        after_state: str | None,
        payload: object,
        source_ip: str,
        client_version: str,
        actor_type: str = "human_or_client",
    ) -> None:
        detail = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        db.execute(
            """
            INSERT INTO audit_events(
                timestamp,case_id,actor_type,action,before_state,after_state,
                payload_sha256,source_ip,client_version,detail_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                utcnow(),
                case_id,
                actor_type,
                action,
                before_state,
                after_state,
                hashlib.sha256(detail.encode("utf-8")).hexdigest(),
                source_ip,
                client_version,
                detail,
            ),
        )
