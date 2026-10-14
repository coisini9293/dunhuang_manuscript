from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from common import (
    FULL_ARTIFACT,
    FULL_CORPORA,
    FULL_DB,
    REPORT_ROOT,
    disk_usage,
    now_iso,
    read_json,
    require_full_budget,
    process_memory,
    sha256_file,
    write_json,
)
from retrieval.rag_v2.compact_ngram import build_compact_ngram_indexes
from retrieval.rag_v2.full_storage import FullRagStorage


SOURCE_VERSION = "CBETA Normal Text 2025 R3"
SOURCE_COMMIT = "2ffcfdb5011e97637295c6589061b9779273ddda"
SHORT_SIZE = 32
SHORT_STEP = 16
MEDIUM_TARGET = 160
MEDIUM_MIN = 128
MEDIUM_MAX = 192
MEDIUM_OVERLAP = 64


def iter_records(path: Path):
    if path.suffix == ".gz":
        stream_context = gzip.open(path, mode="rt", encoding="utf-8")
    else:
        stream_context = path.open(mode="r", encoding="utf-8")
    with stream_context as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def virtual_short_count(length: int) -> int:
    if length <= SHORT_SIZE:
        return 1 if length else 0
    starts = list(range(0, max(length - SHORT_SIZE + 1, 1), SHORT_STEP))
    tail = max(0, length - SHORT_SIZE)
    return len(starts) + int(tail not in starts)


def chunk_id_for(group: dict, selected: list[dict], start: int, end: int) -> str:
    identity = (
        f"{group['sutra_id']}|{group['volume']}|{group['chapter_name']}|medium|"
        f"{selected[0]['record_id']}|{selected[-1]['record_id']}|{start}|{end}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"{group['sutra_id']}_v{group['volume']:03d}_medium_{digest}"


def medium_chunks(group: dict, records: list[dict]) -> list[dict]:
    valid = [record for record in records if record["normalized_text"]]
    prefix = [0]
    for record in valid:
        prefix.append(prefix[-1] + len(record["normalized_text"]))
    chunks: list[dict] = []
    start_index = 0
    while start_index < len(valid):
        end_index = start_index
        while (
            end_index < len(valid)
            and prefix[end_index + 1] - prefix[start_index] < MEDIUM_TARGET
        ):
            end_index += 1
        end_index = min(end_index, len(valid) - 1)
        while (
            end_index + 1 < len(valid)
            and prefix[end_index + 2] - prefix[start_index] <= MEDIUM_MAX
            and prefix[end_index + 1] - prefix[start_index] < MEDIUM_MIN
        ):
            end_index += 1
        selected = valid[start_index : end_index + 1]
        normalized = "".join(item["normalized_text"] for item in selected)
        original = "\n".join(item["original_text"] for item in selected)
        start = prefix[start_index]
        end = prefix[end_index + 1]
        chunks.append(
            {
                "chunk_id": chunk_id_for(group, selected, start, end),
                "group_id": group["group_id"],
                "sutra_id": group["sutra_id"],
                "canonical_title": group["canonical_title"],
                "translator": group["translator"],
                "edition": group["edition"],
                "volume": group["volume"],
                "chapter_name": group["chapter_name"],
                "record_start_id": selected[0]["record_id"],
                "record_end_id": selected[-1]["record_id"],
                "page_start": selected[0]["page"],
                "column_start": selected[0]["column"],
                "line_start": selected[0]["line"],
                "page_end": selected[-1]["page"],
                "column_end": selected[-1]["column"],
                "line_end": selected[-1]["line"],
                "line_heads": json.dumps(
                    [item["line_head"] for item in selected],
                    ensure_ascii=False,
                ),
                "original_text": original,
                "normalized_text": normalized,
                "char_start": start,
                "char_end": end,
                "source_version": SOURCE_VERSION,
                "source_commit": SOURCE_COMMIT,
                "text_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            }
        )
        if end_index == len(valid) - 1:
            break
        desired_next = max(prefix[start_index], prefix[end_index + 1] - MEDIUM_OVERLAP)
        next_index = start_index + 1
        while next_index < end_index and prefix[next_index] < desired_next:
            next_index += 1
        start_index = max(start_index + 1, next_index)
    return chunks


def insert_batch(db: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    db.executemany(
        f"INSERT INTO {table}({','.join(columns)}) "
        f"VALUES({','.join('?' for _ in columns)})",
        [[row[column] for column in columns] for row in rows],
    )


def build_metadata(resume: bool) -> dict:
    storage = FullRagStorage(FULL_DB)
    storage.initialize()
    meta = storage.get_meta()
    if resume and meta.get("metadata_ready") == "true":
        storage.create_runtime_indexes()
        return {"counts": storage.counts(), "resumed": True}
    registry = read_json(FULL_CORPORA.parent / "corpus_registry.json")
    started = time.perf_counter()
    with storage.connect() as db:
        progress = {}
        if resume and meta.get("metadata_progress"):
            progress = json.loads(meta["metadata_progress"])
        completed_corpora = int(progress.get("completed_corpora", 0))
        if not completed_corpora:
            db.execute("DELETE FROM vector_map")
            db.execute("DELETE FROM medium_chunks")
            db.execute("DELETE FROM records")
            db.execute("DELETE FROM groups")
            db.execute("DELETE FROM meta WHERE key='metadata_progress'")
            db.commit()
        global_cursor = int(
            db.execute("SELECT COALESCE(MAX(global_end),0) FROM groups").fetchone()[0]
        )
        short_cursor = int(
            db.execute("SELECT COALESCE(SUM(short_count),0) FROM groups").fetchone()[0]
        )
        group_id = int(
            db.execute("SELECT COALESCE(MAX(group_id),0) FROM groups").fetchone()[0]
        )
        record_key = int(
            db.execute("SELECT COALESCE(MAX(record_key),0) FROM records").fetchone()[0]
        )
        chunk_key = int(
            db.execute(
                "SELECT COALESCE(MAX(chunk_key),0) FROM medium_chunks"
            ).fetchone()[0]
        )
        for corpus_index, corpus in enumerate(registry["corpora"], 1):
            if corpus_index <= completed_corpora:
                continue
            record_path = FULL_CORPORA.parent / corpus["records_file"]
            current_key = None
            current_records: list[dict] = []

            def flush_group() -> None:
                nonlocal global_cursor, short_cursor, group_id, record_key, chunk_key
                if not current_records:
                    return
                group_id += 1
                normalized = "".join(item["normalized_text"] for item in current_records)
                first = current_records[0]
                group = {
                    "group_id": group_id,
                    "sutra_id": first["sutra_id"],
                    "canonical_title": first["canonical_title"],
                    "translator": first["translator"],
                    "edition": first["edition"],
                    "volume": int(first["volume"]),
                    "chapter_name": first.get("chapter_name") or "",
                    "global_start": global_cursor,
                    "global_end": global_cursor + len(normalized),
                    "short_id_start": short_cursor,
                    "short_count": virtual_short_count(len(normalized)),
                    "normalized_text": normalized,
                    "text_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                }
                insert_batch(db, "groups", [group])
                record_rows = []
                local_cursor = 0
                for item in current_records:
                    record_key += 1
                    length = len(item["normalized_text"])
                    record_rows.append(
                        {
                            "record_key": record_key,
                            "record_id": item["record_id"],
                            "group_id": group_id,
                            "sutra_id": item["sutra_id"],
                            "canonical_title": item["canonical_title"],
                            "translator": item["translator"],
                            "edition": item["edition"],
                            "volume": int(item["volume"]),
                            "chapter_name": item.get("chapter_name") or "",
                            "page": int(item["page"]),
                            "column_name": item["column"],
                            "line": int(item["line"]),
                            "line_head": item["line_head"],
                            "original_text": item["original_text"],
                            "normalized_text": item["normalized_text"],
                            "norm_start": local_cursor,
                            "norm_end": local_cursor + length,
                            "source_file": item["source_file"],
                            "source_sha256": item["source_sha256"],
                            "text_hash": item["text_hash"],
                        }
                    )
                    local_cursor += length
                insert_batch(db, "records", record_rows)
                chunks = medium_chunks(group, current_records)
                for chunk in chunks:
                    chunk_key += 1
                    chunk["chunk_key"] = chunk_key
                insert_batch(db, "medium_chunks", chunks)
                global_cursor += len(normalized)
                short_cursor += group["short_count"]

            for record in iter_records(record_path):
                key = (
                    record["sutra_id"],
                    int(record["volume"]),
                    record.get("chapter_name") or "",
                )
                if current_key is not None and key != current_key:
                    flush_group()
                    current_records = []
                current_key = key
                current_records.append(record)
            flush_group()
            if corpus_index % 25 == 0:
                db.execute(
                    "INSERT OR REPLACE INTO meta(key,value) "
                    "VALUES('metadata_progress',?)",
                    (
                        json.dumps(
                            {
                                "completed_corpora": corpus_index,
                                "total_corpora": len(registry["corpora"]),
                                "records": record_key,
                                "groups": group_id,
                                "medium_chunks": chunk_key,
                                "normalized_chars": global_cursor,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
                db.commit()
                print(
                    f"metadata: {corpus_index}/{len(registry['corpora'])} corpora, "
                    f"{record_key} records, {chunk_key} medium chunks",
                    flush=True,
                )
        db.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('metadata_ready','true')"
        )
        db.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('metadata_progress',?)",
            (
                json.dumps(
                    {
                        "completed_corpora": len(registry["corpora"]),
                        "total_corpora": len(registry["corpora"]),
                        "records": record_key,
                        "groups": group_id,
                        "medium_chunks": chunk_key,
                        "normalized_chars": global_cursor,
                        "complete": True,
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        db.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('corpus_version',?)",
            (registry["source_version"],),
        )
        db.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('source_commit',?)",
            (registry["source_commit"],),
        )
        db.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('sutra_count',?)",
            (str(registry["sutra_count"]),),
        )
    index_started = time.perf_counter()
    storage.create_runtime_indexes()
    with storage.connect() as db:
        medium_lengths = dict(
            db.execute(
                """
                SELECT
                    MIN(length(normalized_text)) AS minimum,
                    MAX(length(normalized_text)) AS maximum,
                    AVG(length(normalized_text)) AS average,
                    SUM(length(normalized_text)<128) AS below_128,
                    SUM(length(normalized_text)>192) AS above_192
                FROM medium_chunks
                """
            ).fetchone()
        )
    return {
        "counts": storage.counts(),
        "normalized_chars": global_cursor,
        "medium_chunk_lengths": medium_lengths,
        "seconds": round(time.perf_counter() - started, 3),
        "runtime_index_seconds": round(time.perf_counter() - index_started, 3),
        "database_bytes": FULL_DB.stat().st_size,
    }


def bm25_tokens(text: str) -> str:
    chars = list(text)
    bigrams = [text[index : index + 2] for index in range(max(0, len(text) - 1))]
    return " ".join(chars + bigrams)


def build_bm25(resume: bool) -> dict:
    storage = FullRagStorage(FULL_DB)
    meta = storage.get_meta()
    if resume and meta.get("bm25_ready") == "true":
        with storage.connect() as db:
            count = db.execute("SELECT COUNT(*) FROM bm25_fts").fetchone()[0]
        return {"documents": count, "resumed": True}
    started = time.perf_counter()
    with storage.connect() as db:
        db.execute("DROP TABLE IF EXISTS bm25_fts")
        db.execute(
            "CREATE VIRTUAL TABLE bm25_fts USING fts5("
            "tokens, content='', tokenize='unicode61')"
        )
        # Make the schema replacement durable before the bulk pass. Keeping a
        # single SELECT cursor open while inserting all documents pins the WAL
        # read snapshot and nearly doubles peak disk usage on the full corpus.
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        total = db.execute("SELECT COUNT(*) FROM medium_chunks").fetchone()[0]
        completed = 0
        last_chunk_key = 0
        page_size = 20_000
        while True:
            rows = db.execute(
                """
                SELECT chunk_key,normalized_text
                FROM medium_chunks
                WHERE chunk_key>?
                ORDER BY chunk_key
                LIMIT ?
                """,
                (last_chunk_key, page_size),
            ).fetchall()
            if not rows:
                break
            batch = [
                (row["chunk_key"], bm25_tokens(row["normalized_text"]))
                for row in rows
            ]
            db.executemany("INSERT INTO bm25_fts(rowid,tokens) VALUES(?,?)", batch)
            last_chunk_key = int(rows[-1]["chunk_key"])
            completed += len(rows)
            db.commit()
            checkpoint = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint[0] != 0:
                raise RuntimeError(f"BM25 WAL checkpoint failed: {checkpoint}")
            print(f"BM25: {completed}/{total}", flush=True)
        # FTS5's automatic merge policy keeps the index queryable without a
        # monolithic final `optimize`, whose rewrite can exceed the 35 GiB
        # task-level disk budget on this host.
        db.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('bm25_ready','true')"
        )
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    return {
        "documents": total,
        "seconds": round(time.perf_counter() - started, 3),
        "database_bytes": FULL_DB.stat().st_size,
        "tokenization": "character plus character-bigram terms",
        "fts5_optimize": False,
        "checkpoint_policy": "20,000 documents; WAL TRUNCATE",
    }


def finalize_index() -> dict:
    storage = FullRagStorage(FULL_DB)
    meta = storage.get_meta()
    required_flags = ("metadata_ready", "ngram_ready", "bm25_ready")
    missing = [key for key in required_flags if meta.get(key) != "true"]
    if missing:
        raise RuntimeError(f"Cannot finalize; incomplete stages: {missing}")
    counts = storage.counts()
    with storage.connect() as db:
        bm25_documents = int(
            db.execute("SELECT COUNT(*) FROM bm25_fts").fetchone()[0]
        )
        vector_map_rows = int(
            db.execute("SELECT COUNT(*) FROM vector_map").fetchone()[0]
        )
        duplicate_chunks = int(
            db.execute(
                "SELECT COUNT(*) FROM (SELECT chunk_id FROM medium_chunks "
                "GROUP BY chunk_id HAVING COUNT(*)>1)"
            ).fetchone()[0]
        )
        invalid_hashes = int(
            db.execute(
                "SELECT COUNT(*) FROM medium_chunks WHERE length(text_hash)<>64"
            ).fetchone()[0]
        )
    if bm25_documents != counts["medium_chunks"]:
        raise RuntimeError(
            f"BM25 integrity failure: {bm25_documents}/{counts['medium_chunks']}"
        )
    faiss_manifest_path = FULL_ARTIFACT / "faiss" / "vector_manifest.json"
    faiss_built = faiss_manifest_path.exists()
    if faiss_built and vector_map_rows != counts["medium_chunks"]:
        raise RuntimeError(
            f"Vector map integrity failure: {vector_map_rows}/{counts['medium_chunks']}"
        )
    if not faiss_built and vector_map_rows:
        raise RuntimeError("Vector map exists without a full-corpus FAISS manifest")
    if duplicate_chunks or invalid_hashes:
        raise RuntimeError(
            f"Chunk integrity failure: duplicates={duplicate_chunks}, "
            f"invalid_hashes={invalid_hashes}"
        )
    ngram: dict[str, dict] = {}
    for n in (2, 3, 4):
        manifest_path = FULL_ARTIFACT / "ngram" / f"{n}gram" / "manifest.json"
        manifest = read_json(manifest_path)
        actual_shards = sorted(manifest_path.parent.glob("*.u64"))
        if len(actual_shards) != int(manifest["shards"]):
            raise RuntimeError(f"{n}-gram shard count mismatch")
        expected = {item["shard"]: item for item in manifest["shard_statistics"]}
        for shard in actual_shards:
            item = expected.get(shard.stem)
            if (
                not item
                or shard.stat().st_size != int(item["bytes"])
                or sha256_file(shard) != item["sha256"]
            ):
                raise RuntimeError(f"{n}-gram shard integrity failure: {shard.name}")
        ngram[str(n)] = {
            "postings": int(manifest["postings"]),
            "shards": int(manifest["shards"]),
            "bytes": int(manifest["bytes"]),
            "manifest_sha256": sha256_file(manifest_path),
        }
    vector_manifest = read_json(faiss_manifest_path) if faiss_built else None
    if vector_manifest and int(vector_manifest["vectors"]) != counts["medium_chunks"]:
        raise RuntimeError("FAISS vector count does not match medium chunks")
    manifests = FULL_ARTIFACT / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        FULL_CORPORA.parent / "source" / "source_manifest.json",
        manifests / "source_manifest.json",
    )
    shutil.copy2(
        FULL_CORPORA.parent / "corpus_registry.json",
        manifests / "corpus_manifest.json",
    )
    try:
        build_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=FULL_ARTIFACT.parents[2],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        build_commit = "unavailable"
    index_version = "rag-v2-full-1.0.0"
    storage.set_meta("index_version", index_version)
    manifest = {
        "index_version": index_version,
        "source_version": meta["corpus_version"],
        "source_commit": meta["source_commit"],
        "build_commit": build_commit,
        "built_at": now_iso(),
        "counts": counts,
        "bm25": {
            "active": True,
            "documents": bm25_documents,
            "storage": "SQLite FTS5",
        },
        "ngram": ngram,
        "faiss": {
            "built": faiss_built,
            "enabled_by_default": False,
            "vectors": int(vector_manifest["vectors"]) if vector_manifest else 0,
            "index_bytes": int(vector_manifest["index_bytes"]) if vector_manifest else 0,
            "index_sha256": vector_manifest["index_sha256"] if vector_manifest else "",
            "vector_manifest_sha256": (
                sha256_file(faiss_manifest_path) if vector_manifest else ""
            ),
            "decision": (
                "optional full-corpus vector index built"
                if vector_manifest
                else "not built; T0278 ablation showed no gain and sparse evaluation governs"
            ),
        },
        "integrity": {
            "chunk_id_duplicates": duplicate_chunks,
            "invalid_text_hashes": invalid_hashes,
            "bm25_document_count_matches": True,
            "vector_map_count_matches": (
                vector_map_rows == counts["medium_chunks"]
                if faiss_built
                else vector_map_rows == 0
            ),
            "ngram_shards_verified": True,
        },
        "budget": require_full_budget(),
    }
    index_manifest_path = manifests / "index_manifest.json"
    write_json(index_manifest_path, manifest)
    manifest_hash = sha256_file(index_manifest_path)
    storage.set_meta("index_manifest_hash", manifest_hash)
    (FULL_ARTIFACT / "VERSION").write_text(index_version + "\n", encoding="utf-8")
    return {
        "status": "FULL_INDEX_PASS",
        "manifest": str(index_manifest_path),
        "manifest_sha256": manifest_hash,
        **manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("metadata", "ngram", "bm25", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    FULL_ARTIFACT.mkdir(parents=True, exist_ok=True)
    status_path = FULL_ARTIFACT / "sparse_build_status.json"
    results: dict[str, object] = (
        read_json(status_path) if args.resume and status_path.exists() else {}
    )
    results["budget_before"] = require_full_budget()
    if args.phase in ("metadata", "all"):
        results["metadata"] = build_metadata(args.resume)
        results["metadata"]["memory"] = process_memory()
        results["budget_after_metadata"] = require_full_budget()
    if args.phase in ("ngram", "all"):
        if not FULL_DB.exists():
            raise RuntimeError("Build full metadata first")
        results["ngram"] = build_compact_ngram_indexes(
            FULL_DB,
            FULL_ARTIFACT / "ngram",
            resume=args.resume,
        )
        FullRagStorage(FULL_DB).set_meta("ngram_ready", "true")
        results["ngram_memory"] = process_memory()
        results["budget_after_ngram"] = require_full_budget()
    if args.phase in ("bm25", "all"):
        results["bm25"] = build_bm25(args.resume)
        results["bm25"]["memory"] = process_memory()
        results["budget_after_bm25"] = require_full_budget()
    if args.phase in ("finalize", "all"):
        results["finalize"] = finalize_index()
        results["budget_after_finalize"] = require_full_budget()
        write_json(
            REPORT_ROOT / "full_index_statistics.json",
            results["finalize"],
        )
    results["disk"] = disk_usage()
    results["completed_at"] = now_iso()
    write_json(status_path, results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
