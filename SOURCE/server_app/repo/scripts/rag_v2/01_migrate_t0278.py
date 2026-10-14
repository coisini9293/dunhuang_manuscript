from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import tracemalloc
from pathlib import Path

from common import (
    INDEX_VERSION,
    REPORT_ROOT,
    SOURCE_CBETA,
    T0278_CORPUS,
    T0278_DB,
    now_iso,
    require_disk,
    write_json,
)

from retrieval.rag_v2.chunker import build_medium_chunks, build_short_chunks
from retrieval.rag_v2.normalize import (
    PROFILE,
    VERSION,
    normalize_with_map,
    simplified_alias,
)
from retrieval.rag_v2.schema import AtomicRecord
from retrieval.rag_v2.storage import RagStorage


CANONICAL_TITLE = "大方廣佛華嚴經"
TRANSLATOR = "佛馱跋陀羅"
EDITION = "T"


def build_record(raw: dict, manifest: dict) -> AtomicRecord:
    original = raw.get("original_text") or ""
    normalized, index_map = normalize_with_map(original)
    line_head = raw["line_head"]
    record_id = f"T0278:{line_head}"
    volume = int(raw["volume_number"])
    source_sha = manifest["volume_sha256"][str(volume)]
    return AtomicRecord(
        record_id=record_id,
        sutra_id="T0278",
        canonical_title=CANONICAL_TITLE,
        translator=TRANSLATOR,
        edition=EDITION,
        volume=volume,
        chapter_name=raw.get("chapter_name") or "",
        page=int(raw["page"]),
        column=raw["column"],
        line=int(raw["line"]),
        line_head=line_head,
        original_text=original,
        normalized_text=normalized,
        simplified_alias=simplified_alias(normalized),
        normalization_profile=PROFILE,
        normalization_version=VERSION,
        normalized_to_original_map=index_map,
        source_version=manifest["release"],
        source_commit=manifest["commit"],
        source_file=f"raw/normal_text/T0278_{volume:03d}.txt",
        source_sha256=source_sha,
        text_hash=hashlib.sha256(original.encode("utf-8")).hexdigest(),
    )


def load_records() -> tuple[list[AtomicRecord], dict]:
    manifest = json.loads((SOURCE_CBETA / "source_manifest.json").read_text(encoding="utf-8"))
    records: list[AtomicRecord] = []
    for path in sorted((SOURCE_CBETA / "parsed").glob("T0278_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(build_record(json.loads(line), manifest))
    return records, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    require_disk()
    storage = RagStorage(T0278_DB)
    storage.initialize()
    if arguments.resume and storage.get_meta().get("migration_complete") == "true":
        print("T0278 migration already complete")
        return 0
    started = time.perf_counter()
    tracemalloc.start()
    records, source_manifest = load_records()
    if len(records) != 33_658:
        raise RuntimeError(f"T0278 record count mismatch: {len(records)}")
    if any(len(record.text_hash) != 64 for record in records):
        raise RuntimeError("Found non-64-bit text hash")
    short_chunks = build_short_chunks(records)
    medium_chunks = build_medium_chunks(records)
    all_ids = [chunk.chunk_id for chunk in short_chunks + medium_chunks]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Duplicate chunk_id")
    storage.clear_corpus()
    storage.insert_records(records)
    storage.insert_chunks(short_chunks)
    storage.insert_chunks(medium_chunks)
    corpus_payload = {
        "sutra_id": "T0278",
        "canonical_title": CANONICAL_TITLE,
        "translator": TRANSLATOR,
        "edition": EDITION,
        "source_version": source_manifest["release"],
        "source_commit": source_manifest["commit"],
        "records": len(records),
        "normalized_characters": sum(len(record.normalized_text) for record in records),
        "short_chunks": len(short_chunks),
        "medium_chunks": len(medium_chunks),
        "normalization_profile": PROFILE,
        "normalization_version": VERSION,
        "built_at": now_iso(),
    }
    T0278_CORPUS.mkdir(parents=True, exist_ok=True)
    records_path = T0278_CORPUS / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    write_json(T0278_CORPUS / "corpus_manifest.json", corpus_payload)
    storage.set_meta("migration_complete", "true")
    storage.set_meta("corpus_version", source_manifest["release"])
    storage.set_meta("source_commit", source_manifest["commit"])
    storage.set_meta("sutra_count", "1")
    storage.set_meta("index_version", INDEX_VERSION)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - started
    by_volume: dict[int, dict[str, int]] = {}
    for record in records:
        row = by_volume.setdefault(record.volume, {"records": 0, "characters": 0})
        row["records"] += 1
        row["characters"] += len(record.normalized_text)
    for chunk in short_chunks + medium_chunks:
        row = by_volume[chunk.volume]
        key = f"{chunk.scale}_chunks"
        row[key] = row.get(key, 0) + 1
    statistics = {
        **corpus_payload,
        "chunk_ids_unique": True,
        "build_seconds": round(elapsed, 4),
        "peak_memory_bytes": peak,
        "peak_memory_mib": round(peak / 1024**2, 3),
        "database_bytes": T0278_DB.stat().st_size,
        "records_jsonl_bytes": records_path.stat().st_size,
        "per_volume": [
            {"volume": volume, **by_volume[volume]} for volume in sorted(by_volume)
        ],
    }
    write_json(REPORT_ROOT / "02_chunk_statistics.json", statistics)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
