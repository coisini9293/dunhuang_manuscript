from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

from common import FULL_CORPORA, FULL_SOURCE, REPORT_ROOT, disk_usage, now_iso, read_json, write_json
from retrieval.rag_v2.normalize import PROFILE, simplified_alias


LINE_RE = re.compile(
    r"^(?P<line_head>(?P<head>(?P<prefix>[A-Z]+)\d+n.+?)_?p"
    r"(?P<section>[a-z]?)(?P<page>\d+)(?P<column>[a-z])(?P<line>\d+))"
    r"║(?P<text>.*)$"
)
SUTRA_DIR_RE = re.compile(r"^(?P<edition>[A-Z]+)\d+n(?P<number>.+)$")
TITLE_VOLUME_RE = re.compile(
    r"^(?P<title>.+?)卷(?:第)?[一二三四五六七八九十百千〇零\d].*$"
)
TITLE_TRAILING_VOLUME_RE = re.compile(
    r"^(?P<title>.+?)[一二三四五六七八九十百千〇零\d]+卷(?P<tail>.*)$"
)
AUTHOR_MARKERS = re.compile(
    r"(?:譯|译|撰|述|編|编|集|輯|辑|錄|录|造|註|注|解|校)(?:者)?$"
)
CHAPTER_MARKERS = re.compile(r"(品第|章第|分第|會第|会第|門第|门第)")
PARSER_VERSION = "rag-v2-full-1.1.3"
FULL_NORMALIZATION_VERSION = "2.1.0-full-unicode"


def normalize_full_with_map(text: str) -> tuple[str, list[int]]:
    """Full CBETA normalization, including Romanized Sanskrit/Siddham text."""
    normalized_chars: list[str] = []
    index_map: list[int] = []
    for original_index, char in enumerate(text or ""):
        for folded_char in unicodedata.normalize("NFKC", char):
            if unicodedata.category(folded_char)[0] in {"L", "N", "M"}:
                normalized_chars.append(folded_char)
                index_map.append(original_index)
    return "".join(normalized_chars), index_map


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sutra_identity(relative: Path) -> tuple[str, str]:
    directory = relative.parts[-2]
    match = SUTRA_DIR_RE.match(directory)
    if not match:
        raise ValueError(f"Unrecognized CBETA sutra directory: {directory}")
    edition = match.group("edition")
    return f"{edition}{match.group('number')}", edition


def line_payloads(path: Path) -> list[tuple[re.Match[str], str]]:
    payloads = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        match = LINE_RE.match(raw_line)
        if not match:
            raise RuntimeError(
                f"Unrecognized official line head: {path}:{line_number}: "
                f"{raw_line[:80]!r}"
            )
        payloads.append((match, match.group("text").strip()))
    return payloads


def detect_metadata(payloads: list[tuple[re.Match[str], str]]) -> dict:
    nonempty = [text for _match, text in payloads[:40] if text]
    title = ""
    title_line = ""
    translator = ""
    for text in nonempty:
        if text.startswith("No."):
            continue
        trailing_match = TITLE_TRAILING_VOLUME_RE.match(text)
        if trailing_match:
            title = trailing_match.group("title").strip("◎＊* ")
            title_line = text
            tail = trailing_match.group("tail").strip("◎＊* ")
            if tail and AUTHOR_MARKERS.search(tail):
                translator = tail
            break
        title_match = TITLE_VOLUME_RE.match(text)
        if title_match:
            title = title_match.group("title").strip("◎＊* ")
            title_line = text
            break
    if not title:
        for text in nonempty:
            if (
                not text.startswith("No.")
                and not AUTHOR_MARKERS.search(text)
                and len(normalize_full_with_map(text)[0]) >= 2
            ):
                title = text.strip("◎＊* ")
                title_line = text
                break
    if not translator:
        for text in nonempty:
            if text != title_line and len(text) <= 80 and AUTHOR_MARKERS.search(text):
                translator = text.strip("◎＊* ")
                break
    return {
        "canonical_title": title or "題名未識別",
        "translator": translator or "不詳",
        "title_line": title_line,
    }


def parse_sutra(
    sutra_id: str,
    edition: str,
    source_files: list[Path],
    source_hashes: dict[str, str],
) -> dict:
    target = FULL_CORPORA / sutra_id
    target.mkdir(parents=True, exist_ok=True)
    record_path = target / "records.jsonl.gz"
    manifest_path = target / "corpus_manifest.json"
    if manifest_path.exists() and record_path.exists():
        existing = read_json(manifest_path)
        if existing.get("parser_version") == PARSER_VERSION:
            return existing

    titles: list[str] = []
    translators: list[str] = []
    source_entries: list[dict] = []
    record_count = 0
    source_line_count = 0
    normalized_chars = 0
    volumes: set[int] = set()
    record_ids: set[str] = set()
    started = time.perf_counter()
    with gzip.open(record_path, "wt", encoding="utf-8", compresslevel=6) as output:
        for source_path in source_files:
            relative = source_path.relative_to(FULL_SOURCE / "normal_text")
            volume = int(source_path.stem)
            volumes.add(volume)
            payloads = line_payloads(source_path)
            source_line_count += len(payloads)
            metadata = detect_metadata(payloads)
            title = metadata["canonical_title"]
            translator = metadata["translator"]
            if title not in titles:
                titles.append(title)
            if translator not in translators:
                translators.append(translator)
            chapter_name = ""
            for match, original_text in payloads:
                if not original_text:
                    continue
                if CHAPTER_MARKERS.search(original_text) and len(original_text) <= 80:
                    chapter_name = original_text.strip("◎＊* ")
                    continue
                if (
                    original_text.startswith("No.")
                    or original_text == metadata["title_line"]
                    or original_text == translator
                ):
                    continue
                normalized, index_map = normalize_full_with_map(original_text)
                if not normalized:
                    continue
                # Keep the exact official line head. Some collections use
                # `_pa`, `_pg`, etc., while appendix records can omit the
                # underscore (`T02n0128ap0835c13`). Reconstructing only `_p`
                # would collapse distinct official references.
                line_head = match.group("line_head")
                if line_head in record_ids:
                    raise RuntimeError(f"Duplicate record id in {sutra_id}: {line_head}")
                record_ids.add(line_head)
                text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                record = {
                    "record_id": line_head,
                    "sutra_id": sutra_id,
                    "canonical_title": title,
                    "translator": translator,
                    "edition": edition,
                    "volume": volume,
                    "chapter_name": chapter_name,
                    "page": int(match.group("page")),
                    "column": match.group("column"),
                    "line": int(match.group("line")),
                    "line_head": line_head,
                    "original_text": original_text,
                    "normalized_text": normalized,
                    "simplified_alias": simplified_alias(normalized),
                    "normalization_profile": PROFILE,
                    "normalization_version": FULL_NORMALIZATION_VERSION,
                    "normalized_to_original_map": index_map,
                    "source_version": "CBETA Normal Text 2025 R3",
                    "source_commit": "2ffcfdb5011e97637295c6589061b9779273ddda",
                    "source_file": relative.as_posix(),
                    "source_sha256": source_hashes[relative.as_posix()],
                    "text_hash": text_hash,
                }
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
                record_count += 1
                normalized_chars += len(normalized)
            source_entries.append(
                {
                    "path": relative.as_posix(),
                    "volume": volume,
                    "bytes": source_path.stat().st_size,
                    "sha256": source_hashes[relative.as_posix()],
                }
            )

    canonical_title = next((title for title in titles if title != "題名未識別"), titles[0])
    translator = next((value for value in translators if value != "不詳"), "不詳")
    manifest = {
        "sutra_id": sutra_id,
        "canonical_title": canonical_title,
        "titles_seen": titles,
        "translator": translator,
        "translators_seen": translators,
        "edition": edition,
        "volumes": sorted(volumes),
        "record_count": record_count,
        "source_line_count": source_line_count,
        "normalized_chars": normalized_chars,
        "records_file": "records.jsonl.gz",
        "records_sha256": sha256_file(record_path),
        "source_version": "CBETA Normal Text 2025 R3",
        "source_commit": "2ffcfdb5011e97637295c6589061b9779273ddda",
        "source_files": source_entries,
        "parser_version": PARSER_VERSION,
        "normalization_profile": PROFILE,
        "normalization_version": FULL_NORMALIZATION_VERSION,
        "parse_seconds": round(time.perf_counter() - started, 3),
        "completed_at": now_iso(),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source_manifest_path = FULL_SOURCE / "source_manifest.json"
    if not source_manifest_path.exists():
        raise RuntimeError("Run 05_download_cbeta_full.py first")
    source_manifest = read_json(source_manifest_path)
    source_hashes = {
        item["path"]: item["sha256"] for item in source_manifest["files"]
    }
    source_root = FULL_SOURCE / "normal_text"
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for source_path in sorted(source_root.rglob("*.txt")):
        relative = source_path.relative_to(source_root)
        sutra_id, edition = sutra_identity(relative)
        grouped[(sutra_id, edition)].append(source_path)

    existing_t0278 = FULL_CORPORA / "T0278" / "corpus_manifest.json"
    registry_entries: list[dict] = []
    total_records = 0
    total_chars = 0
    started = time.perf_counter()
    for index, ((sutra_id, edition), files) in enumerate(sorted(grouped.items()), 1):
        if sutra_id == "T0278" and existing_t0278.exists():
            # Preserve the already audited V2 baseline and verify every source
            # volume is byte-identical to the fixed full-corpus commit.
            baseline = read_json(existing_t0278)
            legacy_source_manifest = read_json(
                FULL_CORPORA.parents[1] / "cbeta" / "source_manifest.json"
            )
            old_sources = set(legacy_source_manifest["volume_sha256"].values())
            verified = all(
                source_hashes[
                    path.relative_to(source_root).as_posix()
                ] in old_sources
                for path in files
            )
            if not verified:
                raise RuntimeError("T0278 full source differs from audited baseline")
            manifest = {
                **baseline,
                "record_count": baseline["records"],
                "normalized_chars": baseline["normalized_characters"],
                "source_line_count": sum(
                    len(line_payloads(path)) for path in files
                ),
                "volumes": list(range(1, 61)),
                "records_file": "records.jsonl",
            }
        else:
            manifest = parse_sutra(
                sutra_id,
                edition,
                sorted(files, key=lambda path: (int(path.stem), path.as_posix())),
                source_hashes,
            )
        record_count = int(manifest["record_count"])
        chars = int(
            manifest.get("normalized_chars", manifest.get("normalized_characters", 0))
        )
        total_records += record_count
        total_chars += chars
        registry_entries.append(
            {
                "sutra_id": sutra_id,
                "canonical_title": manifest["canonical_title"],
                "translator": manifest.get("translator", "不詳"),
                "edition": edition,
                "volumes": manifest.get("volumes", []),
                "record_count": record_count,
                "normalized_chars": chars,
                "source_line_count": int(manifest.get("source_line_count", 0)),
                "records_file": (
                    f"corpora/{sutra_id}/{manifest.get('records_file', 'records.jsonl.gz')}"
                ),
                "manifest_file": f"corpora/{sutra_id}/corpus_manifest.json",
            }
        )
        if index % 50 == 0:
            write_json(
                FULL_SOURCE / "parse_progress.json",
                {
                    "completed_sutras": index,
                    "total_sutras": len(grouped),
                    "records": total_records,
                    "normalized_chars": total_chars,
                    "updated_at": now_iso(),
                },
            )
            print(
                f"parse: {index}/{len(grouped)} sutras, {total_records} records",
                flush=True,
            )

    registry = {
        "version": "cbeta-rag-v2-full-1.0.0",
        "source_version": source_manifest["release"],
        "source_commit": source_manifest["commit"],
        "sutra_count": len(registry_entries),
        "record_count": total_records,
        "normalized_chars": total_chars,
        "source_line_count": sum(
            entry["source_line_count"] for entry in registry_entries
        ),
        "corpora": registry_entries,
        "built_at": now_iso(),
    }
    write_json(FULL_CORPORA.parent / "corpus_registry.json", registry)
    statistics = {
        **{key: value for key, value in registry.items() if key != "corpora"},
        "source_file_count": source_manifest["source_file_count"],
        "source_bytes": source_manifest["source_bytes"],
        "parse_seconds": round(time.perf_counter() - started, 3),
        "disk_after_parse": disk_usage(),
        "unknown_title_count": sum(
            entry["canonical_title"] == "題名未識別" for entry in registry_entries
        ),
        "unknown_translator_count": sum(
            entry["translator"] == "不詳" for entry in registry_entries
        ),
    }
    write_json(REPORT_ROOT / "07_full_corpus_statistics.json", statistics)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
