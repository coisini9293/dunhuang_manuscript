from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import httpx

from common import FULL_SOURCE, PROJECT_ROOT, REPORT_ROOT, disk_usage, now_iso, write_json


REPOSITORY = "https://github.com/DILA-edu/cbeta-normal-text"
COMMIT = "2ffcfdb5011e97637295c6589061b9779273ddda"
RELEASE = "CBETA Normal Text 2025 R3"
TREE_URL = (
    "https://api.github.com/repos/DILA-edu/cbeta-normal-text/git/trees/"
    f"{COMMIT}?recursive=1"
)
ARCHIVE_URL = f"https://codeload.github.com/DILA-edu/cbeta-normal-text/zip/{COMMIT}"
LICENSE_URL = "https://cbeta.org/en/copyright"
MAX_NEW_BYTES = 35 * 1024**3
MIN_FREE_AFTER_BYTES = 8 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_tree() -> dict:
    tree_path = FULL_SOURCE / "tree_manifest.raw.json"
    if tree_path.exists():
        return json.loads(tree_path.read_text(encoding="utf-8"))
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Codex-RAG-V2-local-build",
    }
    response = httpx.get(TREE_URL, headers=headers, timeout=120)
    if response.status_code == 403:
        reset = response.headers.get("x-ratelimit-reset", "unknown")
        raise RuntimeError(f"GitHub metadata rate limited; reset={reset}")
    response.raise_for_status()
    payload = response.json()
    if payload.get("truncated"):
        raise RuntimeError("Official Git tree response was truncated")
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    tree_path.write_bytes(response.content)
    return payload


def estimate_budget(tree: dict) -> dict:
    blobs = [entry for entry in tree["tree"] if entry["type"] == "blob"]
    text_blobs = [
        entry
        for entry in blobs
        if entry["path"].lower().endswith(".txt")
    ]
    raw_bytes = sum(int(entry.get("size", 0)) for entry in text_blobs)
    # Conservative estimates for the compact full-corpus implementation:
    # compressed JSONL, SQLite atom metadata, custom delta-coded n-grams/BM25,
    # one 512-d float vector per medium chunk, and one HNSW index.
    normalized_chars = int(raw_bytes * (494_122 / 2_468_987))
    medium_chunks = max(1, int(normalized_chars / 110))
    estimates = {
        "archive_bytes": int(raw_bytes * 0.45),
        "extracted_source_bytes": raw_bytes,
        "parsed_compressed_bytes": int(raw_bytes * 1.40),
        "metadata_sqlite_bytes": int(raw_bytes * 2.50),
        "compact_sparse_indexes_bytes": int(raw_bytes * 7.00),
        "embedding_work_bytes": medium_chunks * 512 * 4,
        "faiss_hnsw_bytes": medium_chunks * (512 * 4 + 32 * 2 * 4 + 64),
        "deploy_hardlinks_incremental_bytes": 0,
    }
    peak_bytes = sum(estimates.values())
    final_bytes = peak_bytes - estimates["archive_bytes"] - estimates["embedding_work_bytes"]
    usage = disk_usage()
    allowed = (
        peak_bytes <= MAX_NEW_BYTES
        and usage["free_bytes"] - peak_bytes >= MIN_FREE_AFTER_BYTES
    )
    return {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "tree_entry_count": len(tree["tree"]),
        "blob_count": len(blobs),
        "text_file_count": len(text_blobs),
        "raw_text_bytes": raw_bytes,
        "estimated_normalized_chars": normalized_chars,
        "estimated_medium_chunks": medium_chunks,
        "components": estimates,
        "estimated_peak_new_bytes": peak_bytes,
        "estimated_peak_new_gib": round(peak_bytes / 1024**3, 3),
        "estimated_final_new_bytes": final_bytes,
        "estimated_final_new_gib": round(final_bytes / 1024**3, 3),
        "max_new_bytes": MAX_NEW_BYTES,
        "minimum_free_after_bytes": MIN_FREE_AFTER_BYTES,
        "disk_before": usage,
        "allowed": allowed,
    }


def download_archive(destination: Path) -> dict:
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        return {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "resumed": False,
        }
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    with httpx.stream(
        "GET",
        ARCHIVE_URL,
        headers=headers,
        timeout=httpx.Timeout(60, read=180),
        follow_redirects=True,
    ) as response:
        response.raise_for_status()
        if downloaded and response.status_code != 206:
            downloaded = 0
        with partial.open("ab" if downloaded else "wb") as handle:
            for block in response.iter_bytes(8 * 1024 * 1024):
                handle.write(block)
                downloaded += len(block)
                if downloaded % (128 * 1024 * 1024) < len(block):
                    print(f"archive: {downloaded} bytes", flush=True)
    partial.replace(destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "resumed": bool(headers),
    }


def extract_archive(archive: Path, expected_files: int) -> dict:
    target = FULL_SOURCE / "normal_text"
    complete = target / ".complete.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    with zipfile.ZipFile(archive) as package:
        members = [
            item
            for item in package.infolist()
            if not item.is_dir() and item.filename.lower().endswith(".txt")
        ]
        for index, member in enumerate(members, 1):
            parts = Path(member.filename).parts
            relative = Path(*parts[1:])
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with package.open(member) as source, destination.open("wb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
            files.append(
                {
                    "path": relative.as_posix(),
                    "bytes": destination.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
            if index % 1000 == 0:
                print(f"extract: {index}/{len(members)}", flush=True)
    if len(files) != expected_files:
        raise RuntimeError(f"Extracted {len(files)} text files, expected {expected_files}")
    payload = {
        "completed_at": now_iso(),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    write_json(complete, payload)
    return payload


def write_license_notice() -> None:
    path = FULL_SOURCE / "CBETA_LICENSE_NOTICE.md"
    path.write_text(
        "\n".join(
            [
                "# CBETA Copyright and Use Notice",
                "",
                f"- Official notice: {LICENSE_URL}",
                f"- Retrieved for local build: {now_iso()}",
                "- Use is limited to personal or organizational non-commercial purposes.",
                "- Commercial use requires prior permission from CBETA and the relevant source-edition rights holders.",
                "- Redistribution, quotation, or reprocessing must preserve substantive content and attach the copyright notice and version information.",
                "- Unless otherwise specified, CBETA states that the archive uses CC BY-NC-SA 3.0 Taiwan; named excluded collections retain separate rights.",
                "- This local RAG build is a non-commercial research artifact and preserves source/version/hash provenance.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    FULL_SOURCE.mkdir(parents=True, exist_ok=True)
    tree = fetch_tree()
    budget = estimate_budget(tree)
    write_json(REPORT_ROOT / "full_download_estimate.json", budget)
    print(json.dumps(budget, ensure_ascii=False, indent=2))
    if not budget["allowed"]:
        print("FULL_CORPUS_BLOCKED_BY_DISK")
        return 20
    if args.estimate_only:
        return 0
    write_license_notice()
    archive = FULL_SOURCE / f"cbeta-normal-text-{COMMIT}.zip"
    archive_info = download_archive(archive)
    extracted = extract_archive(archive, budget["text_file_count"])
    source_manifest = {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "release": RELEASE,
        "archive_url": ARCHIVE_URL,
        "archive": archive_info,
        "license_notice": "CBETA_LICENSE_NOTICE.md",
        "license_url": LICENSE_URL,
        "download_completed_at": now_iso(),
        "source_file_count": extracted["file_count"],
        "source_bytes": extracted["total_bytes"],
        "files": extracted["files"],
    }
    write_json(FULL_SOURCE / "source_manifest.json", source_manifest)
    # Archive hash remains in the manifest; remove the redundant compressed copy.
    archive.unlink()
    source_manifest["archive_removed_after_verified_extraction"] = True
    source_manifest["disk_after"] = disk_usage()
    write_json(FULL_SOURCE / "source_manifest.json", source_manifest)
    print("FULL_CBETA_DOWNLOAD_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
