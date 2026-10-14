from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


class AdapterContractError(RuntimeError):
    """Fail-closed adapter contract violation."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdapterContractError(f"Expected JSON object: {path}")
    return value


def verify_official_tree(source: Path, expected_commit: str) -> dict:
    source = source.resolve()
    if not (source / ".git").is_dir():
        raise AdapterContractError(f"Official source is not a Git worktree: {source}")

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(source), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            raise AdapterContractError(result.stderr.strip() or "Git check failed")
        return result.stdout.strip()

    actual = git("rev-parse", "HEAD")
    if actual != expected_commit:
        raise AdapterContractError(
            f"OFFICIAL_COMMIT_MISMATCH expected={expected_commit} actual={actual}"
        )
    dirty = git("status", "--porcelain=v1")
    if dirty:
        raise AdapterContractError("OFFICIAL_WORKTREE_DIRTY")
    return {"source": str(source), "commit": actual, "clean": True}


def stable_region_id(
    image_sha256: str, reading_order: int, bbox_xyxy: list[int]
) -> str:
    payload = {
        "image_sha256": image_sha256,
        "reading_order": reading_order,
        "bbox_xyxy": bbox_xyxy,
    }
    return f"region-{reading_order:04d}-{canonical_sha256(payload)[:16]}"

