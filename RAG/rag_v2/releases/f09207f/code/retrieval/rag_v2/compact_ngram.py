from __future__ import annotations

import json
import hashlib
import sqlite3
import time
from functools import lru_cache
from pathlib import Path

import numpy as np


HASH_MULTIPLIER = np.uint64(16_777_619)
HASH_SEED = np.uint64(2_166_136_261)
HASH_SALT = np.uint64(0x9E3779B9)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gram_hash(text: str) -> int:
    value = HASH_SEED
    for char in text:
        value = (value * HASH_MULTIPLIER + np.uint64(ord(char)) + HASH_SALT) & np.uint64(
            0xFFFFFFFF
        )
    return int(value)


def vector_hashes(codes: np.ndarray, n: int) -> np.ndarray:
    count = len(codes) - n + 1
    if count <= 0:
        return np.empty(0, dtype=np.uint32)
    values = np.full(count, HASH_SEED, dtype=np.uint64)
    for offset in range(n):
        values = (
            values * HASH_MULTIPLIER
            + codes[offset : offset + count].astype(np.uint64)
            + HASH_SALT
        ) & np.uint64(0xFFFFFFFF)
    return values.astype(np.uint32)


def _write_group(
    handles: dict[int, object],
    text: str,
    global_start: int,
    n: int,
) -> int:
    codes = np.fromiter((ord(char) for char in text), dtype=np.uint32, count=len(text))
    hashes = vector_hashes(codes, n)
    if not len(hashes):
        return 0
    positions = np.arange(
        global_start,
        global_start + len(hashes),
        dtype=np.uint64,
    )
    packed = (hashes.astype(np.uint64) << np.uint64(32)) | positions
    shards = hashes >> np.uint32(24)
    order = np.argsort(shards, kind="stable")
    sorted_shards = shards[order]
    boundaries = np.flatnonzero(np.diff(sorted_shards)) + 1
    for indexes in np.split(order, boundaries):
        shard = int(shards[indexes[0]])
        handle = handles.get(shard)
        if handle is None:
            path = Path(handles["_root"]) / f"{shard:02x}.u64"
            handle = path.open("ab")
            handles[shard] = handle
        handle.write(packed[indexes].tobytes())
    return len(packed)


def build_compact_ngram_indexes(
    database: str | Path,
    output_root: str | Path,
    resume: bool = False,
) -> dict:
    database = Path(database)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    statistics: dict[str, dict] = {}
    try:
        group_count = connection.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
        for n in (2, 3, 4):
            root = output_root / f"{n}gram"
            manifest_path = root / "manifest.json"
            if resume and manifest_path.exists():
                statistics[str(n)] = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                continue
            root.mkdir(parents=True, exist_ok=True)
            # An interrupted n-gram pass has no valid manifest. Remove only
            # that pass's partial shard files before rebuilding it; completed
            # 2/3/4-gram passes remain resumable independently.
            for partial in root.glob("*.u64"):
                partial.unlink()
            manifest_path.unlink(missing_ok=True)
            started = time.perf_counter()
            handles: dict = {"_root": str(root)}
            postings = 0
            for index, row in enumerate(
                connection.execute(
                    "SELECT global_start, normalized_text FROM groups ORDER BY group_id"
                ),
                1,
            ):
                postings += _write_group(
                    handles,
                    row["normalized_text"],
                    int(row["global_start"]),
                    n,
                )
                if index % 500 == 0:
                    print(
                        f"{n}-gram emit: {index}/{group_count} groups",
                        flush=True,
                    )
            for key, handle in list(handles.items()):
                if key != "_root":
                    handle.close()
            shard_stats = []
            for path in sorted(root.glob("*.u64")):
                values = np.memmap(path, dtype=np.uint64, mode="r+")
                values.sort()
                values.flush()
                shard_stats.append(
                    {
                        "shard": path.stem,
                        "postings": int(len(values)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
                del values
            manifest = {
                "n": n,
                "hash": "fnv-style-polynomial-32",
                "hash_multiplier": int(HASH_MULTIPLIER),
                "hash_seed": int(HASH_SEED),
                "hash_salt": int(HASH_SALT),
                "position_bits": 32,
                "postings": postings,
                "shards": len(shard_stats),
                "bytes": sum(item["bytes"] for item in shard_stats),
                "build_seconds": round(time.perf_counter() - started, 3),
                "shard_statistics": shard_stats,
                "collision_policy": "verify source characters after recall",
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            statistics[str(n)] = manifest
    finally:
        connection.close()
    return statistics


@lru_cache(maxsize=128)
def _open_shard(path_text: str) -> np.memmap:
    return np.memmap(path_text, dtype=np.uint64, mode="r")


def lookup_positions(
    output_root: str | Path,
    gram: str,
    limit: int = 5000,
) -> np.ndarray:
    n = len(gram)
    if n not in (2, 3, 4):
        return np.empty(0, dtype=np.uint32)
    value = gram_hash(gram)
    shard = value >> 24
    path = Path(output_root) / f"{n}gram" / f"{shard:02x}.u64"
    if not path.exists():
        return np.empty(0, dtype=np.uint32)
    packed = _open_shard(str(path.resolve()))
    low = np.uint64(value) << np.uint64(32)
    left = int(np.searchsorted(packed, low, side="left"))
    if value == 0xFFFFFFFF:
        right = len(packed)
    else:
        high = np.uint64(value + 1) << np.uint64(32)
        right = int(np.searchsorted(packed, high, side="left"))
    return (packed[left : min(right, left + limit)] & np.uint64(0xFFFFFFFF)).astype(
        np.uint32
    )
