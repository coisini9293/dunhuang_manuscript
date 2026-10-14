from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SOURCE_CBETA = PROJECT_ROOT / "data" / "cbeta"
DATA_ROOT = PROJECT_ROOT / "data" / "cbeta_v2"
T0278_CORPUS = DATA_ROOT / "corpora" / "T0278"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "rag_v2"
T0278_ARTIFACT = ARTIFACT_ROOT / "t0278"
T0278_DB = T0278_ARTIFACT / "metadata.sqlite3"
FULL_SOURCE = DATA_ROOT / "source"
FULL_CORPORA = DATA_ROOT / "corpora"
FULL_ARTIFACT = ARTIFACT_ROOT / "full"
FULL_DB = FULL_ARTIFACT / "metadata.sqlite3"
REPORT_ROOT = PROJECT_ROOT / "reports" / "rag_v2"
CACHE_ROOT = PROJECT_ROOT / "cache" / "rag_v2"
INDEX_VERSION = "rag-v2-t0278-2.0.0"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def disk_usage() -> dict:
    usage = shutil.disk_usage(PROJECT_ROOT.anchor)
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": round(usage.free / (1024**3), 3),
    }


def require_disk(minimum_free_gib: float = 8.0) -> dict:
    usage = disk_usage()
    if usage["free_bytes"] < minimum_free_gib * 1024**3:
        raise RuntimeError(
            f"Disk hard blocker: free={usage['free_gib']} GiB < {minimum_free_gib} GiB"
        )
    return usage


def require_full_budget(
    maximum_increment_gib: float = 35.0,
    minimum_free_gib: float = 8.0,
) -> dict:
    usage = require_disk(minimum_free_gib)
    baseline_path = REPORT_ROOT / "disk_before.json"
    if not baseline_path.exists():
        raise RuntimeError("Task disk baseline is missing")
    baseline_used = int(read_json(baseline_path)["used_bytes"])
    increment = max(0, int(usage["used_bytes"]) - baseline_used)
    maximum = int(maximum_increment_gib * 1024**3)
    if increment > maximum:
        raise RuntimeError(
            "Disk hard blocker: full-corpus increment "
            f"{increment / (1024**3):.3f} GiB > {maximum_increment_gib:.3f} GiB"
        )
    return {
        **usage,
        "baseline_used_bytes": baseline_used,
        "increment_bytes": increment,
        "increment_gib": round(increment / (1024**3), 3),
        "maximum_increment_gib": maximum_increment_gib,
        "minimum_free_gib": minimum_free_gib,
    }


def process_memory() -> dict:
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = int(getattr(info, "peak_wset", info.rss))
        return {
            "rss_bytes": int(info.rss),
            "peak_working_set_bytes": peak,
            "peak_working_set_gib": round(peak / (1024**3), 3),
        }
    except (ImportError, OSError):
        return {"measurement": "unavailable"}
