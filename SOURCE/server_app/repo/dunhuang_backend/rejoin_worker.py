from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_state(path: Path, **changes) -> dict:
    current = json.loads(path.read_text(encoding="utf-8"))
    current.update(changes)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job_path = Path(args.job).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job_dir = job_path.parent
    output = job_dir / "output"
    write_state(job_path, status="RUNNING", phase="PAIRWISE_GEOMETRY", progress=10, started_at=time.time())
    base = Path(os.environ["DUNHUANG_BASE"])
    repo = base / "SOURCE/server_app/repo"
    command = [
        os.environ["AUTOHDR_PYTHON"], "-B", str(repo / "pipelines/true_stitching.py"),
        "--case-id", job["job_id"], "--fragment-a", str(job_dir / "input/fragment_a"),
        "--fragment-b", str(job_dir / "input/fragment_b"), "--output-dir", str(output),
        "--repo-root", str(repo), "--checkpoint", str(base / "MODELS/weights/rejoin.pth"), "--device", "cuda",
    ]
    try:
        write_state(job_path, status="RUNNING", phase="GPU_HOMOLOGY_AND_STITCH", progress=35)
        result = subprocess.run(command, cwd=repo, capture_output=True, text=True, env=os.environ.copy(), timeout=900)
        (job_dir / "worker.stdout.log").write_text(result.stdout, encoding="utf-8")
        (job_dir / "worker.stderr.log").write_text(result.stderr, encoding="utf-8")
        metadata_path = output / "transform_metadata.json"
        if result.returncode != 0 or not metadata_path.is_file():
            write_state(job_path, status="FAILED", phase="FAILED", progress=100, error_code="TRUE_STITCHING_FAILED", error_detail=f"worker exit {result.returncode}", finished_at=time.time())
            return 1
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("stitch_pass"):
            write_state(job_path, status="FAILED", phase="FAILED", progress=100, error_code="TRUE_STITCHING_GATE_FAILED", error_detail=json.dumps({k: metadata.get(k) for k in ("adjacency_gate", "geometry_gate", "composition_gate")}), finished_at=time.time())
            return 2
        stitched = output / "stitched.png"
        write_state(job_path, status="COMPLETE", phase="COMPLETE", progress=100, finished_at=time.time(), result={
            "stitched_sha256": sha256(stitched), "metadata": "output/transform_metadata.json",
            "pairwise_transform_source": metadata.get("PAIRWISE_TRANSFORM_SOURCE"), "reference_usage": metadata.get("REFERENCE_USAGE"),
        })
        return 0
    except Exception as exc:
        write_state(job_path, status="FAILED", phase="FAILED", progress=100, error_code="TRUE_STITCHING_EXCEPTION", error_detail=f"{type(exc).__name__}: {exc}", finished_at=time.time())
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
