from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import os
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

from dunhuang_backend.db import StateStore, utcnow

from . import ADAPTER_VERSION, OFFICIAL_COMMIT
from .candidate_exporter import finalize_candidate_bundle
from .common import file_sha256, load_json


BASE = Path(os.environ.get("DUNHUANG_BASE", "/path/to/dunhuang-autohdr"))
CASE_ROOT = Path(os.environ.get("CASE_ROOT", BASE / "DATA/cases"))
STATE_DB = Path(os.environ.get("STATE_DB", BASE / "RAG/state/dunhuang.sqlite3"))
AUTOHDR_SOURCE = Path(os.environ.get("AUTOHDR_SOURCE", BASE / "SOURCE/src/AutoHDR"))
AUTOHDR_RUNTIME = Path(
    os.environ.get("AUTOHDR_RUNTIME", BASE / "RUNTIME/AutoHDR_official_run")
)
AUTOHDR_PYTHON = Path(
    os.environ.get("AUTOHDR_PYTHON", BASE / "RUNTIME/envs/autohdr/bin/python")
)
repo_root = Path(__file__).resolve().parents[2]


def update_failure(store: StateStore, case_id: str, run_id: str, error: str) -> None:
    detail = {
        "execution": "FAILED",
        "error": error,
        "official_commit": OFFICIAL_COMMIT,
        "worker_version": ADAPTER_VERSION,
    }
    with store.transaction() as db:
        db.execute(
            """
            UPDATE analysis_runs
            SET status='FAILED',finished_at=?,detail_json=?
            WHERE run_id=? AND case_id=?
            """,
            (utcnow(), json.dumps(detail, ensure_ascii=False), run_id, case_id),
        )
        db.execute(
            """
            UPDATE cases SET status='ANALYSIS_FAILED',updated_at=?,
            error_code='OFFICIAL_ANALYSIS_FAILED',error_detail=?
            WHERE case_id=?
            """,
            (utcnow(), error[-2000:], case_id),
        )
        store.audit(
            db,
            case_id=case_id,
            action="official_analysis_failed",
            before_state="ANALYSIS_RUNNING",
            after_state="ANALYSIS_FAILED",
            payload=detail,
            source_ip="server-worker",
            client_version=ADAPTER_VERSION,
            actor_type="system_worker",
        )


def import_bundle(
    store: StateStore,
    case_id: str,
    run_id: str,
    input_path: Path,
    bundle: dict,
) -> None:
    if bundle.get("case_id") != case_id:
        raise RuntimeError("WORKER_CASE_ID_MISMATCH")
    if bundle.get("image_sha256") != file_sha256(input_path):
        raise RuntimeError("WORKER_IMAGE_SHA256_MISMATCH")
    case_dir = CASE_ROOT / case_id
    # Worker region IDs are deterministic for an image and therefore collide
    # when the same input is retried in a fresh case. Persist a case-scoped
    # bundle/ID namespace while retaining the worker evidence and pixels.
    worker_bundle_sha256 = bundle.get("candidate_bundle_sha256")
    persisted_bundle = deepcopy(bundle)
    for region in persisted_bundle["damage_regions"]:
        region["region_id"] = f"{case_id}:{region['region_id']}"
    bundle = finalize_candidate_bundle(
        persisted_bundle, case_dir / "analysis" / "candidate_bundle.json"
    )
    original = Image.open(input_path).convert("RGB")
    overlay = original.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for region in bundle["damage_regions"]:
        x1, y1, x2, y2 = region["bbox"]
        draw.rectangle((x1, y1, x2, y2), fill=(5, 167, 188, 42), outline=(0, 111, 130, 230), width=2)
    overlay.save(case_dir / "analysis" / "overlay.png", format="PNG")
    detail = {
        "execution": "COMPLETE",
        "official_commit": OFFICIAL_COMMIT,
        "worker_version": ADAPTER_VERSION,
        "worker_bundle": bundle,
        "worker_candidate_bundle_sha256": worker_bundle_sha256,
        "persisted_candidate_bundle_sha256": bundle["candidate_bundle_sha256"],
        "region_count": len(bundle["damage_regions"]),
        "allow_restore": False,
        "final_restore_generated": False,
    }
    with store.transaction() as db:
        existing = db.execute(
            "SELECT COUNT(*) FROM damage_regions WHERE case_id=?", (case_id,)
        ).fetchone()[0]
        if existing:
            raise RuntimeError("ANALYSIS_REGIONS_ALREADY_EXIST")
        for region in bundle["damage_regions"]:
            db.execute(
                "INSERT INTO damage_regions VALUES(?,?,?,?,?)",
                (
                    region["region_id"],
                    case_id,
                    json.dumps(region["bbox"]),
                    "ANALYZED",
                    json.dumps(
                        region.get("autohdr_candidates", []), ensure_ascii=False
                    ),
                ),
            )
            db.execute(
                "INSERT INTO ocr_contexts VALUES(?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    region["region_id"],
                    region.get("ocr_before", ""),
                    region.get("ocr_after", ""),
                    region.get("ocr_raw", ""),
                    utcnow(),
                ),
            )
        db.execute(
            """
            UPDATE analysis_runs SET status='COMPLETE',finished_at=?,detail_json=?
            WHERE run_id=? AND case_id=?
            """,
            (utcnow(), json.dumps(detail, ensure_ascii=False), run_id, case_id),
        )
        db.execute(
            """
            UPDATE cases SET status='ANALYSIS_COMPLETE',updated_at=?,
            error_code=NULL,error_detail=NULL WHERE case_id=?
            """,
            (utcnow(), case_id),
        )
        store.audit(
            db,
            case_id=case_id,
            action="official_analysis_completed",
            before_state="ANALYSIS_RUNNING",
            after_state="ANALYSIS_COMPLETE",
            payload={
                "analysis_run_id": run_id,
                "worker_candidate_bundle_sha256": worker_bundle_sha256,
                "persisted_candidate_bundle_sha256": bundle[
                    "candidate_bundle_sha256"
                ],
                "region_count": len(bundle["damage_regions"]),
                "allow_restore": False,
                "final_restore_generated": False,
            },
            source_ip="server-worker",
            client_version=ADAPTER_VERSION,
            actor_type="system_worker",
        )


def run(case_id: str, run_id: str, input_path: Path) -> int:
    store = StateStore(STATE_DB)
    case_dir = CASE_ROOT / case_id
    lock_path = BASE / "RUNTIME" / "autohdr_gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            update_failure(store, case_id, run_id, "GPU_WORKER_LOCK_BUSY")
            return 73
        bundle_path = case_dir / "analysis" / "candidate_bundle.json"
        command = [
            str(AUTOHDR_PYTHON),
            "-m",
            "adapters.autohdr_official.analysis_worker",
            "--input",
            str(input_path),
            "--case-id",
            case_id,
            "--case-dir",
            str(case_dir),
            "--official-source",
            str(AUTOHDR_SOURCE),
            "--official-runtime",
            str(AUTOHDR_RUNTIME),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repo_root)
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        try:
            result = subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(f"ANALYSIS_WORKER_EXIT_{result.returncode}")
            bundle = load_json(bundle_path)
            import_bundle(store, case_id, run_id, input_path, bundle)
            return 0
        except Exception as exc:
            traceback.print_exc()
            update_failure(store, case_id, run_id, str(exc))
            return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    return run(args.case_id, args.run_id, args.input.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
