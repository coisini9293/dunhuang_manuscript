from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import traceback
import uuid
from pathlib import Path

from dunhuang_backend.db import StateStore, utcnow

from . import ADAPTER_VERSION
from .common import file_sha256, load_json, write_json


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


def update_failure(
    store: StateStore, case_id: str, job_id: str, error: str
) -> None:
    detail = {
        "execution": "FAILED",
        "error": error,
        "worker_version": ADAPTER_VERSION,
        "human_review_status": "not_reviewed",
    }
    with store.transaction() as db:
        db.execute(
            """
            UPDATE restore_jobs SET status='FAILED',detail_json=?,updated_at=?
            WHERE job_id=? AND case_id=?
            """,
            (json.dumps(detail, ensure_ascii=False), utcnow(), job_id, case_id),
        )
        db.execute(
            """
            UPDATE cases SET status='RESTORE_FAILED',updated_at=?,
            error_code='OFFICIAL_RESTORE_FAILED',error_detail=?
            WHERE case_id=?
            """,
            (utcnow(), error[-2000:], case_id),
        )
        store.audit(
            db,
            case_id=case_id,
            action="official_restore_failed",
            before_state="RESTORE_RUNNING",
            after_state="RESTORE_FAILED",
            payload=detail,
            source_ip="server-worker",
            client_version=ADAPTER_VERSION,
            actor_type="system_worker",
        )


def import_outputs(
    store: StateStore, case_id: str, job_id: str, result: dict
) -> None:
    case_dir = CASE_ROOT / case_id
    restore_dir = case_dir / "restore"
    output_names = (
        "constrained_restored.png",
        "difference.png",
        "approved_mask.png",
        "comparison.png",
        "outside_mask_audit.json",
        "official_restore_worker.json",
    )
    outputs = []
    for name in output_names:
        path = restore_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"RESTORE_OUTPUT_MISSING:{name}")
        outputs.append(
            {
                "output_id": str(uuid.uuid4()),
                "relative_path": str(path.relative_to(CASE_ROOT)),
                "sha256": file_sha256(path),
            }
        )
    detail = {
        "execution": "COMPLETE",
        "worker_version": ADAPTER_VERSION,
        "plan_sha256": result["plan_sha256"],
        "approved_region_ids": result["approved_region_ids"],
        "confirmed_texts": result["confirmed_texts"],
        "rejected_region_ids": result["rejected_region_ids"],
        "abstained_region_ids": result["abstained_region_ids"],
        "official_commit": result["official_commit"],
        "gpu_peak_bytes": result["gpu_peak_bytes"],
        "patches": result["patches"],
        "compositor": result["compositor"],
        "outputs": outputs,
        "human_review_status": "not_reviewed",
    }
    with store.transaction() as db:
        for output in outputs:
            db.execute(
                "INSERT INTO outputs VALUES(?,?,?,?,?,?,?)",
                (
                    output["output_id"], case_id, job_id,
                    output["relative_path"], output["sha256"],
                    "not_reviewed", utcnow(),
                ),
            )
        db.execute(
            """
            UPDATE restore_jobs SET status='COMPLETE',detail_json=?,updated_at=?
            WHERE job_id=? AND case_id=?
            """,
            (json.dumps(detail, ensure_ascii=False), utcnow(), job_id, case_id),
        )
        db.execute(
            """
            UPDATE cases SET status='RESTORE_COMPLETE',updated_at=?,
            error_code=NULL,error_detail=NULL WHERE case_id=?
            """,
            (utcnow(), case_id),
        )
        store.audit(
            db,
            case_id=case_id,
            action="official_restore_completed",
            before_state="RESTORE_RUNNING",
            after_state="RESTORE_COMPLETE",
            payload=detail,
            source_ip="server-worker",
            client_version=ADAPTER_VERSION,
            actor_type="system_worker",
        )


def run(
    case_id: str,
    job_id: str,
    input_path: Path,
    bundle_path: Path,
    plan_path: Path,
) -> int:
    store = StateStore(STATE_DB)
    case_dir = CASE_ROOT / case_id
    lock_path = BASE / "RUNTIME" / "autohdr_gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            update_failure(store, case_id, job_id, "GPU_WORKER_LOCK_BUSY")
            return 73
        bundle = load_json(bundle_path)
        command = [
            str(AUTOHDR_PYTHON),
            "-m",
            "adapters.autohdr_official.official_restore_worker",
            "--input",
            str(input_path),
            "--bundle",
            str(bundle_path),
            "--plan",
            str(plan_path),
            "--case-dir",
            str(case_dir),
            "--official-source",
            str(AUTOHDR_SOURCE),
            "--official-runtime",
            str(AUTOHDR_RUNTIME),
            "--rag-version",
            bundle["rag_version"],
            "--cbeta-commit",
            bundle["cbeta_source_commit"],
            "--phase-file",
            str(case_dir / "restore" / "phase.json"),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repo_root)
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        try:
            write_json(
                case_dir / "restore" / "phase.json",
                {"phase": "LOADING", "updated_at": utcnow()},
            )
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(f"RESTORE_WORKER_EXIT_{completed.returncode}")
            result = load_json(case_dir / "restore" / "official_restore_worker.json")
            if result.get("status") != "OFFICIAL_RESTORE_WORKER_PASS":
                raise RuntimeError("RESTORE_WORKER_STATUS_INVALID")
            write_json(
                case_dir / "restore" / "phase.json",
                {"phase": "COMPLETE", "updated_at": utcnow()},
            )
            import_outputs(store, case_id, job_id, result)
            return 0
        except Exception as exc:
            traceback.print_exc()
            update_failure(store, case_id, job_id, str(exc))
            return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    return run(
        args.case_id,
        args.job_id,
        args.input.resolve(),
        args.bundle.resolve(),
        args.plan.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
