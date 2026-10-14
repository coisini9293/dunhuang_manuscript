"""任务持久化存储 — JSON + 文件锁。"""
import json, os, uuid, hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from core.state_machine import JobStatus

RUNS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


class JobStore:
    @staticmethod
    def create_job(image_name: str = "") -> dict:
        run_id = uuid.uuid4().hex[:12]
        run_dir = RUNS_ROOT / run_id
        for sub in ["input", "analysis", "evidence", "confirmation", "restoration", "logs"]:
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        manifest = {
            "run_id": run_id,
            "status": JobStatus.UPLOADED.value,
            "image_name": image_name,
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat(),
            "backend": os.environ.get("AUTOHDR_BACKEND", "mock"),
            "is_mock": os.environ.get("AUTOHDR_BACKEND", "mock") == "mock",
            "damages": [],
            "ocr_chars": [],
            "fusions": [],
            "confirmed": [],
            "restored_path": None,
            "error": None,
        }
        JobStore._save(run_id, manifest)
        return manifest

    @staticmethod
    def get_job(run_id: str) -> Optional[dict]:
        p = RUNS_ROOT / run_id / "manifest.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def update_job(run_id: str, updates: dict):
        m = JobStore.get_job(run_id)
        if m is None:
            raise ValueError(f"Job {run_id} not found")
        m.update(updates)
        m["updated"] = datetime.now(timezone.utc).isoformat()
        JobStore._save(run_id, m)

    @staticmethod
    def transition_status(run_id: str, target: JobStatus):
        from core.state_machine import transition
        m = JobStore.get_job(run_id)
        if m is None:
            raise ValueError(f"Job {run_id} not found")
        current = JobStatus(m["status"])
        new_status = transition(current, target)
        m["status"] = new_status.value
        m["updated"] = datetime.now(timezone.utc).isoformat()
        JobStore._save(run_id, m)

    @staticmethod
    def sha256_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _save(run_id: str, manifest: dict):
        p = RUNS_ROOT / run_id / "manifest.json"
        p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def list_jobs() -> list[dict]:
        jobs = []
        for d in sorted(RUNS_ROOT.iterdir(), reverse=True):
            if d.is_dir() and (d / "manifest.json").exists():
                jobs.append(json.loads((d / "manifest.json").read_text(encoding="utf-8")))
        return jobs
