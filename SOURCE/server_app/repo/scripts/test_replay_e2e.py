"""Replay end-to-end validation through both backend contract and HTTP API."""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ["AUTOHDR_BACKEND"] = "replay"
os.environ["ENABLE_LEGACY_DEBUG"] = "false"
os.environ["ENABLE_LEGACY_DIFFHDR"] = "false"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from demo.backend.main import app
from integrations.autohdr.base import get_backend
from integrations.autohdr.replay_backend import REPLAY_DIR, _verify_fixture


client = TestClient(app)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


manifest = _verify_fixture()
require(manifest.get("fixture_type") == "synthetic_contract_fixture", "Wrong fixture type")
print("1. Fixture path and declared hashes verified")

backend = get_backend()
require(backend["is_mock"] is False, "Replay silently degraded to Mock")
probe = backend["detect"]()
require(probe.get("is_replay") is True and probe.get("mock") is False, "Replay markers missing")
print("2. Backend routing is Replay and cannot silently degrade to Mock")

input_bytes = (REPLAY_DIR / "input.png").read_bytes()
created = client.post(
    "/api/jobs",
    files={"image": ("replay_input.png", input_bytes, "image/png")},
)
require(created.status_code == 200, f"Upload failed: {created.status_code} {created.text}")
job = created.json()
run_id = job["run_id"]
require(job.get("backend") == "replay" and job.get("is_mock") is False, "Job backend marker is wrong")
print(f"3. Upload completed: {run_id}")

analyzed = client.post(f"/api/jobs/{run_id}/analyze")
require(analyzed.status_code == 200, f"Analyze failed: {analyzed.status_code} {analyzed.text}")
analysis = analyzed.json()
require(analysis.get("is_replay") is True and analysis.get("mock") is False, "Analyze is not marked Replay")
require(len(analysis.get("damages", [])) > 0, "Replay returned no damages")
print("4. Analyze returned verified Replay fixture data")

evidence = client.post(f"/api/jobs/{run_id}/evidence")
require(evidence.status_code == 200 and "fusions" in evidence.json(), "Evidence failed")
print("5. Evidence generated and persisted")

damage_id = analysis["damages"][0]["damage_id"]
confirmed = client.post(
    f"/api/jobs/{run_id}/confirm",
    json={
        "decisions": [
            {
                "damage_id": damage_id,
                "selected_text": "佛",
                "selected_source": "manual",
                "status": "manual_confirmed",
            }
        ]
    },
)
require(confirmed.status_code == 200 and confirmed.json().get("status") == "CONFIRMED", "Confirm failed")
print("6. Manual confirmation accepted")

restored = client.post(f"/api/jobs/{run_id}/restore")
require(restored.status_code == 200, f"Restore failed: {restored.status_code} {restored.text}")
restore_data = restored.json()
require(restore_data.get("is_replay") is True and restore_data.get("is_mock") is False, "Restore markers are wrong")
restored_url = restore_data.get("restored_url")
require(bool(restored_url), "Restore result URL missing")
artifact = client.get(restored_url)
require(artifact.status_code == 200 and len(artifact.content) > 0, "Restore result URL is invalid")
print("7. Replay restoration image URL is valid")

report = client.get(f"/api/jobs/{run_id}/report")
require(report.status_code == 200, "JSON report failed")
report_data = report.json()
require(report_data.get("backend") == "replay" and report_data.get("is_mock") is False, "Report backend marker is wrong")
require(report_data.get("is_replay") is True, "Report Replay marker missing")

download = client.get(f"/api/jobs/{run_id}/download")
require(download.status_code == 200 and download.content, "Task ZIP download failed")
with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
    names = set(archive.namelist())
    require("manifest.json" in names, "Task ZIP manifest missing")
    require("restoration/restored.png" in names, "Task ZIP restored image missing")
print("8. JSON report and task ZIP are valid")

frontend_js = (ROOT / "demo" / "frontend" / "static" / "app.js").read_text(encoding="utf-8")
require("Replay（回放）" in frontend_js, "Frontend lacks the bilingual Replay label")
require(
    'r.is_replay ? "Official' not in frontend_js,
    "Frontend incorrectly labels Replay as Official",
)
print("9. Frontend labels Replay explicitly and does not label it Official")

tamper_target = REPLAY_DIR / "analysis.json"
original = tamper_target.read_bytes()
tamper_rejected = False
try:
    tamper_target.write_bytes(original + b"\n")
    try:
        _verify_fixture()
    except ValueError as exc:
        tamper_rejected = "SHA-256 mismatch" in str(exc)
finally:
    tamper_target.write_bytes(original)
require(tamper_rejected, "Tampered fixture was not rejected")
_verify_fixture()
print("10. Tampered fixture rejected; original fixture restored and reverified")

print("=== REPLAY_PASS: full API chain validated ===")
