#!/usr/bin/env python3
"""Run real images through the local Mock or Replay API contract.

This is deliberately not an Official AutoHDR inference.  Replay ignores image
content and Mock produces synthetic placeholders.  The manual confirmation
below is tagged as a contract-test sentinel and must never enter a real case
registry as an expert transcription.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports" / "autohdr"
SELECTED = [
    (
        "CASE05-PREVIEW",
        ROOT
        / "data"
        / "real_cases"
        / "incoming"
        / "digital_dunhuang"
        / "0100bd6eba9c42f5b62dbcebffc928c0"
        / "original.jpg",
    ),
    (
        "EDH-DX01962-02052",
        ROOT
        / "data"
        / "real_cases"
        / "incoming"
        / "digital_dunhuang"
        / "0009fffb279e4d2fb965090b8a281db3"
        / "original.jpg",
    ),
    (
        "SCIENCEDB-S4901",
        ROOT
        / "data"
        / "raw"
        / "sciencedb"
        / "面向残片缀合的敦煌文献图像数据集"
        / "真实缀合组"
        / "韩朋赋（图4）"
        / "S.4901.jpg",
    ),
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mock", "replay"), required=True)
    args = parser.parse_args()
    os.environ["AUTOHDR_BACKEND"] = args.backend
    os.environ["ENABLE_LEGACY_DEBUG"] = "false"
    os.environ["ENABLE_LEGACY_DIFFHDR"] = "false"
    sys.path.insert(0, str(ROOT))

    from fastapi.testclient import TestClient
    from demo.backend.main import app

    client = TestClient(app)
    health = client.get("/api/system/health")
    require(health.status_code == 200, "health endpoint failed")
    require(
        health.json()["autohdr_backend"] == args.backend,
        "backend routing mismatch",
    )
    results = []
    for label, path in SELECTED:
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        created = client.post(
            "/api/jobs",
            files={"image": (path.name, data, "image/jpeg")},
        )
        require(created.status_code == 200, f"{label}: upload failed")
        job = created.json()
        require(job["input_sha256"] == sha, f"{label}: upload hash mismatch")
        run_id = job["run_id"]

        analyzed = client.post(f"/api/jobs/{run_id}/analyze")
        require(analyzed.status_code == 200, f"{label}: analyze failed")
        analysis = analyzed.json()
        require(analysis.get("damages"), f"{label}: no placeholder damage")
        overlay_url = analysis.get("overlay_url")
        require(overlay_url, f"{label}: overlay URL missing")
        overlay = client.get(overlay_url)
        require(overlay.status_code == 200 and overlay.content, "overlay missing")

        evidence = client.post(f"/api/jobs/{run_id}/evidence")
        require(evidence.status_code == 200, f"{label}: evidence endpoint failed")

        damage_id = analysis["damages"][0]["damage_id"]
        confirmed = client.post(
            f"/api/jobs/{run_id}/confirm",
            json={
                "decisions": [
                    {
                        "damage_id": damage_id,
                        "selected_text": "契约测试占位",
                        "selected_source": "manual",
                        "status": "manual_confirmed",
                        "reviewer_note": (
                            "AUTOMATED_CONTRACT_TEST_SENTINEL;"
                            "NOT_EXPERT_TRANSCRIPTION;DO_NOT_REUSE"
                        ),
                    }
                ]
            },
        )
        require(confirmed.status_code == 200, f"{label}: confirm failed")
        restored = client.post(f"/api/jobs/{run_id}/restore")
        require(restored.status_code == 200, f"{label}: restore failed")
        restored_data = restored.json()
        require(
            bool(restored_data.get("is_mock")) == (args.backend == "mock"),
            f"{label}: mode marker mismatch",
        )
        require(restored_data.get("restored_url"), f"{label}: result URL missing")
        result_image = client.get(restored_data["restored_url"])
        require(
            result_image.status_code == 200 and result_image.content,
            f"{label}: result artifact missing",
        )
        bundle = client.get(f"/api/jobs/{run_id}/download")
        require(bundle.status_code == 200 and bundle.content, "ZIP missing")
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            names = set(archive.namelist())
            require("manifest.json" in names, "ZIP manifest missing")
            require(
                any(name.startswith("restoration/") for name in names),
                "ZIP restoration artifact missing",
            )
        results.append(
            {
                "label": label,
                "run_id": run_id,
                "backend": args.backend,
                "input_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "input_sha256": sha,
                "input_bytes": len(data),
                "upload_pass": True,
                "analyze_pass": True,
                "overlay_pass": True,
                "evidence_endpoint_pass": True,
                "confirmation_type": "AUTOMATED_CONTRACT_TEST_SENTINEL",
                "restore_contract_pass": True,
                "result_download_pass": True,
                "task_zip_pass": True,
                "is_mock": restored_data.get("is_mock"),
                "is_replay": restored_data.get("is_replay", False),
                "not_official": True,
                "content_semantics": (
                    "synthetic_placeholder_on_real_input"
                    if args.backend == "mock"
                    else "synthetic_fixture_ignores_real_input"
                ),
            }
        )
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "case_count": len(results),
        "contract_pass": True,
        "official_autohdr_run": False,
        "expert_transcription_used": False,
        "warning": (
            "These are API contract checks. The confirmation text is a test sentinel, "
            "not a transcription; outputs are not restoration results."
        ),
        "results": results,
    }
    (REPORT_ROOT / f"real_case_{args.backend}_contract_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"REAL_CASE_{args.backend.upper()}_CONTRACT_PASS cases={len(results)}")


if __name__ == "__main__":
    main()
