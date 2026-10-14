"""Regression checks for the minimal real-case browser/API integration."""

from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ["AUTOHDR_BACKEND"] = "mock"
os.environ["ENABLE_LEGACY_DEBUG"] = "false"
os.environ["ENABLE_LEGACY_DIFFHDR"] = "false"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from demo.backend.main import app


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    client = TestClient(app)
    listing = client.get("/api/real-cases?limit=12")
    require(listing.status_code == 200, "real-case list endpoint failed")
    payload = listing.json()
    require(payload["count"] >= 3, "too few real cases exposed")
    require(
        payload["official_autohdr_status"] == "OFFICIAL_STUB",
        "Official status is inaccurate",
    )
    case = next(item for item in payload["cases"] if item["case05_candidate"])
    require(case["needs_manual_transcription"], "Case05 transcription gate missing")
    require(case["upload_ready"], "Case05 smoke input should fit upload limit")
    idp = next(
        item for item in payload["cases"]
        if item["li_shengduo_evidence_level"] == "confirmed"
    )
    require(
        idp["image_status"] == "unavailable_placeholder"
        and not idp["upload_ready"],
        "IDP placeholder was exposed as a manuscript input",
    )

    image = client.get(case["image_url"])
    require(image.status_code == 200 and len(image.content) > 0, "case image failed")
    created = client.post(f"/api/real-cases/{case['case_id']}/jobs")
    require(created.status_code == 200, "real-case job creation failed")
    job = created.json()
    require(job["real_case"]["case_id"] == case["case_id"], "source metadata lost")
    require(job["input_sha256"], "input hash missing")
    require(job["backend"] == "mock" and job["is_mock"], "mode marker lost")

    rag = client.get("/api/rag/health")
    require(rag.status_code == 200 and rag.json()["loaded"], "RAG health regressed")
    require(rag.json()["vector_enabled"] is False, "full vector index was enabled")
    root = client.get("/")
    require(root.status_code == 200, "existing root route failed")

    html = (ROOT / "demo/frontend/index.html").read_text(encoding="utf-8")
    js = (ROOT / "demo/frontend/static/app.js").read_text(encoding="utf-8")
    css = (ROOT / "demo/frontend/static/styles.css").read_text(encoding="utf-8")
    for marker in [
        "从真实案例库选择",
        "Real manuscript sources",
        'id="m2CaseGrid"',
        "Official（未运行/待部署）",
        "m2LoadRealCases",
        "m2SelectRealCase",
        "provenance-strip",
    ]:
        require(marker in html + js + css, f"frontend marker missing: {marker}")
    require("真实修复完成" not in html + js, "frontend overclaims Official output")
    print("REAL_CASE_UI_API_PASS")


if __name__ == "__main__":
    main()
