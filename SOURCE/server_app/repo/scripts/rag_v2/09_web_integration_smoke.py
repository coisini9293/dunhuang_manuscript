from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.job_store import JobStore, RUNS_ROOT
from demo.backend.main import app
from integrations import cbeta_evidence_fusion


EXPECTED_TEXT = "眾寶雜華"
EXPECTED_SOURCE_VERSION = "CBETA Normal Text 2025 R3"
QUERY = {
    "context_before": "成正覺其地金剛具足嚴淨",
    "context_after": "以為",
    "expected_missing_length": 4,
    "sutra_ids": ["T0278"],
    "top_k": 5,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    started = datetime.now(timezone.utc)
    client = TestClient(app)
    checks: dict[str, object] = {}

    health_response = client.get("/api/rag/health")
    require(health_response.status_code == 200, "RAG health endpoint failed")
    health = health_response.json()
    require(health.get("loaded") is True, "RAG V2 was not loaded")
    require(health.get("ngram_ready") is True, "n-gram index is not ready")
    require(health.get("bm25_ready") is True, "BM25 index is not ready")
    require(int(health.get("sutra_count", 0)) >= 1, "corpus registry is empty")
    checks["health"] = health

    query_response = client.post("/api/rag/query", json=QUERY)
    require(query_response.status_code == 200, "RAG query endpoint failed")
    query = query_response.json()
    require(query.get("decision_status") == "retrieved_exact", "query was not exact")
    require(query.get("candidates"), "query returned no candidates")
    query_top = query["candidates"][0]
    require(query_top.get("candidate_text") == EXPECTED_TEXT, "wrong retrieved text")
    require(query_top.get("sutra_id") == "T0278", "wrong sutra")
    require(query_top.get("source_version") == EXPECTED_SOURCE_VERSION, "wrong corpus version")
    require(query_top.get("citation"), "citation is empty")
    require(query_top.get("retrieval_components", {}).get("ngram", 0) > 0, "n-gram inactive")
    require(query_top.get("retrieval_components", {}).get("bm25", 0) > 0, "BM25 inactive")
    checks["query"] = query

    job = JobStore.create_job("rag_v2_web_smoke.png")
    run_id = job["run_id"]
    try:
        JobStore.update_job(
            run_id,
            {
                "damages": [
                    {
                        "damage_id": "rag-v2-smoke-1",
                        "left_context": QUERY["context_before"],
                        "right_context": QUERY["context_after"],
                        "expected_missing_length": 4,
                        "sutra_ids": ["T0278"],
                        "autohdr_candidates": [],
                    }
                ]
            },
        )
        cbeta_evidence_fusion._agent = None
        evidence_response = client.post(f"/api/jobs/{run_id}/evidence")
        require(evidence_response.status_code == 200, "Evidence endpoint failed")
        evidence = evidence_response.json()
        require(len(evidence.get("fusions", [])) == 1, "Evidence fusion count mismatch")
        fusion = evidence["fusions"][0]
        require(fusion.get("status") == "cbeta_only", "CBETA-only candidate was discarded")
        require(fusion.get("selected_text") == EXPECTED_TEXT, "CBETA-only text mismatch")
        require(fusion.get("allow_restore") is False, "Evidence bypassed manual review")
        require(len(fusion.get("cbeta_candidates", [])) >= 1, "candidate details missing")
        require(len(evidence.get("citations", [])) >= 1, "top-level citations missing")
        evidence_top = fusion["cbeta_candidates"][0]
        require(evidence_top.get("source_version") == EXPECTED_SOURCE_VERSION, "fixture source detected")
        require(evidence_top.get("sutra_id") == "T0278", "non-T0278 source")
        require(len(evidence_top.get("text_hash", "")) == 64, "source text hash missing")
        checks["evidence"] = evidence
    finally:
        cbeta_evidence_fusion._agent = None
        shutil.rmtree(RUNS_ROOT / run_id, ignore_errors=True)

    finished = datetime.now(timezone.utc)
    report_path = PROJECT_ROOT / "reports" / "rag_v2" / "09_web_integration.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# RAG V2 Web Integration Smoke Test",
                "",
                f"- Result: `WEB_REAL_RAG_PASS`",
                f"- Started: `{started.isoformat()}`",
                f"- Finished: `{finished.isoformat()}`",
                f"- Corpus version: `{health['corpus_version']}`",
                f"- Sutra count: `{health['sutra_count']}`",
                f"- Index version: `{health['index_version']}`",
                f"- Index manifest SHA-256: `{health['index_manifest_hash']}`",
                (
                    "- Query source: persistent local full CBETA RAG V2 indexes"
                    if int(health["sutra_count"]) > 1
                    else "- Query source: persistent local T0278 RAG V2 SQLite indexes"
                ),
                f"- Exact candidate: `{EXPECTED_TEXT}`",
                f"- Citation: `{query_top['citation']}`",
                f"- Evidence status: `{checks['evidence']['fusions'][0]['status']}`",
                "- Manual review gate: preserved (`allow_restore=false`)",
                "- Synthetic fixture: not used",
                "",
                "## Machine-readable check snapshot",
                "",
                "```json",
                json.dumps(checks, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("WEB_REAL_RAG_PASS")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
