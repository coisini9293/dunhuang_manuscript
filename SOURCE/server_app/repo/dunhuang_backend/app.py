from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .db import SCHEMA_VERSION, StateStore, utcnow


BASE = Path(os.environ.get("DUNHUANG_BASE", "/path/to/dunhuang-autohdr"))
RAG_ROOT = Path(os.environ.get("RAG_ROOT", BASE / "RAG/rag_v2/current"))
RAG_RUNTIME = RAG_ROOT / "rag_v2_runtime"
CASE_ROOT = Path(os.environ.get("CASE_ROOT", BASE / "DATA/cases"))
STATE_DB = Path(os.environ.get("STATE_DB", BASE / "RAG/state/dunhuang.sqlite3"))
AUTOHDR_SOURCE = Path(os.environ.get("AUTOHDR_SOURCE", BASE / "SOURCE/src/AutoHDR"))
AUTOHDR_RUNTIME = Path(
    os.environ.get("AUTOHDR_RUNTIME", BASE / "RUNTIME/AutoHDR_official_run")
)
AUTOHDR_PYTHON = Path(
    os.environ.get("AUTOHDR_PYTHON", BASE / "RUNTIME/envs/autohdr/bin/python")
)
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", 100_000_000))
CLIENT_VERSION_DEFAULT = "thin-client-unknown"
ALLOWED_MIME = {"image/png", "image/jpeg", "image/tiff", "image/webp"}
ALLOWED_FORMAT = {"PNG", "JPEG", "TIFF", "WEBP"}
EXTENSIONS = {"PNG": ".png", "JPEG": ".jpg", "TIFF": ".tif", "WEBP": ".webp"}

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

store = StateStore(STATE_DB)
store.initialize()
CASE_ROOT.mkdir(parents=True, exist_ok=True)
REJOIN_ROOT = CASE_ROOT / "rejoin_jobs"
REJOIN_ROOT.mkdir(parents=True, exist_ok=True)
_rag = None
_rag_lock = Lock()

app = FastAPI(title="Dunhuang RAG + AutoHDR Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Client-Version"],
)


def client_meta(request: Request) -> tuple[str, str]:
    ip = request.client.host if request.client else "unknown"
    version = request.headers.get("x-client-version", CLIENT_VERSION_DEFAULT)[:128]
    return ip, version


def row_dict(row) -> dict:
    return dict(row) if row is not None else {}


def get_case(case_id: str) -> dict:
    with store.connect() as db:
        row = db.execute(
            "SELECT * FROM cases WHERE case_id=?", (case_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Case not found")
    return row_dict(row)


def set_status(
    db, case_id: str, status: str, request: Request, action: str, payload: object
) -> None:
    row = db.execute(
        "SELECT status FROM cases WHERE case_id=?", (case_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Case not found")
    before = row["status"]
    db.execute(
        "UPDATE cases SET status=?,updated_at=? WHERE case_id=?",
        (status, utcnow(), case_id),
    )
    ip, version = client_meta(request)
    store.audit(
        db,
        case_id=case_id,
        action=action,
        before_state=before,
        after_state=status,
        payload=payload,
        source_ip=ip,
        client_version=version,
    )


def gpu_status() -> dict:
    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        visible = probe.returncode == 0 and bool(probe.stdout.strip())
        return {
            "available": visible,
            "detail": probe.stdout.strip() if visible else "GPU_NOT_VISIBLE",
        }
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "detail": "GPU_NOT_VISIBLE"}


def rag():
    global _rag
    if _rag is None:
        with _rag_lock:
            if _rag is None:
                from retrieval.rag_v2.full_retriever import FullRagV2Retriever

                _rag = FullRagV2Retriever(
                    RAG_RUNTIME / "metadata.sqlite3",
                    RAG_RUNTIME / "ngram",
                    vector_dir=RAG_RUNTIME / "faiss",
                    vector_enabled=False,
                )
    return _rag


def latest_plan(case_id: str) -> dict | None:
    with store.connect() as db:
        row = db.execute(
            """
            SELECT * FROM repair_plans WHERE case_id=?
            ORDER BY created_at DESC LIMIT 1
            """,
            (case_id,),
        ).fetchone()
    if row is None:
        return None
    result = row_dict(row)
    result["allow_restore"] = bool(result["allow_restore"])
    result["plan"] = json.loads(result.pop("plan_json"))
    return result


def canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_candidate_bundle(case_id: str) -> dict:
    case = get_case(case_id)
    with store.connect() as db:
        analysis = db.execute(
            """
            SELECT * FROM analysis_runs WHERE case_id=?
            ORDER BY rowid DESC LIMIT 1
            """,
            (case_id,),
        ).fetchone()
        analysis_detail = (
            json.loads(analysis["detail_json"]) if analysis is not None else {}
        )
        worker_bundle = analysis_detail.get("worker_bundle", {})
        worker_regions = {
            item["region_id"]: item
            for item in worker_bundle.get("damage_regions", [])
            if isinstance(item, dict) and item.get("region_id")
        }
        regions = []
        for region in db.execute(
            "SELECT * FROM damage_regions WHERE case_id=? ORDER BY region_id",
            (case_id,),
        ):
            item = row_dict(region)
            item["bbox"] = json.loads(item.pop("bbox_json"))
            item["autohdr_candidates"] = json.loads(
                item.pop("autohdr_candidates_json")
            )
            worker_region = worker_regions.get(item["region_id"], {})
            for field in (
                "crop_path",
                "mask_path",
                "ocr_before",
                "ocr_after",
                "ocr_raw",
                "official_placeholder",
                "autohdr_candidate",
                "autohdr_confidence",
                "reading_order",
            ):
                if field in worker_region:
                    item[field] = worker_region[field]
            item["crop_url"] = (
                f"/api/cases/{case_id}/regions/{item['region_id']}/crop"
                if worker_region.get("crop_path")
                else None
            )
            item["mask_url"] = (
                f"/api/cases/{case_id}/regions/{item['region_id']}/mask"
                if worker_region.get("mask_path")
                else None
            )
            item["ocr_contexts"] = [
                row_dict(row)
                for row in db.execute(
                    "SELECT * FROM ocr_contexts WHERE region_id=? ORDER BY created_at",
                    (item["region_id"],),
                )
            ]
            queries = []
            for query in db.execute(
                "SELECT * FROM rag_queries WHERE region_id=? ORDER BY created_at",
                (item["region_id"],),
            ):
                query_item = row_dict(query)
                query_item["request"] = json.loads(
                    query_item.pop("request_json")
                )
                query_item["candidates"] = []
                for candidate in db.execute(
                    """
                    SELECT rank,candidate_json,text_hash FROM rag_candidates
                    WHERE query_id=? ORDER BY rank
                    """,
                    (query_item["query_id"],),
                ):
                    candidate_item = json.loads(candidate["candidate_json"])
                    candidate_item["candidate_id"] = (
                        f"{query_item['query_id']}:{candidate['rank']}"
                    )
                    candidate_item["citation_hash"] = canonical_hash(
                        {
                            "citation": candidate_item.get("citation", ""),
                            "text_hash": candidate["text_hash"],
                        }
                    )
                    query_item["candidates"].append(candidate_item)
                queries.append(query_item)
            item["rag_queries"] = queries
            regions.append(item)
    core = {
        "case_id": case_id,
        "image_sha256": case["input_sha256"],
        "analysis_run_id": analysis["run_id"] if analysis else None,
        "analysis_status": analysis["status"] if analysis else "NOT_STARTED",
        "worker_candidate_bundle_sha256": worker_bundle.get(
            "candidate_bundle_sha256"
        ),
        "rag_version": rag().health()["index_version"],
        "cbeta_source_commit": rag().health()["source_commit"],
        "regions": regions,
    }
    return {
        **core,
        "candidate_bundle_sha256": canonical_hash(core),
        "allow_restore": False,
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "dunhuang-backend",
        "schema_version": SCHEMA_VERSION,
        "host": "127.0.0.1",
        "manual_confirmed": False,
        "allow_restore": False,
    }


@app.get("/api/system/status")
def system_status() -> dict:
    gpu = gpu_status()
    return {
        "status": "ok",
        "gpu": gpu,
        "paths": {
            "rag_root": str(RAG_ROOT),
            "case_root": str(CASE_ROOT),
            "state_db": str(STATE_DB),
            "autohdr_source_exists": AUTOHDR_SOURCE.is_dir(),
            "autohdr_runtime_exists": AUTOHDR_RUNTIME.is_dir(),
            "autohdr_python_exists": AUTOHDR_PYTHON.is_file(),
        },
        "autohdr_execution": (
            "GPU_REQUIRED" if not gpu["available"] else "READY_FOR_ADAPTER_AUDIT"
        ),
    }


@app.get("/api/system/artifacts/{artifact_id}")
def system_artifact(artifact_id: str):
    allowed = {
        "human-gate-contract": (
            BASE / "RESTORATION/contracts/human_gate_contract.json",
            "human_gate_contract.json",
            "application/json",
        ),
    }
    if artifact_id not in allowed:
        raise HTTPException(404, "Artifact not found")
    path, filename, media_type = allowed[artifact_id]
    if not path.is_file():
        raise HTTPException(404, "Artifact is not ready")
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/api/rag/health")
def rag_health() -> dict:
    return rag().health()


@app.get("/api/rag/stats")
def rag_stats() -> dict:
    engine = rag()
    return {**engine.health(), "weights": engine.weights}


@app.post("/api/cases")
async def create_case(request: Request, image: UploadFile = File(...), source_rejoin_job_id: str = Form(""), source_rejoin_sha256: str = Form("")) -> dict:
    if image.content_type not in ALLOWED_MIME:
        raise HTTPException(415, "Unsupported image MIME type")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is empty or exceeds upload limit")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = (probe.format or "").upper()
            width, height = probe.size
            probe.verify()
        if image_format not in ALLOWED_FORMAT:
            raise HTTPException(415, "Unsupported decoded image format")
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise HTTPException(413, "Image pixel count exceeds limit")
        with Image.open(io.BytesIO(data)) as decoded:
            decoded.load()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Image decode failed: {type(exc).__name__}")

    sha = hashlib.sha256(data).hexdigest()
    stitch_provenance = None
    if source_rejoin_job_id or source_rejoin_sha256:
        if not source_rejoin_job_id or not source_rejoin_sha256:
            raise HTTPException(422, "Stitch provenance requires both job ID and SHA-256")
        _, rejoin_job = _read_rejoin_job(source_rejoin_job_id)
        if rejoin_job.get("status") != "COMPLETE":
            raise HTTPException(409, "SOURCE_REJOIN_JOB_NOT_COMPLETE")
        expected_sha = rejoin_job.get("result", {}).get("stitched_sha256", "")
        if expected_sha != source_rejoin_sha256 or expected_sha != sha:
            raise HTTPException(409, "STITCH_RESTORATION_SHA_PROVENANCE_MISMATCH")
        stitch_provenance = {
            "rejoin_job_id": source_rejoin_job_id,
            "stitched_sha256": expected_sha,
            "verified_server_side": True,
        }
    with store.connect() as db:
        duplicate = db.execute(
            """
            SELECT c.*,f.file_id FROM cases c
            JOIN uploaded_files f USING(case_id)
            WHERE f.sha256=? ORDER BY c.created_at LIMIT 1
            """,
            (sha,),
        ).fetchone()
    # A failed analysis is retryable in a fresh case. The official worker
    # intentionally rejects a case whose damage_regions already exist, so
    # deduplicating an ANALYSIS_FAILED case would make every retry fail with
    # ANALYSIS_REGIONS_ALREADY_EXIST while also hiding a new validation run.
    if duplicate is not None and duplicate["status"] != "ANALYSIS_FAILED":
        return {**row_dict(duplicate), "deduplicated": True}

    case_id = str(uuid.uuid4())
    file_id = str(uuid.uuid4())
    server_name = uuid.uuid4().hex + EXTENSIONS[image_format]
    case_dir = CASE_ROOT / case_id / "input"
    case_dir.mkdir(parents=True, exist_ok=False)
    destination = case_dir / server_name
    destination.write_bytes(data)
    relative = destination.relative_to(CASE_ROOT).as_posix()
    now = utcnow()
    original_suffix = Path(image.filename or "").suffix.lower()[:16]
    payload = {"sha256": sha, "size_bytes": len(data), "width": width, "height": height, "stitch_provenance": stitch_provenance}
    ip, version = client_meta(request)
    with store.transaction() as db:
        db.execute(
            """
            INSERT INTO cases VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                case_id, "UPLOADED", now, now, Path(image.filename or "upload").name,
                sha, image.content_type, width, height, None, None,
            ),
        )
        db.execute(
            """
            INSERT INTO uploaded_files VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                file_id, case_id, server_name, relative, original_suffix,
                image.content_type, len(data), sha, now,
            ),
        )
        store.audit(
            db, case_id=case_id, action="case_uploaded", before_state=None,
            after_state="UPLOADED", payload=payload, source_ip=ip,
            client_version=version,
        )
    return {**get_case(case_id), "file_id": file_id, "deduplicated": False}


@app.get("/api/cases/{case_id}")
def case_detail(case_id: str) -> dict:
    case = get_case(case_id)
    with store.connect() as db:
        files = [
            row_dict(row) for row in db.execute(
                "SELECT * FROM uploaded_files WHERE case_id=?", (case_id,)
            )
        ]
    return {**case, "files": files, "repair_plan": latest_plan(case_id)}


@app.get("/api/cases/{case_id}/audit")
def case_audit(case_id: str) -> dict:
    get_case(case_id)
    with store.connect() as db:
        rows = [
            row_dict(row) for row in db.execute(
                "SELECT * FROM audit_events WHERE case_id=? ORDER BY event_id",
                (case_id,),
            )
        ]
    return {"case_id": case_id, "events": rows}


@app.post("/api/cases/{case_id}/analyze")
def analyze(case_id: str, request: Request) -> dict:
    get_case(case_id)
    gpu = gpu_status()
    if not gpu["available"]:
        raise HTTPException(503, "GPU_REQUIRED")
    with store.connect() as db:
        active = db.execute(
            "SELECT run_id,case_id FROM analysis_runs WHERE status='RUNNING' LIMIT 1"
        ).fetchone()
        upload = db.execute(
            """
            SELECT relative_path FROM uploaded_files
            WHERE case_id=? ORDER BY created_at LIMIT 1
            """,
            (case_id,),
        ).fetchone()
    if active is not None:
        raise HTTPException(
            409, f"GPU_ANALYSIS_ALREADY_RUNNING:{active['case_id']}:{active['run_id']}"
        )
    if upload is None:
        raise HTTPException(409, "CASE_INPUT_NOT_FOUND")
    input_path = (CASE_ROOT / upload["relative_path"]).resolve()
    try:
        input_path.relative_to(CASE_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(500, "Stored input path escaped case root") from exc
    if not input_path.is_file():
        raise HTTPException(409, "CASE_INPUT_FILE_MISSING")
    run_id = str(uuid.uuid4())
    now = utcnow()
    case_dir = CASE_ROOT / case_id
    log_dir = case_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"analysis-{run_id}.stdout.log"
    stderr_path = log_dir / f"analysis-{run_id}.stderr.log"
    detail = {
        "execution": "RUNNING",
        "official_source_modified": False,
        "allow_restore": False,
        "stdout_path": str(stdout_path.relative_to(case_dir)),
        "stderr_path": str(stderr_path.relative_to(case_dir)),
    }
    with store.transaction() as db:
        db.execute(
            "INSERT INTO analysis_runs VALUES(?,?,?,?,?,?,?)",
            (
                run_id,
                case_id,
                "RUNNING",
                now,
                None,
                "official_external_adapter",
                json.dumps(detail),
            ),
        )
        set_status(
            db, case_id, "ANALYSIS_RUNNING", request, "analysis_requested", detail
        )
    command = [
        sys.executable,
        "-m",
        "adapters.autohdr_official.analysis_job",
        "--case-id",
        case_id,
        "--run-id",
        run_id,
        "--input",
        str(input_path),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        worker = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    return {
        "run_id": run_id,
        "case_id": case_id,
        **detail,
        "worker_pid": worker.pid,
    }


def worker_region_artifact(case_id: str, region_id: str, kind: str) -> Path:
    if kind not in {"crop", "mask"}:
        raise HTTPException(404, "Unknown region artifact")
    bundle = build_candidate_bundle(case_id)
    region = next(
        (item for item in bundle["regions"] if item["region_id"] == region_id),
        None,
    )
    if region is None:
        raise HTTPException(404, "Damage region not found")
    relative = region.get(f"{kind}_path")
    if not relative:
        raise HTTPException(404, "Region artifact not available")
    case_dir = (CASE_ROOT / case_id).resolve()
    path = (case_dir / relative).resolve()
    try:
        path.relative_to(case_dir)
    except ValueError as exc:
        raise HTTPException(403, "Artifact path escaped case root") from exc
    if not path.is_file():
        raise HTTPException(404, "Region artifact file missing")
    return path


@app.get("/api/cases/{case_id}/regions/{region_id}/{kind}")
def region_artifact(case_id: str, region_id: str, kind: str):
    get_case(case_id)
    path = worker_region_artifact(case_id, region_id, kind)
    return FileResponse(path, media_type="image/png", filename=path.name)


@app.get("/api/cases/{case_id}/analysis-overlay")
def analysis_overlay(case_id: str):
    get_case(case_id)
    path = (CASE_ROOT / case_id / "analysis" / "overlay.png").resolve()
    case_dir = (CASE_ROOT / case_id).resolve()
    try:
        path.relative_to(case_dir)
    except ValueError as exc:
        raise HTTPException(403, "Overlay path escaped case root") from exc
    if not path.is_file():
        raise HTTPException(404, "Analysis overlay not available")
    return FileResponse(path, media_type="image/png", filename="overlay.png")


class RagSearch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    region_id: str | None = None
    context_before: str = ""
    context_after: str = ""
    sutra_ids: list[str] | None = None
    translators: list[str] | None = None
    editions: list[str] | None = None
    max_missing: int | None = 10
    expected_missing_length: int | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: Literal["baseline", "noise_aware"] = "baseline"


@app.post("/api/cases/{case_id}/rag/search")
def rag_search(case_id: str, body: RagSearch, request: Request) -> dict:
    get_case(case_id)
    if not body.context_before and not body.context_after:
        raise HTTPException(422, "At least one context side is required")
    if body.region_id:
        with store.connect() as db:
            known = db.execute(
                "SELECT 1 FROM damage_regions WHERE region_id=? AND case_id=?",
                (body.region_id, case_id),
            ).fetchone()
        if known is None:
            raise HTTPException(404, "Damage region not found")
    from retrieval.noise_aware import NoiseAwareRetriever

    result = NoiseAwareRetriever(rag()).query(
        body.context_before,
        body.context_after,
        retrieval_mode=body.retrieval_mode,
        sutra_ids=body.sutra_ids,
        translators=body.translators,
        editions=body.editions,
        max_missing=body.max_missing,
        expected_missing_length=body.expected_missing_length,
        top_k=body.top_k,
    )
    engine_query_id = result["query_id"]
    query_id = uuid.uuid4().hex
    result["engine_query_id"] = engine_query_id
    result["query_id"] = query_id
    with store.transaction() as db:
        db.execute(
            """
            INSERT INTO rag_queries VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                query_id, case_id, body.region_id,
                body.model_dump_json(), result["decision_status"],
                result["latency_ms"], result["corpus_version"],
                result["index_version"], utcnow(),
            ),
        )
        for candidate in result["candidates"]:
            db.execute(
                "INSERT INTO rag_candidates VALUES(?,?,?,?)",
                (
                    query_id, candidate["rank"],
                    json.dumps(candidate, ensure_ascii=False),
                    candidate["text_hash"],
                ),
            )
        set_status(
            db, case_id, "HUMAN_REVIEW_PENDING", request, "rag_search_completed",
            {"query_id": query_id, "candidate_count": len(result["candidates"])},
        )
    return {**result, "case_id": case_id, "region_id": body.region_id, "allow_restore": False}


@app.get("/api/cases/{case_id}/evidence")
def evidence(case_id: str) -> dict:
    get_case(case_id)
    with store.connect() as db:
        queries = []
        for query in db.execute(
            "SELECT * FROM rag_queries WHERE case_id=? ORDER BY created_at",
            (case_id,),
        ):
            item = row_dict(query)
            item["request"] = json.loads(item.pop("request_json"))
            item["candidates"] = [
                json.loads(row["candidate_json"])
                for row in db.execute(
                    "SELECT candidate_json FROM rag_candidates WHERE query_id=? ORDER BY rank",
                    (item["query_id"],),
                )
            ]
            queries.append(item)
    return {"case_id": case_id, "queries": queries, "allow_restore": False}


@app.get("/api/cases/{case_id}/candidate-bundle")
def candidate_bundle(case_id: str) -> dict:
    bundle = build_candidate_bundle(case_id)
    plan = latest_plan(case_id)
    return {
        **bundle,
        "repair_plan": plan,
        "allow_restore": bool(plan and plan["allow_restore"]),
    }


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    region_id: str
    decision: str
    selected_text: str = ""
    selected_source: str = ""
    candidate_id: str = ""
    citation_hash: str = ""
    note: str = ""


class DecisionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisions: list[Decision]
    reviewer: str = ""
    candidate_bundle_sha256: str
    request_id: str


@app.post("/api/cases/{case_id}/decisions")
def decisions(case_id: str, body: DecisionBatch, request: Request) -> dict:
    case = get_case(case_id)
    allowed = {"accept", "manual_modify", "reject", "abstain"}
    if not body.reviewer.strip():
        raise HTTPException(422, "Reviewer identity is required")
    bundle = build_candidate_bundle(case_id)
    if body.candidate_bundle_sha256 != bundle["candidate_bundle_sha256"]:
        raise HTTPException(409, "CANDIDATE_BUNDLE_VERSION_CONFLICT")
    with store.connect() as db:
        previous_rows = db.execute(
            """
            SELECT plan_id,allow_restore,plan_json FROM repair_plans
            WHERE case_id=? ORDER BY created_at DESC
            """,
            (case_id,),
        ).fetchall()
    for previous in previous_rows:
        previous_plan = json.loads(previous["plan_json"])
        if previous_plan.get("request_id") == body.request_id:
            if previous_plan.get("request_payload_hash") != canonical_hash(
                body.model_dump()
            ):
                raise HTTPException(409, "REQUEST_ID_REUSE_CONFLICT")
            return {
                "case_id": case_id,
                "plan_id": previous["plan_id"],
                "allow_restore": bool(previous["allow_restore"]),
                "plan_sha256": previous_plan["plan_sha256"],
                "idempotent": True,
            }
    with store.transaction() as db:
        region_rows = list(
            db.execute("SELECT region_id FROM damage_regions WHERE case_id=?", (case_id,))
        )
        region_ids = {row["region_id"] for row in region_rows}
        if not region_ids:
            raise HTTPException(409, "No analyzed damage regions")
        submitted = {item.region_id for item in body.decisions}
        if submitted != region_ids or len(submitted) != len(body.decisions):
            raise HTTPException(422, "Exactly one decision is required for every region")
        normalized = []
        for item in body.decisions:
            if item.decision not in allowed:
                raise HTTPException(422, f"Invalid decision: {item.decision}")
            if item.decision in {"accept", "manual_modify"} and not item.selected_text:
                raise HTTPException(422, "Accepted/modified decisions require text")
            if (
                item.decision in {"accept", "manual_modify"}
                and len(item.selected_text.strip()) != 1
            ):
                raise HTTPException(
                    422, "Accepted/modified decisions require exactly one character"
                )
            if item.decision == "manual_modify" and not item.note.strip():
                raise HTTPException(422, "Manual modification requires review note")
            if item.decision in {"reject", "abstain"} and item.selected_text:
                raise HTTPException(422, "Rejected/abstained decisions cannot supply text")
            if item.decision == "accept":
                candidates = [
                    candidate
                    for region in bundle["regions"]
                    if region["region_id"] == item.region_id
                    for query in region["rag_queries"]
                    for candidate in query["candidates"]
                ]
                selected = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate["candidate_id"] == item.candidate_id
                    ),
                    None,
                )
                if selected is None:
                    raise HTTPException(422, "Accepted candidate_id is not in bundle")
                if (
                    selected["candidate_text"] != item.selected_text
                    or selected["citation_hash"] != item.citation_hash
                ):
                    raise HTTPException(422, "Accepted candidate evidence mismatch")
            decision_id = str(uuid.uuid4())
            db.execute(
                "INSERT INTO human_decisions VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    decision_id, case_id, item.region_id, item.decision,
                    item.selected_text, item.selected_source, item.note,
                    body.reviewer.strip(), utcnow(),
                ),
            )
            normalized.append(item.model_dump())
        approved = [
            item for item in normalized
            if item["decision"] in {"accept", "manual_modify"}
        ]
        # A constrained run is valid with one or more approved regions while
        # explicit reject/abstain decisions remain immutable negative controls.
        allow = bool(approved) and all(
            bool(item["selected_text"]) for item in approved
        )
        plan_id = str(uuid.uuid4())
        plan_version = len(previous_rows) + 1
        plan_core = {
            "case_id": case_id,
            "image_sha256": case["input_sha256"],
            "analysis_run_id": bundle["analysis_run_id"],
            "candidate_bundle_sha256": bundle["candidate_bundle_sha256"],
            "rag_version": bundle["rag_version"],
            "cbeta_source_commit": bundle["cbeta_source_commit"],
            "plan_version": plan_version,
            "approved_regions": approved,
            "rejected_regions": [
                item for item in normalized if item["decision"] == "reject"
            ],
            "abstained_regions": [
                item for item in normalized if item["decision"] == "abstain"
            ],
            "manual_edits": [
                item for item in normalized if item["decision"] == "manual_modify"
            ],
            "evidence_citations": [
                {
                    "region_id": item["region_id"],
                    "candidate_id": item["candidate_id"],
                    "citation_hash": item["citation_hash"],
                }
                for item in approved
                if item["candidate_id"]
            ],
            "created_at": utcnow(),
            "confirmed_at": utcnow(),
            "actor": body.reviewer.strip(),
            "request_id": body.request_id,
            "request_payload_hash": canonical_hash(body.model_dump()),
            "allow_restore": allow,
        }
        plan_payload = {**plan_core, "plan_sha256": canonical_hash(plan_core)}
        db.execute(
            "INSERT INTO repair_plans VALUES(?,?,?,?,?,?)",
            (
                plan_id, case_id, "PLAN_READY" if allow else "PLAN_BLOCKED",
                int(allow), json.dumps(plan_payload, ensure_ascii=False), utcnow(),
            ),
        )
        plan_dir = CASE_ROOT / case_id / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / f"confirmed_repair_plan.v{plan_version}.json").write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (plan_dir / "confirmed_repair_plan.json").write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        set_status(
            db, case_id, "PLAN_READY" if allow else "HUMAN_REVIEW_PENDING",
            request, "human_decisions_recorded",
            {"plan_id": plan_id, "allow_restore": allow},
        )
    return {
        "case_id": case_id,
        "plan_id": plan_id,
        "allow_restore": allow,
        "plan_sha256": plan_payload["plan_sha256"],
        "idempotent": False,
    }


@app.get("/api/cases/{case_id}/repair-plan")
def repair_plan(case_id: str) -> dict:
    get_case(case_id)
    plan = latest_plan(case_id)
    if plan is None:
        return {"case_id": case_id, "status": "NOT_READY", "allow_restore": False}
    return plan


@app.post("/api/cases/{case_id}/restore")
def restore(case_id: str, request: Request) -> dict:
    get_case(case_id)
    plan = latest_plan(case_id)
    if not plan or not plan["allow_restore"]:
        raise HTTPException(409, "Server repair plan does not allow restore")
    current_bundle = build_candidate_bundle(case_id)
    if (
        plan["plan"].get("candidate_bundle_sha256")
        != current_bundle["candidate_bundle_sha256"]
    ):
        raise HTTPException(409, "CANDIDATE_BUNDLE_VERSION_CONFLICT")
    if not gpu_status()["available"]:
        raise HTTPException(503, "GPU_REQUIRED")
    with store.connect() as db:
        active = db.execute(
            """
            SELECT job_id,case_id FROM restore_jobs
            WHERE status='RUNNING' ORDER BY created_at LIMIT 1
            """
        ).fetchone()
        previous = db.execute(
            """
            SELECT * FROM restore_jobs
            WHERE case_id=? AND plan_id=? AND status='COMPLETE'
            ORDER BY created_at DESC LIMIT 1
            """,
            (case_id, plan["plan_id"]),
        ).fetchone()
        upload = db.execute(
            """
            SELECT relative_path FROM uploaded_files
            WHERE case_id=? ORDER BY created_at LIMIT 1
            """,
            (case_id,),
        ).fetchone()
    if previous is not None:
        return {**row_dict(previous), "idempotent": True}
    if active is not None:
        raise HTTPException(
            409, f"GPU_RESTORE_ALREADY_RUNNING:{active['case_id']}:{active['job_id']}"
        )
    if upload is None:
        raise HTTPException(409, "CASE_INPUT_NOT_FOUND")

    input_path = (CASE_ROOT / upload["relative_path"]).resolve()
    try:
        input_path.relative_to(CASE_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(500, "Stored input path escaped case root") from exc
    if not input_path.is_file():
        raise HTTPException(409, "CASE_INPUT_FILE_MISSING")

    job_id = str(uuid.uuid4())
    now = utcnow()
    case_dir = CASE_ROOT / case_id
    request_dir = case_dir / "restore_requests" / job_id
    log_dir = case_dir / "logs"
    request_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = request_dir / "candidate_bundle.json"
    plan_path = request_dir / "confirmed_repair_plan.json"
    worker_bundle = {
        **current_bundle,
        "damage_regions": current_bundle["regions"],
    }
    bundle_path.write_text(
        json.dumps(worker_bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plan_path.write_text(
        json.dumps(plan["plan"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    stdout_path = log_dir / f"restore-{job_id}.stdout.log"
    stderr_path = log_dir / f"restore-{job_id}.stderr.log"
    detail = {
        "execution": "QUEUED",
        "phase": "QUEUED",
        "plan_sha256": plan["plan"]["plan_sha256"],
        "candidate_bundle_sha256": current_bundle["candidate_bundle_sha256"],
        "bundle_snapshot": str(bundle_path.relative_to(case_dir)),
        "plan_snapshot": str(plan_path.relative_to(case_dir)),
        "stdout_path": str(stdout_path.relative_to(case_dir)),
        "stderr_path": str(stderr_path.relative_to(case_dir)),
        "human_review_status": "not_reviewed",
    }
    phase_path = case_dir / "restore" / "phase.json"
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    phase_path.write_text(
        json.dumps(
            {"phase": "QUEUED", "updated_at": now},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with store.transaction() as db:
        db.execute(
            "INSERT INTO restore_jobs VALUES(?,?,?,?,?,?,?)",
            (
                job_id, case_id, plan["plan_id"], "RUNNING",
                json.dumps(detail, ensure_ascii=False), now, now,
            ),
        )
        set_status(
            db, case_id, "RESTORE_RUNNING", request, "restore_requested", detail
        )
    command = [
        sys.executable,
        "-m",
        "adapters.autohdr_official.restore_job",
        "--case-id",
        case_id,
        "--job-id",
        job_id,
        "--input",
        str(input_path),
        "--bundle",
        str(bundle_path),
        "--plan",
        str(plan_path),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        worker = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    return {
        "job_id": job_id,
        "case_id": case_id,
        **detail,
        "worker_pid": worker.pid,
        "idempotent": False,
    }


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict:
    with store.connect() as db:
        row = db.execute(
            "SELECT * FROM restore_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Job not found")
    payload = row_dict(row)
    phase_path = CASE_ROOT / payload["case_id"] / "restore" / "phase.json"
    if phase_path.is_file():
        try:
            phase = json.loads(phase_path.read_text(encoding="utf-8"))
            payload["phase"] = phase.get("phase", payload["status"])
            payload["phase_updated_at"] = phase.get("updated_at", "")
            payload["phase_detail"] = phase.get("detail", "")
        except (OSError, ValueError, TypeError):
            payload["phase"] = payload["status"]
    else:
        payload["phase"] = payload["status"]
    return payload


@app.get("/api/cases/{case_id}/result")
def result(case_id: str) -> dict:
    get_case(case_id)
    with store.connect() as db:
        outputs = [
            row_dict(row) for row in db.execute(
                "SELECT * FROM outputs WHERE case_id=? ORDER BY created_at",
                (case_id,),
            )
        ]
    for output in outputs:
        output["url"] = (
            f"/api/cases/{case_id}/outputs/{output['output_id']}"
        )
    return {"case_id": case_id, "outputs": outputs, "human_review_status": "not_reviewed"}


@app.get("/api/cases/{case_id}/outputs/{output_id}")
def output_file(case_id: str, output_id: str):
    get_case(case_id)
    with store.connect() as db:
        row = db.execute(
            """
            SELECT * FROM outputs WHERE case_id=? AND output_id=?
            """,
            (case_id, output_id),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Output not found")
    root = CASE_ROOT.resolve()
    path = (CASE_ROOT / row["relative_path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(403, "Output path escaped case root") from exc
    if not path.is_file():
        raise HTTPException(404, "Output file unavailable")
    media_types = {
        ".png": "image/png",
        ".json": "application/json",
    }
    return FileResponse(
        path,
        media_type=media_types.get(path.suffix.lower(), "application/octet-stream"),
        filename=path.name,
    )


@app.get("/api/cases/{case_id}/files/{file_id}")
def case_file(case_id: str, file_id: str):
    get_case(case_id)
    with store.connect() as db:
        row = db.execute(
            "SELECT * FROM uploaded_files WHERE case_id=? AND file_id=?",
            (case_id, file_id),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "File not found")
    root = CASE_ROOT.resolve()
    path = (CASE_ROOT / row["relative_path"]).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404, "File unavailable")
    return FileResponse(path, media_type=row["mime_type"], filename=row["server_name"])


def _rejoin_dir(job_id: str) -> Path:
    try:
        parsed = str(uuid.UUID(job_id))
    except ValueError as exc:
        raise HTTPException(404, "Rejoin job not found") from exc
    path = (REJOIN_ROOT / parsed).resolve()
    if path.parent != REJOIN_ROOT.resolve() or not path.is_dir():
        raise HTTPException(404, "Rejoin job not found")
    return path


def _read_rejoin_job(job_id: str) -> tuple[Path, dict]:
    directory = _rejoin_dir(job_id)
    path = directory / "job.json"
    if not path.is_file():
        raise HTTPException(404, "Rejoin job metadata not found")
    return directory, json.loads(path.read_text(encoding="utf-8"))


async def _read_rejoin_image(upload: UploadFile, label: str) -> tuple[bytes, str, int, int]:
    if upload.content_type not in ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported {label} MIME type")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{label} is empty or exceeds upload limit")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = (probe.format or "").upper()
            width, height = probe.size
            probe.verify()
        if image_format not in ALLOWED_FORMAT or width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise HTTPException(415, f"Unsupported {label} image")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"{label} decode failed: {type(exc).__name__}") from exc
    return data, image_format, width, height


@app.post("/api/rejoin")
async def create_rejoin_job(request: Request, fragment_a: UploadFile = File(...), fragment_b: UploadFile = File(...)) -> dict:
    if not gpu_status()["available"]:
        raise HTTPException(503, "GPU_REQUIRED")
    a_data, a_format, a_width, a_height = await _read_rejoin_image(fragment_a, "fragment_a")
    b_data, b_format, b_width, b_height = await _read_rejoin_image(fragment_b, "fragment_b")
    job_id = str(uuid.uuid4())
    directory = REJOIN_ROOT / job_id
    (directory / "input").mkdir(parents=True, exist_ok=False)
    a_path, b_path = directory / "input/fragment_a", directory / "input/fragment_b"
    a_path.write_bytes(a_data); b_path.write_bytes(b_data)
    payload = {"job_id": job_id, "status": "QUEUED", "phase": "QUEUED", "progress": 0, "created_at": utcnow(),
        "inputs": {"fragment_a": {"sha256": hashlib.sha256(a_data).hexdigest(), "format": a_format, "dimensions": [a_width, a_height]},
                   "fragment_b": {"sha256": hashlib.sha256(b_data).hexdigest(), "format": b_format, "dimensions": [b_width, b_height]}}}
    job_path = directory / "job.json"
    job_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    environment = os.environ.copy(); environment.update({"DUNHUANG_BASE": str(BASE), "AUTOHDR_PYTHON": str(AUTOHDR_PYTHON), "PYTHONDONTWRITEBYTECODE": "1"})
    worker = subprocess.Popen([str(AUTOHDR_PYTHON), "-B", "-m", "dunhuang_backend.rejoin_worker", "--job", str(job_path)], cwd=repo_root, env=environment, stdin=subprocess.DEVNULL, start_new_session=True)
    payload["worker_pid"] = worker.pid
    return payload


@app.get("/api/rejoin/{job_id}")
def rejoin_job_status(job_id: str) -> dict:
    _, job = _read_rejoin_job(job_id)
    return job


@app.get("/api/rejoin/{job_id}/result")
def rejoin_result(job_id: str):
    directory, job = _read_rejoin_job(job_id)
    if job.get("status") != "COMPLETE":
        raise HTTPException(409, "REJOIN_RESULT_NOT_READY")
    path = directory / "output/stitched.png"
    if not path.is_file():
        raise HTTPException(500, "REJOIN_RESULT_MISSING")
    return FileResponse(path, media_type="image/png", filename=f"{job_id}-stitched.png")


@app.get("/api/rejoin/{job_id}/download")
def rejoin_download(job_id: str):
    return rejoin_result(job_id)
