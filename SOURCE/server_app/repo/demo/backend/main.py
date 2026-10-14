"""Gurou AutoHDR + CBETA RAG — Demo Backend."""
import base64, cv2, hashlib, io, json, os, re, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import zipfile as zip_mod

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
REAL_CASE_REGISTRY = PROJECT_ROOT / "data" / "real_cases" / "registry" / "real_case_registry.json"
AUTOHDR_BACKEND = os.environ.get("AUTOHDR_BACKEND", "mock")
LEGACY = os.environ.get("ENABLE_LEGACY_DIFFHDR", "false").lower() == "true"
DEBUG_ROUTES = os.environ.get("ENABLE_LEGACY_DEBUG", "false").lower() == "true"
MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_MIME = {"image/png", "image/jpeg", "image/tiff", "image/tif"}

app = FastAPI(title="Gurou AutoHDR + CBETA RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

# Lazy legacy pipeline
_pipeline = None
def _get_pipeline():
    global _pipeline
    if _pipeline is None and LEGACY:
        from pipelines.full_pipeline import FullPipeline
        _pipeline = FullPipeline(device="auto")
    return _pipeline


# ── Health ──
@app.get("/api/system/health")
async def api_health():
    return {"status":"ok","autohdr_backend":AUTOHDR_BACKEND,"is_mock":AUTOHDR_BACKEND=="mock","legacy":LEGACY}


# ── Image validation ──
def _validate_image(data: bytes) -> tuple[bytes, str, int, int]:
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large (max 20MB)")
    # Verify with cv2
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not a valid image")
    return data, hashlib.sha256(data).hexdigest(), img.shape[1], img.shape[0]


# ── Job Store ──
from core.job_store import JobStore, RUNS_ROOT
from core.state_machine import JobStatus, transition as _transition


def _real_cases() -> list[dict]:
    if not REAL_CASE_REGISTRY.exists():
        return []
    records = json.loads(REAL_CASE_REGISTRY.read_text(encoding="utf-8"))
    downloaded_groups = {
        row["group_id"] for row in records if row["source_type"] == "downloaded"
    }
    usable = [
        row for row in records
        if row.get("needs_manual_transcription")
        and not (
            row["source_type"] == "local_iiif"
            and row["group_id"].split("_", 1)[-1] in downloaded_groups
        )
        and (PROJECT_ROOT / row["local_image_path"]).is_file()
    ]
    return sorted(
        usable,
        key=lambda row: (
            not row.get("case05_candidate", False),
            row.get("li_shengduo_evidence_level") != "confirmed",
            row.get("source_type") != "downloaded",
            row["case_id"],
        ),
    )


def _real_case(case_id: str) -> dict:
    for row in _real_cases():
        if row["case_id"] == case_id:
            return row
    raise HTTPException(404, "Real case not found")


def _public_real_case(row: dict) -> dict:
    path = PROJECT_ROOT / row["local_image_path"]
    return {
        key: row.get(key)
        for key in (
            "case_id",
            "source_type",
            "institution",
            "collection",
            "shelfmark",
            "group_id",
            "suitable_for_rejoin",
            "suitable_for_autohdr",
            "li_shengduo_evidence_level",
            "cbeta_relation_status",
            "case05_candidate",
            "needs_manual_transcription",
            "needs_expert_review",
            "download_source_url",
            "image_status",
            "notes",
        )
    } | {
        "image_url": f"/api/real-cases/{row['case_id']}/image",
        "file_size": path.stat().st_size,
        "upload_ready": (
            row.get("image_status", "available") == "available"
            and path.stat().st_size <= MAX_FILE_SIZE
        ),
        "transcription_status": "not_started",
        "official_autohdr_status": "OFFICIAL_STUB",
    }


def _create_job_from_bytes(
    data: bytes,
    filename: str,
    source_metadata: dict | None = None,
) -> dict:
    _, sha, w, h = _validate_image(data)
    job = JobStore.create_job(image_name=filename or "upload")
    run_dir = RUNS_ROOT / job["run_id"] / "input"
    run_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename or "png").suffix or ".png"
    orig_path = run_dir / f"original{ext}"
    input_path = run_dir / "input.png"
    orig_path.write_bytes(data)
    shutil.copyfile(orig_path, input_path)
    updates = {
        "input_sha256": sha,
        "width": w,
        "height": h,
        "original_filename": filename,
        "file_size": len(data),
        "input_path": str(input_path),
    }
    if source_metadata:
        updates["real_case"] = source_metadata
    JobStore.update_job(job["run_id"], updates)
    return JobStore.get_job(job["run_id"])


# ── Jobs ──
@app.get("/api/real-cases")
async def api_real_cases(limit: int = 12):
    safe_limit = max(1, min(limit, 50))
    cases = [_public_real_case(row) for row in _real_cases()[:safe_limit]]
    return JSONResponse(
        {
            "cases": cases,
            "count": len(cases),
            "registry_status": "real_sources_with_manual_review_pending",
            "official_autohdr_status": "OFFICIAL_STUB",
        }
    )


@app.get("/api/real-cases/{case_id}/image")
async def api_real_case_image(case_id: str):
    row = _real_case(case_id)
    path = (PROJECT_ROOT / row["local_image_path"]).resolve()
    if PROJECT_ROOT.resolve() not in path.parents:
        raise HTTPException(400, "Invalid real-case path")
    return FileResponse(str(path))


@app.post("/api/real-cases/{case_id}/jobs")
async def api_create_real_case_job(case_id: str):
    row = _real_case(case_id)
    path = PROJECT_ROOT / row["local_image_path"]
    public = _public_real_case(row)
    job = _create_job_from_bytes(path.read_bytes(), path.name, public)
    return JSONResponse(job)


@app.post("/api/jobs")
async def api_create_job(image: UploadFile = File(...)):
    if image.content_type and image.content_type not in ALLOWED_MIME:
        raise HTTPException(400, f"Unsupported type: {image.content_type}")
    data = await image.read()
    return JSONResponse(
        _create_job_from_bytes(data, image.filename or "upload")
    )


@app.get("/api/jobs/{run_id}")
async def api_get_job(run_id: str):
    j = JobStore.get_job(run_id)
    if j is None: raise HTTPException(404, "Job not found")
    return JSONResponse(j)


@app.post("/api/jobs/{run_id}/analyze")
async def api_analyze(run_id: str):
    j = JobStore.get_job(run_id)
    if j is None: raise HTTPException(404, "Job not found")
    try:
        _transition(JobStatus(j["status"]), JobStatus.ANALYZING)
    except ValueError:
        raise HTTPException(409, f"Cannot transition from {j['status']} to ANALYZING")
    JobStore.transition_status(run_id, JobStatus.ANALYZING)
    # Read image for hash record
    img_path = j.get("input_path", "")
    img_data = open(img_path, "rb").read() if img_path and Path(img_path).exists() else None
    from integrations.autohdr.base import detect
    r = detect(img_data)
    if img_data:
        overlay = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if overlay is not None:
            for damage in r.get("damages", []):
                x, y, w, h = [int(v) for v in damage.get("bbox", [0, 0, 0, 0])]
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), max(2, min(overlay.shape[:2]) // 250))
            overlay_path = RUNS_ROOT / run_id / "analysis" / "overlay.png"
            ok, encoded = cv2.imencode(".png", overlay)
            if ok:
                encoded.tofile(overlay_path)
                r["overlay_url"] = f"/api/jobs/{run_id}/artifacts/overlay.png"
    updates = {"damages": r.get("damages", []), "ocr_chars": r.get("ocr_chars", []),
               "input_sha256": hashlib.sha256(img_data).hexdigest() if img_data else j.get("input_sha256",""),
               "overlay_url": r.get("overlay_url")}
    JobStore.update_job(run_id, updates)
    _transition(JobStatus.ANALYZING, JobStatus.NEEDS_REVIEW)
    JobStore.transition_status(run_id, JobStatus.NEEDS_REVIEW)
    return JSONResponse({**r, "run_id": run_id})


@app.post("/api/jobs/{run_id}/evidence")
async def api_evidence(run_id: str):
    j = JobStore.get_job(run_id)
    if j is None: raise HTTPException(404, "Job not found")
    damages = j.get("damages", [])
    from integrations.cbeta_evidence_fusion import fuse_batch
    from integrations.cbeta_evidence_fusion import get_agent as _get_rag
    fs = fuse_batch(damages)
    fusions_dict = [f.to_dict() if hasattr(f, 'to_dict') else f for f in fs]

    # Save evidence
    ev_dir = RUNS_ROOT / run_id / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    evidence_doc = {"fusions": fusions_dict, "timestamp": datetime.now(timezone.utc).isoformat()}
    (ev_dir / "evidence.json").write_text(json.dumps(evidence_doc, indent=2))
    JobStore.update_job(run_id, {"fusions": fusions_dict, "evidence_timestamp": evidence_doc["timestamp"]})

    return JSONResponse({
        "fusions": fusions_dict,
        "citations": [
            candidate.get("citation", "")
            for fusion in fusions_dict
            for candidate in fusion.get("cbeta_candidates", [])
            if candidate.get("citation")
        ],
    })


class RagQueryRequest(BaseModel):
    context_before: str = ""
    context_after: str = ""
    sutra_ids: list[str] | None = None
    translators: list[str] | None = None
    editions: list[str] | None = None
    max_missing: int | None = 10
    expected_missing_length: int | None = None
    top_k: int = 5
    normalization_profile: str = "cbeta_rag_v2"


@app.get("/api/rag/health")
async def api_rag_health():
    from services.rag_v2_service import rag_v2_health

    return JSONResponse(rag_v2_health())


@app.get("/api/rag/stats")
async def api_rag_stats():
    from services.rag_v2_service import get_rag_v2

    retriever = get_rag_v2()
    if retriever is None:
        raise HTTPException(503, "RAG V2 index is not available")
    return JSONResponse({**retriever.health(), "weights": retriever.weights})


@app.post("/api/rag/query")
async def api_rag_query(request: RagQueryRequest):
    from services.rag_v2_service import get_rag_v2

    retriever = get_rag_v2()
    if retriever is None:
        raise HTTPException(503, "RAG V2 index is not available")
    query_options = {
        "sutra_ids": request.sutra_ids,
        "translators": request.translators,
        "max_missing": request.max_missing,
        "expected_missing_length": request.expected_missing_length,
        "top_k": max(1, min(request.top_k, 20)),
        "normalization_profile": request.normalization_profile,
    }
    if getattr(retriever, "supports_editions", False):
        query_options["editions"] = request.editions
    result = retriever.query(
        request.context_before,
        request.context_after,
        **query_options,
    )
    return JSONResponse(result.to_dict())


# Confirm validation
VALID_SOURCES = {"autohdr", "cbeta", "qwen", "manual", "abstain"}

class ConfirmDecisionItem(BaseModel):
    damage_id: str
    selected_text: str = ""
    selected_source: str = "abstain"
    status: str = "abstain"
    reviewer_note: str = ""

class ConfirmBatchRequest(BaseModel):
    decisions: list[ConfirmDecisionItem] = []

@app.post("/api/jobs/{run_id}/confirm")
async def api_confirm_job(run_id: str, req: ConfirmBatchRequest):
    j = JobStore.get_job(run_id)
    if j is None: raise HTTPException(404, "Job not found")
    if JobStatus(j["status"]) is not JobStatus.NEEDS_REVIEW:
        raise HTTPException(409, f"Cannot confirm from {j['status']}")

    damages = j.get("damages", [])
    damage_ids = {d["damage_id"]: d for d in damages}
    damage_set = set(damage_ids.keys())
    seen = set()

    confirmed = []
    for d in req.decisions:
        if d.damage_id not in damage_set:
            raise HTTPException(400, f"Unknown damage_id: {d.damage_id}")
        if d.damage_id in seen:
            raise HTTPException(400, f"Duplicate damage_id: {d.damage_id}")
        seen.add(d.damage_id)
        if d.selected_source not in VALID_SOURCES:
            raise HTTPException(400, f"Invalid source: {d.selected_source}")

        dm = damage_ids[d.damage_id]

        if d.status == "manual_confirmed":
            if not d.selected_text:
                raise HTTPException(400, f"{d.damage_id}: manual_confirmed requires text")
            # Validate source-specific
            if d.selected_source == "autohdr":
                candidates = [c["text"] for c in dm.get("autohdr_candidates", [])]
                if d.selected_text not in candidates:
                    raise HTTPException(400, f"{d.damage_id}: text not in AutoHDR candidates")
            confirmed.append(d.model_dump())
        elif d.status == "abstain":
            if d.selected_text:
                raise HTTPException(400, f"{d.damage_id}: abstain must have empty text")
        else:
            raise HTTPException(400, f"Invalid status: {d.status}")

    st = JobStatus.CONFIRMED if confirmed else JobStatus.ABSTAINED
    JobStore.transition_status(run_id, st)
    JobStore.update_job(run_id, {
        "confirmed": confirmed,
        "evidence/confirm/decisions.json": json.dumps(confirmed, ensure_ascii=False, indent=2)
    })
    return JSONResponse({"confirmed_count": len(confirmed), "status": st.value})


@app.post("/api/jobs/{run_id}/restore")
async def api_restore_job(run_id: str):
    j = JobStore.get_job(run_id)
    if j is None: raise HTTPException(404, "Job not found")
    conf = j.get("confirmed", [])
    status = JobStatus(j["status"])
    if not conf:
        if status in {JobStatus.NEEDS_REVIEW, JobStatus.ABSTAINED}:
            return JSONResponse({"status": "abstain", "message": "No manual_confirmed decisions"})
        raise HTTPException(409, f"Cannot restore from {j['status']}")

    # Server-side re-validation
    damages = j.get("damages", [])
    damage_ids = {d["damage_id"]: d for d in damages}
    for d in conf:
        if d["damage_id"] not in damage_ids:
            raise HTTPException(400, f"Unknown damage_id: {d['damage_id']}")
    try:
        _transition(JobStatus(j["status"]), JobStatus.RESTORING)
    except ValueError:
        raise HTTPException(409, f"Cannot transition from {j['status']}")

    JobStore.transition_status(run_id, JobStatus.RESTORING)
    from integrations.autohdr.base import restore, is_mock
    input_path = j.get("input_path", "")
    img_data = open(input_path, "rb").read() if input_path and Path(input_path).exists() else None
    r = restore(img_data, conf)
    r["is_mock"] = is_mock()
    r["status"] = "completed"
    artifact_name = "restored_mock.png"
    if is_mock():
        _generate_mock_restored(run_id, j)
    else:
        restored_path = Path(r.get("restored_path") or "")
        if not restored_path.is_file():
            raise HTTPException(503, "Restoration backend returned no result image")
        artifact_name = "restored.png"
        artifact_path = RUNS_ROOT / run_id / "restoration" / artifact_name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(restored_path, artifact_path)
        r["restored_path"] = str(artifact_path)
    r["restored_url"] = f"/api/jobs/{run_id}/artifacts/{artifact_name}"
    r["original_url"] = f"/api/jobs/{run_id}/artifacts/input.png"
    r["overlay_url"] = j.get("overlay_url")

    JobStore.transition_status(run_id, JobStatus.COMPLETED)
    JobStore.update_job(run_id, {
        "restored_path": r.get("restored_path", ""),
        "restored_url": r["restored_url"],
        "original_url": r["original_url"],
        "overlay_url": r.get("overlay_url"),
        "is_replay": bool(r.get("is_replay")),
    })
    return JSONResponse(r)


def _generate_mock_restored(run_id: str, job: dict):
    """Generate a mock restoration image with watermark."""
    input_path = job.get("input_path", "")
    img = None
    if input_path and Path(input_path).exists():
        img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = np.full((256, 256, 3), (240, 235, 220), dtype=np.uint8)
    # Add watermark
    cv2.putText(img, "MOCK RESTORATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "NOT AUTOHDR OUTPUT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    out_dir = RUNS_ROOT / run_id / "restoration"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "Failed to encode mock restoration image")
    encoded.tofile(out_dir / "restored_mock.png")


# ── Legacy routes (conditional) ──
if LEGACY:
    def _b64(img):
        if img is None:
            return None
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return None
        return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")

    @app.post("/api/rejoin")
    async def api_rejoin(img1: UploadFile = File(...), img2: UploadFile = File(...)):
        p = _get_pipeline()
        if p is None: raise HTTPException(503, "Legacy pipeline not available")
        raw1 = await img1.read()
        raw2 = await img2.read()
        arr1 = np.frombuffer(raw1, np.uint8)
        arr2 = np.frombuffer(raw2, np.uint8)
        bgr1 = cv2.imdecode(arr1, cv2.IMREAD_COLOR)
        bgr2 = cv2.imdecode(arr2, cv2.IMREAD_COLOR)
        if bgr1 is None or bgr2 is None: raise HTTPException(400, "Image decode failed")
        r = p.rejoin_only(bgr1, bgr2)
        pano = r.get("panorama")
        return {
            "rejoin": r["rejoin"],
            "images": {
                "fragment_a": _b64(bgr1),
                "fragment_b": _b64(bgr2),
                "panorama": _b64(pano),
            },
        }

    @app.post("/api/restore")
    async def api_restore_legacy(img1: UploadFile = File(...), img2: UploadFile = File(...), mask: UploadFile = File(...)):
        p = _get_pipeline()
        if p is None: raise HTTPException(503, "Legacy pipeline not available")
        raw1 = await img1.read()
        raw2 = await img2.read()
        bgr1 = cv2.imdecode(np.frombuffer(raw1, np.uint8), cv2.IMREAD_COLOR)
        bgr2 = cv2.imdecode(np.frombuffer(raw2, np.uint8), cv2.IMREAD_COLOR)
        from utils.mask_gen import mask_has_damage
        raw = await mask.read()
        user_mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE) if raw else None
        if user_mask is None or not mask_has_damage(user_mask):
            raise HTTPException(400, "Invalid mask")
        r = p.restore_only(bgr1, bgr2, user_mask)
        return {
            "mask_area_ratio": r["mask_area_ratio"],
            "content_source": r.get("content_source"),
            "images": {
                "before": _b64(r.get("panorama")),
                "after": _b64(r.get("repaired")),
                "mask": _b64(r.get("mask")),
            },
        }


# Debug routes (conditional)
if DEBUG_ROUTES:
    @app.post("/api/confirm")
    async def api_confirm_debug(req: dict):
        return JSONResponse({"status": "debug", "note": "Use /api/jobs/{id}/confirm for production"})

    @app.post("/api/restore-with-rag")
    async def api_restore_with_rag_debug(req: dict):
        return JSONResponse({"status": "debug", "note": "Use /api/jobs/{id}/restore for production"})



@app.get("/api/jobs/{run_id}/artifacts/{artifact_name:path}")
async def api_artifact(run_id: str, artifact_name: str):
    """Serve registered artifacts only."""
    j = JobStore.get_job(run_id)
    if j is None: raise HTTPException(404)
    name = Path(artifact_name).name
    for sub in ["analysis", "restoration", "input"]:
        p = RUNS_ROOT / run_id / sub / name
        if p.exists(): return FileResponse(str(p))
    raise HTTPException(404, "Artifact not found")

@app.get("/api/jobs/{run_id}/report")
async def api_report(run_id: str):
    j = JobStore.get_job(run_id)
    if j is None: raise HTTPException(404)
    return JSONResponse(j)

@app.get("/api/jobs/{run_id}/download")
async def api_download(run_id: str):
    j = JobStore.get_job(run_id)
    if j is None: raise HTTPException(404)
    zf_path = RUNS_ROOT / run_id / "task.zip"
    with zip_mod.ZipFile(str(zf_path), 'w', zip_mod.ZIP_DEFLATED) as z:
        for sub in ['input','analysis','evidence','confirmation','restoration']:
            d = RUNS_ROOT / run_id / sub
            if d.exists():
                for f in d.rglob('*'):
                    if f.is_file(): z.write(f, '{}/{}'.format(sub, f.name))
        z.writestr('manifest.json', json.dumps(j, ensure_ascii=False, indent=2))
    return FileResponse(str(zf_path), media_type='application/zip', filename=f'{run_id}.zip')

@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    print(f"Gurou — {AUTOHDR_BACKEND} mode, legacy={LEGACY}, debug={DEBUG_ROUTES}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
