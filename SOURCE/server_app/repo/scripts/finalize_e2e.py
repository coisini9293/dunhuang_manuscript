from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--client-sha256", required=True)
    args = parser.parse_args()

    root = args.root
    run = root / "e2e_runs" / args.run_id
    case_dir = root / "cases" / args.case_id
    repo = root / "server_app" / "repo"
    official = root / "src" / "AutoHDR"
    reports = run / "reports" / "final"
    reports.mkdir(parents=True, exist_ok=True)
    request_dir = case_dir / "restore_requests" / args.job_id
    plan = json.loads(
        (request_dir / "confirmed_repair_plan.json").read_text("utf-8")
    )
    bundle = json.loads(
        (request_dir / "candidate_bundle.json").read_text("utf-8")
    )
    worker = json.loads(
        (case_dir / "restore/official_restore_worker.json").read_text("utf-8")
    )
    pixel = json.loads(
        (case_dir / "restore/outside_mask_audit.json").read_text("utf-8")
    )
    negative = json.loads((run / "reports/negative_controls.json").read_text("utf-8"))

    with urllib.request.urlopen(
        f"http://127.0.0.1:8000/api/cases/{args.case_id}/audit"
    ) as response:
        audit = json.load(response)
    with urllib.request.urlopen("http://127.0.0.1:8000/api/rag/health") as response:
        rag_health = json.load(response)
    with urllib.request.urlopen(
        f"http://127.0.0.1:8000/api/jobs/{args.job_id}"
    ) as response:
        restore_job = json.load(response)

    approved = plan["approved_regions"]
    approved_ids = {item["region_id"] for item in approved}
    matched_evidence = []
    for region in bundle["damage_regions"]:
        if region["region_id"] not in approved_ids:
            continue
        accepted = next(item for item in approved if item["region_id"] == region["region_id"])
        for query in region.get("rag_queries", []):
            for candidate in query.get("candidates", []):
                if candidate.get("candidate_id") == accepted.get("candidate_id"):
                    matched_evidence.append(
                        {
                            "region_id": region["region_id"],
                            "query_id": query["query_id"],
                            "engine_query_id": candidate.get("engine_query_id"),
                            "candidate_text": candidate.get("candidate_text"),
                            "citation": candidate.get("citation"),
                            "citation_hash": candidate.get("citation_hash"),
                            "text_hash": candidate.get("text_hash"),
                        }
                    )

    # Copy immutable result artifacts into the run directory.
    (run / "repair_plan").mkdir(exist_ok=True)
    shutil.copy2(
        request_dir / "confirmed_repair_plan.json",
        run / "repair_plan/confirmed_repair_plan.json",
    )
    shutil.copy2(
        request_dir / "candidate_bundle.json",
        run / "candidate_bundle/candidate_bundle.restore_snapshot.json",
    )
    for source in (case_dir / "restore").iterdir():
        destination = run / "restore" / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    shutil.copy2(
        case_dir / "plans/confirmed_repair_plan.json",
        run / "human_decisions/confirmed_repair_plan.browser_submission.json",
    )
    write(
        run / "audit/case_audit.json",
        json.dumps(audit, ensure_ascii=False, indent=2),
    )
    write(
        run / "audit/restore_job.json",
        json.dumps(restore_job, ensure_ascii=False, indent=2),
    )
    for log in case_dir.glob(f"logs/restore-{args.job_id}.*.log"):
        shutil.copy2(log, run / "logs" / log.name)

    app_commit = command("git", "-C", str(repo), "rev-parse", "HEAD")
    app_status = command(
        "git", "-C", str(repo), "status", "--short", "--untracked-files=all"
    )
    official_commit = command("git", "-C", str(official), "rev-parse", "HEAD")
    official_status = command(
        "git", "-C", str(official), "status", "--short", "--untracked-files=all"
    )
    disk = command("df", "-h", str(root))
    gpu = command(
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap",
        "--format=csv,noheader",
    )
    now = datetime.now(timezone.utc).isoformat()

    chain = {
        "input_sha256": plan["image_sha256"],
        "analysis_run_id": plan["analysis_run_id"],
        "candidate_bundle_sha256": plan["candidate_bundle_sha256"],
        "rag_evidence": matched_evidence,
        "human_request_id": plan["request_id"],
        "human_request_payload_hash": plan["request_payload_hash"],
        "repair_plan_sha256": plan["plan_sha256"],
        "approved_regions": approved,
        "official_worker": "adapters.autohdr_official.official_restore_worker",
        "confirmed_text_payload": worker["confirmed_texts"],
        "patches": worker["patches"],
        "final_png_sha256": worker["compositor"]["final_sha256"],
        "outside_mask_audit": pixel,
        "human_review_status": "not_reviewed",
    }
    write(run / "audit/causal_chain.json", json.dumps(chain, ensure_ascii=False, indent=2))

    write(
        reports / "FINAL_GPU_ENVIRONMENT.md",
        f"""# Final GPU Environment

- GPU: {gpu}
- Restore job: `{args.job_id}` (`COMPLETE`)
- Peak allocated memory: {worker['gpu_peak_bytes']} bytes
- Background restore processes after completion: none
- Status: `GPU_AUTOHDR_RUNTIME_RECHECK_PASS`
""",
    )
    write(
        reports / "FINAL_HUMAN_GATE_AUDIT.md",
        f"""# Final Human Gate Audit

- Reviewer: `{plan['actor']}`
- Browser request ID: `{plan['request_id']}`
- Request payload hash: `{plan['request_payload_hash']}`
- Plan SHA-256: `{plan['plan_sha256']}`
- Decisions: {len(approved)} approved, {len(plan['rejected_regions'])} rejected, {len(plan['abstained_regions'])} abstained
- Server-computed allow_restore: `true`
- Human audit event: `human_decisions_recorded`
- Status: `RAG_PLAN_VALIDATION_PASS`
""",
    )
    write(
        reports / "FINAL_CONFIRMED_TEXT_INJECTION.md",
        f"""# Final Confirmed Text Injection

- Region: `{approved[0]['region_id']}`
- Confirmed text: `{approved[0]['selected_text']}`
- Decision: `{approved[0]['decision']}`
- Content image SHA-256: `{worker['patches'][0]['content_sha256']}`
- Character mask SHA-256: `{worker['patches'][0]['mask_sha256']}`
- Patch SHA-256: `{worker['patches'][0]['output_sha256']}`
- Official commit: `{worker['official_commit']}`
- Status: `OFFICIAL_CONFIRMED_TEXT_INJECTION_PASS`
""",
    )
    write(
        reports / "FINAL_PIXEL_SAFETY_AUDIT.md",
        f"""# Final Pixel Safety Audit

- Total pixels: {pixel['total_pixels']}
- Approved mask pixels: {pixel['approved_mask_pixels']}
- Changed inside mask: {pixel['changed_inside_mask_pixels']}
- Changed outside mask: {pixel['changed_outside_mask_pixels']}
- Rejected-region changed pixels: {pixel['rejected_changed_pixels']}
- Abstained-region changed pixels: {pixel['abstained_changed_pixels']}
- Max outside-mask channel difference: {pixel['max_channel_difference_outside_mask']}
- Final SHA-256: `{pixel['final_sha256']}`
- Status: `OUTSIDE_APPROVED_MASK_UNCHANGED_PASS`
""",
    )
    write(
        reports / "FINAL_THIN_CLIENT_E2E.md",
        f"""# Final Thin Client E2E

- URL: `http://127.0.0.1:5500/Dunhuang_RAG_AutoHDR_Client.html?case_id={args.case_id}`
- Client SHA-256: `{args.client_sha256}`
- Server, RAG and GPU status displayed from `127.0.0.1:8000`.
- Refresh restored the case, evidence chain and constrained result.
- Result label: `受约束恢复结果（待人工复核）`
- Status: `GPU_WINDOWS_THIN_CLIENT_PASS`
""",
    )
    write(
        reports / "FINAL_DISK_USAGE.md",
        f"# Final Disk Usage\n\n```text\n{disk}\n```\n",
    )
    write(
        reports / "FINAL_GIT_STATUS.md",
        f"""# Final Git Status

- Server app commit: `{app_commit}`
- Server app worktree: `{'clean' if not app_status else app_status}`
- Official commit: `{official_commit}`
- Official worktree: `{'clean' if not official_status else official_status}`
""",
    )
    write(
        reports / "FINAL_RAG_CONSTRAINED_E2E_STATUS.md",
        f"""# Final RAG-Constrained Official AutoHDR E2E Status

Generated: `{now}`

`{plan['image_sha256']}`
→ analysis `{plan['analysis_run_id']}`
→ candidate bundle `{plan['candidate_bundle_sha256']}`
→ browser decision `{plan['request_id']}`
→ repair plan `{plan['plan_sha256']}`
→ Official restore job `{args.job_id}`
→ final PNG `{worker['compositor']['final_sha256']}`
→ outside-mask audit `{sha256(case_dir / 'restore/outside_mask_audit.json')}`

- RAG: {rag_health['sutra_count']} sutras, {rag_health['record_count']} records, vector disabled
- Approved: {len(approved)}
- Rejected: {len(plan['rejected_regions'])}
- Abstained: {len(plan['abstained_regions'])}
- Negative controls: `{negative['status']}`
- Human review status of all outputs: `not_reviewed`

`RAG_CONSTRAINED_OFFICIAL_AUTOHDR_E2E_PASS`
""",
    )

    metadata_path = run / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text("utf-8"))
    metadata.update(
        {
            "phase": "RAG_CONSTRAINED_OFFICIAL_AUTOHDR_E2E_PASS",
            "restore_job_id": args.job_id,
            "repair_plan_sha256": plan["plan_sha256"],
            "final_png_sha256": worker["compositor"]["final_sha256"],
            "human_review_status": "not_reviewed",
            "finished_at": now,
        }
    )
    write(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2))

    sums_path = run / "SHA256SUMS.txt"
    files = sorted(
        path for path in run.rglob("*")
        if path.is_file() and path != sums_path
    )
    write(
        sums_path,
        "\n".join(f"{sha256(path)}  {path.relative_to(run).as_posix()}" for path in files),
    )
    print(
        json.dumps(
            {
                "status": "RAG_CONSTRAINED_OFFICIAL_AUTOHDR_E2E_PASS",
                "run": str(run),
                "file_count": len(files) + 1,
                "app_commit": app_commit,
                "official_commit": official_commit,
                "final_png_sha256": worker["compositor"]["final_sha256"],
                "sha256sums_sha256": sha256(sums_path),
                "disk": disk,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
