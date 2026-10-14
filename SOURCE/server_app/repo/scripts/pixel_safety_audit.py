from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def union_bbox_mask(
    shape: tuple[int, int], regions: list[dict], selected: set[str]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    by_id = {item["region_id"]: item for item in regions}
    for region_id in selected:
        x1, y1, x2, y2 = [int(value) for value in by_id[region_id]["bbox"]]
        mask[y1:y2, x1:x2] = True
    return mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--final", required=True, type=Path)
    parser.add_argument("--approved-mask", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--state-db", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--job-id")
    args = parser.parse_args()

    original = np.asarray(Image.open(args.original).convert("RGB"))
    final = np.asarray(Image.open(args.final).convert("RGB"))
    approved = np.asarray(Image.open(args.approved_mask).convert("L")) > 0
    if original.shape != final.shape or original.shape[:2] != approved.shape:
        raise RuntimeError("PIXEL_AUDIT_DIMENSION_MISMATCH")
    bundle = json.loads(args.bundle.read_text("utf-8"))
    plan = json.loads(args.plan.read_text("utf-8"))
    regions = bundle["damage_regions"]
    rejected_ids = {item["region_id"] for item in plan["rejected_regions"]}
    abstained_ids = {item["region_id"] for item in plan["abstained_regions"]}
    rejected = union_bbox_mask(approved.shape, regions, rejected_ids)
    abstained = union_bbox_mask(approved.shape, regions, abstained_ids)

    channel_difference = np.abs(final.astype(np.int16) - original.astype(np.int16))
    changed = np.any(channel_difference != 0, axis=2)
    outside = ~approved
    report = {
        "status": "PIXEL_SAFETY_AUDIT_PASS",
        "total_pixels": int(approved.size),
        "approved_mask_pixels": int(approved.sum()),
        "changed_inside_mask_pixels": int((changed & approved).sum()),
        "changed_outside_mask_pixels": int((changed & outside).sum()),
        "rejected_changed_pixels": int((changed & rejected).sum()),
        "abstained_changed_pixels": int((changed & abstained).sum()),
        "max_channel_difference_outside_mask": int(
            channel_difference[outside].max(initial=0)
        ),
        "final_is_nonempty": bool(final.size),
        "final_is_all_black": bool(np.all(final == 0)),
        "final_is_all_white": bool(np.all(final == 255)),
        "final_differs_from_input": bool(changed.any()),
        "original_sha256": sha256(args.original),
        "final_sha256": sha256(args.final),
        "approved_mask_sha256": sha256(args.approved_mask),
        "plan_sha256": plan["plan_sha256"],
        "candidate_bundle_sha256": bundle["candidate_bundle_sha256"],
        "human_review_status": "not_reviewed",
    }
    required_zero = (
        report["changed_outside_mask_pixels"],
        report["rejected_changed_pixels"],
        report["abstained_changed_pixels"],
        report["max_channel_difference_outside_mask"],
    )
    if any(required_zero):
        report["status"] = "PIXEL_SAFETY_AUDIT_FAIL"
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.state_db:
        if not args.case_id or not args.job_id:
            raise RuntimeError("DB_SYNC_REQUIRES_CASE_AND_JOB")
        output_hash = sha256(args.output)
        relative_path = str(
            args.output.resolve().relative_to(args.original.resolve().parents[2])
        )
        now = datetime.now(timezone.utc).isoformat()
        detail = json.dumps(
            {
                "audit_file": relative_path,
                "audit_sha256": output_hash,
                "status": report["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        with sqlite3.connect(args.state_db) as db:
            cursor = db.execute(
                """
                UPDATE outputs SET sha256=?
                WHERE case_id=? AND job_id=? AND relative_path=?
                """,
                (output_hash, args.case_id, args.job_id, relative_path),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("OUTPUT_HASH_SYNC_ROW_MISMATCH")
            before = db.execute(
                "SELECT status FROM cases WHERE case_id=?", (args.case_id,)
            ).fetchone()[0]
            db.execute(
                """
                INSERT INTO audit_events(
                    timestamp,case_id,actor_type,action,before_state,after_state,
                    payload_sha256,source_ip,client_version,detail_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    now,
                    args.case_id,
                    "system_worker",
                    "pixel_safety_audit_completed",
                    before,
                    before,
                    hashlib.sha256(detail.encode("utf-8")).hexdigest(),
                    "server-worker",
                    "pixel-safety-audit-1.0.0",
                    detail,
                ),
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"].endswith("_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
