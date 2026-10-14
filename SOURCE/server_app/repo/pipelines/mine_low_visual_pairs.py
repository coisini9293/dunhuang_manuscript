#!/usr/bin/env python3
"""Reference-free visual scan of archived real manuscript fragments."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visual-threshold", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(args.image_root).resolve()
    images = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"} and "总图" not in path.name)
    groups: dict[Path, list[Path]] = defaultdict(list)
    for path in images:
        groups[path.parent].append(path)
    pairs = [(group, a, b) for group, members in sorted(groups.items()) for a, b in itertools.combinations(sorted(members), 2)]

    from rejoin import RejoinPipeline

    pipeline = RejoinPipeline(args.checkpoint, device=args.device)
    started = time.time()
    results, skipped = [], []
    for index, (group, fragment_a, fragment_b) in enumerate(pairs, 1):
        image_a = cv2.imread(str(fragment_a), cv2.IMREAD_COLOR)
        image_b = cv2.imread(str(fragment_b), cv2.IMREAD_COLOR)
        if image_a is None or image_b is None:
            skipped.append({"fragment_a_path": str(fragment_a), "fragment_b_path": str(fragment_b), "reason": "UNREADABLE_IMAGE"})
            continue
        score, pose = pipeline.predict_pair(image_a, image_b)
        results.append({
            "group_id": str(group.relative_to(root)),
            "fragment_a_path": str(fragment_a),
            "fragment_b_path": str(fragment_b),
            "fragment_a_sha256": sha256(fragment_a),
            "fragment_b_sha256": sha256(fragment_b),
            "visual_score": float(score),
            "visual_threshold": args.visual_threshold,
            "visual_branch_decision": "LOW_VISUAL_ROUTE_TO_TEXT" if score < args.visual_threshold else "VISUAL_MATCH",
            "learned_pose_raw": pose.tolist(),
            "reference_used_for_inference": False,
            "geometry_not_yet_decided": True,
        })
        if index % 25 == 0:
            print(f"scanned {index}/{len(pairs)}", flush=True)

    results.sort(key=lambda item: item["visual_score"])
    low_visual = [item for item in results if item["visual_score"] < args.visual_threshold]
    payload = {
        "run_id": "LOW_VISUAL_PAIR_SEARCH_001",
        "created_at_unix": time.time(),
        "dataset_root": str(root),
        "dataset_provenance": "Science Data Bank true-stitching group; train/val/test labels and reference images were not read",
        "fragment_count": len(images),
        "group_count": len(groups),
        "pair_count": len(pairs),
        "visual_threshold": args.visual_threshold,
        "low_visual_count": len(low_visual),
        "reference_used_for_inference": False,
        "geometry_not_yet_decided": True,
        "results": results,
        "skipped": skipped,
        "runtime_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fragment_count": len(images), "pair_count": len(pairs), "low_visual_count": len(low_visual), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
